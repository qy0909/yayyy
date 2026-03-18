import json
import os
import re
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client

URL = "https://jtksm.mohr.gov.my/ms/hubungi-kami/alamat-ibu-pejabat-dan-cawangan?field_kategori_target_id=All"
TABLE = os.getenv("LABOUR_OFFICES_TABLE_NAME", "labour_offices")


def load_environment() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    backend_root = os.path.abspath(os.path.join(script_dir, ".."))

    # Load repo-level first, then backend-specific overrides.
    load_dotenv(os.path.join(repo_root, ".env"), override=False)
    load_dotenv(os.path.join(backend_root, ".env"), override=True)


def normalize_space(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def clean_email(value: str) -> str:
    txt = normalize_space(value).lower()
    txt = txt.replace("[at]", "@").replace("[dot]", ".")
    txt = txt.replace(" ", "")
    return txt


def extract_city_state(address: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    states = [
        "Perlis", "Kedah", "Pulau Pinang", "Perak", "Selangor", "Negeri Sembilan",
        "Melaka", "Johor", "Pahang", "Terengganu", "Kelantan", "Kuala Lumpur",
        "Wilayah Persekutuan", "Putrajaya", "Labuan",
    ]

    address_clean = normalize_space(address)
    postcode_match = re.search(r"\b(\d{5})\b", address_clean)
    postcode = postcode_match.group(1) if postcode_match else None

    state = None
    for candidate in states:
        if re.search(rf"\b{re.escape(candidate)}\b", address_clean, flags=re.IGNORECASE):
            state = candidate
            break

    city = None
    # Try to infer city from "<postcode> <city>" pattern
    if postcode:
        city_match = re.search(rf"\b{postcode}\b\s*([^,\.]+)", address_clean)
        if city_match:
            city = normalize_space(city_match.group(1))

    # Fallback city from last comma segments if needed
    if not city:
        parts = [p.strip() for p in re.split(r",", address_clean) if p.strip()]
        if parts:
            city = parts[-1]

    return city, state, postcode


def parse_contact_fields(contact_text: str) -> Dict[str, Optional[str]]:
    txt = normalize_space(contact_text)

    phone_match = re.search(r"Telefon\s*:\s*(.*?)(?:Faksimili|Faksimil|E-mel|$)", txt, flags=re.IGNORECASE)
    fax_match = re.search(r"Faksimili?\s*:\s*(.*?)(?:E-mel|$)", txt, flags=re.IGNORECASE)
    email_match = re.search(r"E-mel\s*:\s*([^\s]+)", txt, flags=re.IGNORECASE)

    phone = normalize_space(phone_match.group(1)) if phone_match else None
    fax = normalize_space(fax_match.group(1)) if fax_match else None
    email = clean_email(email_match.group(1)) if email_match else None

    # Keep fax in open_hours suffix if present and open_hours is not available in source.
    return {
        "phone": phone,
        "fax": fax,
        "email": email,
    }


def parse_rows(html: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.select("div.views-row")

    parsed: List[Dict] = []
    for row in rows:
        office_name_el = row.select_one("div.views-field-nothing strong")
        address_el = row.select_one("div.views-field-nothing p")
        contact_el = row.select_one("div.views-field-field-hubungi")
        map_link_el = row.select_one("div.views-field-nothing-1 a[href^='http']")

        office_name = normalize_space(office_name_el.get_text(" ", strip=True) if office_name_el else "")
        address = normalize_space(address_el.get_text(" ", strip=True) if address_el else "")
        contact_text = normalize_space(contact_el.get_text(" ", strip=True) if contact_el else "")

        if not office_name or not address:
            continue

        contact = parse_contact_fields(contact_text)
        city, state_region, _ = extract_city_state(address)

        is_national = "semenanjung malaysia" in office_name.lower() and "negeri" not in office_name.lower()

        website = map_link_el.get("href", "") if map_link_el else URL
        if not website:
            website = URL

        record = {
            "office_name": office_name,
            "country_code": "MY",
            "state_region": state_region,
            "district": None,
            "city": city,
            "address": address,
            "phone": contact["phone"],
            "email": contact["email"],
            "website": website,
            "open_hours": f"Source page does not list uniform hours. Fax: {contact['fax']}" if contact["fax"] else None,
            "is_national": is_national,
            "is_active": True,
        }
        parsed.append(record)

    # Deduplicate by office_name + address (page contains repeated variants).
    deduped: Dict[Tuple[str, str], Dict] = {}
    for rec in parsed:
        key = (rec["office_name"].lower(), rec["address"].lower())
        deduped[key] = rec

    return list(deduped.values())


def get_existing_keys(supabase_client) -> set:
    existing = supabase_client.table(TABLE).select("office_name,address").eq("is_active", True).execute()
    rows = existing.data or []
    keys = set()
    for row in rows:
        office_name = normalize_space(row.get("office_name", "")).lower()
        address = normalize_space(row.get("address", "")).lower()
        if office_name and address:
            keys.add((office_name, address))
    return keys


def insert_records(supabase_client, records: List[Dict]) -> Tuple[int, int]:
    existing_keys = get_existing_keys(supabase_client)

    new_records = []
    for rec in records:
        key = (rec["office_name"].lower(), rec["address"].lower())
        if key not in existing_keys:
            new_records.append(rec)

    if not new_records:
        return 0, len(existing_keys)

    batch_size = 100
    inserted = 0
    for i in range(0, len(new_records), batch_size):
        batch = new_records[i:i + batch_size]
        supabase_client.table(TABLE).insert(batch).execute()
        inserted += len(batch)

    return inserted, len(existing_keys)


def write_sql_seed(records: List[Dict], output_path: str) -> None:
    def sql_value(value: Optional[str]) -> str:
        if value is None:
            return "null"
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    lines = [
        "-- Auto-generated from scrape_and_load_labour_offices.py",
        f"-- Source: {URL}",
        "",
        f"insert into public.{TABLE} (",
        "    office_name, country_code, state_region, district, city,",
        "    address, phone, email, website, open_hours, is_national, is_active",
        ") values",
    ]

    value_rows = []
    for rec in records:
        value_rows.append(
            "(" + ", ".join([
                sql_value(rec.get("office_name")),
                sql_value(rec.get("country_code")),
                sql_value(rec.get("state_region")),
                sql_value(rec.get("district")),
                sql_value(rec.get("city")),
                sql_value(rec.get("address")),
                sql_value(rec.get("phone")),
                sql_value(rec.get("email")),
                sql_value(rec.get("website")),
                sql_value(rec.get("open_hours")),
                "true" if rec.get("is_national") else "false",
                "true" if rec.get("is_active", True) else "false",
            ]) + ")"
        )

    lines.append(",\n".join(value_rows))
    lines.append("on conflict do nothing;")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    load_environment()

    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in environment.")

    print(f"Fetching: {URL}")
    response = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
    response.raise_for_status()

    records = parse_rows(response.text)
    print(f"Parsed offices: {len(records)}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_output = os.path.join(script_dir, "labour_offices_scraped.json")
    sql_output = os.path.join(script_dir, "labour_offices_scraped_seed.sql")

    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    write_sql_seed(records, sql_output)

    print(f"Saved parsed JSON: {json_output}")
    print(f"Saved SQL seed file: {sql_output}")

    client = create_client(supabase_url, supabase_key)
    try:
        inserted, existing_before = insert_records(client, records)
        print("--- Import Summary ---")
        print(f"Table: {TABLE}")
        print(f"Existing active rows before: {existing_before}")
        print(f"Parsed rows from source: {len(records)}")
        print(f"Inserted new rows: {inserted}")
        print(f"Skipped duplicates: {len(records) - inserted}")
    except Exception as error:
        print("--- Import Blocked ---")
        print(f"Direct insert failed: {error}")
        print("Use the generated SQL seed file in Supabase SQL Editor to import all rows.")


if __name__ == "__main__":
    main()
