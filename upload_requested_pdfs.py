import os

import uploader as u
from scraper import clean_filename, scrape_with_jina

PDF_URLS = [
    "https://jtksm.mohr.gov.my/sites/default/files/2023-03/12.%20Peraturan%20Standard%20Minimum%20Perumahan%20dan%20Kemudahan%20Pekerja%20%28Fi%20Pemprosesan%20bagi%20Permohonan%20Perakuan%20Penginapan%29%202020%20%281%29_0.pdf",
    "https://jtksm.mohr.gov.my/sites/default/files/2023-03/2.%20Employees%20Minimum%20Standards%20of%20Housing%2C%20Accomodations%20and%20emenities%20Act%201990.pdf",
    "https://jtksm.mohr.gov.my/sites/default/files/2023-03/15.%20Peraturan%20Standard%20Minimum%20Perumahan%20dan%20Kemudahan%20Pekerja%20%28Pekerja%20Yang%20Dikehendaki%20untuk%20Disediakan%20Penginapan%29%202021%20%282%29_0.pdf",
]


def scrape_to_markdown(url: str) -> str:
    content = scrape_with_jina(url)
    if not content:
        raise RuntimeError(f"Failed to scrape: {url}")

    file_name = clean_filename(url)
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(f"SOURCE_URL: {url}\n---\n{content}")

    print(f"Saved markdown: {file_name} ({len(content)} chars)")
    return file_name


def upload_markdown(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not (lines and lines[0].startswith("SOURCE_URL:")):
        raise RuntimeError(f"Missing SOURCE_URL header: {file_path}")

    source_url = lines[0].replace("SOURCE_URL:", "").strip()
    content = "".join(lines[2:])

    content = u.normalize_markdown_tables(content)
    content = u.clean_images(content)
    content = u.clean_boilerplate(content)
    content = u.clean_navigation_links(content)

    source_type = u.infer_source_type(file_path, source_url)
    content_type = u.detect_content_type(content)
    chunk_size = u.get_optimal_chunk_size(content_type)
    chunks = u.create_chunks_semantic(content, chunk_size, u.OVERLAP_SENTENCES)
    chunks = u.add_page_numbers_to_chunks(chunks, content, source_type)
    doc_type = u.infer_document_type(file_path, source_url, content)

    u.supabase.table("embeddings").delete().eq("source_url", source_url).execute()

    total = len(chunks)
    print(f"Uploading {total} chunks for {source_url}")

    for index, chunk_data in enumerate(chunks):
        chunk_text = chunk_data["text"]
        metadata = chunk_data.get("metadata") or {}
        page_number, page_start, page_end = u.extract_page_metadata(metadata, source_type)

        title = u.format_smart_title(
            file_path,
            source_url,
            metadata.get("section"),
            metadata.get("subsection"),
            chunk_text,
            index,
            total,
        )

        data = {
            "content": chunk_text,
            "embedding": u.get_embedding(chunk_text),
            "title": title,
            "source_url": source_url,
            "source_type": source_type,
            "language": "ms",
            "document_type": doc_type,
            "region": "Malaysia",
            "page_number": page_number,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_index": metadata.get("chunk_index"),
            "total_chunks": metadata.get("total_chunks"),
            "section": metadata.get("section"),
            "subsection": metadata.get("subsection"),
        }

        u.supabase.table("embeddings").upsert(data, on_conflict="title").execute()

        if (index + 1) % 25 == 0 or index + 1 == total:
            print(f"  Uploaded {index + 1}/{total}")


if __name__ == "__main__":
    created_files = []

    for url in PDF_URLS:
        print("\n" + "=" * 90)
        print(f"Processing URL: {url}")
        print("=" * 90)

        md_file = scrape_to_markdown(url)
        created_files.append(md_file)
        upload_markdown(md_file)

    print("\nAll requested PDFs have been uploaded to Supabase.")
