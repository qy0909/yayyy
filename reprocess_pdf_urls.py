import glob

import uploader as u

TARGET_URLS = [
    "https://jtksm.mohr.gov.my/sites/default/files/2023-11/Akta%20Kerja%201955%20%28Akta%20265%29_0.pdf",
    "https://jtksm.mohr.gov.my/sites/default/files/2023-03/2.%20Employees%20Minimum%20Standards%20of%20Housing%2C%20Accomodations%20and%20emenities%20Act%201990.pdf",
    "https://jtksm.mohr.gov.my/sites/default/files/2023-03/15.%20Peraturan%20Standard%20Minimum%20Perumahan%20dan%20Kemudahan%20Pekerja%20%28Pekerja%20Yang%20Dikehendaki%20untuk%20Disediakan%20Penginapan%29%202021%20%282%29_0.pdf",
    "https://jtksm.mohr.gov.my/sites/default/files/2023-03/12.%20Peraturan%20Standard%20Minimum%20Perumahan%20dan%20Kemudahan%20Pekerja%20%28Fi%20Pemprosesan%20bagi%20Permohonan%20Perakuan%20Penginapan%29%202020%20%281%29_0.pdf",
]


def build_source_file_map():
    mapping = {}
    for fp in glob.glob("*.md"):
        with open(fp, "r", encoding="utf-8") as f:
            first = f.readline().strip()
        if first.startswith("SOURCE_URL:"):
            mapping[first.replace("SOURCE_URL:", "").strip()] = fp
    return mapping


def upload_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    source_url = lines[0].replace("SOURCE_URL:", "").strip()
    content = "".join(lines[2:])

    content = u.normalize_markdown_tables(content)
    content = u.clean_images(content)
    content = u.clean_boilerplate(content)
    content = u.clean_navigation_links(content)

    source_type = u.infer_source_type(file_path, source_url)
    content_type = u.detect_content_type(content)
    chunk_size = u.get_optimal_chunk_size(content_type)
    if source_type == "pdf":
        chunks = u.create_pdf_chunks_with_real_pages(source_url)
        if not chunks:
            chunks = u.create_chunks_semantic(content, chunk_size, u.OVERLAP_SENTENCES)
    else:
        chunks = u.create_chunks_semantic(content, chunk_size, u.OVERLAP_SENTENCES)

    chunks = u.add_page_numbers_to_chunks(chunks, content, source_type)
    doc_type = u.infer_document_type(file_path, source_url, content)

    u.supabase.table("embeddings").delete().eq("source_url", source_url).execute()

    total = len(chunks)
    print(f"Uploading {total} chunks for {source_url}")

    for i, ch in enumerate(chunks):
        m = ch.get("metadata") or {}
        pn, ps, pe = u.extract_page_metadata(m, source_type)
        title = u.format_smart_title(
            file_path,
            source_url,
            m.get("section"),
            m.get("subsection"),
            ch["text"],
            i,
            total,
        )

        data = {
            "content": ch["text"],
            "embedding": u.get_embedding(ch["text"]),
            "title": title,
            "source_url": source_url,
            "source_type": source_type,
            "language": "ms",
            "document_type": doc_type,
            "region": "Malaysia",
            "page_number": pn,
            "page_start": ps,
            "page_end": pe,
            "chunk_index": m.get("chunk_index"),
            "total_chunks": m.get("total_chunks"),
            "section": m.get("section"),
            "subsection": m.get("subsection"),
        }

        u.supabase.table("embeddings").upsert(data, on_conflict="title").execute()

        if (i + 1) % 25 == 0 or i + 1 == total:
            print(f"  Uploaded {i + 1}/{total}")


if __name__ == "__main__":
    source_map = build_source_file_map()

    for url in TARGET_URLS:
        fp = source_map.get(url)
        if not fp:
            print(f"SKIP missing markdown source for {url}")
            continue
        print("\n" + "=" * 80)
        print(f"Reprocessing: {fp}")
        print("=" * 80)
        upload_file(fp)

    print("\nDone reprocessing target PDF URLs.")
