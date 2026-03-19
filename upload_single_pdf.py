import uploader as u

FILE_PATH = "jtksm_mohr_gov_my_sites_default_files_2023-11_Akta.md"

with open(FILE_PATH, "r", encoding="utf-8") as f:
    lines = f.readlines()

if not (lines and lines[0].startswith("SOURCE_URL:")):
    raise RuntimeError("Missing SOURCE_URL header in PDF markdown file")

source_url = lines[0].replace("SOURCE_URL:", "").strip()
content = "".join(lines[2:])

content = u.normalize_markdown_tables(content)
content = u.clean_images(content)
content = u.clean_boilerplate(content)
content = u.clean_navigation_links(content)

source_type = u.infer_source_type(FILE_PATH, source_url)
content_type = u.detect_content_type(content)
chunk_size = u.get_optimal_chunk_size(content_type)
semantic_chunks = u.create_chunks_semantic(content, chunk_size, u.OVERLAP_SENTENCES)
semantic_chunks = u.add_page_numbers_to_chunks(semantic_chunks, content, source_type)
doc_type = u.infer_document_type(FILE_PATH, source_url, content)

total = len(semantic_chunks)
print(f"Preparing to upload {total} chunks for: {source_url}")

u.supabase.table("embeddings").delete().eq("source_url", source_url).execute()

for index, chunk_data in enumerate(semantic_chunks):
    chunk_text = chunk_data["text"]
    metadata = chunk_data.get("metadata") or {}
    page_number, page_start, page_end = u.extract_page_metadata(metadata, source_type)

    readable_title = u.format_smart_title(
        FILE_PATH,
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
        "title": readable_title,
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
        print(f"Uploaded {index + 1}/{total}")

print("Upload complete")
