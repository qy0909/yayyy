import os
import glob
from dotenv import load_dotenv 
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv() 

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CHUNK_SIZE = 1000  # Number of characters per chunk
CHUNK_OVERLAP = 200 # Overlap to keep context between chunks

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: Could not find Supabase credentials in .env file!")
    exit()
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

def get_embedding(text):
    return model.encode(text).tolist()

# NEW: Helper function to slice the text
def create_chunks(text, size, overlap):
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i : i + size])
    return chunks

def upload_all_markdown_files():
    files = glob.glob("*.md")
    
    if not files:
        print("Empty pantry! No .md files found to upload.")
        return

    print(f"📚 Found {len(files)} files. Starting chunked sync...")

    for file_path in files:
        file_name = os.path.basename(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        source_url = "https://www.malaysia.gov.my/" 
        if lines and lines[0].startswith("SOURCE_URL:"):
            source_url = lines[0].replace("SOURCE_URL:", "").strip()
            content = "".join(lines[2:]) 
        else:
            content = "".join(lines)

        # NEW: Split content into chunks
        chunks = create_chunks(content, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"✂️ Slicing {file_name} into {len(chunks)} chunks...")

        for index, chunk_text in enumerate(chunks):
            # Create a unique title for each chunk to avoid 'upsert' overwriting
            unique_title = f"{file_name}_chunk_{index}"
            
            data = {
                "content": chunk_text,
                "embedding": get_embedding(chunk_text),
                "title": unique_title,
                "source_url": source_url,
                "language": "ms",
                "document_type": "government_doc", 
                "region": "Malaysia"
            }

            try:
                # Use unique_title for the on_conflict check
                supabase.table("embeddings").upsert(data, on_conflict="title").execute()
            except Exception as e:
                print(f"❌ Error uploading {unique_title}: {e}")
        
        print(f"✅ Finished {file_name}")

if __name__ == "__main__":
    upload_all_markdown_files()