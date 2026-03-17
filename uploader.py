import os
import glob
from multiprocessing import freeze_support
from dotenv import load_dotenv 
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv() 

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 200 

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: Could not find Supabase credentials in .env file!")
    exit()
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

model = None


def get_model():
    """Lazy-load the embedding model to avoid Windows spawn import issues."""
    global model
    if model is None:
        print("Loading BAAI/bge-m3 model (this may take a moment)...")
        model = SentenceTransformer('BAAI/bge-m3')
    return model


def infer_document_type(file_name, source_url, content):
    """Infer a coarse category for Supabase document_type."""
    text = f"{file_name} {source_url} {content[:400]}".lower()

    if any(k in text for k in ["welfare", "sara", "str", "aid", "bantuan", "mykasih"]):
        return "welfare"
    if any(k in text for k in ["immigration", "immigrant", "permit", "visa", "migrant", "pekerja asing", "fomema"]):
        return "immigration"
    if any(k in text for k in ["health", "hospital", "clinic", "medical", "kesihatan", "covid"]):
        return "healthcare"

    return "government_doc"

def get_embedding(text):
    # bge-m3 is optimized for dense retrieval
    return get_model().encode(text, normalize_embeddings=True).tolist()

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

    print(f"📚 Found {len(files)} files. Starting chunked sync with BGE-M3...")

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

        chunks = create_chunks(content, CHUNK_SIZE, CHUNK_OVERLAP)
        doc_type = infer_document_type(file_name, source_url, content)
        print(f"✂️ Slicing {file_name} into {len(chunks)} chunks...")

        for index, chunk_text in enumerate(chunks):
            unique_title = f"{file_name}_chunk_{index}"
            
            data = {
                "content": chunk_text,
                "embedding": get_embedding(chunk_text), # Now 1024 dims
                "title": unique_title,
                "source_url": source_url,
                "language": "ms",
                "document_type": doc_type,
                "region": "Malaysia"
            }

            try:
                supabase.table("embeddings").upsert(data, on_conflict="title").execute()
            except Exception as e:
                print(f"❌ Error uploading {unique_title}: {e}")
        
        print(f"✅ Finished {file_name}")

if __name__ == "__main__":
    freeze_support()
    upload_all_markdown_files()