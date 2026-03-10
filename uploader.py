import os
import glob
from dotenv import load_dotenv  # <-- ADD THIS
from supabase import create_client
from sentence_transformers import SentenceTransformer

# 1. Load the variables from your .env file
load_dotenv() 

# 2. Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# 3. Initialize Clients carefully
if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: Could not find Supabase credentials in .env file!")
    exit() # Stop the script if keys are missing
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Using DistilUSE for 512-dimension vectors
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')

def get_embedding(text):
    return model.encode(text).tolist()

def upload_all_markdown_files():
    files = glob.glob("*.md")
    
    if not files:
        print("Empty pantry! No .md files found to upload.")
        return

    print(f"📚 Found {len(files)} files. Starting sync to Supabase...")

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

        data = {
            "content": content,
            "embedding": get_embedding(content),
            "title": file_name,
            "source_url": source_url,
            "language": "ms",
            "document_type": "government_doc", 
            "region": "Malaysia"
        }

        try:
            supabase.table("embeddings").upsert(data, on_conflict="title").execute()
            print(f"✅ Successfully synced {file_name} to the team table!")
        except Exception as e:
            print(f"❌ Error uploading {file_name}: {e}")

if __name__ == "__main__":
    upload_all_markdown_files()