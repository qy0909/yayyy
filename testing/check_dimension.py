"""Check actual embedding dimension in database"""
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# Get one embedding and check its dimension
response = supabase.table('embeddings').select('embedding').limit(1).execute()

if response.data:
    embedding = response.data[0]['embedding']
    print(f"✅ Actual embedding dimension in database: {len(embedding)}")
    
    # Check what model creates this dimension
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer('BAAI/bge-m3')
    test_embedding = model.encode("test")
    print(f"   BAAI/bge-m3 dimension: {len(test_embedding)}")
else:
    print("No data found")
