"""Check Supabase table schema and fix dimension mismatch"""
import os
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv()
supabase = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

print("=" * 60)
print("ISSUE DIAGNOSIS")
print("=" * 60)

# 1. Check what's in the database
response = supabase.table('embeddings').select('id, embedding, content').limit(1).execute()

if response.data:
    data = response.data[0]
    embedding = data['embedding']
    
    print(f"\n📊 Database Status:")
    print(f"   Embedding type: {type(embedding)}")
    print(f"   Embedding length: {len(embedding) if isinstance(embedding, list) else 'N/A'}")
    
    if isinstance(embedding, list):
        print(f"   First 5 values: {embedding[:5]}")
        print(f"   All numeric?: {all(isinstance(x, (int, float)) for x in embedding[:10])}")

# 2. Check the model
print(f"\n🤖 Model Status:")
model = SentenceTransformer('BAAI/bge-m3')
test_embedding = model.encode("Hello world")
print(f"   Model: BAAI/bge-m3")  
print(f"   Produces dimension: {len(test_embedding)}")
print(f"   Type: {type(test_embedding)}")

# 3. Solution
print(f"\n💡 SOLUTION:")
if len(embedding) != len(test_embedding):
    print(f"   ❌ MISMATCH: Database has {len(embedding)} dims, model creates {len(test_embedding)} dims")
    print(f"\n   You have 2 options:")
    print(f"\n   Option 1: DELETE and re-upload embeddings")
    print(f"      1. Clear the embeddings table")
    print(f"      2. Re-run: python uploader.py")
    print(f"\n   Option 2: Use a different embedding model that matches {len(embedding)} dimensions")
    print(f"      (This is very unusual, check if embeddings were created correctly)")
else:
    print(f"   ✅ Dimensions match! The problem is elsewhere.")

print("=" * 60)
