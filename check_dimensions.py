"""
Quick Check: Supabase Vector Dimension
========================================
This script shows you the current dimension of vectors in your Supabase database
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# Connect to Supabase
supabase = create_client(
    os.getenv('SUPABASE_URL'),
    os.getenv('SUPABASE_KEY')
)

print("=" * 70)
print("CHECKING SUPABASE VECTOR DIMENSIONS")
print("=" * 70)

try:
    # Get one row from embeddings table
    response = supabase.table('embeddings').select('embedding, id, title').limit(1).execute()
    
    if not response.data:
        print("\n❌ No data found in 'embeddings' table")
        print("   Run: python uploader.py to add documents")
    else:
        data = response.data[0]
        embedding = data['embedding']
        
        print(f"\n✅ Found document: {data.get('title', 'Untitled')}")
        print(f"\n📊 Embedding Information:")
        print(f"   Type: {type(embedding)}")
        
        if isinstance(embedding, list):
            # It's a proper vector (list of numbers)
            print(f"   ✅ Stored as: VECTOR")
            print(f"   Dimension: {len(embedding)}")
            print(f"   First 5 values: {embedding[:5]}")
            
            if len(embedding) == 1024:
                print(f"\n   ✅ PERFECT! Matches BAAI/bge-m3 (1024 dims)")
            elif len(embedding) == 512:
                print(f"\n   ⚠️  Uses 512 dimensions (distiluse-base-multilingual-cased-v2)")
            elif len(embedding) == 384:
                print(f"\n   ⚠️  Uses 384 dimensions (all-MiniLM-L6-v2 model)")
                print(f"   Update RAG.py to use: 'all-MiniLM-L6-v2'")
            elif len(embedding) == 768:
                print(f"\n   ⚠️  Uses 768 dimensions (larger model)")
            else:
                print(f"\n   ⚠️  Unusual dimension: {len(embedding)}")
                
        elif isinstance(embedding, str):
            # It's stored as text - BIG PROBLEM
            print(f"   ❌ Stored as: TEXT STRING (length: {len(embedding)} chars)")
            print(f"\n   🚨 PROBLEM: Embeddings are stored as text, not vectors!")
            print(f"   This prevents vector search from working.")
            print(f"\n   FIX: Run the SQL commands in FIX_SUPABASE.md to:")
            print(f"   1. Recreate table with vector(1024) type")
            print(f"   2. Re-upload documents using uploader.py")
        else:
            print(f"   ❌ Unexpected type: {type(embedding)}")

    # Also check if using RPC function
    print(f"\n🔍 Checking vector search function...")
    try:
        # Try calling match_documents with a dummy vector
        test_vector = [0.0] * 1024
        result = supabase.rpc('match_documents', {
            'query_embedding': test_vector,
            'match_count': 1,
            'similarity_threshold': 0.1
        }).execute()
        print(f"   ✅ RPC function 'match_documents' exists")
        print(f"   Expected dimension: 1024")
    except Exception as e:
        error_msg = str(e).lower()
        if 'different vector dimensions' in error_msg:
            # Extract dimensions from error message
            import re
            dims = re.findall(r'\d+', str(e))
            if len(dims) >= 2:
                print(f"   ❌ Dimension mismatch!")
                print(f"   Database expects: {dims[0]} dimensions")
                print(f"   You sent: {dims[1]} dimensions")
        elif 'function' in error_msg or 'not exist' in error_msg:
            print(f"   ❌ RPC function 'match_documents' does not exist")
            print(f"   Run the SQL in FIX_SUPABASE.md to create it")
        else:
            print(f"   ⚠️  Error: {e}")

except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"\n   Possible issues:")
    print(f"   - Table 'embeddings' doesn't exist")
    print(f"   - Connection credentials are wrong")
    print(f"   - Database is empty")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("\nWhat dimension should you use?")
print("  • 1024 dims →  BAAI/bge-m3 (RECOMMENDED)")
print("  • 512 dims  →  distiluse-base-multilingual-cased-v2")
print("  • 384 dims  →  all-MiniLM-L6-v2")
print("  • 768 dims  →  paraphrase-multilingual-mpnet-base-v2")
print("\nMake sure:")
print("  1. Supabase table uses vector(XXX) type, not text")
print("  2. Uploader uses the SAME model as RAG.py")
print("  3. RPC function expects the SAME dimension")
print("=" * 70)
