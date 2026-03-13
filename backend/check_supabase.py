"""
Check Supabase Database Status
================================
This script checks if your Supabase database has embeddings and the RPC function
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
EMBEDDINGS_TABLE = os.getenv('EMBEDDINGS_TABLE_NAME', 'embeddings')

print("=" * 60)
print("Checking Supabase Database Status")
print("=" * 60)

# Initialize Supabase client
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Connected to Supabase")
    print(f"   URL: {SUPABASE_URL}")
except Exception as e:
    print(f"❌ Failed to connect to Supabase: {e}")
    exit(1)

# Check if table exists and has data
try:
    print(f"\n📊 Checking table: {EMBEDDINGS_TABLE}")
    response = supabase.table(EMBEDDINGS_TABLE).select("*", count='exact').limit(5).execute()
    
    total_count = response.count if hasattr(response, 'count') else len(response.data)
    
    print(f"   Total documents: {total_count}")
    
    if response.data:
        print(f"   Sample documents found: {len(response.data)}")
        print("\n   First document structure:")
        sample = response.data[0]
        for key in sample.keys():
            value = sample[key]
            if isinstance(value, list) and len(value) > 5:
                print(f"      - {key}: [vector with {len(value)} dimensions]")
            else:
                print(f"      - {key}: {str(value)[:50]}...")
    else:
        print("   ⚠️  No documents found in the table!")
        print("\n   💡 To fix this, you need to:")
        print("      1. Run your scraper to get documents")
        print("      2. Convert documents to embeddings")
        print("      3. Upload to Supabase")
        print("\n   Try running:")
        print("      python scraper.py")
        print("      python uploader.py")
        
except Exception as e:
    print(f"   ❌ Error querying table: {e}")
    print(f"\n   The table '{EMBEDDINGS_TABLE}' might not exist yet.")
    print("   You need to create it in your Supabase dashboard.")

# Check if match_documents function exists
print(f"\n🔍 Checking RPC function: match_documents")
try:
    # Try calling the function with dummy data
    test_embedding = [0.0] * 384  # Dummy embedding
    response = supabase.rpc(
        'match_documents',
        {
            'query_embedding': test_embedding,
            'match_count': 1,
            'similarity_threshold': 0.1
        }
    ).execute()
    
    print("   ✅ RPC function exists and is callable")
except Exception as e:
    error_msg = str(e)
    if 'function' in error_msg.lower() or 'not found' in error_msg.lower():
        print("   ❌ RPC function 'match_documents' does not exist!")
        print("\n   💡 You need to create this function in Supabase SQL Editor:")
        print("""
   CREATE OR REPLACE FUNCTION match_documents (
     query_embedding vector(384),
     match_count int DEFAULT 5,
     similarity_threshold float DEFAULT 0.7
   )
   RETURNS TABLE (
     id bigint,
     content text,
     metadata jsonb,
     similarity float
   )
   LANGUAGE plpgsql
   AS $$
   BEGIN
     RETURN QUERY
     SELECT
       embeddings.id,
       embeddings.content,
       embeddings.metadata,
       1 - (embeddings.embedding <=> query_embedding) AS similarity
     FROM embeddings
     WHERE 1 - (embeddings.embedding <=> query_embedding) > similarity_threshold
     ORDER BY embeddings.embedding <=> query_embedding
     LIMIT match_count;
   END;
   $$;
        """)
    else:
        print(f"   ⚠️  Error testing function: {error_msg}")

print("\n" + "=" * 60)
print("Diagnosis Complete")
print("=" * 60)
