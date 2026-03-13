import os
from dotenv import load_dotenv
from supabase import create_client
from sentence_transformers import SentenceTransformer

load_dotenv(r"c:\JY\UM\V Hack\Multilingual_Bot\yayyy-VHack\backend\.env")

sb = create_client(os.getenv('SUPABASE_URL'), os.getenv('SUPABASE_KEY'))

# 1. Check how embeddings are actually stored
print("=== CHECKING RAW DATA ===")
row = sb.table('embeddings').select('id, title, embedding').limit(1).execute().data[0]
emb = row['embedding']
print(f"Embedding type   : {type(emb)}")
print(f"Embedding preview: {str(emb)[:80]}")

# If it's a string, parse it to check dimension
if isinstance(emb, str):
    parsed = [float(x) for x in emb.strip('[]').split(',')]
    print(f"Parsed dimension : {len(parsed)}  <-- this is the REAL dimension")
    real_dim = len(parsed)
elif isinstance(emb, list):
    print(f"Vector dimension : {len(emb)}")
    real_dim = len(emb)
else:
    print(f"Unknown type, raw: {emb[:100]}")
    real_dim = 512

# 2. Check query embedding dimension
print("\n=== CHECKING QUERY EMBEDDING ===")
model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
q = model.encode('financial assistance').tolist()
print(f"Query dimension  : {len(q)}")

# 3. Run search with very low threshold
print("\n=== RUNNING VECTOR SEARCH (threshold=0.1) ===")
try:
    result = sb.rpc('match_documents', {
        'query_embedding': q,
        'match_count': 3,
        'similarity_threshold': 0.1
    }).execute()
    
    print(f"Results found: {len(result.data)}")
    for r in result.data:
        sim = r.get('similarity', 0)
        meta = r.get('metadata', {})
        print(f"  similarity={sim:.3f}  title={meta.get('title', '?')}")

except Exception as e:
    print(f"RPC error: {e}")

# 4. Summary
print("\n=== CONCLUSION ===")
if real_dim != len(q):
    print(f"MISMATCH: DB has {real_dim} dims, model creates {len(q)} dims")
    print("FIX: Re-upload documents using the correct model")
else:
    print(f"Dimensions match ({real_dim}). Problem is elsewhere.")
    print("Try lowering the similarity threshold in RAG.py")
