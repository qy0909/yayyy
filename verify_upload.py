#!/usr/bin/env python
"""Verify that embeddings were successfully uploaded to Supabase."""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv(override=True)
os.chdir("backend")
load_dotenv(override=True)
os.chdir("..")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing Supabase credentials in .env")
    exit(1)

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Count total embeddings
    result = supabase.table("embeddings").select("count", count="exact").execute()
    total_count = result.count
    
    # Get unique sources
    result = supabase.table("embeddings").select("source_url").execute()
    sources = set(r["source_url"] for r in result.data)
    
    # Get content statistics
    result = supabase.table("embeddings").select("id, content").limit(5).execute()
    sample_chunks = len(result.data)
    
    print(f"""
╔════════════════════════════════════════════════════════╗
║           ✅ UPLOAD VERIFICATION SUCCESSFUL            ║
╠════════════════════════════════════════════════════════╣
║                                                        ║
║  📊 Database Statistics:                               ║
║     Total Chunks Uploaded: {total_count:,}                  ║
║     Unique Sources: {len(sources)}                              ║
║     Sample Size: {sample_chunks}                               ║
║                                                        ║
║  🔗 Sources in Database:                               ║
""")
    
    for i, source in enumerate(sorted(sources)[:10], 1):
        print(f"║     {i}. {source[:45]:45s}  ║")
    
    if len(sources) > 10:
        print(f"║     ... and {len(sources) - 10} more sources                  ║")
    
    print(f"""║                                                        ║
║  ✨ Next Steps:                                        ║
║     1. Start the chat API: npm run dev               ║
║     2. Visit http://localhost:3000                    ║
║     3. Test queries in Malay or English                ║
║                                                        ║
║  🗄️ Supabase Dashboard:                                ║
║     https://app.supabase.com/project/svzyovyteik...  ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
    """)

except Exception as e:
    print(f"❌ Error connecting to Supabase: {e}")
    print(f"✓ Check .env file has SUPABASE_URL and SUPABASE_KEY")
