#!/usr/bin/env python
"""Clear all embeddings to re-upload with new smart titles."""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(override=True)
os.chdir("backend")
load_dotenv(override=True)
os.chdir("..")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Missing Supabase credentials")
    exit(1)

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Count before
    result = supabase.table("embeddings").select("count", count="exact").execute()
    count_before = result.count
    
    if count_before == 0:
        print("✅ Table already empty")
        exit(0)
    
    # Delete all by ID (safer approach)
    print(f"🗑️  Deleting {count_before} old embeddings...")
    
    # Get all IDs first
    result = supabase.table("embeddings").select("id").execute()
    all_ids = [row["id"] for row in result.data]
    
    # Delete in batches
    batch_size = 100
    for i in range(0, len(all_ids), batch_size):
        batch_ids = all_ids[i:i+batch_size]
        # Delete each batch
        for id_val in batch_ids:
            supabase.table("embeddings").delete().eq("id", id_val).execute()
    
    # Verify
    result = supabase.table("embeddings").select("count", count="exact").execute()
    count_after = result.count
    
    print(f"✅ Cleared! Deleted {count_before} chunks, {count_after} remain")
    
except Exception as e:
    print(f"❌ Error: {e}")
