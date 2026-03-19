#!/usr/bin/env python
"""Show current chunk titles from Supabase to compare formatting."""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment
load_dotenv(override=True)
os.chdir("backend")
load_dotenv(override=True)
os.chdir("..")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Get sample titles with context
result = supabase.table("embeddings").select(
    "title, content, section, subsection, source_url, chunk_index, total_chunks"
).limit(20).execute()

print("\n" + "="*90)
print("CURRENT CHUNK TITLE EXAMPLES".center(90))
print("="*90 + "\n")

for i, row in enumerate(result.data, 1):
    print(f"{i}. TITLE:")
    print(f"   {row['title']}\n")
    
    if row['section']:
        print(f"   📑 Section: {row['section']}")
    if row['subsection']:
        print(f"   📖 Subsection: {row['subsection']}")
    
    print(f"   💬 Content preview: {row['content'][:70]}...")
    print(f"   📄 Chunk {row['chunk_index']}/{row['total_chunks']} | {row['source_url'][:40]}...")
    print("-" * 90 + "\n")

print("\n" + "="*90)
print("TITLE FORMAT ANALYSIS".center(90))
print("="*90)
print("""
CURRENT APPROACH:
  Document Name - Section - Subsection... (if available)
  -or- Document Name (Chunk N) (fallback)

OBSERVATIONS:
  ✓ Shows document & section hierarchy
  ✓ Good for context
  ✗ Can be long/truncated
  ✗ Doesn't preview content
  ✗ Generic fallback for chunks without sections

POTENTIAL IMPROVEMENTS:
  A) Add content preview:
     "Employment Act - Definitions - 'Employment is defined as any work...'"
     
  B) Add metadata badges:
     "Employment Act - Section 2 📋 [List] (1.5 min read, Chunk 3/38)"
     
  C) Smarter titles from content:
     "Employment Act - Definitions - 'Employment' term explained"
     
  D) Concise mode (better for UI):
     "Employment Act: Definitions"
     
  E) Add emojis by content type:
     "📋 Employment Act - Definitions" (list)
     "📝 FAQ - Benefits Overview" (narrative)
     "🔢 Table Data - Salary Ranges" (table)
""")
