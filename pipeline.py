#!/usr/bin/env python3
"""
Orchestrator for the complete document pipeline:
1. Scrape URLs from links.txt → Save as .md files
2. Upload .md files → Chunk, embed, and store in Supabase

Usage:
    python pipeline.py              # Run full pipeline
    python pipeline.py --scrape-only # Only scrape
    python pipeline.py --upload-only # Only upload
"""

import sys
import os
import time
import glob
from datetime import datetime


def print_banner(text):
    """Print formatted section banner."""
    width = 80
    print("\n" + "=" * width)
    print(f"  {text}".ljust(width))
    print("=" * width)


def print_step(step_num, description):
    """Print step indicator."""
    print(f"\n📍 Step {step_num}: {description}")
    print("-" * 60)


def run_scraper():
    """Execute scraper.py and return success status."""
    print_step(1, "Scraping URLs to Markdown files")
    
    try:
        import scraper
        
        # Check if links.txt exists
        if not os.path.exists("links.txt"):
            print("❌ Error: links.txt not found!")
            return False
        
        print("📂 Reading URLs from: links.txt")
        with open("links.txt", "r") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            print(f"   Found {len(urls)} URLs to scrape\n")
        
        # Run scraper
        start_time = time.time()
        scraper.main()
        elapsed = time.time() - start_time
        
        # Count output files
        md_files = glob.glob("*.md")
        print(f"\n✅ Scraping completed in {elapsed:.1f}s")
        print(f"   Generated {len(md_files)} .md files")
        
        return True
        
    except Exception as e:
        print(f"❌ Scraper failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_uploader():
    """Execute uploader.py and return success status."""
    print_step(2, "Uploading to Supabase with semantic chunking")
    
    try:
        import uploader
        
        # Check if .md files exist
        md_files = glob.glob("*.md")
        if not md_files:
            print("❌ Error: No .md files found!")
            print("   Run scraper first with: python pipeline.py --scrape-only")
            return False
        
        print(f"📂 Found {len(md_files)} .md files to upload:")
        for f in md_files[:5]:
            print(f"   - {f}")
        if len(md_files) > 5:
            print(f"   ... and {len(md_files) - 5} more")
        
        # Check Supabase config
        if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
            print("❌ Error: Supabase credentials not configured in .env!")
            print("   Add SUPABASE_URL and SUPABASE_KEY to .env file")
            return False
        
        print("\n⚙️ Configuration:")
        print(f"   Chunk size: 1000 chars")
        print(f"   Overlap: 2 sentences (semantic)")
        print(f"   Embedding model: BAAI/bge-m3")
        
        # Run uploader
        print("\n🚀 Starting upload...\n")
        start_time = time.time()
        uploader.upload_all_markdown_files()
        elapsed = time.time() - start_time
        
        print(f"\n✅ Upload completed in {elapsed:.1f}s")
        return True
        
    except Exception as e:
        print(f"❌ Uploader failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_statistics():
    """Show pipeline statistics."""
    print_step(3, "Pipeline Statistics")
    
    # Count .md files
    md_files = glob.glob("*.md")
    print(f"📊 Local Markdown files: {len(md_files)}")
    
    total_size = sum(os.path.getsize(f) for f in md_files if os.path.isfile(f))
    print(f"📊 Total content size: {total_size / 1024 / 1024:.2f} MB")
    
    # Check links.txt
    if os.path.exists("links.txt"):
        with open("links.txt", "r") as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
            print(f"📊 URLs in links.txt: {len(urls)}")
    
    # Show recent files
    if md_files:
        print(f"\n🔄 Recent files:")
        recent = sorted(md_files, key=lambda x: os.path.getmtime(x), reverse=True)[:3]
        for f in recent:
            mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
            size = os.path.getsize(f) / 1024
            print(f"   {f:<40} {size:>6.1f}KB  {mtime}")


def main():
    """Main orchestrator."""
    print_banner("DOCUMENT PIPELINE ORCHESTRATOR")
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working directory: {os.getcwd()}")
    
    # Parse arguments
    scrape_only = "--scrape-only" in sys.argv
    upload_only = "--upload-only" in sys.argv
    
    # Run pipeline
    overall_success = True
    
    if not upload_only:
        success = run_scraper()
        overall_success = overall_success and success
        if not success and not scrape_only:
            print("\n⚠️  Scraping failed, skipping upload")
            upload_only = True
    
    if not scrape_only:
        if overall_success or (os.path.exists("*.md") and glob.glob("*.md")):
            success = run_uploader()
            overall_success = overall_success and success
    
    # Show statistics
    show_statistics()
    
    # Final summary
    print_banner("PIPELINE SUMMARY")
    if overall_success:
        print("✅ Pipeline completed successfully!")
        print("\n📝 Next steps:")
        print("   1. Check Supabase console to verify embeddings were uploaded")
        print("   2. Test RAG system with: npm run dev:all")
        print("   3. Send a query to verify retrieval works")
    else:
        print("❌ Pipeline encountered errors. Check output above.")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(main())
