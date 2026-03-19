#!/usr/bin/env python
"""
Complete Pipeline Runner: Scrape URLs → Process Markdown → Upload to Supabase
Run: python run_pipeline.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Execute a command and report results."""
    print(f"\n{'='*70}")
    print(f"🚀 {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ {description} FAILED!")
        return False
    
    print(f"\n✅ {description} COMPLETED!")
    return True

def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if Path(filepath).exists():
        print(f"✅ {description}: Found")
        return True
    else:
        print(f"⚠️ {description}: Not found - {filepath}")
        return False

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                    MULTILINGUAL BOT PIPELINE                        ║
║                  Scrape → Process → Upload to Supabase               ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check Prerequisites
    print("\n📋 CHECKING PREREQUISITES...")
    check_file_exists("links.txt", "URL list")
    check_file_exists("backend/.env", "Supabase credentials")
    check_file_exists("scraper.py", "Scraper script")
    check_file_exists("uploader.py", "Uploader script")
    check_file_exists("semantic_chunker.py", "Semantic chunker")
    
    # Step 2: Install Dependencies
    if not run_command(
        f"{sys.executable} -m pip install -q -r requirements.txt",
        "STEP 1: Installing Dependencies"
    ):
        print("⚠️ Failed to install dependencies. Please run manually:")
        print(f"   {sys.executable} -m pip install -r requirements.txt")
        return
    
    # Step 3: Run Scraper
    markdown_files = list(Path(".").glob("*.md"))
    if not markdown_files:
        if not run_command(
            f"{sys.executable} scraper.py",
            "STEP 2: Scraping URLs to Markdown Files"
        ):
            print("❌ Scraping failed. Check links.txt and internet connection.")
            return
    else:
        print(f"\n✅ STEP 2: Scraping skipped ({len(markdown_files)} .md files already exist)")
        response = input("Rescrap? (y/n) [n]: ").strip().lower()
        if response == 'y':
            # Delete old files
            for f in markdown_files:
                f.unlink()
            if not run_command(
                f"{sys.executable} scraper.py",
                "STEP 2: Re-scraping URLs"
            ):
                print("❌ Scraping failed.")
                return
    
    # Step 4: Run Uploader
    if not run_command(
        f"{sys.executable} uploader.py",
        "STEP 3: Processing & Uploading to Supabase"
    ):
        print("❌ Upload failed. Check Supabase credentials in backend/.env")
        return
    
    # Success Message
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                         ✅ PIPELINE COMPLETE!                        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  ✓ Web content scraped from links.txt                               ║
║  ✓ Markdown processed with semantic chunking                        ║
║  ✓ Embeddings generated (BGE-M3 1024-dim)                           ║
║  ✓ Data uploaded to Supabase (embeddings table)                     ║
║                                                                      ║
║  📊 Check Supabase Dashboard:                                        ║
║     https://app.supabase.com                                         ║
║                                                                      ║
║  🤖 Next: Start the chat API                                         ║
║     npm run dev                                                       ║
║     # Then visit http://localhost:3000                               ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    main()
