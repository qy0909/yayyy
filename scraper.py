import requests
import os
import re

def clean_filename(url):
    """Converts URL to a clean filename for your .md files."""
    # Removes 'https://', 'www', and special chars
    name = url.replace("https://", "").replace("http://", "").replace("www.", "")
    clean_name = re.sub(r'[^\w\-]', '_', name).strip('_')
    return f"{clean_name[:50]}.md" # Limit length for Windows paths

def scrape_with_jina(url):
    """Uses Jina Reader to bypass gov firewalls and get clean Markdown."""
    print(f"🔍 Jina is reading: {url}")
    jina_url = f"https://r.jina.ai/{url}"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'X-Return-Format': 'markdown'
    }

    try:
        response = requests.get(jina_url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"❌ Failed to scrape {url}: {e}")
        return None

def main():
    if not os.path.exists("links.txt"):
        print("❌ Error: links.txt not found!")
        return

    with open("links.txt", "r") as f:
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    for url in urls:
        content = scrape_with_jina(url)
        if content:
            filename = clean_filename(url)
            
            # --- IMPORTANT: This matches your uploader's logic ---
            # Line 0: SOURCE_URL
            # Line 1: Separator
            # Line 2+: Content
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"SOURCE_URL: {url}\n---\n{content}")
            
            print(f"✅ Saved as: {filename}")

if __name__ == "__main__":
    main()