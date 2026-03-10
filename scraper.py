import requests
from bs4 import BeautifulSoup
import os
import re

def clean_filename(url):
    # Converts 'https://hasil.gov.my/faq' -> 'hasil_gov_my_faq'
    name = re.sub(r'https?://', '', url)
    name = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return name[:50] # Keep it short

def scrape_to_markdown(url):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 1. REMOVE NOISE: Strip out unwanted sections
        for noise in soup(["nav", "footer", "script", "style", "header", "aside"]):
            noise.decompose() # This deletes these tags and their content

        # 2. TARGET CONTENT: Try to find the main article first
        # Many gov sites use 'main', 'article', or specific ID/Classes
        main_content = soup.find('main') or soup.find('article') or soup.find('div', id='main')
        
        # If we found a specific container, use it; otherwise, use the cleaned body
        target = main_content if main_content else soup.body
        
        if target:
            text = target.get_text(separator='\n')
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            clean_text = '\n'.join(lines)
            
            # Save the file (same logic as before)
            filename = f"{clean_filename(url)}.md"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"SOURCE_URL: {url}\n" + "-"*20 + "\n" + clean_text)
            print(f"✅ Clean Scrape: {url}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

# Read links from your links.txt
if os.path.exists("links.txt"):
    with open("links.txt", "r") as f:
        # This line now filters out comments and empty lines
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        for url in urls:
            scrape_to_markdown(url)