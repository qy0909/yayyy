import requests
from pypdf import PdfReader
import io
import os

def download_and_convert_pdf(url, filename):
    try:
        print(f"📥 Downloading: {url}")
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        pdf_file = io.BytesIO(response.content)
        reader = PdfReader(pdf_file)
        
        text_content = ""
        for page in reader.pages:
            text_content += page.extract_text() + "\n"
        
        md_filename = filename.replace(".pdf", ".md")
        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(f"SOURCE_URL: {url}\n" + "-"*20 + "\n" + text_content)
            
        print(f"✅ Created: {md_filename}")
        
    except Exception as e:
        print(f"❌ Failed to process {url}: {e}")

pdf_links = {
    "mof_media_release_march2026.pdf": "https://www.mof.gov.my/portal/images/2026/03/09/Siaran-Media-STR-Fasa-2.pdf",
    
    # Official Employment Act (Latest available consolidated version)
    "employment_act_1955.pdf": "https://jtksm.mohr.gov.my/sites/default/files/2023-11/Akta%20Kerja%201955%20%28Akta%20265%29_0.pdf",
    
    # 2026 STR FAQ specific document
    "str_2026_faq_status.pdf": "https://bantuantunai.hasil.gov.my/FAQ/FAQ%20SEMAKAN%20STATUS%20STR%202026.pdf",
}

for filename, url in pdf_links.items():
    download_and_convert_pdf(url, filename)