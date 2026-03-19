import os
import glob
import re
import sys
import io
from urllib.parse import unquote
from multiprocessing import freeze_support
import requests
from dotenv import load_dotenv 
from supabase import create_client
from sentence_transformers import SentenceTransformer
from semantic_chunker import create_chunks_semantic

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Try to import pypdf for PDF support, but don't fail if not available
try:
    from pypdf import PdfReader
    HAS_PYPDF = True
except ImportError:
    HAS_PYPDF = False

load_dotenv() 

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Chunk sizes optimized for different content types
CHUNK_SIZE_TABLE = 500      # Smaller chunks for structured data (better retrieval)
CHUNK_SIZE_LIST = 800       # Medium for lists/bullets
CHUNK_SIZE_NARRATIVE = 1200 # Larger for continuous prose
DEFAULT_CHUNK_SIZE = 1000   # Default fallback

OVERLAP_SENTENCES = 2  # Semantic overlap: carry forward 2 sentences 

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ Error: Could not find Supabase credentials in .env file!")
    exit()
else:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

model = None

def get_model():
    """Lazy-load the embedding model to avoid Windows spawn import issues."""
    global model
    if model is None:
        print("Loading BAAI/bge-m3 model (this may take a moment)...")
        model = SentenceTransformer('BAAI/bge-m3')
    return model

def infer_document_type(file_name, source_url, content):
    """Infer a coarse category for Supabase document_type."""
    text = f"{file_name} {source_url} {content[:400]}".lower()

    if any(k in text for k in ["welfare", "sara", "str", "aid", "bantuan", "mykasih"]):
        return "welfare"
    if any(k in text for k in ["immigration", "immigrant", "permit", "visa", "migrant", "pekerja asing", "fomema"]):
        return "immigration"
    if any(k in text for k in ["health", "hospital", "clinic", "medical", "kesihatan", "covid"]):
        return "healthcare"

    return "government_doc"

def get_embedding(text):
    return get_model().encode(text, normalize_embeddings=True).tolist()

def infer_source_type(file_name, source_url):
    """Classify source type so PDFs can carry page-level metadata."""
    source = f"{file_name} {source_url}".lower()
    return "pdf" if re.search(r"\.pdf(?:$|[?#\s])", source) else "web"

def as_positive_int(value):
    """Normalize numeric-like values to positive int or None."""
    if value is None:
        return None
    try:
        number = int(value)
        return number if number > 0 else None
    except (TypeError, ValueError):
        return None

def extract_page_metadata(metadata, source_type):
    """Extract page metadata for PDF chunks; keep null for web chunks."""
    if source_type != "pdf":
        return None, None, None

    page_number = as_positive_int(metadata.get("page_number") or metadata.get("page"))
    page_start = as_positive_int(metadata.get("page_start") or metadata.get("start_page"))
    page_end = as_positive_int(metadata.get("page_end") or metadata.get("end_page"))

    if page_number is not None:
        page_start = page_start or page_number
        page_end = page_end or page_number

    if page_start is not None and page_end is not None and page_start > page_end:
        page_start, page_end = page_end, page_start

    return page_number, page_start, page_end

def detect_page_breaks(text):
    """Detect page breaks in web content and assign rough page numbers.
    
    Detects:
    - Horizontal dividers (---, ***, ===)
    - "Page X" markers
    - Large spacing patterns
    - "Page Break" text markers
    
    Returns: Dict mapping text positions to page numbers
    """
    pages = {}
    page_num = 1
    lines = text.split('\n')
    char_pos = 0
    
    # Patterns indicating page breaks
    page_break_patterns = [
        r'^\s*[-*=]{5,}\s*$',                      # Horizontal rules
        r'^\s*\*\s*\*\s*\*\s*$',                   # *** separator
        r'^\s*(?:page|halaman)\s*\d+\s*$',         # "Page 1" or "Halaman 1"
        r'^\s*(?:----+|====|____)\s*$',             # Long dividers
    ]
    
    page_break_compiled = [re.compile(p, re.IGNORECASE) for p in page_break_patterns]
    
    for line_num, line in enumerate(lines):
        # Check if line indicates a page break
        if any(pattern.match(line) for pattern in page_break_compiled):
            page_num += 1
        
        # Also check for excessive blank lines (4+ consecutive = possible page break)
        if line_num > 0 and line.strip() == '':
            # Count consecutive blanks
            blank_count = 1
            for next_line_idx in range(line_num + 1, min(line_num + 5, len(lines))):
                if lines[next_line_idx].strip() == '':
                    blank_count += 1
                else:
                    break
            if blank_count >= 4:
                page_num += 1
        
        # Map character position to page number
        pages[char_pos] = page_num
        char_pos += len(line) + 1  # +1 for newline
    
    return pages, page_num

def detect_pdf_pages(text):
    """Detect likely PDF page markers and build char-position to page map.

    This is heuristic-based for scraped PDF markdown where explicit metadata
    is unavailable. It looks for patterns like:
    - "Page 12" / "Halaman 12"
    - Standalone page numbers on their own line
    """
    pages = {0: 1}
    current_page = 1
    lines = text.split('\n')
    char_pos = 0

    page_label_re = re.compile(r'^\s*(?:page|halaman)\s+(\d{1,4})(?:\s+of\s+\d{1,4})?\s*$', re.IGNORECASE)
    standalone_num_re = re.compile(r'^\s*(\d{1,4})\s*$')

    for i, line in enumerate(lines):
        stripped = line.strip()

        match_label = page_label_re.match(stripped)
        if match_label:
            detected = int(match_label.group(1))
            if 1 <= detected <= 5000:
                current_page = detected
                pages[char_pos] = current_page
        else:
            # Common in PDF text extraction: page numbers appear as a lone number line.
            match_num = standalone_num_re.match(stripped)
            if match_num:
                detected = int(match_num.group(1))
                prev_blank = (i == 0) or (lines[i - 1].strip() == '')
                next_blank = (i == len(lines) - 1) or (lines[i + 1].strip() == '')

                # Avoid treating list numbers as pages: require isolated line and monotonic increase.
                if prev_blank and next_blank and 1 <= detected <= 5000 and detected >= current_page:
                    current_page = detected
                    pages[char_pos] = current_page

        pages[char_pos] = current_page
        char_pos += len(line) + 1

    return pages, current_page

def extract_pdf_pages_from_url(source_url):
    """Download PDF from source URL and return extracted text per real page number.

    Returns:
        List of tuples: [(page_number, page_text), ...] or None on failure.
    """
    if not HAS_PYPDF:
        return None

    if not source_url or not source_url.lower().endswith('.pdf'):
        return None

    try:
        response = requests.get(source_url, timeout=60)
        response.raise_for_status()

        reader = PdfReader(io.BytesIO(response.content))
        page_data = []

        for page_number, page in enumerate(reader.pages, start=1):
            page_text = (page.extract_text() or "").strip()
            # Keep placeholder for empty pages so numbering stays accurate.
            page_data.append((page_number, page_text))

        return page_data
    except Exception as e:
        print(f"⚠️ Could not extract real PDF pages from {source_url}: {e}")
        return None

def create_pdf_chunks_with_real_pages(source_url):
    """Create chunks with exact PDF page_number from the original PDF binary."""
    page_data = extract_pdf_pages_from_url(source_url)
    if not page_data:
        return None

    all_chunks = []

    for page_number, page_text in page_data:
        if not page_text:
            continue

        page_text = normalize_markdown_tables(page_text)
        page_text = clean_images(page_text)
        page_text = clean_boilerplate(page_text)
        page_text = clean_navigation_links(page_text)

        if not page_text.strip():
            continue

        page_content_type = detect_content_type(page_text)
        page_chunk_size = get_optimal_chunk_size(page_content_type)
        page_chunks = create_chunks_semantic(page_text, page_chunk_size, OVERLAP_SENTENCES)

        for chunk in page_chunks:
            metadata = chunk.get('metadata') or {}
            metadata['page_number'] = page_number
            metadata['page_start'] = page_number
            metadata['page_end'] = page_number
            chunk['metadata'] = metadata
            all_chunks.append(chunk)

    total = len(all_chunks)
    for idx, chunk in enumerate(all_chunks):
        metadata = chunk.get('metadata') or {}
        metadata['chunk_index'] = idx
        metadata['total_chunks'] = total
        chunk['metadata'] = metadata

    return all_chunks

def get_page_for_position(pages_map, char_pos, total_pages):
    """Get page number for a character position in text."""
    # Find closest page break before this position
    relevant_pages = [p for p in pages_map.keys() if p <= char_pos]
    if relevant_pages:
        return pages_map[max(relevant_pages)]
    return 1

def add_page_numbers_to_chunks(chunks, content, source_type):
    """Add page numbers to chunk metadata based on source type."""
    if source_type == "pdf":
        # If page numbers are already set from real PDF extraction, keep them.
        has_real_page_metadata = any((chunk.get('metadata') or {}).get('page_number') is not None for chunk in chunks)
        if has_real_page_metadata:
            return chunks

        # Scraped PDFs usually lack explicit page metadata; infer from text markers.
        pages_map, total_pages = detect_pdf_pages(content)

        char_pos = 0
        for chunk_data in chunks:
            chunk_text = chunk_data['text']
            metadata = chunk_data.get('metadata') or {}

            page_num = get_page_for_position(pages_map, char_pos, total_pages)
            metadata['page_number'] = page_num
            metadata['page_start'] = page_num
            metadata['page_end'] = page_num
            metadata['total_pages'] = total_pages

            chunk_data['metadata'] = metadata
            char_pos += len(chunk_text)

        return chunks
    
    if source_type == "web":
        # For web content, detect page breaks and assign page numbers
        pages_map, total_pages = detect_page_breaks(content)
        
        # Calculate approximate character position of each chunk
        char_pos = 0
        for chunk_data in chunks:
            chunk_text = chunk_data['text']
            metadata = chunk_data.get('metadata') or {}
            
            # Find which page this chunk belongs to
            page_num = get_page_for_position(pages_map, char_pos, total_pages)
            
            # Add page info to metadata
            metadata['page_number'] = page_num
            metadata['total_pages'] = total_pages
            
            chunk_data['metadata'] = metadata
            char_pos += len(chunk_text)
        
        return chunks
    
    return chunks

def extract_pdf_with_pages(pdf_path):
    """Extract text from PDF with page tracking.
    
    Returns:
        (full_text, pages_dict) where pages_dict maps text position to page number
    """
    if not HAS_PYPDF:
        print(f"⚠️ pypdf not installed. Skipping PDF parsing for {pdf_path}")
        print("   Install with: pip install pypdf")
        return None, None
    
    try:
        reader = PdfReader(pdf_path)
        full_text = []
        pages_dict = {}
        char_pos = 0
        
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            pages_dict[char_pos] = page_num
            full_text.append(text)
            char_pos += len(text) + 50  # +50 for page break spacing
        
        return '\n\n[PAGE BREAK]\n\n'.join(full_text), pages_dict
    except Exception as e:
        print(f"❌ Error extracting PDF {pdf_path}: {e}")
        return None, None

def detect_content_type(text):
    """Detect content type to optimize chunk size."""
    lower_text = text.lower()
    line_count = len(text.split('\n'))
    
    # Count table indicators
    table_count = text.count('|')
    if table_count > 20:  # Lots of pipes = likely tabular
        return 'table'
    
    # Count list indicators
    list_indicators = text.count('- ') + text.count('* ') + len(re.findall(r'^\d+\.', text, re.MULTILINE))
    if list_indicators > (line_count * 0.3):  # >30% lines are list items
        return 'list'
    
    return 'narrative'

def get_optimal_chunk_size(content_type):
    """Return optimal chunk size based on content type."""
    return {
        'table': CHUNK_SIZE_TABLE,
        'list': CHUNK_SIZE_LIST,
        'narrative': CHUNK_SIZE_NARRATIVE
    }.get(content_type, DEFAULT_CHUNK_SIZE)

def clean_images(text):
    """Convert image references to text citations, preserving alt text without URLs.
    
    Converts:
      ![alt text](url) → [Image: alt text]
      <img alt="alt" /> → [Image: alt]
    Removes images without alt text entirely.
    """
    # Markdown images: ![alt](url) → [Image: alt]
    text = re.sub(r'!\[([^\]]+)\]\([^)]*\)', r'[Image: \1]', text)
    
    # HTML images with alt: <img alt="..." ... /> → [Image: ...]
    alt_match = re.findall(r'<img[^>]*alt="([^"]*)"[^>]*>', text)
    for alt_text in alt_match:
        if alt_text.strip():
            text = re.sub(
                r'<img[^>]*alt="' + re.escape(alt_text) + r'"[^>]*>',
                f'[Image: {alt_text}]',
                text
            )
    
    # Remove remaining images without alt text
    text = re.sub(r'<img[^>]+>', '', text)
    # Remove remaining markdown images without alt (edge case)
    text = re.sub(r'!\[\]\([^)]*\)', '', text)
    
    return text

def clean_boilerplate(text):
    """Remove common boilerplate, navigation, and non-useful content.
    
    Removes:
    - Navigation menus, breadcrumbs
    - Footers, copyright, legal disclaimers
    - Social sharing widgets
    - Metadata (dates, authors, reading time)
    - CTAs (call-to-action), ads, promotions
    - Related articles, recommendations
    - Share/conversation buttons (English & Malay)
    """
    lines = text.split('\n')
    filtered_lines = []
    
    # Patterns to skip entire lines (case-insensitive)
    skip_patterns = [
        # Navigation & menus
        r'^\s*(home|about|contact|services|products|blog|login|sign\s*up|register|download)\s*[\|•/]',
        r'^\s*(home\s*[>→]\s*|breadcrumb)',
        r'^\s*menu:\s*',
        
        # Navigation alternatives
        r'^\s*skip\s+to\s+(content|main)',
        r'^\s*(table\s+of\s+)?contents?\s*:?',
        
        # Social & sharing
        r'(share|post|kongsikan|berkongsi)\s+(on|to|di)',
        r'(like|tweet|follow|subscribe)\s+us',
        r'^\s*rating:\s*\d+',
        
        # Share/feedback buttons (English & Malay)
        r'^\s*(kongsikan|berkongsi|hantar\s+maklum|maklum\s+balas)',
        r'^\s*(share|tweet|pin|email)\s+this',
        
        # Metadata & dates (English & Malay)
        r'(last\s+updated?|published|posted)\s+on:?',
        r'(tarikh|updated?)\s+(kemaskini|tersimpan)',
        r'(author|by|written\s+by|contributed\s+by):?',
        r'reading\s+time:?',
        r'(word\s+)?count:?',
        
        # Contact info & footer sections
        r'^\s*(phone|fax|email|tel|call|contact|hubungi)',
        r'^\s*\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{4,4}',  # Phone numbers
        r'[@\.]\s*(mohr|gov|my)',  # Email domains
        r'(aras|blok|kompleks|pusat|jalan|jln)\s+',   # Address keywords (Malay)
        r'^\s*\d+,?\s*\d+',  # Postal codes
        
        # System/browser requirements
        r'(chrome|firefox|safari|opera|explorer|edge)',  # Browser names
        r'(resolution|screen\s+size|minimum|recommended)',
        r'(system\s+requirement|compatible)',
        r'piksel|pixel|pixels?|x\s*\d+|×\s*\d+',  # Resolution indicators
        r'(terkini|kini|latest|update)',  # Latest version keywords
        
        # Ads & promotions
        r'^\s*(subscribe|sign\s*up|request\s+demo|download\s+pdf|learn\s+more)',
        r'^\s*special\s+offer|limited\s+time|act\s+now',
        r'^\s*advertisement|ad:',
        
        # Disclaimers & legal
        r'^\s*(disclaimer|warning|notice|important|legal)',
        r'(terms\s+(of\s+)?service|privacy\s+polic|cookie)',
        r'©.*|copyright|all\s+rights\s+reserved',
        
        # Related content
        r'(you\s+might\s+also|read\s+next|similar\s+posts|related\s+articles)',
        r'^\s*see\s+also:',
        
        # Empty or minimal lines (preserve intentional spacing)
        r'^\s*$',
    ]
    
    skip_compiled = [re.compile(pattern, re.IGNORECASE) for pattern in skip_patterns]
    
    for line in lines:
        # Skip if matches any boilerplate pattern
        if any(pattern.search(line) for pattern in skip_compiled):
            continue
        
        # Skip lines that are just navigation separators
        if re.match(r'^\s*[\|\-•\*]+\s*$', line):
            continue
        
        # Skip HTML/markdown nav tags
        if re.search(r'<nav[^>]*>.*?</nav>', line, re.IGNORECASE | re.DOTALL):
            continue
        if re.search(r'<footer[^>]*>.*?</footer>', line, re.IGNORECASE | re.DOTALL):
            continue
        
        filtered_lines.append(line)
    
    # Remove excessive blank lines (keep max 2 consecutive)
    cleaned_text = '\n'.join(filtered_lines)
    cleaned_text = re.sub(r'\n\n\n+', '\n\n', cleaned_text)
    
    # Strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def clean_navigation_links(text):
    """Remove navigation links, language switchers, decorative images, and empty links.
    
    Removes:
    - Empty/orphaned links: [](url)
    - Language switchers: [ms](url), [en](url), [ENG](url), [BM](url), etc.
    - Decorative images: ![Image 1](url), ![Icon](url), ![jata](url)
    - Image-link combos: [![Image](url)Text](link)
    - Navigation-only lists (lines with only links, no prose)
    - Footer link lists (multiple legal/nav links separated by |)
    """
    
    # 1. Remove empty links: [](url)
    text = re.sub(r'\[\]\([^)]*\)', '', text)
    
    # 1b. Remove anchor-only empty links: [](#anchor)
    text = re.sub(r'\[\]\s*#[a-z\-]+', '', text, flags=re.IGNORECASE)
    
    # 1c. Remove malformed links: text](url) without opening bracket
    text = re.sub(r'\w+\]\([^)]*\)', '', text)
    
    # 2. Remove language switcher links
    # Patterns: [ms](url), [en](url), [ENG](url), [BM](url), [en-US](url), etc.
    text = re.sub(r'\[(en|ms|bm|eng|mal|bahasa\s+\w+|english|malay)[^\]]*\]\([^)]*\)', '', text, flags=re.IGNORECASE)
    
    # 3. Remove decorative image markdown (low-semantic-value alt text)
    # ![Image N](url), ![Icon](url), ![Logo](url), ![Jata](url), etc.
    decorative_patterns = [
        r'!\[image\s+\d+[^\]]*\]\([^)]*\)',          # ![Image 1](url), ![Image 2: etc](url)
        r'!\[icon[^\]]*\]\([^)]*\)',                  # ![Icon](url), ![icon_name](url)
        r'!\[logo[^\]]*\]\([^)]*\)',                  # ![Logo](url)
        r'!\[jata[^\]]*\]\([^)]*\)',                  # ![jata](url)
        r'!\[banner[^\]]*\]\([^)]*\)',                # ![banner](url)
        r'!\[flag[^\]]*\]\([^)]*\)',                  # ![flag](url)
        r'!\[shield[^\]]*\]\([^)]*\)',                # ![shield](url)
        r'!\[thumbnail[^\]]*\]\([^)]*\)',             # ![thumbnail](url)
        r'!\[screenshot[^\]]*\]\([^)]*\)',            # ![screenshot](url)
    ]
    
    for pattern in decorative_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 4. Remove image-text links: [![Image](url)Text](link)
    # Also catches variations like [![Image...](url)Organization](link)
    text = re.sub(r'\[!\[[^\]]*\]\([^)]*\)[^\]]*\]\([^)]*\)', '', text)
    
    # 5. Remove footer link lists (multiple links separated by |)
    # Patterns: [Link 1](url) | [Link 2](url) | [Link 3](url)
    # Also catches footer keywords in English and Malay
    footer_keywords = [
        # English
        r'privacy\s+polic|terms\s+of\s+service|disclaimer|legal|copy',
        r'user\s+help|support|contact\s+us|feedback|sitemap',
        r'accessibility|cookie|compliance',
        # Malay
        r'dasar\s+privasi|dasar\s+keselamatan|penafian|peta\s+laman',
        r'bantuan\s+pengguna|hubungi|maklum\s+balas|syarat',
        r'mengenai\s+kami|profil|fungsi|organisasi',
        r'pendaftaran|perkhidmatan|layanan',
    ]
    footer_pattern = r'(?:' + '|'.join(footer_keywords) + r')'
    
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Skip lines starting with image-link pattern (e.g., [![Image...])
        if re.match(r'^\s*\[!', line):
            continue
        
        # Check if line is a footer-style link list
        # Pattern: multiple links separated by pipes, with footer keywords
        pipe_count = line.count('|')
        link_count = len(re.findall(r'\[[^\]]*\]\([^)]*\)', line))
        
        # If 2+ pipes with footer keywords, it's a footer → skip
        if pipe_count >= 2 and link_count >= 2:
            if re.search(footer_pattern, line, re.IGNORECASE):
                continue
        
        # Check if line is a single link that matches footer keywords
        if link_count == 1 and not pipe_count:
            if re.search(footer_pattern, line, re.IGNORECASE):
                continue
        
        # Skip lines with only links (navigation lists)
        stripped = re.sub(r'^\s*[\*\-•]\s*', '', line).strip()
        
        # If line is ONLY markdown links (nothing else meaningful)
        link_content = re.sub(r'\[[^\]]*\]\([^)]*\)', '', stripped)
        link_content = link_content.strip()
        
        # If almost nothing left after removing links, it's nav-only
        if link_content and len(link_content) < 5:
            continue
        
        filtered_lines.append(line)
    
    text = '\n'.join(filtered_lines)
    
    return text

def normalize_markdown_tables(text):
    """Convert markdown table blocks into readable key-value lines for better retrieval."""
    lines = text.splitlines()
    out = []
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Detect markdown table header + separator
        has_pipe = '|' in stripped
        if (
            has_pipe
            and i + 1 < len(lines)
            and re.match(r"^\s*\|?\s*[-:]+(?:\s*\|\s*[-:]+)+\s*\|?\s*$", lines[i + 1])
        ):
            headers = [cell.strip() for cell in stripped.strip('|').split('|')]
            out.append("Table extracted:")
            i += 2

            while i < len(lines):
                row_line = lines[i].strip()
                if not row_line or '|' not in row_line:
                    break

                # Stop if we hit another separator or malformed structural line.
                if re.match(r"^\s*\|?\s*[-:]+(?:\s*\|\s*[-:]+)+\s*\|?\s*$", row_line):
                    i += 1
                    continue

                cells = [cell.strip() for cell in row_line.strip('|').split('|')]
                if len(cells) < 2:
                    break

                # Match row cells with headers, padding missing cells to keep structure stable.
                if len(cells) < len(headers):
                    cells.extend([''] * (len(headers) - len(cells)))

                parts = []
                for idx, header in enumerate(headers):
                    if not header:
                        continue
                    value = cells[idx].strip() if idx < len(cells) else ''
                    if value:
                        parts.append(f"{header}: {value}")

                if parts:
                    out.append("- " + "; ".join(parts))

                i += 1

            out.append("")
            continue

        out.append(line)
        i += 1

    return "\n".join(out)

def create_chunks(text, size, overlap):
    """
    Deprecated: Use create_chunks_semantic instead.
    Kept for backwards compatibility.
    """
    return [chunk['text'] for chunk in create_chunks_semantic(text, size, max(1, overlap // 250))]

def clean_markdown_formatting(text):
    """Remove markdown formatting (**, #, etc) from section titles."""
    if not text:
        return text
    # Remove markdown bold/italic/etc
    text = re.sub(r'\*{1,2}(.+?)\*{1,2}', r'\1', text)
    # Remove markdown headers
    text = re.sub(r'^#{1,6}\s*', '', text)
    # Clean up pipe separators
    text = text.replace(' | ', ' - ').strip()
    return text

def infer_category_label(file_name, section, subsection, content):
    """Infer document category for smart labels.
    
    Returns category label like [FAQ], [LAW], [GUIDE], [FORM], [HEALTH], etc.
    """
    # Combine all text to search
    combined = f"{file_name} {section} {subsection} {content[:200]}".lower()
    
    # Category patterns (order matters - most specific first)
    category_patterns = {
        "[FAQ]": r'(soalan|lazim|faq|frequently asked|jawapan)',
        "[LAW]": r'(act|akta|enactment|statute|ordinance|regulation|peraturan)',
        "[FORM]": r'(form|borang|application|permohonan|register|daftar)',
        "[HEALTH]": r'(health|kesihatan|medical|clinic|hospital|spital)',
        "[BENEFITS]": r'(benefit|faedah|allowance|elaun|aid|bantuan|gaji|salary)',
        "[HOUSING]": r'(housing|rumah|perumahan|house|accommodation)',
        "[WORK]": r'(work|kerja|employment|pekerja|job|occupation|pekerjaan)',
        "[EDUCATION]": r'(education|pendidikan|school|sekolah|university|universiti)',
        "[IMMIGRATION]": r'(immigration|imigrasi|foreigner|asing|foreign|permit)',
        "[GUIDE]": r'(guide|guideline|panduan|prosedur|steps|langkah)',
        "[CONTACT]": r'(contact|hubungi|address|alamat|phone|telefon|office)',
        "[POLICY]": r'(policy|dasar|principle|prinsip|procedure)',
    }
    
    for category, pattern in category_patterns.items():
        if re.search(pattern, combined):
            return category
    
    # Default based on content type (should rarely reach here)
    return "[INFO]"

def infer_readable_document_name(file_name, source_url, section, content):
    """Build a human-readable document name for titles.

    Prefers source_url-derived names when filenames are URL-hash-like.
    """
    raw_name = file_name.replace(".md", "").replace("_", " ").strip()
    titled_name = raw_name.title()

    # Known source shortcuts.
    if "jtksm.mohr.gov.my" in (source_url or ""):
        return "JTKSM"

    if "iom.int" in (source_url or ""):
        if "guidance-for-employers-of-migrant-workers-legal-obligations-in-malaysia" in (source_url or ""):
            return "IOM Employer Guidance (Malaysia)"
        return "IOM Guidance"

    # If filename already looks good, keep it.
    if len(titled_name) <= 40 and not re.search(r"\b[a-z0-9]{8,}\b", raw_name):
        return titled_name

    # Try to derive from URL path segment for readable labels.
    if source_url:
        path_tail = unquote(source_url.split("/")[-1])
        path_tail = re.sub(r"\.pdf$", "", path_tail, flags=re.IGNORECASE)
        path_tail = re.sub(r"^\d+[\.-]?\s*", "", path_tail)  # remove numeric prefixes like "12."
        path_tail = path_tail.replace("_", " ").replace("-", " ")
        path_tail = re.sub(r"\s+", " ", path_tail).strip()
        if path_tail:
            readable = path_tail.title()
            # Keep titles concise.
            if len(readable) > 55:
                readable = readable[:55].rstrip() + "..."
            return readable

    # Last fallback.
    parts = titled_name.split(" ")
    if len(parts) > 4:
        return " ".join(parts[-4:])
    return titled_name

def format_smart_title(file_name, source_url, section, subsection, content, index, total_chunks):
    """Generate smart category-labeled title.
    
    Format: [CATEGORY] Document Name - Section - Brief Description
    Example: [FAQ] Employment Act - Definitions - What qualifies as employment?
    """
    # Infer category
    category = infer_category_label(file_name, section or "", subsection or "", content)
    
    # Clean section/subsection titles
    clean_section = clean_markdown_formatting(section) if section else None
    clean_subsection = clean_markdown_formatting(subsection) if subsection else None
    
    short_name = infer_readable_document_name(file_name, source_url, clean_section, content)
    
    # Build title hierarchy
    if clean_subsection and clean_section and clean_section != clean_subsection:
        # Section + Subsection: Full hierarchy
        brief_sub = clean_subsection[:45]
        if len(clean_subsection) > 45:
            brief_sub += "..."
        title = f"{category} {short_name} - {clean_section} - {brief_sub}"
    elif clean_section:
        # Just section: Section as main context
        section_brief = clean_section[:50]
        if len(clean_section) > 50:
            section_brief += "..."
        title = f"{category} {short_name} - {section_brief}"
    else:
        # No section: Use chunk indicator
        title = f"{category} {short_name} ({index + 1}/{total_chunks})"
    
    return title

def upload_all_markdown_files():
    files = glob.glob("*.md")
    
    if not files:
        print("Empty pantry! No .md files found to upload.")
        return

    print(f"📚 Found {len(files)} files. Starting semantic chunking with BGE-M3...")

    for file_path in files:
        file_name = os.path.basename(file_path)
        
        # 1. CLEAN TITLE: 'employment_act_1955.md' -> 'Employment Act 1955'
        display_name = file_name.replace(".md", "").replace("_", " ").title()

        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        # 2. SOURCE & CONTENT LOGIC
        if not (lines and lines[0].startswith("SOURCE_URL:")):
            print(f"⏭️ Skipping non-scraped markdown file: {file_name}")
            continue

        source_url = "https://www.malaysia.gov.my/"
        if lines and lines[0].startswith("SOURCE_URL:"):
            source_url = lines[0].replace("SOURCE_URL:", "").strip()
            content = "".join(lines[2:]) 


        content = normalize_markdown_tables(content)
        content = clean_images(content)  # Convert images to citations, remove URLs
        content = clean_boilerplate(content)  # Remove navigation, footers, ads, metadata
        content = clean_navigation_links(content)  # Remove nav links, language switchers, decorative images
        source_type = infer_source_type(file_name, source_url)
        
        # Detect content type for optimal chunking
        content_type = detect_content_type(content)
        optimal_chunk_size = get_optimal_chunk_size(content_type)

        # Replace existing chunks for this source so old chunk artifacts don't persist.
        try:
            supabase.table("embeddings").delete().eq("source_url", source_url).execute()
        except Exception as e:
            print(f"⚠️ Could not clear old chunks for {source_url}: {e}")

        # Use exact PDF page extraction when available; fallback to semantic chunking of scraped text.
        if source_type == "pdf":
            semantic_chunks = create_pdf_chunks_with_real_pages(source_url)
            if not semantic_chunks:
                semantic_chunks = create_chunks_semantic(content, optimal_chunk_size, OVERLAP_SENTENCES)
        else:
            semantic_chunks = create_chunks_semantic(content, optimal_chunk_size, OVERLAP_SENTENCES)

        doc_type = infer_document_type(file_name, source_url, content)
        
        # Add page numbers to chunks (detects page breaks in web content)
        semantic_chunks = add_page_numbers_to_chunks(semantic_chunks, content, source_type)
        
        print(f"✂️ Slicing {display_name} using semantic chunking ({content_type} mode) → {len(semantic_chunks)} chunks...")

        for index, chunk_data in enumerate(semantic_chunks):
            chunk_text = chunk_data['text']
            metadata = chunk_data.get('metadata') or {}
            page_number, page_start, page_end = extract_page_metadata(metadata, source_type)
            
            # Generate smart category-labeled title
            readable_title = format_smart_title(
                file_name,
                source_url,
                metadata.get('section'),
                metadata.get('subsection'),
                chunk_text,
                index,
                len(semantic_chunks)
            )

            data = {
                "content": chunk_text,
                "embedding": get_embedding(chunk_text),
                "title": readable_title,
                "source_url": source_url,
                "source_type": source_type,
                "language": "ms",
                "document_type": doc_type,
                "region": "Malaysia",
                "page_number": page_number,
                "page_start": page_start,
                "page_end": page_end,
                # Additional metadata for better retrieval context
                "chunk_index": metadata.get('chunk_index'),
                "total_chunks": metadata.get('total_chunks'),
                "section": metadata.get('section'),
                "subsection": metadata.get('subsection'),
            }

            try:
                # Upsert based on the new readable title
                supabase.table("embeddings").upsert(data, on_conflict="title").execute()
            except Exception as e:
                print(f"❌ Error uploading {readable_title}: {e}")
        
        print(f"✅ Finished {file_name} with semantic context preservation")

if __name__ == "__main__":
    freeze_support()
    upload_all_markdown_files()