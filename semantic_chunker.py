"""
Semantic chunker with context-aware splitting, header preservation, and metadata enrichment.
Solves issues with naive character-based chunking by understanding document structure.
"""

import re
from typing import List, Dict, Any, Optional, Tuple


class SemanticChunker:
    """
    Context-aware text chunker that:
    - Preserves document headers and sections
    - Uses smart sentence splitting (handles abbreviations, decimals, etc)
    - Maintains semantic overlap instead of character overlap
    - Tracks metadata about context (section name, depth, etc)
    - Recognizes hard boundaries (section breaks)
    """
    
    def __init__(self, chunk_size: int = 1000, overlap_sentences: int = 2):
        """
        Args:
            chunk_size: Maximum characters per chunk
            overlap_sentences: Number of sentences to carry forward for context
        """
        self.chunk_size = chunk_size
        self.overlap_sentences = overlap_sentences
        
        # Pattern to detect section headers (e.g., "Section 38", "Part 2", "Chapter 1")
        self.section_pattern = re.compile(
            r'^(Section|Part|Chapter|Article|Clause|Schedule|Appendix)\s+(\d+[A-Za-z]*)',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Pattern for subsection headers (e.g., "38.1", "(a)", "(i)")
        self.subsection_pattern = re.compile(
            r'^([\d.]+\.\d+|[a-z]\)|[ivx]+\))',
            re.IGNORECASE | re.MULTILINE
        )
        
        # Abbreviations that should NOT cause sentence splits
        self.abbreviations = {
            'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'sr.', 'jr.',
            'e.g.', 'i.e.', 'etc.', 'vs.', 'inc.', 'ltd.', 'co.',
            'u.s.', 'u.k.', 'u.n.', 'ph.d.', 'm.a.', 'b.a.',
            'jan.', 'feb.', 'mar.', 'apr.', 'jun.', 'jul.', 'aug.', 'sept.', 'oct.', 'nov.', 'dec.',
        }

    def _split_sentences(self, text: str) -> List[str]:
        """
        Smart sentence splitting that handles abbreviations, decimals, and special cases.
        
        Returns:
            List of sentences
        """
        # Replace common abbreviations with placeholders to protect them
        protected_text = text
        protection_map = {}
        
        for i, abbr in enumerate(self.abbreviations):
            placeholder = f"__ABBR_{i}__"
            if abbr in protected_text.lower():
                # Case-insensitive replacement
                protected_text = re.sub(
                    re.escape(abbr),
                    placeholder,
                    protected_text,
                    flags=re.IGNORECASE
                )
                protection_map[placeholder] = abbr
        
        # Protect decimal numbers (e.g., "0.5", "3.14")
        protected_text = re.sub(r'(\d)\.(\d)', r'\1__DECIMAL__\2', protected_text)
        
        # Split on sentence boundaries
        # Handles: . ! ? followed by space and capital letter (or end of string)
        sentences = re.split(
            r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])$',
            protected_text
        )
        
        # Restore abbreviations and decimals
        restored_sentences = []
        for sentence in sentences:
            for placeholder, abbr in protection_map.items():
                sentence = sentence.replace(placeholder, abbr)
            sentence = sentence.replace('__DECIMAL__', '.')
            if sentence.strip():
                restored_sentences.append(sentence.strip())
        
        return restored_sentences

    def _detect_header_level(self, line: str) -> Tuple[Optional[int], str]:
        """
        Detect if line is a header and its level.
        
        Returns:
            (level, header_text) or (None, "") if not a header
            Levels: 1=Section, 2=Subsection, 3=Paragraph
        """
        line = line.strip()
        
        # Markdown headers
        match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if match:
            level = len(match.group(1))
            return level, match.group(2).strip()
        
        # Section headers
        if re.match(r'^Section\s+\d+[A-Za-z]*\s*[-–]?\s*', line, re.IGNORECASE):
            return 1, line
        
        # Subsection headers
        if re.match(r'^\(\d+\)|^\(\w\)|^\d+\.\d+', line):
            return 2, line
        
        # All caps headers
        if len(line) > 5 and line.isupper() and not line.endswith('.'):
            return 1, line
        
        return None, ""

    def _build_chunk_metadata(
        self,
        current_section: Optional[str],
        current_subsection: Optional[str],
        chunk_index: int,
        total_chunks: int
    ) -> Dict[str, Any]:
        """Build metadata about the chunk's context."""
        return {
            'section': current_section,
            'subsection': current_subsection,
            'chunk_index': chunk_index,
            'total_chunks': total_chunks,
        }

    def chunk(self, text: str) -> List[Dict[str, Any]]:
        """
        Semantic chunking with context preservation.
        
        Returns:
            List of dicts with 'text' and 'metadata' keys
        """
        lines = text.split('\n')
        chunks = []
        
        # State tracking
        current_chunk_lines = []
        current_chunk_text = ""
        current_section = None
        current_subsection = None
        buffer_sentences = []  # For overlap
        
        for line in lines:
            stripped_line = line.strip()
            
            # Detect headers
            header_level, header_text = self._detect_header_level(stripped_line)
            
            # Update section tracking
            if header_level == 1:
                # Hard boundary: new top-level section
                if current_chunk_text.strip():
                    chunks.append({
                        'text': current_chunk_text.strip(),
                        'metadata': self._build_chunk_metadata(
                            current_section,
                            current_subsection,
                            len(chunks),
                            None  # Will be set later
                        )
                    })
                    current_chunk_text = ""
                    buffer_sentences = []
                
                current_section = header_text
                current_subsection = None
                current_chunk_lines = []
            
            elif header_level == 2:
                current_subsection = header_text
            
            # Add line to current chunk
            current_chunk_lines.append(line)
            candidate = '\n'.join(current_chunk_lines)
            
            # Check if we should chunk
            if len(candidate) > self.chunk_size and current_chunk_text:
                # Finalize current chunk with context
                # Prepend section header if not already included
                chunk_to_save = current_chunk_text.strip()
                if current_section and current_section not in chunk_to_save[:200]:
                    chunk_to_save = f"{current_section}\n\n{chunk_to_save}"
                
                if current_subsection and current_subsection not in chunk_to_save[:300]:
                    chunk_to_save = f"{current_subsection}\n{chunk_to_save}"
                
                chunks.append({
                    'text': chunk_to_save,
                    'metadata': self._build_chunk_metadata(
                        current_section,
                        current_subsection,
                        len(chunks),
                        None
                    )
                })
                
                # Overlap: keep last N sentences
                sentences = self._split_sentences(current_chunk_text)
                buffer_sentences = sentences[-self.overlap_sentences:]
                
                # Start new chunk with overlap context
                overlap_text = ' '.join(buffer_sentences)
                current_chunk_text = overlap_text + '\n' + stripped_line if overlap_text else stripped_line
                current_chunk_lines = [stripped_line]
            else:
                current_chunk_text = candidate
        
        # Finalize last chunk
        if current_chunk_text.strip():
            chunk_to_save = current_chunk_text.strip()
            if current_section and current_section not in chunk_to_save[:200]:
                chunk_to_save = f"{current_section}\n\n{chunk_to_save}"
            
            if current_subsection and current_subsection not in chunk_to_save[:300]:
                chunk_to_save = f"{current_subsection}\n{chunk_to_save}"
            
            chunks.append({
                'text': chunk_to_save,
                'metadata': self._build_chunk_metadata(
                    current_section,
                    current_subsection,
                    len(chunks),
                    len(chunks)  # This is the last one
                )
            })
        
        # Set total_chunks for all
        total = len(chunks)
        for chunk in chunks:
            chunk['metadata']['total_chunks'] = total
        
        return chunks


def create_chunks_semantic(text: str, chunk_size: int = 1000, overlap_sentences: int = 2) -> List[Dict[str, Any]]:
    """
    Convenience function matching the old interface but with semantic chunking.
    
    Args:
        text: Document text
        chunk_size: Max chunk size in characters
        overlap_sentences: Number of sentences to overlap between chunks
    
    Returns:
        List of dicts: [{'text': str, 'metadata': dict}, ...]
    """
    chunker = SemanticChunker(chunk_size=chunk_size, overlap_sentences=overlap_sentences)
    return chunker.chunk(text)


# Backwards compatibility: if uploader uses create_chunks, this replaces it
def create_chunks(text: str, size: int, overlap: int) -> List[str]:
    """
    Drop-in replacement for the old create_chunks function.
    Returns just the text list (for backwards compatibility).
    """
    chunker = SemanticChunker(chunk_size=size, overlap_sentences=max(1, overlap // 250))
    chunks = chunker.chunk(text)
    return [chunk['text'] for chunk in chunks]
