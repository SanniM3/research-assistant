"""Text chunking utilities for document processing."""
from typing import List, Optional, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ..models.chunk import Chunk, SourceType, ChunkQuality, ChunkMetadata
from ..config.settings import get_settings


def chunk_text(
    text: str,
    paper_id: str,
    source_type: SourceType,
    section_path: str = "",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """
    Split text into chunks suitable for embedding and retrieval.
    
    Uses section-aware splitting that tries to keep logical units together.
    
    Args:
        text: The text to chunk
        paper_id: ID of the source paper
        source_type: Type of source document
        section_path: Path to this section in document hierarchy
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of Chunk objects
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    # Use recursive splitter with academic-friendly separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",      # Major headings
            "\n### ",     # Subheadings
            "\n#### ",    # Sub-subheadings
            "\n\n",       # Paragraphs
            "\n",         # Lines
            ". ",         # Sentences
            ", ",         # Clauses
            " ",          # Words
            ""            # Characters
        ],
        length_function=len,
    )
    
    # Split text
    text_chunks = splitter.split_text(text)
    
    # Convert to Chunk objects
    chunks = []
    for i, chunk_text in enumerate(text_chunks):
        chunk = Chunk.create(
            paper_id=paper_id,
            text=chunk_text,
            source_type=source_type,
            section_path=section_path,
            metadata=ChunkMetadata(
                heading=_extract_heading(chunk_text),
            )
        )
        chunks.append(chunk)
    
    return chunks


def chunk_document(
    documents: List[Document],
    paper_id: str,
    source_type: SourceType,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """
    Chunk a list of LangChain Documents (e.g., from PDF loader).
    
    Preserves page information and document metadata.
    
    Args:
        documents: List of Document objects
        paper_id: ID of the source paper
        source_type: Type of source document
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
    
    Returns:
        List of Chunk objects
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    # Split documents
    split_docs = splitter.split_documents(documents)
    
    # Convert to Chunks
    chunks = []
    for doc in split_docs:
        page_num = doc.metadata.get("page", 0)
        
        chunk = Chunk.create(
            paper_id=paper_id,
            text=doc.page_content,
            source_type=source_type,
            section_path=f"page_{page_num + 1}",
            page_span=(page_num, page_num),
            quality=_assess_chunk_quality(doc.page_content),
        )
        chunks.append(chunk)
    
    return chunks


def chunk_with_sections(
    text: str,
    paper_id: str,
    source_type: SourceType,
    chunk_size: Optional[int] = None,
) -> List[Chunk]:
    """
    Chunk text while preserving section boundaries.
    
    Attempts to identify section headings and keep them with their content.
    
    Args:
        text: The text to chunk
        paper_id: ID of the source paper
        source_type: Type of source document
        chunk_size: Target chunk size
    
    Returns:
        List of Chunk objects with section paths
    """
    settings = get_settings()
    chunk_size = chunk_size or settings.chunk_size
    
    # Identify sections
    sections = _identify_sections(text)
    
    all_chunks = []
    section_path = ""
    
    for section in sections:
        section_path = section.get("heading", "")
        section_text = section.get("content", "")
        
        if not section_text.strip():
            continue
        
        # Chunk this section
        section_chunks = chunk_text(
            text=section_text,
            paper_id=paper_id,
            source_type=source_type,
            section_path=section_path,
            chunk_size=chunk_size,
        )
        
        # Update metadata with section info
        for chunk in section_chunks:
            chunk.metadata.heading = section_path
        
        all_chunks.extend(section_chunks)
    
    # If no sections found, chunk the whole text
    if not all_chunks:
        all_chunks = chunk_text(
            text=text,
            paper_id=paper_id,
            source_type=source_type,
            chunk_size=chunk_size,
        )
    
    return all_chunks


def _identify_sections(text: str) -> List[dict]:
    """
    Identify sections in text based on heading patterns.
    
    Looks for markdown-style headings (##), numbered sections (1., 1.1),
    and common academic section names.
    """
    import re
    
    # Common section patterns
    heading_patterns = [
        r"^#{1,4}\s+(.+)$",           # Markdown headings
        r"^(\d+\.?\s+[A-Z].+)$",       # Numbered sections
        r"^(Abstract|Introduction|Related Work|Background|Method|Methodology|"
        r"Approach|Experiments|Results|Discussion|Conclusion|References|"
        r"Acknowledgments|Appendix)s?:?\s*$",  # Common section names
    ]
    
    combined_pattern = "|".join(f"({p})" for p in heading_patterns)
    
    sections = []
    current_section = {"heading": "Preamble", "content": ""}
    
    for line in text.split("\n"):
        is_heading = False
        for pattern in heading_patterns:
            if re.match(pattern, line.strip(), re.IGNORECASE | re.MULTILINE):
                # Save previous section
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Start new section
                heading = re.sub(r"^#+\s*", "", line.strip())
                heading = re.sub(r"^\d+\.?\s*", "", heading)
                current_section = {"heading": heading, "content": ""}
                is_heading = True
                break
        
        if not is_heading:
            current_section["content"] += line + "\n"
    
    # Add final section
    if current_section["content"].strip():
        sections.append(current_section)
    
    return sections


def _extract_heading(text: str) -> Optional[str]:
    """Extract heading from chunk text if present."""
    import re
    
    lines = text.strip().split("\n")
    if not lines:
        return None
    
    first_line = lines[0].strip()
    
    # Check for markdown heading
    if first_line.startswith("#"):
        return re.sub(r"^#+\s*", "", first_line)
    
    # Check for numbered heading
    if re.match(r"^\d+\.?\s+[A-Z]", first_line):
        return first_line
    
    return None


def _assess_chunk_quality(text: str) -> ChunkQuality:
    """
    Assess the quality of a text chunk.
    
    Considers factors like:
    - Length (too short or too long)
    - Presence of meaningful content
    - Formatting issues
    """
    # Too short chunks are likely low quality
    if len(text.strip()) < 50:
        return ChunkQuality.LOW
    
    # Check for excessive special characters (possible OCR errors)
    special_chars = sum(1 for c in text if not c.isalnum() and c not in " .,;:!?'\"-\n")
    if special_chars / len(text) > 0.3:
        return ChunkQuality.LOW
    
    # Check for reasonable word count
    words = text.split()
    if len(words) < 10:
        return ChunkQuality.LOW
    
    # Check for very long unbroken strings (possible encoding issues)
    max_word_len = max(len(w) for w in words) if words else 0
    if max_word_len > 100:
        return ChunkQuality.LOW
    
    # If none of the above, assume medium quality
    # High quality would require more sophisticated analysis
    return ChunkQuality.MEDIUM
