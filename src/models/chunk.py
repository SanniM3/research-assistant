"""Chunk model for grounding units of text."""
from datetime import datetime
from typing import Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field
import hashlib


class SourceType(str, Enum):
    """Type of source the chunk was extracted from."""
    PDF_TEXT = "pdf_text"
    ARXIV_HTML = "arxiv_html"
    WEB_HTML = "web_html"


class ChunkQuality(str, Enum):
    """Quality assessment of the extracted chunk."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ChunkMetadata(BaseModel):
    """Additional metadata for a chunk."""
    heading: Optional[str] = None
    subheading: Optional[str] = None
    is_table: bool = False
    is_figure_caption: bool = False
    is_equation: bool = False
    contains_citation: bool = False
    language: str = "en"


class Chunk(BaseModel):
    """
    Chunk record representing a grounding unit of text.
    
    Chunks are the fundamental units for RAG retrieval and citation.
    Every claim in the synthesized paper must trace back to chunk evidence.
    
    Attributes:
        chunk_id: Stable, deterministic ID (hash-based)
        paper_id: Reference to parent paper
        source_type: Type of source (PDF, HTML, etc.)
        section_path: Hierarchical section path (e.g., "Introduction > Related Work")
        page_span: Page range if from PDF (start, end)
        paragraph_span: Paragraph indices or DOM path
        text: The actual chunk text content
        hash: Content hash for deduplication
        quality: Quality assessment of extraction
        metadata: Additional structured metadata
        embedding: Optional pre-computed embedding
        created_at: Timestamp of creation
    """
    chunk_id: str = Field(..., description="Stable chunk identifier")
    paper_id: str = Field(..., description="Parent paper ID")
    source_type: SourceType
    section_path: str = Field(default="", description="Hierarchical section path")
    page_span: Optional[Tuple[int, int]] = None
    paragraph_span: Optional[Tuple[int, int]] = None
    text: str
    hash: str = Field(default="", description="Content hash for dedup")
    quality: ChunkQuality = ChunkQuality.MEDIUM
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    embedding: Optional[list] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def model_post_init(self, __context) -> None:
        """Generate hash after initialization if not provided."""
        if not self.hash:
            self.hash = self._compute_hash()
        if not self.chunk_id:
            self.chunk_id = self._generate_chunk_id()
    
    def _compute_hash(self) -> str:
        """Compute content hash for deduplication."""
        content = f"{self.paper_id}:{self.text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_chunk_id(self) -> str:
        """Generate a stable chunk ID."""
        return f"chunk:{self.hash[:16]}"
    
    @classmethod
    def create(cls, paper_id: str, text: str, source_type: SourceType,
               section_path: str = "", **kwargs) -> "Chunk":
        """Factory method to create a chunk with auto-generated IDs."""
        content_hash = hashlib.sha256(f"{paper_id}:{text}".encode()).hexdigest()
        chunk_id = f"chunk:{content_hash[:16]}"
        return cls(
            chunk_id=chunk_id,
            paper_id=paper_id,
            text=text,
            source_type=source_type,
            section_path=section_path,
            hash=content_hash,
            **kwargs
        )
