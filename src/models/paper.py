"""Paper model for tracking academic papers."""
from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field
import hashlib


class MetadataConfidence(str, Enum):
    """Confidence level for paper metadata."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PaperMetadata(BaseModel):
    """Additional metadata for a paper."""
    categories: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    domain: Optional[str] = None
    task: Optional[str] = None
    method_type: Optional[str] = None
    dataset_names: List[str] = Field(default_factory=list)
    is_seminal: bool = False
    citation_count: Optional[int] = None


class Paper(BaseModel):
    """
    Paper record for tracking ingested academic papers.
    
    Attributes:
        paper_id: Stable internal ID (prefer DOI or arXiv ID)
        title: Paper title
        authors: List of author names
        year: Publication year
        venue: Publication venue (conference/journal)
        doi: Digital Object Identifier
        arxiv_id: arXiv identifier
        url_list: List of URLs where paper can be accessed
        abstract: Paper abstract
        language: Primary language of the paper
        version_group_id: Groups arXiv versions / extensions
        retrieved_at: Timestamp of retrieval
        license_notes: License/usage information
        metadata_confidence: Confidence in metadata accuracy
        metadata: Additional structured metadata
        is_ingested: Whether full text has been ingested
        ingestion_status: Status of ingestion process
    """
    paper_id: str = Field(..., description="Stable internal ID")
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    arxiv_id: Optional[str] = None
    url_list: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    language: str = "en"
    version_group_id: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    license_notes: Optional[str] = None
    metadata_confidence: MetadataConfidence = MetadataConfidence.MEDIUM
    metadata: PaperMetadata = Field(default_factory=PaperMetadata)
    is_ingested: bool = False
    ingestion_status: str = "pending"
    
    @classmethod
    def generate_paper_id(cls, doi: Optional[str] = None, arxiv_id: Optional[str] = None, 
                          title: Optional[str] = None) -> str:
        """Generate a stable paper ID from available identifiers."""
        if doi:
            return f"doi:{doi}"
        if arxiv_id:
            return f"arxiv:{arxiv_id}"
        if title:
            # Normalize title and create hash
            normalized = title.lower().strip()
            hash_val = hashlib.sha256(normalized.encode()).hexdigest()[:12]
            return f"title:{hash_val}"
        raise ValueError("At least one identifier (doi, arxiv_id, or title) is required")
    
    def get_primary_url(self) -> Optional[str]:
        """Get the primary URL for accessing this paper."""
        if self.arxiv_id:
            return f"https://arxiv.org/abs/{self.arxiv_id}"
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return self.url_list[0] if self.url_list else None
    
    def get_pdf_url(self) -> Optional[str]:
        """Get URL for PDF version if available."""
        if self.arxiv_id:
            return f"https://arxiv.org/pdf/{self.arxiv_id}.pdf"
        # Check for direct PDF links in url_list
        for url in self.url_list:
            if url.endswith(".pdf"):
                return url
        return None
