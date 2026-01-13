"""Claim model for structured assertions extracted from papers."""
from datetime import datetime
from typing import Optional, List, Tuple
from enum import Enum
from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    """Types of claims that can be extracted."""
    DEFINITION = "definition"
    METHOD_SUMMARY = "method_summary"
    EMPIRICAL_RESULT = "empirical_result"
    THEORETICAL_RESULT = "theoretical_result"
    LIMITATION = "limitation"
    COMPARISON = "comparison"
    OPEN_PROBLEM = "open_problem"


class ConfidenceLevel(str, Enum):
    """Confidence in the claim extraction."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Evidence(BaseModel):
    """
    Evidence pointer linking a claim to its source.
    
    Attributes:
        chunk_id: ID of the chunk containing evidence
        quote_span: Character offsets of the supporting quote (start, end)
        snippet_hash: Hash of the specific supporting snippet
        relevance_score: How relevant this evidence is to the claim
    """
    chunk_id: str
    quote_span: Optional[Tuple[int, int]] = None
    snippet_hash: Optional[str] = None
    relevance_score: float = 1.0
    
    def get_citation_anchor(self) -> str:
        """Get citation anchor for internal references."""
        if self.quote_span:
            return f"{self.chunk_id}:{self.quote_span[0]}-{self.quote_span[1]}"
        return self.chunk_id


class Claim(BaseModel):
    """
    Structured assertion extracted from academic papers.
    
    Claims form the Claim Bank that drives grounded synthesis.
    Every factual statement in the output must map to claims with evidence.
    
    Attributes:
        claim_id: Unique identifier
        claim_type: Category of claim
        text: The claim text in canonical language
        normalized_form: Optional template form (e.g., "X improves Y on Z by Δ")
        entity_ids: List of entity IDs involved in this claim
        evidence: List of evidence pointers to source chunks
        confidence: Confidence level of extraction
        notes: Additional context, assumptions, evaluation details
        paper_id: Source paper ID
        extracted_by: Agent/version that extracted this claim
        created_at: Timestamp of extraction
        is_verified: Whether claim has been verified
        contradicts: IDs of claims this contradicts
    """
    claim_id: str
    claim_type: ClaimType
    text: str
    normalized_form: Optional[str] = None
    entity_ids: List[str] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    notes: Optional[str] = None
    paper_id: str
    extracted_by: str = "extractor_agent"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    is_verified: bool = False
    contradicts: List[str] = Field(default_factory=list)
    
    def has_sufficient_evidence(self) -> bool:
        """Check if claim has at least one evidence pointer."""
        return len(self.evidence) > 0
    
    def get_primary_evidence(self) -> Optional[Evidence]:
        """Get the highest relevance evidence."""
        if not self.evidence:
            return None
        return max(self.evidence, key=lambda e: e.relevance_score)
    
    def add_evidence(self, chunk_id: str, quote_span: Optional[Tuple[int, int]] = None,
                     relevance_score: float = 1.0) -> None:
        """Add evidence pointer to this claim."""
        evidence = Evidence(
            chunk_id=chunk_id,
            quote_span=quote_span,
            relevance_score=relevance_score
        )
        self.evidence.append(evidence)
    
    def to_citation_format(self) -> str:
        """Format claim with internal citation markers."""
        if not self.evidence:
            return f"{self.text} [MISSING_CITATION]"
        
        citations = [f"[@{e.chunk_id}]" for e in self.evidence]
        return f"{self.text} {' '.join(citations)}"
