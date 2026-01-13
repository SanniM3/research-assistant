"""Issue model for tracking problems that drive iteration."""
from datetime import datetime
from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, Field


class IssueSeverity(str, Enum):
    """Severity level of an issue."""
    BLOCKER = "blocker"
    MAJOR = "major"
    MINOR = "minor"


class IssueCategory(str, Enum):
    """Category of issue."""
    MISSING_CITATION = "missing_citation"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTION = "contradiction"
    THIN_COVERAGE = "thin_coverage"
    TAXONOMY_GAP = "taxonomy_gap"
    BENCHMARK_GAP = "benchmark_gap"
    MISSING_SEMINAL = "missing_seminal"
    MISSING_RECENT = "missing_recent"
    WEAK_EVIDENCE = "weak_evidence"
    STRUCTURAL = "structural"


class IssueStatus(str, Enum):
    """Current status of an issue."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"


class Issue(BaseModel):
    """
    Issue record for tracking problems that need resolution.
    
    Issues drive the iteration loop - the system continues researching
    and revising until all blockers are resolved and majors are below threshold.
    
    Attributes:
        issue_id: Unique identifier
        severity: How critical this issue is
        category: Type of issue
        description: Detailed description of the problem
        linked_section: Which outline section this affects
        linked_claim_ids: Claims related to this issue
        suggested_queries: Search queries that might resolve this
        suggested_papers: Specific papers to look for
        status: Current resolution status
        resolution_notes: How the issue was resolved
        created_at: When issue was identified
        resolved_at: When issue was resolved
        created_by: Agent that created this issue
    """
    issue_id: str
    severity: IssueSeverity
    category: IssueCategory
    description: str
    linked_section: Optional[str] = None
    linked_claim_ids: List[str] = Field(default_factory=list)
    suggested_queries: List[str] = Field(default_factory=list)
    suggested_papers: List[str] = Field(default_factory=list)
    status: IssueStatus = IssueStatus.OPEN
    resolution_notes: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None
    created_by: str = "verifier_agent"
    
    def resolve(self, notes: str = "") -> None:
        """Mark issue as resolved."""
        self.status = IssueStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.resolution_notes = notes
    
    def is_blocking(self) -> bool:
        """Check if this issue blocks completion."""
        return self.severity == IssueSeverity.BLOCKER and self.status == IssueStatus.OPEN
    
    def start_progress(self) -> None:
        """Mark issue as being worked on."""
        self.status = IssueStatus.IN_PROGRESS
