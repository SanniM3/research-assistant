"""Research state model for the LangGraph workflow."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage

from .paper import Paper
from .chunk import Chunk
from .claim import Claim
from .entity import Entity
from .relation import Relation
from .issue import Issue, IssueStatus, IssueSeverity, IssueCategory


class QuestionStatus(str, Enum):
    """Status of a research question."""
    OPEN = "open"
    PARTIALLY_ANSWERED = "partially_answered"
    ANSWERED = "answered"
    DEFERRED = "deferred"


class ResearchQuestion(BaseModel):
    """
    A research question that drives the investigation.
    
    Questions evolve during research: initial questions may spawn
    follow-ups, and new questions emerge from findings.
    """
    question_id: str
    text: str
    status: QuestionStatus = QuestionStatus.OPEN
    parent_id: Optional[str] = None
    supporting_claim_ids: List[str] = Field(default_factory=list)
    answer_summary: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    iteration_created: int = 0
    is_follow_up: bool = False


class OutlineSection(BaseModel):
    """A section in the survey outline."""
    section_id: str
    title: str
    description: str
    parent_id: Optional[str] = None
    order: int = 0
    required_elements: List[str] = Field(default_factory=list)
    min_claims: int = 3
    draft_content: str = ""
    is_complete: bool = False


class AcceptanceCriteria(BaseModel):
    """
    ARR-style acceptance criteria for research quality.
    
    Uses reviewer-style scoring (1-5 scale) as primary quality gate,
    supplemented by hard minimum requirements. Research continues until
    scores meet thresholds AND minimum requirements are satisfied.
    """
    # ARR-style score thresholds (1-5 scale, research continues below these)
    min_soundness: float = 3.0
    min_contribution: float = 3.0
    min_clarity: float = 3.0
    min_coverage: float = 3.0
    min_question_sufficiency: float = 3.5

    # Hard minimum requirements
    min_papers: int = 10
    min_answered_questions_ratio: float = 0.7
    require_seminal_papers: bool = True
    require_recent_papers: bool = True
    max_open_blockers: int = 0
    max_open_majors: int = 2
    max_open_follow_ups: int = 3


class ReviewScores(BaseModel):
    """ARR-style scores for the current research state."""
    soundness: float = 0.0
    contribution: float = 0.0
    clarity: float = 0.0
    coverage: float = 0.0
    question_sufficiency: float = 0.0


class CoverageScores(BaseModel):
    """Coverage metrics for the current research state."""
    taxonomy_coverage: float = 0.0
    benchmark_coverage: float = 0.0
    timeline_coverage: float = 0.0
    venue_diversity: float = 0.0
    claims_per_section: Dict[str, int] = Field(default_factory=dict)
    papers_per_category: Dict[str, int] = Field(default_factory=dict)
    

class QueryRecord(BaseModel):
    """Record of a search query execution."""
    query_id: str
    query_text: str
    source: str  # "arxiv", "web", etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    results_count: int = 0
    selected_count: int = 0


class ResearchState(BaseModel):
    """
    Complete state for the research workflow.
    
    This is the central state object passed through all LangGraph nodes.
    It contains everything needed to track research progress and make decisions.
    """
    # Core research parameters
    topic: str = ""
    user_constraints: Optional[str] = None
    output_language: str = "en"
    
    # Scope and planning
    scope: str = ""
    research_questions: List[ResearchQuestion] = Field(default_factory=list)
    outline: List[OutlineSection] = Field(default_factory=list)
    outline_finalized: bool = False
    acceptance_criteria: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    
    # Iteration tracking
    iteration: int = 0
    max_iterations: int = 5
    phase: str = "init"
    
    # Search and retrieval
    queries_run: List[QueryRecord] = Field(default_factory=list)
    pending_queries: List[str] = Field(default_factory=list)
    
    # Paper management
    candidate_papers: List[Paper] = Field(default_factory=list)
    selected_papers: List[str] = Field(default_factory=list)
    papers_ingested: Dict[str, Paper] = Field(default_factory=dict)
    
    # Knowledge base
    chunks: Dict[str, Chunk] = Field(default_factory=dict)
    claims: Dict[str, Claim] = Field(default_factory=dict)
    entities: Dict[str, Entity] = Field(default_factory=dict)
    relations: Dict[str, Relation] = Field(default_factory=dict)
    
    # Draft sections
    draft_sections: Dict[str, str] = Field(default_factory=dict)
    
    # Quality tracking
    issues: List[Issue] = Field(default_factory=list)
    coverage_scores: CoverageScores = Field(default_factory=CoverageScores)
    review_scores: ReviewScores = Field(default_factory=ReviewScores)
    
    # Bibliography and citation tracking
    bib_entries: Dict[str, Any] = Field(default_factory=dict)
    
    # Final output
    final_report: str = ""
    
    # Messages for user-facing communication (replacement semantics — not
    # accumulated across nodes to avoid unbounded state growth).
    messages: List[AnyMessage] = Field(default_factory=list)
    
    # Audit trail
    audit_log: List[Dict[str, Any]] = Field(default_factory=list)
    
    _MAX_AUDIT_LOG = 500

    def log_action(self, agent: str, action: str, details: Dict[str, Any] = None) -> None:
        """Add entry to audit log, capping at _MAX_AUDIT_LOG entries."""
        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": self.iteration,
            "agent": agent,
            "action": action,
            "details": details or {}
        })
        if len(self.audit_log) > self._MAX_AUDIT_LOG:
            self.audit_log = self.audit_log[-self._MAX_AUDIT_LOG:]
    
    def get_open_issues(self, severity: Optional[IssueSeverity] = None) -> List[Issue]:
        """Get all open issues, optionally filtered by severity."""
        open_issues = [i for i in self.issues if i.status == IssueStatus.OPEN]
        if severity:
            open_issues = [i for i in open_issues if i.severity == severity]
        return open_issues

    def get_follow_up_issues(self) -> List[Issue]:
        """Get all open NEEDS_FOLLOW_UP issues."""
        return [i for i in self.issues 
                if i.category == IssueCategory.NEEDS_FOLLOW_UP 
                and i.status == IssueStatus.OPEN]
    
    def has_blocking_issues(self) -> bool:
        """Check if there are any blocking issues."""
        return any(i.is_blocking() for i in self.issues)
    
    # --- Research question helpers ---

    def get_open_questions(self) -> List[ResearchQuestion]:
        """Get all questions not yet fully answered."""
        return [q for q in self.research_questions 
                if q.status in (QuestionStatus.OPEN, QuestionStatus.PARTIALLY_ANSWERED)]

    def get_answered_ratio(self) -> float:
        """Fraction of research questions that are answered."""
        if not self.research_questions:
            return 0.0
        answered = sum(1 for q in self.research_questions if q.status == QuestionStatus.ANSWERED)
        return answered / len(self.research_questions)
    
    def get_section_by_id(self, section_id: str) -> Optional[OutlineSection]:
        """Get outline section by ID."""
        for section in self.outline:
            if section.section_id == section_id:
                return section
        return None
