"""Research state model for the LangGraph workflow."""
from datetime import datetime
from typing import Optional, List, Dict, Any, Annotated
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from .paper import Paper
from .chunk import Chunk
from .claim import Claim
from .entity import Entity
from .relation import Relation
from .issue import Issue, IssueStatus, IssueSeverity


class OutlineSection(BaseModel):
    """A section in the survey outline."""
    section_id: str
    title: str
    description: str
    parent_id: Optional[str] = None
    order: int = 0
    required_elements: List[str] = Field(default_factory=list)  # e.g., ["taxonomy", "comparison_table"]
    min_claims: int = 3
    draft_content: str = ""
    is_complete: bool = False


class AcceptanceCriteria(BaseModel):
    """Criteria that must be met to complete research."""
    min_papers: int = 10
    min_claims_per_section: int = 3
    taxonomy_coverage: float = 0.7
    benchmark_coverage: float = 0.6
    require_seminal_papers: bool = True
    require_recent_papers: bool = True  # Last 2 years
    max_open_blockers: int = 0
    max_open_majors: int = 2


class CoverageScores(BaseModel):
    """Coverage metrics for the current research state."""
    taxonomy_coverage: float = 0.0
    benchmark_coverage: float = 0.0
    timeline_coverage: float = 0.0  # Mix of seminal + recent
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
    research_questions: List[str] = Field(default_factory=list)
    outline: List[OutlineSection] = Field(default_factory=list)
    acceptance_criteria: AcceptanceCriteria = Field(default_factory=AcceptanceCriteria)
    
    # Iteration tracking
    iteration: int = 0
    max_iterations: int = 5
    phase: str = "init"  # init, planning, research, synthesis, review, finalize
    
    # Search and retrieval
    queries_run: List[QueryRecord] = Field(default_factory=list)
    pending_queries: List[str] = Field(default_factory=list)
    
    # Paper management
    candidate_papers: List[Paper] = Field(default_factory=list)
    selected_papers: List[str] = Field(default_factory=list)  # paper_ids
    papers_ingested: Dict[str, Paper] = Field(default_factory=dict)  # paper_id -> Paper
    
    # Knowledge base
    chunks: Dict[str, Chunk] = Field(default_factory=dict)  # chunk_id -> Chunk
    claims: Dict[str, Claim] = Field(default_factory=dict)  # claim_id -> Claim
    entities: Dict[str, Entity] = Field(default_factory=dict)  # entity_id -> Entity
    relations: Dict[str, Relation] = Field(default_factory=dict)  # relation_id -> Relation
    
    # Draft sections
    draft_sections: Dict[str, str] = Field(default_factory=dict)  # section_id -> content
    
    # Quality tracking
    issues: List[Issue] = Field(default_factory=list)
    coverage_scores: CoverageScores = Field(default_factory=CoverageScores)
    
    # Bibliography
    bib_entries: Dict[str, str] = Field(default_factory=dict)  # citekey -> bibtex
    
    # Final output
    final_report: str = ""
    
    # Message history for agent communication
    messages: Annotated[List[AnyMessage], add_messages] = Field(default_factory=list)
    
    # Audit trail
    audit_log: List[Dict[str, Any]] = Field(default_factory=list)
    
    def log_action(self, agent: str, action: str, details: Dict[str, Any] = None) -> None:
        """Add entry to audit log."""
        self.audit_log.append({
            "timestamp": datetime.utcnow().isoformat(),
            "iteration": self.iteration,
            "agent": agent,
            "action": action,
            "details": details or {}
        })
    
    def get_open_issues(self, severity: Optional[IssueSeverity] = None) -> List[Issue]:
        """Get all open issues, optionally filtered by severity."""
        open_issues = [i for i in self.issues if i.status == IssueStatus.OPEN]
        if severity:
            open_issues = [i for i in open_issues if i.severity == severity]
        return open_issues
    
    def has_blocking_issues(self) -> bool:
        """Check if there are any blocking issues."""
        return any(i.is_blocking() for i in self.issues)
    
    def get_papers_for_section(self, section_id: str) -> List[Paper]:
        """Get papers relevant to a specific section."""
        section_claims = [c for c in self.claims.values() 
                         if any(section_id in str(e) for e in c.evidence)]
        paper_ids = set(c.paper_id for c in section_claims)
        return [self.papers_ingested[pid] for pid in paper_ids if pid in self.papers_ingested]
    
    def get_claims_for_section(self, section_id: str) -> List[Claim]:
        """Get claims relevant to a specific section (to be implemented with section mapping)."""
        # For now, return all claims - section mapping to be refined
        return list(self.claims.values())
    
    def should_stop(self) -> bool:
        """Check if research loop should terminate."""
        # Check iteration limit
        if self.iteration >= self.max_iterations:
            return True
        
        # Check blocking issues
        if self.has_blocking_issues():
            return False
        
        # Check coverage criteria
        criteria = self.acceptance_criteria
        scores = self.coverage_scores
        
        if scores.taxonomy_coverage < criteria.taxonomy_coverage:
            return False
        if scores.benchmark_coverage < criteria.benchmark_coverage:
            return False
        
        # Check minimum papers
        if len(self.papers_ingested) < criteria.min_papers:
            return False
        
        # Check open majors
        open_majors = len(self.get_open_issues(IssueSeverity.MAJOR))
        if open_majors > criteria.max_open_majors:
            return False
        
        return True
    
    def get_section_by_id(self, section_id: str) -> Optional[OutlineSection]:
        """Get outline section by ID."""
        for section in self.outline:
            if section.section_id == section_id:
                return section
        return None
