"""Gap Scorer agent - computes coverage metrics."""
import json
from typing import Dict, Any, List
from datetime import datetime
import uuid

from ..models.state import ResearchState, CoverageScores
from ..models.issue import Issue, IssueSeverity, IssueCategory, IssueStatus
from ..models.claim import ClaimType
from ..models.entity import EntityType
from .base import get_llm, create_agent_message


def gap_scorer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Gap Scorer node - computes coverage metrics and identifies gaps.
    
    Responsibilities:
    - Compute taxonomy coverage
    - Compute benchmark coverage
    - Compute timeline coverage (seminal + recent)
    - Compute venue diversity
    - Create issues for gaps
    - Decide if iteration should continue
    """
    llm = get_llm()
    
    state.log_action("gap_scorer", "starting", {"iteration": state.iteration})
    
    # Compute coverage scores
    coverage_scores = compute_coverage_scores(state)
    
    # Identify gaps and create issues
    gap_issues = identify_gaps(state, coverage_scores)
    
    # Merge with existing issues (don't duplicate)
    existing_issue_descriptions = {i.description for i in state.issues}
    new_issues = [i for i in gap_issues if i.description not in existing_issue_descriptions]
    all_issues = list(state.issues) + new_issues
    
    # Determine next phase
    should_continue = should_continue_research(state, coverage_scores)
    
    if should_continue and state.iteration < state.max_iterations:
        next_phase = "search_planning"
        next_iteration = state.iteration + 1
    else:
        next_phase = "review"
        next_iteration = state.iteration
    
    state.log_action("gap_scorer", "completed", {
        "coverage": coverage_scores.model_dump(),
        "new_issues": len(new_issues),
        "continue_research": should_continue,
    })
    
    return {
        "coverage_scores": coverage_scores,
        "issues": all_issues,
        "phase": next_phase,
        "iteration": next_iteration,
    }


def compute_coverage_scores(state: ResearchState) -> CoverageScores:
    """Compute all coverage metrics."""
    scores = CoverageScores()
    
    # Taxonomy coverage - based on entity types and claims
    scores.taxonomy_coverage = compute_taxonomy_coverage(state)
    
    # Benchmark coverage - datasets and evaluation entities
    scores.benchmark_coverage = compute_benchmark_coverage(state)
    
    # Timeline coverage - mix of seminal and recent
    scores.timeline_coverage = compute_timeline_coverage(state)
    
    # Venue diversity
    scores.venue_diversity = compute_venue_diversity(state)
    
    # Claims per section
    scores.claims_per_section = compute_claims_per_section(state)
    
    # Papers per category
    scores.papers_per_category = compute_papers_per_category(state)
    
    return scores


def compute_taxonomy_coverage(state: ResearchState) -> float:
    """Compute taxonomy coverage based on methods and their organization."""
    # Count method entities
    method_entities = [e for e in state.entities.values() 
                       if e.entity_type == EntityType.METHOD]
    
    # Count taxonomy-related claims
    taxonomy_claims = [c for c in state.claims.values()
                       if c.claim_type in [ClaimType.DEFINITION, ClaimType.COMPARISON]]
    
    # Expected minimums
    expected_methods = 5
    expected_taxonomy_claims = 10
    
    method_score = min(1.0, len(method_entities) / expected_methods)
    claim_score = min(1.0, len(taxonomy_claims) / expected_taxonomy_claims)
    
    return (method_score + claim_score) / 2


def compute_benchmark_coverage(state: ResearchState) -> float:
    """Compute benchmark/dataset coverage."""
    # Count dataset and benchmark entities
    benchmark_entities = [e for e in state.entities.values()
                         if e.entity_type in [EntityType.DATASET, EntityType.BENCHMARK]]
    
    # Count empirical result claims
    result_claims = [c for c in state.claims.values()
                     if c.claim_type == ClaimType.EMPIRICAL_RESULT]
    
    expected_benchmarks = 3
    expected_results = 5
    
    benchmark_score = min(1.0, len(benchmark_entities) / expected_benchmarks)
    result_score = min(1.0, len(result_claims) / expected_results)
    
    return (benchmark_score + result_score) / 2


def compute_timeline_coverage(state: ResearchState) -> float:
    """Compute timeline coverage (seminal + recent papers)."""
    current_year = datetime.now().year
    recent_threshold = current_year - 2
    seminal_threshold = current_year - 5
    
    recent_papers = 0
    seminal_papers = 0
    
    for paper in state.papers_ingested.values():
        if paper.year:
            if paper.year >= recent_threshold:
                recent_papers += 1
            if paper.year <= seminal_threshold or paper.metadata.is_seminal:
                seminal_papers += 1
    
    # Require at least 3 recent and 2 seminal
    recent_score = min(1.0, recent_papers / 3)
    seminal_score = min(1.0, seminal_papers / 2)
    
    return (recent_score + seminal_score) / 2


def compute_venue_diversity(state: ResearchState) -> float:
    """Compute venue diversity."""
    venues = set()
    
    for paper in state.papers_ingested.values():
        if paper.venue:
            venues.add(paper.venue.lower())
    
    # Require at least 3 different venues/sources
    return min(1.0, len(venues) / 3)


def compute_claims_per_section(state: ResearchState) -> Dict[str, int]:
    """Count claims assigned to each section."""
    # Simple heuristic - distribute claims based on type
    counts = {}
    
    for section in state.outline:
        section_claims = len([c for c in state.claims.values()])  # Simplified
        counts[section.section_id] = section_claims // len(state.outline) if state.outline else 0
    
    return counts


def compute_papers_per_category(state: ResearchState) -> Dict[str, int]:
    """Count papers per category/domain."""
    categories: Dict[str, int] = {}
    
    for paper in state.papers_ingested.values():
        domain = paper.metadata.domain or "uncategorized"
        categories[domain] = categories.get(domain, 0) + 1
    
    return categories


def identify_gaps(state: ResearchState, scores: CoverageScores) -> List[Issue]:
    """Identify gaps and create issues."""
    issues = []
    criteria = state.acceptance_criteria
    
    # Taxonomy gap
    if scores.taxonomy_coverage < criteria.taxonomy_coverage:
        issues.append(Issue(
            issue_id=f"issue_{uuid.uuid4().hex[:8]}",
            severity=IssueSeverity.MAJOR,
            category=IssueCategory.TAXONOMY_GAP,
            description=f"Taxonomy coverage is {scores.taxonomy_coverage:.0%}, below threshold {criteria.taxonomy_coverage:.0%}",
            suggested_queries=["taxonomy survey methods", "classification approaches"],
            status=IssueStatus.OPEN,
            created_by="gap_scorer_agent",
        ))
    
    # Benchmark gap
    if scores.benchmark_coverage < criteria.benchmark_coverage:
        issues.append(Issue(
            issue_id=f"issue_{uuid.uuid4().hex[:8]}",
            severity=IssueSeverity.MAJOR,
            category=IssueCategory.BENCHMARK_GAP,
            description=f"Benchmark coverage is {scores.benchmark_coverage:.0%}, below threshold {criteria.benchmark_coverage:.0%}",
            suggested_queries=["benchmark dataset evaluation", "performance comparison"],
            status=IssueStatus.OPEN,
            created_by="gap_scorer_agent",
        ))
    
    # Timeline gaps
    if scores.timeline_coverage < 0.5:
        if criteria.require_seminal_papers:
            issues.append(Issue(
                issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                severity=IssueSeverity.MAJOR,
                category=IssueCategory.MISSING_SEMINAL,
                description="Missing seminal/foundational papers",
                suggested_queries=["seminal foundational original"],
                status=IssueStatus.OPEN,
                created_by="gap_scorer_agent",
            ))
        
        if criteria.require_recent_papers:
            issues.append(Issue(
                issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                severity=IssueSeverity.MAJOR,
                category=IssueCategory.MISSING_RECENT,
                description="Missing recent papers (last 2 years)",
                suggested_queries=["2023 2024 recent latest"],
                status=IssueStatus.OPEN,
                created_by="gap_scorer_agent",
            ))
    
    # Paper count
    if len(state.papers_ingested) < criteria.min_papers:
        issues.append(Issue(
            issue_id=f"issue_{uuid.uuid4().hex[:8]}",
            severity=IssueSeverity.MAJOR,
            category=IssueCategory.THIN_COVERAGE,
            description=f"Only {len(state.papers_ingested)} papers ingested, minimum is {criteria.min_papers}",
            status=IssueStatus.OPEN,
            created_by="gap_scorer_agent",
        ))
    
    return issues


def should_continue_research(state: ResearchState, scores: CoverageScores) -> bool:
    """Determine if research loop should continue."""
    criteria = state.acceptance_criteria
    
    # Must meet coverage thresholds
    if scores.taxonomy_coverage < criteria.taxonomy_coverage:
        return True
    if scores.benchmark_coverage < criteria.benchmark_coverage:
        return True
    
    # Must have minimum papers
    if len(state.papers_ingested) < criteria.min_papers:
        return True
    
    # Check blocking issues
    blockers = [i for i in state.issues 
                if i.severity == IssueSeverity.BLOCKER and i.status == IssueStatus.OPEN]
    if blockers:
        return True
    
    # Check major issues
    majors = [i for i in state.issues
              if i.severity == IssueSeverity.MAJOR and i.status == IssueStatus.OPEN]
    if len(majors) > criteria.max_open_majors:
        return True
    
    return False
