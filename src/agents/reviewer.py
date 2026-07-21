"""Reviewer agent - ARR-style academic review."""
import json
from typing import Dict, Any, List
import uuid

from ..models.state import ResearchState
from ..models.issue import Issue, IssueSeverity, IssueCategory, IssueStatus
from .base import get_llm, create_agent_message, parse_llm_json


def reviewer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Reviewer node - conducts ARR-style review of the survey.
    
    Responsibilities:
    - Review full draft using academic rubric
    - Output strengths and weaknesses
    - Categorize issues
    - Provide actionable required changes
    - Suggest retrieval/extraction tasks for gaps
    """
    llm = get_llm(role="reviewer")
    kb = state.kb()
    
    state.log_action("reviewer", "starting", {})
    
    # Compile full draft
    full_draft = compile_draft_for_review(state, kb)
    
    # Conduct review
    review_result = conduct_review(
        draft=full_draft,
        topic=state.topic,
        papers_count=kb.reviewed_count(),
        claims_count=len(kb.all_claims()),
        llm=llm
    )
    
    # Convert review findings to issues
    new_issues = create_issues_from_review(review_result, state.issues)
    
    # Merge issues
    all_issues = list(state.issues) + new_issues

    # Actionable issues drive the loops: BLOCKERs always, MAJORs when they
    # accumulate. Retries are bounded by counters to prevent thrashing.
    actionable = [i for i in all_issues if i.status == IssueStatus.OPEN
                  and i.severity in (IssueSeverity.BLOCKER, IssueSeverity.MAJOR)]

    research_categories = {
        IssueCategory.MISSING_SEMINAL, IssueCategory.MISSING_RECENT,
        IssueCategory.BENCHMARK_GAP, IssueCategory.TAXONOMY_GAP,
        IssueCategory.THIN_COVERAGE, IssueCategory.NEEDS_FOLLOW_UP,
    }
    writing_categories = {
        IssueCategory.STRUCTURAL, IssueCategory.UNSUPPORTED_CLAIM,
        IssueCategory.MISSING_CITATION, IssueCategory.WEAK_EVIDENCE,
        IssueCategory.CONTRADICTION,
    }

    open_research = [i for i in actionable if i.category in research_categories]
    open_writing = [i for i in actionable if i.category in writing_categories]

    def _triggers(items):
        return any(i.severity == IssueSeverity.BLOCKER for i in items) or len(items) >= 2

    revision_count = state.revision_count
    resynth_count = state.resynth_count
    next_phase = "finalize"

    if (open_research and _triggers(open_research)
            and state.iteration < state.max_iterations
            and revision_count < state.max_revisions):
        next_phase = "revision"
        revision_count += 1
    elif (open_writing and _triggers(open_writing)
            and resynth_count < state.max_resynths):
        next_phase = "resynthesize"
        resynth_count += 1

    state.log_action("reviewer", "completed", {
        "new_issues": len(new_issues),
        "actionable_research": len(open_research),
        "actionable_writing": len(open_writing),
        "next_phase": next_phase,
        "revision_count": revision_count,
        "resynth_count": resynth_count,
    })

    return {
        "issues": all_issues,
        "phase": next_phase,
        "revision_count": revision_count,
        "resynth_count": resynth_count,
        "estimated_cost_usd": _cost(),
    }


def _cost() -> float:
    from .base import get_cost
    return round(get_cost().get("usd", 0.0), 4)


def compile_draft_for_review(state: ResearchState, kb) -> str:
    """Compile draft sections for review."""
    parts = []
    
    parts.append(f"# Survey: {state.topic}\n")
    parts.append(f"Scope: {state.scope}\n")
    parts.append(f"Papers reviewed: {kb.reviewed_count()}\n")
    parts.append(f"Claims extracted: {len(kb.all_claims())}\n\n")
    
    for section in sorted(state.outline, key=lambda s: s.order):
        content = state.draft_sections.get(section.section_id, "[Section not yet written]")
        parts.append(f"\n## {section.title}\n")
        parts.append(content[:3000])  # Limit per section
    
    return "\n".join(parts)


def conduct_review(draft: str, topic: str, papers_count: int, 
                   claims_count: int, llm) -> Dict[str, Any]:
    """Conduct ARR-style review of the survey."""
    
    prompt = f"""Conduct an academic review of this survey draft using the ARR (ACL Rolling Review) rubric.

TOPIC: {topic}
PAPERS REVIEWED: {papers_count}
CLAIMS EXTRACTED: {claims_count}

DRAFT:
{draft[:10000]}

Provide your review in JSON format:
{{
    "overall_assessment": "accept|revise|reject",
    "confidence": 1-5,
    "strengths": [
        "Strength 1",
        "Strength 2"
    ],
    "weaknesses": [
        {{
            "category": "missing_seminal|missing_benchmarks|unsupported_claims|structural|coverage|clarity",
            "severity": "blocker|major|minor",
            "description": "Detailed description",
            "section": "affected section or null",
            "suggested_action": "how to address this",
            "suggested_queries": ["query1", "query2"]
        }}
    ],
    "required_changes": [
        {{
            "section": "section name",
            "change": "what needs to change",
            "priority": "high|medium|low"
        }}
    ],
    "missing_elements": {{
        "seminal_papers": ["paper topics that should be included"],
        "recent_work": ["recent developments to cover"],
        "benchmarks": ["benchmarks/datasets to discuss"],
        "comparisons": ["comparisons to add"]
    }},
    "scores": {{
        "soundness": 1-5,
        "contribution": 1-5,
        "clarity": 1-5,
        "coverage": 1-5
    }},
    "summary": "One paragraph summary of the review"
}}

Review criteria:
1. SOUNDNESS: Are claims properly supported by evidence?
2. CONTRIBUTION: Does the survey provide valuable synthesis?
3. CLARITY: Is the writing clear and well-organized?
4. COVERAGE: Does it cover the important work in the field?

Be constructive but thorough. The goal is to help improve the survey.
Output ONLY valid JSON."""

    messages = create_agent_message("reviewer", prompt)
    response = llm.invoke(messages)
    
    parsed = parse_llm_json(response.content, fallback=None, agent="reviewer")
    if parsed and isinstance(parsed, dict):
        return parsed

    return {
            "overall_assessment": "revise",
            "confidence": 3,
            "strengths": ["Survey addresses an important topic"],
            "weaknesses": [
                {
                    "category": "coverage",
                    "severity": "major",
                    "description": "Unable to fully assess due to parsing error",
                    "suggested_action": "Manual review recommended",
                }
            ],
            "required_changes": [],
            "missing_elements": {},
            "scores": {"soundness": 3, "contribution": 3, "clarity": 3, "coverage": 3},
            "summary": "Review parsing failed - manual review recommended",
        }


def create_issues_from_review(review: Dict[str, Any], existing_issues: List[Issue]) -> List[Issue]:
    """Convert review findings to Issue objects."""
    new_issues = []
    existing_descriptions = {i.description.lower() for i in existing_issues}
    
    category_map = {
        "missing_seminal": IssueCategory.MISSING_SEMINAL,
        "missing_benchmarks": IssueCategory.BENCHMARK_GAP,
        "unsupported_claims": IssueCategory.UNSUPPORTED_CLAIM,
        "structural": IssueCategory.STRUCTURAL,
        "coverage": IssueCategory.THIN_COVERAGE,
        "clarity": IssueCategory.STRUCTURAL,
        "needs_follow_up": IssueCategory.NEEDS_FOLLOW_UP,
    }
    
    for weakness in review.get("weaknesses", []):
        desc = weakness.get("description", "")
        
        # Skip if similar issue exists
        if desc.lower() in existing_descriptions:
            continue
        
        # Map category
        category_str = weakness.get("category", "coverage")
        category = category_map.get(category_str, IssueCategory.THIN_COVERAGE)
        
        # Map severity
        severity_str = weakness.get("severity", "minor")
        severity_map = {
            "blocker": IssueSeverity.BLOCKER,
            "major": IssueSeverity.MAJOR,
            "minor": IssueSeverity.MINOR,
        }
        severity = severity_map.get(severity_str, IssueSeverity.MINOR)
        
        issue = Issue(
            issue_id=f"issue_{uuid.uuid4().hex[:8]}",
            severity=severity,
            category=category,
            description=desc,
            linked_section=weakness.get("section"),
            suggested_queries=weakness.get("suggested_queries", []),
            status=IssueStatus.OPEN,
            created_by="reviewer_agent",
        )
        new_issues.append(issue)
    
    # Create issues for missing elements
    missing = review.get("missing_elements", {})
    
    if missing.get("seminal_papers"):
        issue = Issue(
            issue_id=f"issue_{uuid.uuid4().hex[:8]}",
            severity=IssueSeverity.MAJOR,
            category=IssueCategory.MISSING_SEMINAL,
            description=f"Missing seminal papers on: {', '.join(missing['seminal_papers'][:3])}",
            suggested_queries=missing["seminal_papers"][:3],
            status=IssueStatus.OPEN,
            created_by="reviewer_agent",
        )
        new_issues.append(issue)
    
    if missing.get("benchmarks"):
        issue = Issue(
            issue_id=f"issue_{uuid.uuid4().hex[:8]}",
            severity=IssueSeverity.MAJOR,
            category=IssueCategory.BENCHMARK_GAP,
            description=f"Missing benchmarks: {', '.join(missing['benchmarks'][:3])}",
            suggested_queries=[f"{b} benchmark" for b in missing["benchmarks"][:3]],
            status=IssueStatus.OPEN,
            created_by="reviewer_agent",
        )
        new_issues.append(issue)
    
    return new_issues


