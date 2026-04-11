"""Gap Scorer agent - computes coverage metrics, evolves research questions, and identifies gaps."""
import json
from typing import Dict, Any, List
from datetime import datetime
import uuid

from ..models.state import (
    ResearchState, CoverageScores, ReviewScores,
    ResearchQuestion, QuestionStatus,
)
from ..models.issue import Issue, IssueSeverity, IssueCategory, IssueStatus
from ..models.claim import ClaimType
from ..models.entity import EntityType
from .base import get_llm, create_agent_message, parse_llm_json


def gap_scorer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Gap Scorer node - computes coverage, evaluates question answering,
    evolves research questions with follow-ups, and decides whether to
    continue iterating.
    """
    llm = get_llm()

    state.log_action("gap_scorer", "starting", {"iteration": state.iteration})

    # 1. Compute coverage scores
    coverage_scores = compute_coverage_scores(state)

    # 2. Evaluate research questions and evolve them
    updated_questions, follow_up_issues = evaluate_and_evolve_questions(state, llm)

    # 3. Compute ARR-style review scores
    review_scores = compute_review_scores(state, coverage_scores, updated_questions)

    # 4. Identify coverage gaps and create issues
    gap_issues = identify_gaps(state, coverage_scores)

    # 5. Merge all new issues
    existing_descriptions = {i.description for i in state.issues}
    new_issues = [i for i in gap_issues + follow_up_issues
                  if i.description not in existing_descriptions]
    all_issues = list(state.issues) + new_issues

    # 6. Determine whether to continue research
    should_continue = should_continue_research(
        state, coverage_scores, review_scores, updated_questions
    )

    if should_continue and state.iteration < state.max_iterations:
        next_phase = "search_planning"
        next_iteration = state.iteration + 1
    else:
        next_phase = "review"
        next_iteration = state.iteration

    state.log_action("gap_scorer", "completed", {
        "coverage": coverage_scores.model_dump(),
        "review_scores": review_scores.model_dump(),
        "questions_answered": sum(1 for q in updated_questions if q.status == QuestionStatus.ANSWERED),
        "questions_total": len(updated_questions),
        "new_issues": len(new_issues),
        "follow_ups_created": len(follow_up_issues),
        "continue_research": should_continue,
    })

    return {
        "coverage_scores": coverage_scores,
        "review_scores": review_scores,
        "research_questions": updated_questions,
        "issues": all_issues,
        "phase": next_phase,
        "iteration": next_iteration,
    }


# ---------------------------------------------------------------------------
# Question evaluation & evolution
# ---------------------------------------------------------------------------

def evaluate_and_evolve_questions(
    state: ResearchState, llm
) -> tuple[List[ResearchQuestion], List[Issue]]:
    """
    Ask the LLM to evaluate each open question against current claims,
    update question statuses, and identify follow-up questions.
    """
    open_qs = state.get_open_questions()
    if not open_qs:
        return list(state.research_questions), []

    claims_summary = [
        {"id": c.claim_id, "type": c.claim_type.value, "text": c.text[:200]}
        for c in list(state.claims.values())[:50]
    ]

    prompt = f"""Evaluate these research questions against the evidence gathered so far.

RESEARCH QUESTIONS:
{json.dumps([{"id": q.question_id, "text": q.text, "status": q.status.value} for q in open_qs], indent=2)}

CLAIMS COLLECTED ({len(state.claims)} total, showing first 50):
{json.dumps(claims_summary, indent=2)}

PAPERS INGESTED: {len(state.papers_ingested)}

For each question, respond in JSON:
{{
    "evaluations": [
        {{
            "question_id": "rq_1",
            "new_status": "open|partially_answered|answered",
            "answer_summary": "brief answer if partially/fully answered, else null",
            "supporting_claim_ids": ["claim_id_1"],
            "follow_up_questions": [
                "A deeper question raised by the current findings"
            ]
        }}
    ]
}}

Guidelines:
- Mark "answered" only if claims fully address the question.
- Mark "partially_answered" if we have some evidence but gaps remain.
- follow_up_questions are NEW questions that emerge from findings and need
  further research. Only add them when findings genuinely raise a probing
  question — do NOT manufacture them.
- Each follow_up_question should be specific and researchable.

Output ONLY valid JSON."""

    messages = create_agent_message("gap_scorer", prompt)
    response = llm.invoke(messages)

    # Build lookup for current questions
    q_map = {q.question_id: q.model_copy() for q in state.research_questions}
    follow_up_issues: List[Issue] = []
    new_follow_up_questions: List[ResearchQuestion] = []

    result = parse_llm_json(response.content, fallback=None, agent="gap_scorer")
    if result and isinstance(result, dict):
        for ev in result.get("evaluations", []):
            qid = ev.get("question_id", "")
            if qid not in q_map:
                continue
            q = q_map[qid]
            new_status = ev.get("new_status", q.status.value)
            valid_values = {m.value for m in QuestionStatus}
            if new_status in valid_values:
                q.status = QuestionStatus(new_status)
            q.answer_summary = ev.get("answer_summary") or q.answer_summary
            q.supporting_claim_ids = ev.get("supporting_claim_ids", q.supporting_claim_ids)

            for fq_text in ev.get("follow_up_questions", []):
                fq_id = f"rq_fu_{uuid.uuid4().hex[:6]}"
                new_follow_up_questions.append(ResearchQuestion(
                    question_id=fq_id,
                    text=fq_text,
                    status=QuestionStatus.OPEN,
                    parent_id=qid,
                    is_follow_up=True,
                    iteration_created=state.iteration,
                ))
                follow_up_issues.append(Issue(
                    issue_id=f"issue_{uuid.uuid4().hex[:8]}",
                    severity=IssueSeverity.MAJOR,
                    category=IssueCategory.NEEDS_FOLLOW_UP,
                    description=f"Follow-up needed: {fq_text}",
                    follow_up_questions=[fq_text],
                    suggested_queries=[fq_text],
                    status=IssueStatus.OPEN,
                    created_by="gap_scorer_agent",
                ))
    else:
        state.log_action("gap_scorer", "question_eval_parse_error", {})

    all_questions = list(q_map.values()) + new_follow_up_questions
    return all_questions, follow_up_issues


# ---------------------------------------------------------------------------
# ARR-style scoring
# ---------------------------------------------------------------------------

def compute_review_scores(
    state: ResearchState,
    coverage: CoverageScores,
    questions: List[ResearchQuestion],
) -> ReviewScores:
    """Compute ARR-style 1-5 scores from current research state."""
    scores = ReviewScores()

    # Soundness: proportion of verified claims
    total_claims = len(state.claims)
    if total_claims > 0:
        verified = sum(1 for c in state.claims.values() if c.is_verified)
        with_evidence = sum(1 for c in state.claims.values() if c.has_sufficient_evidence())
        scores.soundness = min(5.0, 1.0 + 4.0 * (0.4 * verified / total_claims + 0.6 * with_evidence / total_claims))
    else:
        scores.soundness = 1.0

    # Coverage: average of taxonomy + benchmark + timeline coverage mapped to 1-5
    avg_cov = (coverage.taxonomy_coverage + coverage.benchmark_coverage + coverage.timeline_coverage) / 3
    scores.coverage = min(5.0, 1.0 + 4.0 * avg_cov)

    # Clarity: based on whether sections exist and issues related to structure
    structural_issues = len([i for i in state.issues
                             if i.category == IssueCategory.STRUCTURAL and i.status == IssueStatus.OPEN])
    sections_written = sum(1 for s in state.outline if s.section_id in state.draft_sections)
    section_ratio = sections_written / max(len(state.outline), 1)
    clarity_penalty = min(1.0, structural_issues * 0.25)
    scores.clarity = min(5.0, max(1.0, 1.0 + 4.0 * section_ratio - clarity_penalty * 4.0))

    # Contribution: entity diversity + relation count
    entity_types = len(set(e.entity_type for e in state.entities.values()))
    relation_count = len(state.relations)
    contrib_signal = min(1.0, entity_types / 5) * 0.5 + min(1.0, relation_count / 10) * 0.5
    scores.contribution = min(5.0, 1.0 + 4.0 * contrib_signal)

    # Question sufficiency: fraction of questions answered
    if questions:
        answered = sum(1 for q in questions if q.status == QuestionStatus.ANSWERED)
        partial = sum(1 for q in questions if q.status == QuestionStatus.PARTIALLY_ANSWERED)
        ratio = (answered + 0.4 * partial) / len(questions)
        scores.question_sufficiency = min(5.0, 1.0 + 4.0 * ratio)
    else:
        scores.question_sufficiency = 1.0

    return scores


# ---------------------------------------------------------------------------
# Coverage computation (unchanged core, refactored)
# ---------------------------------------------------------------------------

def compute_coverage_scores(state: ResearchState) -> CoverageScores:
    """Compute all coverage metrics."""
    scores = CoverageScores()
    scores.taxonomy_coverage = compute_taxonomy_coverage(state)
    scores.benchmark_coverage = compute_benchmark_coverage(state)
    scores.timeline_coverage = compute_timeline_coverage(state)
    scores.venue_diversity = compute_venue_diversity(state)
    scores.claims_per_section = compute_claims_per_section(state)
    scores.papers_per_category = compute_papers_per_category(state)
    return scores


def compute_taxonomy_coverage(state: ResearchState) -> float:
    method_entities = [e for e in state.entities.values()
                       if e.entity_type == EntityType.METHOD]
    taxonomy_claims = [c for c in state.claims.values()
                       if c.claim_type in [ClaimType.DEFINITION, ClaimType.COMPARISON]]
    expected_methods = 5
    expected_taxonomy_claims = 10
    method_score = min(1.0, len(method_entities) / expected_methods)
    claim_score = min(1.0, len(taxonomy_claims) / expected_taxonomy_claims)
    return (method_score + claim_score) / 2


def compute_benchmark_coverage(state: ResearchState) -> float:
    benchmark_entities = [e for e in state.entities.values()
                         if e.entity_type in [EntityType.DATASET, EntityType.BENCHMARK]]
    result_claims = [c for c in state.claims.values()
                     if c.claim_type == ClaimType.EMPIRICAL_RESULT]
    expected_benchmarks = 3
    expected_results = 5
    benchmark_score = min(1.0, len(benchmark_entities) / expected_benchmarks)
    result_score = min(1.0, len(result_claims) / expected_results)
    return (benchmark_score + result_score) / 2


def compute_timeline_coverage(state: ResearchState) -> float:
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
    recent_score = min(1.0, recent_papers / 3)
    seminal_score = min(1.0, seminal_papers / 2)
    return (recent_score + seminal_score) / 2


def compute_venue_diversity(state: ResearchState) -> float:
    venues = set()
    for paper in state.papers_ingested.values():
        if paper.venue:
            venues.add(paper.venue.lower())
    return min(1.0, len(venues) / 3)


def compute_claims_per_section(state: ResearchState) -> Dict[str, int]:
    counts = {}
    for section in state.outline:
        section_claims = len([c for c in state.claims.values()])
        counts[section.section_id] = section_claims // len(state.outline) if state.outline else 0
    return counts


def compute_papers_per_category(state: ResearchState) -> Dict[str, int]:
    categories: Dict[str, int] = {}
    for paper in state.papers_ingested.values():
        domain = paper.metadata.domain or "uncategorized"
        categories[domain] = categories.get(domain, 0) + 1
    return categories


# ---------------------------------------------------------------------------
# Gap identification
# ---------------------------------------------------------------------------

def identify_gaps(state: ResearchState, scores: CoverageScores) -> List[Issue]:
    """Identify coverage gaps and create issues."""
    issues = []
    criteria = state.acceptance_criteria

    if scores.taxonomy_coverage < 0.7:
        issues.append(Issue(
            issue_id=f"issue_{uuid.uuid4().hex[:8]}",
            severity=IssueSeverity.MAJOR,
            category=IssueCategory.TAXONOMY_GAP,
            description=f"Taxonomy coverage is {scores.taxonomy_coverage:.0%}, below 70% threshold",
            suggested_queries=["taxonomy survey methods", "classification approaches"],
            status=IssueStatus.OPEN,
            created_by="gap_scorer_agent",
        ))

    if scores.benchmark_coverage < 0.6:
        issues.append(Issue(
            issue_id=f"issue_{uuid.uuid4().hex[:8]}",
            severity=IssueSeverity.MAJOR,
            category=IssueCategory.BENCHMARK_GAP,
            description=f"Benchmark coverage is {scores.benchmark_coverage:.0%}, below 60% threshold",
            suggested_queries=["benchmark dataset evaluation", "performance comparison"],
            status=IssueStatus.OPEN,
            created_by="gap_scorer_agent",
        ))

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


# ---------------------------------------------------------------------------
# Stopping decision
# ---------------------------------------------------------------------------

def should_continue_research(
    state: ResearchState,
    coverage: CoverageScores,
    review_scores: ReviewScores,
    questions: List[ResearchQuestion],
) -> bool:
    """
    Determine if the research loop should continue.

    Primary drivers (in priority order):
    1. Are there unanswered research questions?
    2. Are there open NEEDS_FOLLOW_UP issues?
    3. Do ARR-style scores meet thresholds?
    4. Are coverage minimums met?
    """
    criteria = state.acceptance_criteria

    # Unanswered questions
    if questions:
        answered_ratio = sum(1 for q in questions if q.status == QuestionStatus.ANSWERED) / len(questions)
        if answered_ratio < criteria.min_answered_questions_ratio:
            return True

    # Open follow-ups
    open_follow_ups = len([i for i in state.issues
                           if i.category == IssueCategory.NEEDS_FOLLOW_UP
                           and i.status == IssueStatus.OPEN])
    if open_follow_ups > criteria.max_open_follow_ups:
        return True

    # ARR score thresholds
    if review_scores.question_sufficiency < criteria.min_question_sufficiency:
        return True
    if review_scores.coverage < criteria.min_coverage:
        return True

    # Hard minimums
    if len(state.papers_ingested) < criteria.min_papers:
        return True

    # Blocking issues
    blockers = [i for i in state.issues
                if i.severity == IssueSeverity.BLOCKER and i.status == IssueStatus.OPEN]
    if blockers:
        return True

    # Major issue cap
    majors = [i for i in state.issues
              if i.severity == IssueSeverity.MAJOR and i.status == IssueStatus.OPEN]
    if len(majors) > criteria.max_open_majors:
        return True

    return False
