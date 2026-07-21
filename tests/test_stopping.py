"""Stopping criteria are driven by extracted knowledge, not raw paper counts."""
import os

from src.agents.gap_scorer import should_continue_research
from src.models.state import (
    ResearchState, AcceptanceCriteria, CoverageScores, ReviewScores,
    ResearchQuestion, QuestionStatus,
)
from src.storage.knowledge_base import KnowledgeBase
from src.models.paper import Paper
from src.models.claim import Claim, ClaimType


def _state(**kw):
    s = ResearchState(topic="t", corpus_id="stop-test")
    s.acceptance_criteria = AcceptanceCriteria(min_papers=3)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def _kb(tmp_path, n_papers=0, n_claims=0):
    kb = KnowledgeBase("stop-test", persist_dir=os.path.join(tmp_path, "s"), enable_persistence=False)
    for i in range(n_papers):
        kb.upsert_paper(Paper(paper_id=f"arxiv:{i}", title=f"p{i}", ingestion_status="complete"))
    for i in range(n_claims):
        kb.upsert_claims([Claim(claim_id=f"c{i}", claim_type=ClaimType.METHOD_SUMMARY,
                                text=f"claim {i}", paper_id="arxiv:0")])
    return kb


def _good_scores():
    cov = CoverageScores(taxonomy_coverage=1.0, benchmark_coverage=1.0, timeline_coverage=1.0)
    rev = ReviewScores(coverage=5.0, question_sufficiency=5.0)
    return cov, rev


def test_continue_when_too_few_papers(tmp_path):
    state = _state()
    kb = _kb(tmp_path, n_papers=1, n_claims=30)
    cov, rev = _good_scores()
    qs = [ResearchQuestion(question_id="q1", text="?", status=QuestionStatus.ANSWERED)]
    assert should_continue_research(state, kb, cov, rev, qs) is True


def test_continue_when_too_few_claims(tmp_path):
    state = _state()
    kb = _kb(tmp_path, n_papers=5, n_claims=2)  # papers ok, but almost no claims
    cov, rev = _good_scores()
    qs = [ResearchQuestion(question_id="q1", text="?", status=QuestionStatus.ANSWERED)]
    assert should_continue_research(state, kb, cov, rev, qs) is True


def test_stop_when_criteria_met(tmp_path):
    state = _state()
    kb = _kb(tmp_path, n_papers=5, n_claims=40)
    cov, rev = _good_scores()
    qs = [ResearchQuestion(question_id="q1", text="?", status=QuestionStatus.ANSWERED)]
    assert should_continue_research(state, kb, cov, rev, qs) is False
