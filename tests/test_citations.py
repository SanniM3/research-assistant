"""Citation resolution must be exact/normalized-id only (no fuzzy guessing)."""
from src.agents.citation_manager import resolve_paper_id, convert_to_ieee_citations
from src.models.paper import Paper


def _papers():
    return {
        "arxiv:2301.12345": Paper(paper_id="arxiv:2301.12345", title="Attention Is All You Need",
                                  authors=["Vaswani"], year=2017, arxiv_id="2301.12345"),
        "arxiv:1810.04805": Paper(paper_id="arxiv:1810.04805", title="BERT Pretraining",
                                  authors=["Devlin"], year=2018, arxiv_id="1810.04805"),
    }


def test_exact_and_normalized_resolution():
    papers = _papers()
    assert resolve_paper_id("arxiv:2301.12345", papers) == "arxiv:2301.12345"
    assert resolve_paper_id("2301.12345", papers) == "arxiv:2301.12345"
    assert resolve_paper_id("2301.12345v2", papers) == "arxiv:2301.12345"


def test_no_fuzzy_title_or_substring_match():
    papers = _papers()
    # Title words must NOT resolve (this used to misattribute citations).
    assert resolve_paper_id("attention need", papers) is None
    assert resolve_paper_id("BERT", papers) is None
    assert resolve_paper_id("unknown_ref", papers) is None


def test_unresolved_citation_is_dropped_not_marked():
    papers = _papers()
    # paper_to_number maps resolved ids to numbers
    p2n = {"arxiv:2301.12345": 1}
    text = "Transformers are strong [@arxiv:2301.12345] and so is [@ghost_paper]."
    out = convert_to_ieee_citations(text, p2n, {}, papers)
    assert "[1]" in out
    assert "?" not in out          # no [?...] debug markers
    assert "ghost_paper" not in out
