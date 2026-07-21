#!/usr/bin/env python3
"""
Grounding / faithfulness evaluation for a Research Assistant run.

Operates on a saved final_state.json (produced by inspect_run.py). It evaluates
two objects:

  (A) The CLAIM BANK   - does each Claim follow from the Chunk it cites?
  (B) The REPORT prose - does each factual sentence follow from the paper it cites?

Tiers:
  L0 Structural (free)  : evidence pointers exist & resolve; citations resolve;
                          no unresolved [?] markers; citation coverage of factual
                          sentences.
  L1 Lexical (free)     : claim<->evidence token overlap; numbers in a claim must
                          appear in its evidence (fabricated-statistic detector).
  L2 LLM-judge (costs $) : NLI-style entailment of claims by evidence, and of
                          report sentences by their cited source. Enable with
                          --llm-judge. Use --sample to cap cost.

Outputs (next to the state file, or --out-dir):
  grounding_eval.json   - machine-readable per-item verdicts + metrics
  grounding_eval.md     - human-readable scorecard + lists of failures

Usage:
  python evaluate_grounding.py run_output/<ts>/final_state.json
  python evaluate_grounding.py run_output/<ts>/final_state.json --llm-judge --sample 40
"""
import argparse
import json
import os
import re
from collections import Counter

from src.models.state import ResearchState

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_STOP = set("""a an the of for to in on and or with without from by as is are was were be been
being this that these those it its their our your his her they we you i he she which who whom
whose what when where why how can could should would may might will shall do does did has have
had not no nor than then thus so such via using used use also more most much many few via into
over under between within across per via et al""".split())

_STRONG_CLAIM_WORDS = [
    "state-of-the-art", "state of the art", "sota", "first", "novel", "best",
    "outperforms", "outperform", "significantly", "achieves", "achieve",
    "improves", "improve", "guarantees", "proves", "demonstrates",
]

_CITE_RE = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")     # [3]  [3, 5]  [3-6]
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")
_UNRESOLVED_RE = re.compile(r"\[\?")


def load_state(path: str) -> ResearchState:
    with open(path, "r", encoding="utf-8") as f:
        return ResearchState(**json.load(f))


def get_kb(state: ResearchState):
    """Return the persistent KnowledgeBase for a run's corpus (chunks/claims live here)."""
    from src.storage.registry import get_knowledge_base, derive_corpus_id
    corpus_id = state.corpus_id or derive_corpus_id(state.topic)
    return get_knowledge_base(corpus_id)


def content_tokens(text: str) -> set:
    toks = re.split(r"[^A-Za-z0-9]+", (text or "").lower())
    return {t for t in toks if len(t) >= 3 and t not in _STOP}


def numbers_in(text: str) -> set:
    # normalise: drop thousands separators so "1,000" matches "1000"
    return set(_NUM_RE.findall((text or "").replace(",", "")))


def split_sentences(report: str) -> list:
    """Crude but adequate sentence splitter; strips markdown headings/bullets."""
    lines = []
    for line in report.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("|") or s.startswith("---"):
            continue
        if s.startswith(("- ", "* ", "> ")):
            s = s[2:]
        lines.append(s)
    text = " ".join(lines)
    # split on . ! ? followed by space + capital / digit / citation
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[])", text)
    return [p.strip() for p in parts if len(p.strip()) > 25]


def cited_numbers(sentence: str) -> list:
    nums = []
    for grp in _CITE_RE.findall(sentence):
        for token in re.split(r"[,\s]+", grp):
            if "-" in token:
                a, b = token.split("-", 1)
                if a.isdigit() and b.isdigit():
                    nums.extend(range(int(a), int(b) + 1))
            elif token.isdigit():
                nums.append(int(token))
    return sorted(set(nums))


def is_factual(sentence: str) -> bool:
    """Heuristic: a sentence that *asserts something checkable*."""
    low = sentence.lower()
    if _NUM_RE.search(sentence):
        return True
    return any(w in low for w in _STRONG_CLAIM_WORDS)


# --------------------------------------------------------------------------- #
# L0 + L1: claim bank
# --------------------------------------------------------------------------- #

def eval_claim_bank(state: ResearchState, kb) -> dict:
    chunks = kb.chunks_map()
    rows = []
    for cid, c in kb.claims_map().items():
        ev_ids = [e.chunk_id for e in c.evidence]
        present = [eid for eid in ev_ids if eid in chunks]
        has_ev = len(ev_ids) > 0
        resolves = has_ev and len(present) == len(ev_ids)

        # L1 lexical overlap + number support against present evidence chunks
        ev_text = " ".join(chunks[eid].text for eid in present)
        ctoks, etoks = content_tokens(c.text), content_tokens(ev_text)
        overlap = (len(ctoks & etoks) / len(ctoks)) if ctoks else 0.0

        claim_nums = numbers_in(c.text)
        ev_nums = numbers_in(ev_text)
        missing_nums = sorted(claim_nums - ev_nums)

        rows.append({
            "claim_id": cid,
            "type": c.claim_type.value,
            "paper_id": c.paper_id,
            "text": c.text,
            "has_evidence": has_ev,
            "evidence_chunk_ids": ev_ids,
            "evidence_resolves": resolves,
            "lexical_overlap": round(overlap, 3),
            "claim_numbers": sorted(claim_nums),
            "numbers_missing_from_evidence": missing_nums,
        })

    n = len(rows) or 1
    return {
        "rows": rows,
        "metrics": {
            "n_claims": len(rows),
            "with_evidence": sum(r["has_evidence"] for r in rows),
            "evidence_resolves": sum(r["evidence_resolves"] for r in rows),
            "claim_grounding_rate": round(sum(r["evidence_resolves"] for r in rows) / n, 3),
            "low_lexical_overlap(<0.05)": sum(r["lexical_overlap"] < 0.05 for r in rows),
            "claims_with_numbers": sum(bool(r["claim_numbers"]) for r in rows),
            "claims_with_unsupported_numbers": sum(bool(r["numbers_missing_from_evidence"]) for r in rows),
        },
    }


# --------------------------------------------------------------------------- #
# L0: report citations
# --------------------------------------------------------------------------- #

def eval_report_citations(state: ResearchState) -> dict:
    report = state.final_report or ""
    # valid citation numbers = those assigned by the citation manager
    p2n = {}
    raw = state.bib_entries.get("_paper_to_number")
    if raw:
        try:
            p2n = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            p2n = {}
    valid_numbers = set(p2n.values())
    num_to_paper = {v: k for k, v in p2n.items()}

    sentences = split_sentences(report)
    factual = [s for s in sentences if is_factual(s)]
    factual_cited = [s for s in factual if cited_numbers(s)]

    all_cited = set()
    for s in sentences:
        all_cited.update(cited_numbers(s))
    invalid = sorted(n for n in all_cited if valid_numbers and n not in valid_numbers)

    return {
        "metrics": {
            "n_sentences": len(sentences),
            "n_factual_sentences": len(factual),
            "factual_sentences_cited": len(factual_cited),
            "citation_coverage": round(len(factual_cited) / (len(factual) or 1), 3),
            "distinct_citations_used": len(all_cited),
            "valid_reference_numbers": len(valid_numbers),
            "invalid_citation_numbers": invalid,
            "unresolved_markers": len(_UNRESOLVED_RE.findall(report)),
        },
        "_num_to_paper": num_to_paper,
        "_sentences": sentences,
        "_factual": factual,
    }


# --------------------------------------------------------------------------- #
# L2: LLM-as-judge (optional)
# --------------------------------------------------------------------------- #

JUDGE_SYS = ("You are a strict scientific fact-checker. Given EVIDENCE and a "
             "STATEMENT, decide whether the evidence supports the statement. "
             "Answer ONLY with JSON.")


def _judge(llm, evidence: str, statement: str, parse_llm_json):
    prompt = f"""EVIDENCE:
\"\"\"{evidence[:4000]}\"\"\"

STATEMENT:
\"\"\"{statement}\"\"\"

Does the EVIDENCE support the STATEMENT? Consider only the evidence shown.
Respond ONLY as JSON:
{{"verdict": "supported|partial|unsupported|contradicted", "reason": "one short sentence"}}"""
    from langchain_core.messages import SystemMessage, HumanMessage
    resp = llm.invoke([SystemMessage(content=JUDGE_SYS), HumanMessage(content=prompt)])
    out = parse_llm_json(resp.content, fallback={"verdict": "unparseable", "reason": ""}, agent="grounding_judge")
    if not isinstance(out, dict):
        out = {"verdict": "unparseable", "reason": ""}
    return out


def judge_claims(state, kb, sample, llm, parse_llm_json):
    chunks = kb.chunks_map()
    items = kb.all_claims()
    if sample:
        items = items[:sample]
    results = []
    for c in items:
        ev = " ".join(chunks[e.chunk_id].text for e in c.evidence if e.chunk_id in chunks)
        if not ev:
            results.append({"claim_id": c.claim_id, "verdict": "no_evidence", "text": c.text})
            continue
        v = _judge(llm, ev, c.text, parse_llm_json)
        results.append({"claim_id": c.claim_id, "verdict": v.get("verdict"),
                        "reason": v.get("reason"), "text": c.text})
    counts = Counter(r["verdict"] for r in results)
    n = len(results) or 1
    return {
        "rows": results,
        "metrics": {
            "n_judged": len(results),
            "verdict_counts": dict(counts),
            "claim_faithfulness": round(counts.get("supported", 0) / n, 3),
        },
    }


def judge_report(state, kb, report_eval, sample, llm, parse_llm_json):
    num_to_paper = report_eval["_num_to_paper"]
    # build per-paper evidence from its chunks
    chunks_by_paper = {}
    for ch in kb.all_chunks():
        chunks_by_paper.setdefault(ch.paper_id, []).append(ch.text)

    cited_sentences = [s for s in report_eval["_sentences"] if cited_numbers(s)]
    if sample:
        cited_sentences = cited_sentences[:sample]

    results = []
    for s in cited_sentences:
        nums = cited_numbers(s)
        ev_parts = []
        for nnum in nums:
            pid = num_to_paper.get(nnum)
            if pid and pid in chunks_by_paper:
                ev_parts.extend(chunks_by_paper[pid][:3])
        ev = "\n---\n".join(p[:800] for p in ev_parts[:6])
        if not ev:
            results.append({"sentence": s, "cited": nums, "verdict": "no_source_text"})
            continue
        v = _judge(llm, ev, s, parse_llm_json)
        results.append({"sentence": s, "cited": nums, "verdict": v.get("verdict"),
                        "reason": v.get("reason")})
    counts = Counter(r["verdict"] for r in results)
    n = len(results) or 1
    return {
        "rows": results,
        "metrics": {
            "n_judged": len(results),
            "verdict_counts": dict(counts),
            "attribution_precision": round(counts.get("supported", 0) / n, 3),
        },
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def write_markdown(path, state, kb, cb, rc, jc, jr):
    L = []
    A = L.append
    A(f"# Grounding Evaluation\n")
    A(f"Topic: **{state.topic or '(empty)'}**  ")
    A(f"Claims: {len(kb.all_claims())} · Chunks: {len(kb.all_chunks())} · "
      f"Papers reviewed: {kb.reviewed_count()}\n")

    A("## Scorecard\n")
    A("| Metric | Value |")
    A("|--------|-------|")
    A(f"| Claim grounding rate (L0: evidence resolves) | {cb['metrics']['claim_grounding_rate']:.1%} |")
    A(f"| Claims with evidence | {cb['metrics']['with_evidence']}/{cb['metrics']['n_claims']} |")
    A(f"| Claims w/ low lexical overlap (<0.05) | {cb['metrics']['low_lexical_overlap(<0.05)']} |")
    A(f"| Claims w/ numbers unsupported by evidence | {cb['metrics']['claims_with_unsupported_numbers']}/{cb['metrics']['claims_with_numbers']} |")
    A(f"| Report citation coverage (factual sentences) | {rc['metrics']['citation_coverage']:.1%} |")
    A(f"| Invalid citation numbers | {rc['metrics']['invalid_citation_numbers'] or 'none'} |")
    A(f"| Unresolved [?] markers | {rc['metrics']['unresolved_markers']} |")
    if jc:
        A(f"| **Claim faithfulness (L2 judge)** | {jc['metrics']['claim_faithfulness']:.1%} ({jc['metrics']['verdict_counts']}) |")
    if jr:
        A(f"| **Attribution precision (L2 judge)** | {jr['metrics']['attribution_precision']:.1%} ({jr['metrics']['verdict_counts']}) |")
    A("")

    # Failures: claims with no/dangling evidence
    bad = [r for r in cb["rows"] if not r["evidence_resolves"]]
    A(f"## Claims with missing/dangling evidence ({len(bad)})\n")
    for r in bad[:50]:
        A(f"- `{r['claim_id']}` — {r['text'][:140]}  (evidence: {r['evidence_chunk_ids']})")
    A("")

    numbad = [r for r in cb["rows"] if r["numbers_missing_from_evidence"]]
    A(f"## Claims whose numbers are absent from their evidence ({len(numbad)})\n")
    A("_Strong signal of a fabricated/misattributed statistic — review these._\n")
    for r in numbad[:50]:
        A(f"- `{r['claim_id']}` — missing {r['numbers_missing_from_evidence']} — {r['text'][:140]}")
    A("")

    if jc:
        unsup = [r for r in jc["rows"] if r["verdict"] in ("unsupported", "contradicted", "no_evidence")]
        A(f"## LLM-judged unsupported/contradicted claims ({len(unsup)})\n")
        for r in unsup[:50]:
            A(f"- `{r['claim_id']}` [{r['verdict']}] — {r['text'][:140]}  \n  _{r.get('reason','')}_")
        A("")
    if jr:
        unsup = [r for r in jr["rows"] if r["verdict"] in ("unsupported", "contradicted", "no_source_text")]
        A(f"## LLM-judged unsupported report sentences ({len(unsup)})\n")
        for r in unsup[:50]:
            A(f"- [{r['verdict']}] (cites {r['cited']}) {r['sentence'][:160]}  \n  _{r.get('reason','')}_")
        A("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser(description="Evaluate grounding/faithfulness of a run.")
    ap.add_argument("state", help="Path to final_state.json")
    ap.add_argument("--llm-judge", action="store_true", help="Run L2 LLM entailment checks (costs money).")
    ap.add_argument("--sample", type=int, default=None, help="Cap items sent to the LLM judge.")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    state = load_state(args.state)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.state))
    kb = get_kb(state)

    cb = eval_claim_bank(state, kb)
    rc = eval_report_citations(state)

    jc = jr = None
    if args.llm_judge:
        from dotenv import load_dotenv
        load_dotenv()
        from src.agents.base import get_llm, parse_llm_json
        llm = get_llm(temperature=0.0)
        print(">>> Running LLM judge on claims ...")
        jc = judge_claims(state, kb, args.sample, llm, parse_llm_json)
        print(">>> Running LLM judge on report sentences ...")
        jr = judge_report(state, kb, rc, args.sample, llm, parse_llm_json)

    payload = {
        "claim_bank": cb["metrics"],
        "report_citations": {k: v for k, v in rc["metrics"].items()},
        "claim_judge": jc["metrics"] if jc else None,
        "report_judge": jr["metrics"] if jr else None,
    }
    with open(os.path.join(out_dir, "grounding_eval.json"), "w", encoding="utf-8") as f:
        json.dump({"metrics": payload,
                   "claim_rows": cb["rows"],
                   "claim_judge_rows": jc["rows"] if jc else None,
                   "report_judge_rows": jr["rows"] if jr else None},
                  f, indent=2, default=str)
    write_markdown(os.path.join(out_dir, "grounding_eval.md"), state, kb, cb, rc, jc, jr)

    print("\n=== Grounding scorecard ===")
    print(f"  claim grounding rate (L0)        : {cb['metrics']['claim_grounding_rate']:.1%}")
    print(f"  claims w/ unsupported numbers    : {cb['metrics']['claims_with_unsupported_numbers']}/{cb['metrics']['claims_with_numbers']}")
    print(f"  report citation coverage         : {rc['metrics']['citation_coverage']:.1%}")
    print(f"  invalid citations / [?] markers  : {rc['metrics']['invalid_citation_numbers']} / {rc['metrics']['unresolved_markers']}")
    if jc:
        print(f"  claim faithfulness (L2 judge)    : {jc['metrics']['claim_faithfulness']:.1%}")
    if jr:
        print(f"  attribution precision (L2 judge) : {jr['metrics']['attribution_precision']:.1%}")
    print(f"\nWrote: {os.path.join(out_dir, 'grounding_eval.md')}")
    print(f"       {os.path.join(out_dir, 'grounding_eval.json')}")


if __name__ == "__main__":
    main()
