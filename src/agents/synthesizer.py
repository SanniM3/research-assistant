"""Synthesizer agent - writes survey sections from the claim bank using RAG.

For each section it builds a query (title + description + required elements +
open questions), retrieves the most relevant claims and evidence chunks from the
knowledge base by embedding similarity, re-ranks them with claim-type / keyword /
entity signals, enforces source-paper diversity, and writes citation-dense prose
in the requested output language. Falls back to keyword scoring when embeddings
are unavailable.
"""
from typing import Dict, Any, List, Set

from ..models.state import ResearchState, OutlineSection
from ..models.claim import Claim, ClaimType
from ..models.chunk import Chunk
from ..config.settings import get_settings
from .base import get_llm, create_agent_message

MAX_PER_PAPER = 8


def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    """Write each section from retrieved claims, then run a coherence pass."""
    llm = get_llm(role="synthesizer", temperature=0.3)
    kb = state.kb()

    state.log_action("synthesizer", "starting", {
        "sections": len(state.outline),
        "papers_available": kb.reviewed_count(),
        "claims_available": len(kb.all_claims()),
        "chunks_available": len(kb.all_chunks()),
    })

    papers = kb.papers_map()
    draft_sections = dict(state.draft_sections)
    used_paper_ids_global: Set[str] = set()

    for section in state.outline:
        query = _section_query(section, state)
        relevant_claims = get_claims_for_section(section, state, kb, query)

        if not relevant_claims and section.section_id in draft_sections:
            continue

        supporting_chunks = get_supporting_chunks(relevant_claims, kb, query)

        section_content = write_section(
            section=section,
            claims=relevant_claims,
            chunks=supporting_chunks,
            papers=papers,
            topic=state.topic,
            output_language=state.output_language,
            all_sections=state.outline,
            llm=llm,
        )

        draft_sections[section.section_id] = section_content

        for pid in papers:
            if f"[@{pid}]" in section_content:
                used_paper_ids_global.add(pid)

        state.log_action("synthesizer", "section_written", {
            "section_id": section.section_id,
            "claims_used": len(relevant_claims),
            "chunks_used": len(supporting_chunks),
            "length": len(section_content),
        })

    draft_sections = coherence_pass(draft_sections, state.outline, state.topic,
                                    state.output_language, llm)
    state.log_action("synthesizer", "coherence_pass_complete", {})

    return {
        "draft_sections": draft_sections,
        "phase": "verification",
        "estimated_cost_usd": _cost(),
    }


def _cost() -> float:
    from .base import get_cost
    return round(get_cost().get("usd", 0.0), 4)


# ---------------------------------------------------------------------------
# RAG retrieval
# ---------------------------------------------------------------------------

def _section_query(section: OutlineSection, state: ResearchState) -> str:
    parts = [section.title, section.description] + list(section.required_elements)
    for q in state.get_open_questions()[:5]:
        parts.append(q.text)
    parts.append(state.topic)
    return " \n".join(p for p in parts if p)


def get_claims_for_section(section: OutlineSection, state: ResearchState, kb,
                           query: str) -> List[Claim]:
    """Retrieve claims for a section via semantic search + multi-signal re-rank."""
    settings = get_settings()
    top_k = settings.retrieval_top_k_claims

    semantic = kb.search_claims(query, k=top_k * 2)  # [(claim, score)]
    if semantic:
        candidates = semantic
    else:
        # Keyword fallback when embeddings are unavailable.
        candidates = [(c, 0.0) for c in kb.all_claims()]

    entities = kb.entities_map()
    target_types = _types_for_title(section.title.lower())
    keywords = _section_keywords(section)
    entity_names = {
        e.name.lower() for e in entities.values()
        if e.name.lower() in section.title.lower() or e.name.lower() in section.description.lower()
    }

    scored = []
    for claim, sem in candidates:
        score = sem * 3.0  # semantic similarity is the primary signal
        claim_text_lower = claim.text.lower()
        if target_types and claim.claim_type in target_types:
            score += 2.0
        elif not target_types:
            score += 0.5
        overlap = keywords & set(claim_text_lower.split())
        score += len(overlap) * 0.5
        for ename in entity_names:
            if ename in claim_text_lower:
                score += 2.0
                break
        scored.append((claim, score))

    ranked = sorted(scored, key=lambda x: -x[1])
    selected: List[Claim] = []
    paper_counts: Dict[str, int] = {}
    for claim, score in ranked:
        if score <= 0:
            continue
        pid = claim.paper_id
        if paper_counts.get(pid, 0) >= MAX_PER_PAPER:
            continue
        selected.append(claim)
        paper_counts[pid] = paper_counts.get(pid, 0) + 1
        if len(selected) >= top_k:
            break
    return selected


def _types_for_title(title_lower: str) -> List[ClaimType]:
    mapping = {
        "introduction": [ClaimType.DEFINITION, ClaimType.METHOD_SUMMARY],
        "background": [ClaimType.DEFINITION, ClaimType.THEORETICAL_RESULT],
        "taxonomy": [ClaimType.DEFINITION, ClaimType.METHOD_SUMMARY, ClaimType.COMPARISON],
        "evolution": [ClaimType.DEFINITION, ClaimType.METHOD_SUMMARY, ClaimType.COMPARISON],
        "methods": [ClaimType.METHOD_SUMMARY, ClaimType.THEORETICAL_RESULT],
        "technique": [ClaimType.METHOD_SUMMARY, ClaimType.THEORETICAL_RESULT],
        "application": [ClaimType.METHOD_SUMMARY, ClaimType.EMPIRICAL_RESULT],
        "dataset": [ClaimType.DEFINITION, ClaimType.EMPIRICAL_RESULT],
        "benchmark": [ClaimType.DEFINITION, ClaimType.EMPIRICAL_RESULT],
        "experiment": [ClaimType.EMPIRICAL_RESULT, ClaimType.COMPARISON],
        "result": [ClaimType.EMPIRICAL_RESULT, ClaimType.COMPARISON],
        "compar": [ClaimType.COMPARISON, ClaimType.EMPIRICAL_RESULT],
        "analysis": [ClaimType.COMPARISON, ClaimType.EMPIRICAL_RESULT, ClaimType.LIMITATION],
        "discussion": [ClaimType.COMPARISON, ClaimType.LIMITATION, ClaimType.OPEN_PROBLEM],
        "challenge": [ClaimType.LIMITATION, ClaimType.OPEN_PROBLEM],
        "limitation": [ClaimType.LIMITATION],
        "open problem": [ClaimType.OPEN_PROBLEM, ClaimType.LIMITATION],
        "future": [ClaimType.OPEN_PROBLEM, ClaimType.LIMITATION],
        "conclusion": [ClaimType.METHOD_SUMMARY, ClaimType.EMPIRICAL_RESULT, ClaimType.COMPARISON],
    }
    types: List[ClaimType] = []
    for key, ctypes in mapping.items():
        if key in title_lower:
            types.extend(ctypes)
    return list(set(types))


def _section_keywords(section: OutlineSection) -> set:
    keywords = set()
    for source in [section.title, section.description] + list(section.required_elements):
        for word in (source or "").lower().split():
            if len(word) > 3:
                keywords.add(word)
    return keywords


def get_supporting_chunks(claims: List[Claim], kb, query: str) -> List[Chunk]:
    """Evidence chunks from the selected claims, enriched by semantic retrieval."""
    settings = get_settings()
    top_k = settings.retrieval_top_k_chunks

    chunk_ids: Set[str] = set()
    chunks: List[Chunk] = []
    for claim in claims:
        for evidence in claim.evidence:
            if evidence.chunk_id and evidence.chunk_id not in chunk_ids:
                chunk = kb.get_chunk(evidence.chunk_id)
                if chunk:
                    chunks.append(chunk)
                    chunk_ids.add(chunk.chunk_id)

    for chunk, _score in kb.search_chunks(query, k=top_k):
        if chunk.chunk_id not in chunk_ids:
            chunks.append(chunk)
            chunk_ids.add(chunk.chunk_id)
        if len(chunks) >= top_k:
            break

    return chunks[:top_k]


# ---------------------------------------------------------------------------
# Section writing
# ---------------------------------------------------------------------------

def write_section(section: OutlineSection, claims: List[Claim], chunks: List[Chunk],
                  papers: Dict[str, Any], topic: str, output_language: str,
                  all_sections: List[OutlineSection], llm) -> str:
    from ..models.paper import Paper

    claims_text = "\n".join(
        f"- [{c.claim_type.value}] {c.text} (source: {c.paper_id})"
        for c in claims[:50]
    )

    chunks_text = ""
    if chunks:
        entries = [f"[Source: {c.paper_id}]\n{c.text[:800]}" for c in chunks[:25]]
        chunks_text = "\n\n".join(entries)

    # Closed citation set: only papers that actually feed this section.
    paper_ids_from_claims = {c.paper_id for c in claims}
    paper_ids_from_chunks = {c.paper_id for c in chunks}
    citable_ids = list(paper_ids_from_claims | paper_ids_from_chunks)

    paper_list_entries = []
    for pid in citable_ids[:60]:
        paper = papers.get(pid)
        if isinstance(paper, Paper):
            title = (paper.title[:100] + "...") if paper.title and len(paper.title) > 100 else (paper.title or "Unknown")
            authors = ", ".join(paper.authors[:2]) if paper.authors else "Unknown"
            if paper.authors and len(paper.authors) > 2:
                authors += " et al."
            year = paper.year or "N/A"
            paper_list_entries.append(f"  [@{pid}]  {authors} ({year}). \"{title}\"")
        else:
            paper_list_entries.append(f"  [@{pid}]")
    paper_list = "\n".join(paper_list_entries)

    unique_source_papers = len(citable_ids)
    sibling_titles = [s.title for s in all_sections if s.section_id != section.section_id]

    lang_note = ""
    if (output_language or "en").lower() != "en":
        lang_note = (
            f"\nWRITE THE ENTIRE SECTION IN THIS LANGUAGE: {output_language}. "
            f"Use correct academic register for that language. Keep citation markers "
            f"[@paper_id] and any numbers/entity names as-is.\n"
        )

    prompt = f"""You are writing the **"{section.title}"** section of an academic survey paper on: **{topic}**
{lang_note}
SECTION GOAL: {section.description}
REQUIRED ELEMENTS: {', '.join(section.required_elements) if section.required_elements else 'N/A'}

OTHER SECTIONS IN THIS SURVEY (for cross-referencing):
{chr(10).join(f'- {t}' for t in sibling_titles)}

=== {len(citable_ids[:60])} AVAILABLE SOURCES (use [@paper_id] to cite) ===
{paper_list if paper_list else "WARNING: No sources available."}

=== {len(claims[:50])} CLAIMS EXTRACTED FROM LITERATURE ===
{claims_text if claims_text else "No claims extracted."}

=== EVIDENCE EXCERPTS FROM PAPERS ===
{chunks_text if chunks_text else "No direct evidence available."}

=== INSTRUCTIONS ===

You are writing a **comprehensive academic survey section**, not a summary or blog post.
Real survey papers are DETAILED and REFERENCE-HEAVY. Follow these rules strictly:

LENGTH: Write **800-2000 words**. This is a section of a full survey paper.

CITATION DENSITY:
- Nearly EVERY sentence that states a fact, result, or method MUST have a citation.
- Aim for at least 1 citation per 2 sentences on average.
- Use this EXACT format: [@paper_id] - the paper_id MUST be one from AVAILABLE SOURCES above.
- CRITICAL: Do NOT cite any paper_id that is not in AVAILABLE SOURCES. Do not invent citations.
- You have {unique_source_papers} source papers for this section; cite the majority of them.

DEPTH AND STRUCTURE:
- Organise with subsections (### headings).
- For each major method/model/approach, dedicate at least a full paragraph.
- Include specific numbers: accuracy figures, model sizes, speedup ratios, etc.
- Compare and contrast approaches - don't just list them.

STYLE:
- Formal academic tone; use discourse markers ("In contrast,", "Building upon this,").
- Present conflicting findings fairly with citations to both sides.
- Every statement should be grounded in a specific source.

DO NOT: write fewer than 800 words; cite papers not in AVAILABLE SOURCES; make claims
without citations; use bullet lists as the primary structure.

Write the section now (markdown, starting with content directly - do NOT include a section heading):"""

    messages = create_agent_message("synthesizer", prompt)
    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Coherence pass
# ---------------------------------------------------------------------------

def coherence_pass(draft_sections: Dict[str, str], outline: List[OutlineSection],
                   topic: str, output_language: str, llm) -> Dict[str, str]:
    compiled = []
    for section in sorted(outline, key=lambda s: s.order):
        content = draft_sections.get(section.section_id, "")
        compiled.append(f"## {section.title}\n\n{content}")
    full_draft = "\n\n".join(compiled)

    if len(full_draft) < 2000:
        return draft_sections

    lang_note = ""
    if (output_language or "en").lower() != "en":
        lang_note = f"Write the transitions in this language: {output_language}.\n"

    prompt = f"""You are reviewing a draft academic survey on: **{topic}**

Below is the full draft. For EACH section that needs it, provide a SHORT paragraph
(2-4 sentences) to PREPEND as a transition from the previous section, and/or APPEND
as a bridge to the next section. Do NOT rewrite the sections.
{lang_note}
FULL DRAFT:
{full_draft[:20000]}

Respond in JSON:
{{
    "section_transitions": [
        {{"section_title": "Section Name", "prepend": "... or null", "append": "... or null"}}
    ]
}}

Skip Introduction (no prepend) and Conclusion (no append). Output ONLY valid JSON."""

    messages = create_agent_message("synthesizer_coherence", prompt)
    response = llm.invoke(messages)

    from .base import parse_llm_json
    result = parse_llm_json(response.content, fallback=None, agent="synthesizer_coherence")
    if not result or not isinstance(result, dict):
        return draft_sections

    title_to_id = {sec.title.lower(): sec.section_id for sec in outline}
    updated = dict(draft_sections)
    for entry in result.get("section_transitions", []):
        title = entry.get("section_title", "")
        sid = title_to_id.get(title.lower())
        if not sid or sid not in updated:
            continue
        content = updated[sid]
        prepend = entry.get("prepend")
        append = entry.get("append")
        if prepend and isinstance(prepend, str):
            content = prepend.strip() + "\n\n" + content
        if append and isinstance(append, str):
            content = content + "\n\n" + append.strip()
        updated[sid] = content

    return updated
