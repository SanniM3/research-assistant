"""Synthesizer agent - writes survey sections from claims."""
import json
from typing import Dict, Any, List, Set

from ..models.state import ResearchState, OutlineSection
from ..models.claim import Claim, ClaimType
from ..models.chunk import Chunk
from ..models.entity import Entity
from .base import get_llm, create_agent_message


def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Synthesizer node - writes survey sections from claim bank,
    then performs a coherence pass over the full draft.
    """
    llm = get_llm(temperature=0.3)

    state.log_action("synthesizer", "starting", {
        "sections": len(state.outline),
        "papers_available": len(state.papers_ingested),
        "claims_available": len(state.claims),
        "chunks_available": len(state.chunks),
    })

    draft_sections = dict(state.draft_sections)
    used_paper_ids_global: Set[str] = set()

    # Phase 1: Write each section individually with deep, citation-dense content
    for section in state.outline:
        relevant_claims = get_claims_for_section(
            section, state.claims, state.entities, state.outline
        )

        if not relevant_claims and section.section_id in draft_sections:
            continue

        supporting_chunks = get_supporting_chunks(relevant_claims, state.chunks)

        if not supporting_chunks and state.chunks:
            supporting_chunks = list(state.chunks.values())[:25]

        section_content = write_section(
            section=section,
            claims=relevant_claims,
            chunks=supporting_chunks,
            papers=state.papers_ingested,
            topic=state.topic,
            all_sections=state.outline,
            used_paper_ids_global=used_paper_ids_global,
            llm=llm,
        )

        draft_sections[section.section_id] = section_content

        # Track which papers this section actually cited
        for pid in state.papers_ingested:
            if f"[@{pid}]" in section_content:
                used_paper_ids_global.add(pid)

        state.log_action("synthesizer", "section_written", {
            "section_id": section.section_id,
            "claims_used": len(relevant_claims),
            "chunks_used": len(supporting_chunks),
            "length": len(section_content),
        })

    # Phase 2: Coherence pass — review the full draft and add
    # cross-references, smooth transitions, fix redundancies.
    draft_sections = coherence_pass(
        draft_sections, state.outline, state.topic, llm
    )

    state.log_action("synthesizer", "coherence_pass_complete", {})

    return {
        "draft_sections": draft_sections,
        "phase": "verification",
    }


# ---------------------------------------------------------------------------
# Section–claim matching
# ---------------------------------------------------------------------------

def get_claims_for_section(
    section: OutlineSection,
    claims: Dict[str, Claim],
    entities: Dict[str, Entity],
    all_sections: List[OutlineSection],
) -> List[Claim]:
    """Select claims relevant to *this* section using multiple signals."""

    # 1. Type-based matching from section title
    title_lower = section.title.lower()
    target_types = _types_for_title(title_lower)

    # 2. Keyword pool: title + description + required_elements
    keywords = set()
    for word in title_lower.split():
        if len(word) > 3:
            keywords.add(word)
    for word in section.description.lower().split():
        if len(word) > 3:
            keywords.add(word)
    for elem in section.required_elements:
        for word in elem.lower().split():
            if len(word) > 3:
                keywords.add(word)

    # 3. Entity-name pool — entities whose names appear in the section
    #    title/description are strong signals.
    entity_names_lower = set()
    for e in entities.values():
        if e.name.lower() in title_lower or e.name.lower() in section.description.lower():
            entity_names_lower.add(e.name.lower())
            for alias in e.aliases:
                entity_names_lower.add(alias.lower())

    scored: Dict[str, float] = {}

    for cid, claim in claims.items():
        score = 0.0
        claim_text_lower = claim.text.lower()

        # Type match
        if target_types and claim.claim_type in target_types:
            score += 2.0
        elif not target_types:
            score += 0.5  # no type filter ⇒ weak match for everything

        # Keyword overlap
        claim_words = set(claim_text_lower.split())
        overlap = keywords & claim_words
        score += len(overlap) * 0.5

        # Entity-name match
        for ename in entity_names_lower:
            if ename in claim_text_lower:
                score += 2.0
                break

        if score > 0:
            scored[cid] = score

    # Sort by score descending, then ensure source-paper diversity:
    # at most 8 claims from a single paper to avoid over-representation.
    ranked = sorted(scored.items(), key=lambda x: -x[1])
    selected: List[Claim] = []
    paper_counts: Dict[str, int] = {}
    MAX_PER_PAPER = 8

    for cid, _ in ranked:
        claim = claims[cid]
        pid = claim.paper_id
        if paper_counts.get(pid, 0) >= MAX_PER_PAPER:
            continue
        selected.append(claim)
        paper_counts[pid] = paper_counts.get(pid, 0) + 1
        if len(selected) >= 80:
            break

    return selected


def _types_for_title(title_lower: str) -> List[ClaimType]:
    """Map section title keywords to claim types."""
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


# ---------------------------------------------------------------------------
# Chunk retrieval
# ---------------------------------------------------------------------------

def get_supporting_chunks(
    claims: List[Claim], chunks: Dict[str, Chunk]
) -> List[Chunk]:
    chunk_ids = set()
    for claim in claims:
        for evidence in claim.evidence:
            chunk_ids.add(evidence.chunk_id)
    return [chunks[cid] for cid in chunk_ids if cid in chunks]


# ---------------------------------------------------------------------------
# Section writing
# ---------------------------------------------------------------------------

def write_section(
    section: OutlineSection,
    claims: List[Claim],
    chunks: List[Chunk],
    papers: Dict[str, Any],
    topic: str,
    all_sections: List[OutlineSection],
    used_paper_ids_global: Set[str],
    llm,
) -> str:
    from ..models.paper import Paper

    # --- Build claims block (up to 50) ---
    claims_text = "\n".join(
        f"- [{c.claim_type.value}] {c.text} (source: {c.paper_id})"
        for c in claims[:50]
    )

    # --- Build chunk evidence block (up to 25, 800 chars each) ---
    chunks_text = ""
    if chunks:
        entries = []
        for c in chunks[:25]:
            entries.append(f"[Source: {c.paper_id}]\n{c.text[:800]}")
        chunks_text = "\n\n".join(entries)

    # --- Build paper reference list (up to 50) ---
    paper_ids_from_claims = {c.paper_id for c in claims}
    paper_ids_from_chunks = {c.paper_id for c in chunks}
    priority = list(paper_ids_from_claims | paper_ids_from_chunks)
    remaining = [pid for pid in papers if pid not in priority]
    all_paper_ids = priority + remaining

    paper_list_entries = []
    for pid in all_paper_ids[:50]:
        paper = papers.get(pid)
        if isinstance(paper, Paper):
            title = (paper.title[:100] + "...") if paper.title and len(paper.title) > 100 else (paper.title or "Unknown")
            authors = ", ".join(paper.authors[:2]) if paper.authors else "Unknown"
            if paper.authors and len(paper.authors) > 2:
                authors += " et al."
            year = paper.year or "N/A"
            paper_list_entries.append(
                f"  [@{pid}]  {authors} ({year}). \"{title}\""
            )
        else:
            paper_list_entries.append(f"  [@{pid}]")
    paper_list = "\n".join(paper_list_entries)

    # --- Count unique papers feeding this section ---
    unique_source_papers = len(paper_ids_from_claims | paper_ids_from_chunks)

    # --- Sibling section titles for cross-referencing ---
    sibling_titles = [s.title for s in all_sections if s.section_id != section.section_id]

    prompt = f"""You are writing the **"{section.title}"** section of an academic survey paper on: **{topic}**

SECTION GOAL: {section.description}
REQUIRED ELEMENTS: {', '.join(section.required_elements) if section.required_elements else 'N/A'}

OTHER SECTIONS IN THIS SURVEY (for cross-referencing):
{chr(10).join(f'- {t}' for t in sibling_titles)}

=== {len(all_paper_ids[:50])} AVAILABLE SOURCES (use [@paper_id] to cite) ===
{paper_list if paper_list else "WARNING: No sources available."}

=== {len(claims[:50])} CLAIMS EXTRACTED FROM LITERATURE ===
{claims_text if claims_text else "No claims extracted."}

=== EVIDENCE EXCERPTS FROM PAPERS ===
{chunks_text if chunks_text else "No direct evidence available."}

=== INSTRUCTIONS ===

You are writing a **comprehensive academic survey section**, not a summary or blog post.
Real survey papers are DETAILED and REFERENCE-HEAVY. Follow these rules strictly:

LENGTH: Write **800–2000 words**. This is a section of a full survey paper — do not be brief.

CITATION DENSITY:
- Nearly EVERY sentence that states a fact, result, or method MUST have at least one citation.
- Aim for **at least 1 citation per 2 sentences** on average. More is better.
- When discussing a method or result, cite its source PAPER specifically.
- Use this EXACT format: [@paper_id] — the paper_id must match one from AVAILABLE SOURCES above.
- Multiple citations: [@paper_id_1] [@paper_id_2]
- You have {unique_source_papers} source papers for this section. Try to cite the MAJORITY of them.

DEPTH AND STRUCTURE:
- Organise with **subsections** (### headings).
- For EACH major method/model/approach, dedicate at least a full paragraph.
- Include specific numbers: accuracy figures, model sizes, speedup ratios, etc.
- Compare and contrast approaches — don't just list them.
- Reference other sections of this survey where appropriate (e.g., "as discussed in the Background section").

STYLE:
- Formal academic tone throughout.
- Use discourse markers: "In contrast,", "Building upon this,", "Furthermore,", "However,"
- Present conflicting findings fairly with citations to both sides.
- Avoid vague generalities — every statement should be grounded in a specific source.

DO NOT:
- Write fewer than 800 words.
- Cite papers not listed in AVAILABLE SOURCES.
- Make claims without citations.
- Use bullet-point lists as the primary structure — write in flowing paragraphs.

Write the section now (markdown format, starting with content directly — do NOT include a section heading):"""

    messages = create_agent_message("synthesizer", prompt)
    response = llm.invoke(messages)
    return response.content


# ---------------------------------------------------------------------------
# Coherence pass
# ---------------------------------------------------------------------------

def coherence_pass(
    draft_sections: Dict[str, str],
    outline: List[OutlineSection],
    topic: str,
    llm,
) -> Dict[str, str]:
    """Review the full compiled draft and add cross-references and transitions."""
    compiled = []
    for section in sorted(outline, key=lambda s: s.order):
        content = draft_sections.get(section.section_id, "")
        compiled.append(f"## {section.title}\n\n{content}")
    full_draft = "\n\n".join(compiled)

    # Only do coherence pass if we have substantial content
    if len(full_draft) < 2000:
        return draft_sections

    prompt = f"""You are reviewing a draft academic survey on: **{topic}**

Below is the full draft. Your task is to identify, for EACH section, specific
improvements to cross-referencing and transitions. Do NOT rewrite the sections —
instead, for each section that needs improvement, provide a SHORT paragraph
(2-4 sentences) that should be PREPENDED to the section as a transition from
the previous section, and/or APPENDED as a bridge to the next section.

FULL DRAFT:
{full_draft[:20000]}

Respond in JSON format:
{{
    "section_transitions": [
        {{
            "section_title": "Section Name",
            "prepend": "Transition paragraph to add at the start (or null)",
            "append": "Bridge paragraph to add at the end (or null)"
        }}
    ]
}}

Only include sections that genuinely need better transitions. Skip Introduction
(needs no prepend) and Conclusion (needs no append). Keep transitions brief and
academic.

Output ONLY valid JSON."""

    messages = create_agent_message("synthesizer", prompt)
    response = llm.invoke(messages)

    from .base import parse_llm_json
    result = parse_llm_json(response.content, fallback=None, agent="synthesizer_coherence")

    if not result or not isinstance(result, dict):
        return draft_sections

    # Build title → section_id mapping
    title_to_id = {}
    for sec in outline:
        title_to_id[sec.title.lower()] = sec.section_id

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
