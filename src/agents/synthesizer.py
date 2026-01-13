"""Synthesizer agent - writes survey sections from claims."""
import json
from typing import Dict, Any, List, Optional

from ..models.state import ResearchState, OutlineSection
from ..models.claim import Claim, ClaimType
from ..models.chunk import Chunk
from .base import get_llm, create_agent_message


def synthesizer_node(state: ResearchState) -> Dict[str, Any]:
    """
    Synthesizer node - writes survey sections from claim bank.
    
    Responsibilities:
    - Write section drafts from claims
    - Include proper citations to evidence
    - Ensure grounded writing
    - Handle contradictions appropriately
    """
    llm = get_llm(temperature=0.3)  # Slightly higher for writing
    
    state.log_action("synthesizer", "starting", {"sections": len(state.outline)})
    
    draft_sections = dict(state.draft_sections)
    
    for section in state.outline:
        # Get relevant claims for this section
        relevant_claims = get_claims_for_section(section, state.claims, state.entities)
        
        if not relevant_claims and section.section_id in draft_sections:
            # Skip if no new claims and draft exists
            continue
        
        # Get supporting chunks for context
        supporting_chunks = get_supporting_chunks(relevant_claims, state.chunks)
        
        # Generate section content
        section_content = write_section(
            section=section,
            claims=relevant_claims,
            chunks=supporting_chunks,
            topic=state.topic,
            llm=llm
        )
        
        draft_sections[section.section_id] = section_content
        
        state.log_action("synthesizer", "section_written", {
            "section_id": section.section_id,
            "claims_used": len(relevant_claims),
            "length": len(section_content),
        })
    
    return {
        "draft_sections": draft_sections,
        "phase": "verification",
    }


def get_claims_for_section(section: OutlineSection, 
                           claims: Dict[str, Claim],
                           entities: Dict) -> List[Claim]:
    """Get claims relevant to a section based on type and content matching."""
    relevant = []
    
    # Map section titles to claim types
    section_claim_types = {
        "introduction": [ClaimType.DEFINITION, ClaimType.METHOD_SUMMARY],
        "background": [ClaimType.DEFINITION, ClaimType.THEORETICAL_RESULT],
        "taxonomy": [ClaimType.DEFINITION, ClaimType.METHOD_SUMMARY, ClaimType.COMPARISON],
        "methods": [ClaimType.METHOD_SUMMARY, ClaimType.THEORETICAL_RESULT],
        "techniques": [ClaimType.METHOD_SUMMARY, ClaimType.THEORETICAL_RESULT],
        "datasets": [ClaimType.DEFINITION, ClaimType.EMPIRICAL_RESULT],
        "benchmarks": [ClaimType.DEFINITION, ClaimType.EMPIRICAL_RESULT],
        "experiments": [ClaimType.EMPIRICAL_RESULT, ClaimType.COMPARISON],
        "results": [ClaimType.EMPIRICAL_RESULT, ClaimType.COMPARISON],
        "discussion": [ClaimType.COMPARISON, ClaimType.LIMITATION, ClaimType.OPEN_PROBLEM],
        "open problems": [ClaimType.OPEN_PROBLEM, ClaimType.LIMITATION],
        "future": [ClaimType.OPEN_PROBLEM],
        "conclusion": [ClaimType.METHOD_SUMMARY, ClaimType.EMPIRICAL_RESULT],
        "limitations": [ClaimType.LIMITATION],
    }
    
    # Find matching claim types
    section_title_lower = section.title.lower()
    target_types = []
    for key, types in section_claim_types.items():
        if key in section_title_lower:
            target_types.extend(types)
    
    # If no specific match, include all types
    if not target_types:
        target_types = list(ClaimType)
    
    # Filter claims
    for claim in claims.values():
        if claim.claim_type in target_types:
            relevant.append(claim)
    
    # Also match by section description keywords
    section_keywords = set(section.description.lower().split())
    for claim in claims.values():
        claim_words = set(claim.text.lower().split())
        if len(section_keywords & claim_words) >= 2 and claim not in relevant:
            relevant.append(claim)
    
    return relevant[:50]  # Limit to avoid token limits


def get_supporting_chunks(claims: List[Claim], 
                          chunks: Dict[str, Chunk]) -> List[Chunk]:
    """Get chunks that support the given claims."""
    chunk_ids = set()
    for claim in claims:
        for evidence in claim.evidence:
            chunk_ids.add(evidence.chunk_id)
    
    return [chunks[cid] for cid in chunk_ids if cid in chunks]


def write_section(section: OutlineSection, claims: List[Claim],
                  chunks: List[Chunk], topic: str, llm) -> str:
    """Write a single section using claims and chunks."""
    
    # Prepare claims for prompt
    claims_text = "\n".join([
        f"- [{c.claim_type.value}] {c.text} (evidence: {[e.chunk_id for e in c.evidence]})"
        for c in claims[:30]
    ])
    
    # Prepare chunk snippets for context
    chunks_text = "\n\n".join([
        f"[{c.chunk_id}] {c.text[:500]}..."
        for c in chunks[:15]
    ])
    
    prompt = f"""Write the "{section.title}" section for an academic survey on: {topic}

SECTION DESCRIPTION: {section.description}
REQUIRED ELEMENTS: {', '.join(section.required_elements)}

CLAIMS FROM LITERATURE (use these as your primary source):
{claims_text if claims_text else "No specific claims available - write based on general knowledge but note limitations."}

SUPPORTING EVIDENCE (for additional context and wording):
{chunks_text if chunks_text else "No additional context available."}

WRITING GUIDELINES:
1. Write in academic survey style
2. Every factual statement must cite evidence using format: [@paper_id:chunk_id]
3. If a claim has evidence, include the citation
4. If making a claim without evidence, explicitly note it as "observed trend" or "commonly held view"
5. Address contradictions by presenting both sides
6. Be comprehensive but concise
7. Use appropriate structure (subsections if needed)
8. For comparison claims, consider using tables or structured comparisons

FORBIDDEN:
- Do not invent facts or statistics
- Do not cite papers not in the evidence
- Do not use phrases like "state-of-the-art" or "first" without explicit evidence

Write the section content (markdown format):"""

    messages = create_agent_message("synthesizer", prompt)
    response = llm.invoke(messages)
    
    return response.content


def compile_full_draft(state: ResearchState) -> str:
    """Compile all sections into a full survey draft."""
    parts = []
    
    # Title
    parts.append(f"# Survey: {state.topic}\n")
    
    # Abstract placeholder
    parts.append("## Abstract\n")
    parts.append("[To be written after full synthesis]\n")
    
    # Sections in order
    sorted_sections = sorted(state.outline, key=lambda s: s.order)
    for section in sorted_sections:
        if section.section_id in state.draft_sections:
            parts.append(f"\n## {section.title}\n")
            parts.append(state.draft_sections[section.section_id])
    
    # References placeholder
    parts.append("\n## References\n")
    parts.append("[References to be compiled by Citation Manager]\n")
    
    return "\n".join(parts)
