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
    
    # Build a paper reference list for the LLM
    paper_refs = []
    for claim in claims[:30]:
        for evidence in claim.evidence:
            paper_id = claim.paper_id
            if paper_id not in [p[0] for p in paper_refs]:
                paper_refs.append((paper_id, evidence.chunk_id))
    
    paper_list = "\n".join([f"- Paper ID: {pid} (cite as: [@{pid}])" for pid, _ in paper_refs[:20]])
    
    prompt = f"""Write the "{section.title}" section for an academic survey on: {topic}

SECTION DESCRIPTION: {section.description}
REQUIRED ELEMENTS: {', '.join(section.required_elements)}

AVAILABLE SOURCES TO CITE:
{paper_list if paper_list else "No sources available yet."}

CLAIMS FROM LITERATURE (use these as your primary source):
{claims_text if claims_text else "No specific claims available - write based on general knowledge but note limitations."}

SUPPORTING EVIDENCE (for additional context and wording):
{chunks_text if chunks_text else "No additional context available."}

WRITING GUIDELINES:
1. Write in academic survey style with proper in-line citations
2. IMPORTANT: Cite sources using this exact format: [@paper_id] - these will be converted to [1], [2], etc.
3. Every factual statement, result, or method description MUST have a citation
4. You can cite multiple sources together: [@paper_id1] [@paper_id2] or combine them
5. Place citations at the end of the sentence or clause they support, before the period
6. If making a general observation without a specific source, phrase it as "it is generally observed that..." without a citation
7. Address contradictions by presenting both sides with their respective citations
8. Be comprehensive but concise
9. Use appropriate structure (subsections if needed)
10. For comparisons, consider using tables with citations in relevant cells

EXAMPLE OF GOOD CITATION USAGE:
"Transformer models have achieved remarkable success in NLP tasks [@arxiv:1706.03762]. Building on this, BERT introduced bidirectional pre-training [@arxiv:1810.04805], which was later extended by RoBERTa [@arxiv:1907.11692] with improved training procedures."

FORBIDDEN:
- Do not invent facts, statistics, or citations
- Do not cite papers not listed in AVAILABLE SOURCES
- Do not use phrases like "state-of-the-art", "best", or "first" without explicit citation
- Do not write paragraphs of factual content without any citations

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
