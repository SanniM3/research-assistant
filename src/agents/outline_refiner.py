"""Outline Refiner agent - finalizes the survey outline based on research findings."""
import json
from typing import Dict, Any, List

from ..models.state import ResearchState, OutlineSection
from .base import get_llm, create_agent_message, parse_llm_json


def outline_refiner_node(state: ResearchState) -> Dict[str, Any]:
    """
    Outline Refiner node - produces the final survey outline based on
    what was actually discovered during research.

    Runs once, after the research loop ends and before synthesis begins.
    Uses the preliminary outline as a starting point, then reshapes it
    based on entities, claims, and relations in the knowledge base.
    """
    if state.outline_finalized:
        return {}

    llm = get_llm(role="outline_refiner")
    kb = state.kb()
    state.log_action("outline_refiner", "starting", {
        "preliminary_sections": len(state.outline),
        "entities": len(kb.all_entities()),
        "claims": len(kb.all_claims()),
    })

    entity_summary = _summarize_entities(kb)
    claim_type_counts = _count_claim_types(kb)
    relation_summary = _summarize_relations(kb)
    question_summary = [
        {"text": q.text, "status": q.status.value, "answer": q.answer_summary}
        for q in state.research_questions
    ]

    prompt = f"""You are refining the outline for a survey paper.

TOPIC: {state.topic}
SCOPE: {state.scope}
OUTPUT LANGUAGE: {state.output_language}

PRELIMINARY OUTLINE:
{json.dumps([{"id": s.section_id, "title": s.title, "desc": s.description} for s in state.outline], indent=2)}

RESEARCH QUESTIONS AND STATUS:
{json.dumps(question_summary, indent=2)}

KNOWLEDGE BASE SUMMARY:
- Papers reviewed: {kb.reviewed_count()}
- Claims extracted: {len(kb.all_claims())}  (by type: {json.dumps(claim_type_counts)})
- Entities: {json.dumps(entity_summary, indent=2)}
- Relations: {json.dumps(relation_summary[:20], indent=2)}

Based on what was actually found in the research, produce the FINAL survey
outline.  The outline should reflect the real structure of the field as
revealed by the evidence — e.g. if methods naturally cluster into 3
families, create subsections for each; if there are notable comparisons,
include a comparison section; if certain topics had no evidence, drop them.

Respond in JSON:
{{
    "outline": [
        {{
            "section_id": "sec_1",
            "title": "Section Title",
            "description": "What this section covers and why",
            "required_elements": ["element1", "element2"],
            "min_claims": 3
        }}
    ]
}}

Guidelines:
- Keep Introduction and Conclusion.
- Group related findings into coherent sections.
- Add comparison/benchmark sections if the data supports them.
- Remove sections that have no supporting evidence.
- Re-order for logical flow.

Output ONLY valid JSON."""

    messages = create_agent_message("outline_refiner", prompt)
    response = llm.invoke(messages)

    result = parse_llm_json(response.content, fallback=None, agent="outline_refiner")

    if result and isinstance(result, dict) and result.get("outline"):
        new_outline = []
        for i, sec_data in enumerate(result["outline"]):
            section = OutlineSection(
                section_id=sec_data.get("section_id", f"sec_{i+1}"),
                title=sec_data.get("title", f"Section {i+1}"),
                description=sec_data.get("description", ""),
                order=i,
                required_elements=sec_data.get("required_elements", []),
                min_claims=sec_data.get("min_claims", 3),
            )
            new_outline.append(section)

        state.log_action("outline_refiner", "completed", {
            "final_sections": len(new_outline),
        })

        return {
            "outline": new_outline,
            "outline_finalized": True,
            "phase": "synthesis",
        }

    state.log_action("outline_refiner", "parse_error_keeping_preliminary", {
        "sections_kept": len(state.outline),
    })
    return {
        "outline": list(state.outline),
        "outline_finalized": True,
        "phase": "synthesis",
    }


def _summarize_entities(kb) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for e in kb.all_entities():
        t = e.entity_type.value
        counts[t] = counts.get(t, 0) + 1
    return counts


def _count_claim_types(kb) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for c in kb.all_claims():
        t = c.claim_type.value
        counts[t] = counts.get(t, 0) + 1
    return counts


def _summarize_relations(kb) -> List[Dict[str, str]]:
    return [
        {"type": r.predicate.value, "source": r.subject_entity_id, "target": r.object_entity_id}
        for r in kb.all_relations()[:30]
    ]
