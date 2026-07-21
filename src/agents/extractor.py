"""Extractor agent - extracts claims, entities, and relations into the KB.

Extraction is the throughput bottleneck for survey depth, so it processes ALL
un-extracted chunks (tracked at chunk level in the KB) in token-bounded batches
rather than truncating each paper to a fixed window. Cheap-tier model by default.
"""
from typing import Dict, Any, List
import uuid

from ..models.state import ResearchState
from ..models.claim import Claim, ClaimType, Evidence, ConfidenceLevel
from ..models.entity import Entity, EntityType
from ..models.relation import Relation, RelationType
from ..models.chunk import Chunk
from ..config.settings import get_settings
from ..utils.logging import get_logger
from .base import get_llm, create_agent_message, parse_llm_json, budget_exceeded

_logger = get_logger("extractor")


def extractor_node(state: ResearchState) -> Dict[str, Any]:
    """Extract structured knowledge from every un-extracted chunk in the KB."""
    settings = get_settings()
    llm = get_llm(role="extractor")
    kb = state.kb()

    chunks_to_extract = kb.unextracted_chunks()
    state.log_action("extractor", "starting", {
        "chunks_pending": len(chunks_to_extract),
        "chunks_total": len(kb.all_chunks()),
    })

    if not chunks_to_extract:
        return {"phase": "kb_update"}

    # Group by paper so claims carry the right source id.
    chunks_by_paper: Dict[str, List[Chunk]] = {}
    for chunk in chunks_to_extract:
        chunks_by_paper.setdefault(chunk.paper_id, []).append(chunk)

    paper_items = list(chunks_by_paper.items())
    cap = settings.max_papers_per_extraction
    if cap and cap > 0:
        paper_items = paper_items[:cap]

    batch_size = max(1, settings.extraction_chunk_batch)
    total_claims = 0
    total_papers = len(paper_items)
    _logger.info("Extracting knowledge from %d paper(s), %d chunk(s) pending...",
                 total_papers, len(chunks_to_extract))

    for p_idx, (paper_id, paper_chunks) in enumerate(paper_items, 1):
        if budget_exceeded():
            _logger.warning("Cost budget reached; stopping extraction at paper %d/%d",
                            p_idx, total_papers)
            state.log_action("extractor", "budget_stop", {"paper_id": paper_id})
            break
        _logger.info("  [%d/%d] extracting %s (%d chunks)",
                     p_idx, total_papers, paper_id, len(paper_chunks))

        paper_claims: List[Claim] = []
        paper_entities: List[Entity] = []
        paper_relations: List[Relation] = []
        processed_ids: List[str] = []

        for i in range(0, len(paper_chunks), batch_size):
            batch = paper_chunks[i:i + batch_size]
            try:
                claims, entities, relations = extract_from_chunks(
                    batch, paper_id, state.topic, llm
                )
                paper_claims.extend(claims)
                paper_entities.extend(entities)
                paper_relations.extend(relations)
            except Exception as e:
                state.log_action("extractor", "extraction_error", {
                    "paper_id": paper_id, "error": str(e),
                })
            processed_ids.extend(c.chunk_id for c in batch)

        if paper_claims:
            kb.upsert_claims(paper_claims)
        for entity in paper_entities:
            kb.upsert_entity(entity)
        for relation in paper_relations:
            kb.upsert_relation(relation)
        # Mark all attempted chunks as extracted so empty ones are not retried.
        kb.mark_chunks_extracted(processed_ids)

        total_claims += len(paper_claims)
        state.log_action("extractor", "paper_extracted", {
            "paper_id": paper_id,
            "chunks": len(paper_chunks),
            "claims": len(paper_claims),
            "entities": len(paper_entities),
            "relations": len(paper_relations),
        })

    return {"phase": "kb_update", "estimated_cost_usd": _cost()}


def _cost() -> float:
    from .base import get_cost
    return round(get_cost().get("usd", 0.0), 4)


def extract_from_chunks(chunks: List[Chunk], paper_id: str, topic: str, llm) -> tuple:
    """Extract claims, entities, and relations from a bounded batch of chunks."""
    combined_text = "\n\n---\n\n".join([
        f"[CHUNK {c.chunk_id}]\nSection: {c.section_path}\n{c.text}"
        for c in chunks
    ])

    prompt = f"""Extract structured knowledge from this academic paper content.

RESEARCH TOPIC: {topic}
PAPER ID: {paper_id}

CONTENT:
{combined_text}

Extract the following in JSON format:
{{
    "claims": [
        {{
            "type": "definition|method_summary|empirical_result|theoretical_result|limitation|comparison|open_problem",
            "text": "The claim text",
            "evidence_chunk_id": "chunk_id that supports this",
            "confidence": "low|medium|high",
            "entities_mentioned": ["entity names"]
        }}
    ],
    "entities": [
        {{
            "type": "method|dataset|metric|task|domain|benchmark|framework|model|technique",
            "name": "Entity name",
            "aliases": ["alternative names"],
            "description": "Brief description from text"
        }}
    ],
    "relations": [
        {{
            "subject": "Entity name",
            "predicate": "evaluated_on|improves_over|uses|assumes|similar_to|extends|based_on|applied_to",
            "object": "Entity name",
            "evidence_chunk_id": "supporting chunk"
        }}
    ]
}}

Guidelines:
1. Only extract claims that are explicitly stated in the text.
2. Always link claims to the exact supporting chunk id (from the [CHUNK ...] markers).
3. Extract entities that are specific and meaningful.
4. Capture relations between methods, datasets, and metrics.
5. Extract EVERY substantive claim in this content - be thorough, not minimal.

Output ONLY valid JSON."""

    messages = create_agent_message("extractor", prompt)
    response = llm.invoke(messages)

    claims, entities, relations = [], [], []
    valid_chunk_ids = {c.chunk_id for c in chunks}

    data = parse_llm_json(response.content, fallback=None, agent="extractor")
    if not (data and isinstance(data, dict)):
        return claims, entities, relations

    for claim_data in data.get("claims", []):
        text = (claim_data.get("text") or "").strip()
        if not text:
            continue
        try:
            claim_type = ClaimType(claim_data.get("type", "method_summary"))
        except ValueError:
            claim_type = ClaimType.METHOD_SUMMARY

        evidence = []
        chunk_id = claim_data.get("evidence_chunk_id")
        # Only trust evidence pointers that actually exist in this batch;
        # otherwise fall back to the first chunk so grounding is never fabricated.
        if chunk_id in valid_chunk_ids:
            evidence.append(Evidence(chunk_id=chunk_id))
        elif chunks:
            evidence.append(Evidence(chunk_id=chunks[0].chunk_id, relevance_score=0.5))

        claims.append(Claim(
            claim_id=f"claim_{uuid.uuid4().hex[:8]}",
            claim_type=claim_type,
            text=text,
            evidence=evidence,
            confidence=_confidence(claim_data.get("confidence", "medium")),
            paper_id=paper_id,
            entity_ids=[],
        ))

    for entity_data in data.get("entities", []):
        name = (entity_data.get("name") or "").strip()
        if not name:
            continue
        try:
            entity_type = EntityType(entity_data.get("type", "method"))
        except ValueError:
            entity_type = EntityType.METHOD
        entities.append(Entity(
            entity_id=f"entity_{uuid.uuid4().hex[:8]}",
            entity_type=entity_type,
            name=name,
            aliases=entity_data.get("aliases", []),
            description=entity_data.get("description"),
            paper_ids=[paper_id],
        ))

    for rel_data in data.get("relations", []):
        subject = (rel_data.get("subject") or "").strip()
        obj = (rel_data.get("object") or "").strip()
        if not subject or not obj:
            continue
        try:
            predicate = RelationType(rel_data.get("predicate", "uses"))
        except ValueError:
            predicate = RelationType.USES
        relations.append(Relation(
            relation_id=f"rel_{uuid.uuid4().hex[:8]}",
            subject_entity_id=subject,   # resolved to entity ids in KB curator
            predicate=predicate,
            object_entity_id=obj,
            paper_id=paper_id,
            evidence_chunks=[rel_data.get("evidence_chunk_id", "")],
        ))

    return claims, entities, relations


def _confidence(value: str) -> ConfidenceLevel:
    try:
        return ConfidenceLevel(value)
    except ValueError:
        return ConfidenceLevel.MEDIUM
