"""Extractor agent - extracts claims, entities, and relations."""
import json
from typing import Dict, Any, List
import uuid

from ..models.state import ResearchState
from ..models.claim import Claim, ClaimType, Evidence, ConfidenceLevel
from ..models.entity import Entity, EntityType
from ..models.relation import Relation, RelationType
from ..models.chunk import Chunk
from .base import get_llm, create_agent_message


def extractor_node(state: ResearchState) -> Dict[str, Any]:
    """
    Extractor node - extracts structured knowledge from chunks.
    
    Responsibilities:
    - Extract claims: definitions, results, limitations
    - Extract entities: methods, datasets, metrics
    - Extract relations: evaluated_on, improves_over, uses
    - Link everything to evidence chunks
    """
    llm = get_llm()
    
    state.log_action("extractor", "starting", {"chunk_count": len(state.chunks)})
    
    # Get chunks from papers not yet extracted
    extracted_paper_ids = set(c.paper_id for c in state.claims.values())
    chunks_to_extract = [
        c for c in state.chunks.values()
        if c.paper_id not in extracted_paper_ids
    ]
    
    if not chunks_to_extract:
        return {"phase": "kb_update"}
    
    new_claims = dict(state.claims)
    new_entities = dict(state.entities)
    new_relations = dict(state.relations)
    
    # Group chunks by paper
    chunks_by_paper: Dict[str, List[Chunk]] = {}
    for chunk in chunks_to_extract:
        if chunk.paper_id not in chunks_by_paper:
            chunks_by_paper[chunk.paper_id] = []
        chunks_by_paper[chunk.paper_id].append(chunk)
    
    # Process each paper's chunks
    for paper_id, paper_chunks in list(chunks_by_paper.items())[:5]:  # Limit per iteration
        try:
            claims, entities, relations = extract_from_chunks(
                paper_chunks, paper_id, state.topic, llm
            )
            
            for claim in claims:
                new_claims[claim.claim_id] = claim
            
            for entity in entities:
                new_entities[entity.entity_id] = entity
            
            for relation in relations:
                new_relations[relation.relation_id] = relation
            
            state.log_action("extractor", "paper_extracted", {
                "paper_id": paper_id,
                "claims": len(claims),
                "entities": len(entities),
                "relations": len(relations),
            })
            
        except Exception as e:
            state.log_action("extractor", "extraction_error", {
                "paper_id": paper_id,
                "error": str(e),
            })
    
    return {
        "claims": new_claims,
        "entities": new_entities,
        "relations": new_relations,
        "phase": "kb_update",
    }


def extract_from_chunks(
    chunks: List[Chunk], 
    paper_id: str,
    topic: str,
    llm
) -> tuple:
    """Extract claims, entities, and relations from a paper's chunks."""
    
    # Combine chunks for context (limit to avoid token limits)
    combined_text = "\n\n---\n\n".join([
        f"[CHUNK {c.chunk_id}]\nSection: {c.section_path}\n{c.text}"
        for c in chunks[:20]
    ])
    
    prompt = f"""Extract structured knowledge from this academic paper content.

RESEARCH TOPIC: {topic}
PAPER ID: {paper_id}

CONTENT:
{combined_text[:15000]}

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
1. Only extract claims that are explicitly stated in the text
2. Always link claims to supporting chunk IDs
3. Extract entities that are specific and meaningful
4. Capture relations between methods, datasets, and metrics
5. Be conservative - only high-confidence extractions

Output ONLY valid JSON."""

    messages = create_agent_message("extractor", prompt)
    response = llm.invoke(messages)
    
    claims = []
    entities = []
    relations = []
    
    try:
        data = json.loads(response.content)
        
        # Process claims
        for claim_data in data.get("claims", []):
            claim_id = f"claim_{uuid.uuid4().hex[:8]}"
            claim_type = ClaimType(claim_data.get("type", "method_summary"))
            
            evidence = []
            chunk_id = claim_data.get("evidence_chunk_id")
            if chunk_id:
                evidence.append(Evidence(chunk_id=chunk_id))
            
            claim = Claim(
                claim_id=claim_id,
                claim_type=claim_type,
                text=claim_data.get("text", ""),
                evidence=evidence,
                confidence=ConfidenceLevel(claim_data.get("confidence", "medium")),
                paper_id=paper_id,
                entity_ids=[],  # Will be linked in KB curator
            )
            claims.append(claim)
        
        # Process entities
        for entity_data in data.get("entities", []):
            entity_id = f"entity_{uuid.uuid4().hex[:8]}"
            
            try:
                entity_type = EntityType(entity_data.get("type", "method"))
            except ValueError:
                entity_type = EntityType.METHOD
            
            entity = Entity(
                entity_id=entity_id,
                entity_type=entity_type,
                name=entity_data.get("name", ""),
                aliases=entity_data.get("aliases", []),
                description=entity_data.get("description"),
                paper_ids=[paper_id],
            )
            entities.append(entity)
        
        # Process relations
        for rel_data in data.get("relations", []):
            rel_id = f"rel_{uuid.uuid4().hex[:8]}"
            
            try:
                predicate = RelationType(rel_data.get("predicate", "uses"))
            except ValueError:
                predicate = RelationType.USES
            
            # Create placeholder entity IDs (will be resolved in KB curator)
            relation = Relation(
                relation_id=rel_id,
                subject_entity_id=rel_data.get("subject", ""),
                predicate=predicate,
                object_entity_id=rel_data.get("object", ""),
                paper_id=paper_id,
                evidence_chunks=[rel_data.get("evidence_chunk_id", "")],
            )
            relations.append(relation)
            
    except json.JSONDecodeError:
        pass
    
    return claims, entities, relations
