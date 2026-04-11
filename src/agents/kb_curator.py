"""KB Curator agent - maintains knowledge base consistency."""
from typing import Dict, Any, List, Optional
from difflib import SequenceMatcher

from ..models.state import ResearchState
from ..models.entity import Entity, EntityType
from ..models.relation import Relation
from ..models.claim import Claim


def kb_curator_node(state: ResearchState) -> Dict[str, Any]:
    """
    KB Curator node - ensures knowledge base consistency.
    
    Responsibilities:
    - Merge duplicate entities
    - Resolve entity aliases
    - Link claims to entity IDs
    - Resolve relation entity references
    - Ensure referential integrity
    """
    state.log_action("kb_curator", "starting", {
        "entities": len(state.entities),
        "relations": len(state.relations),
        "claims": len(state.claims),
    })
    
    # Work with copies
    entities = dict(state.entities)
    relations = dict(state.relations)
    claims = dict(state.claims)
    
    # Step 1: Merge duplicate entities
    entities, merge_map = merge_duplicate_entities(entities)
    
    # Step 2: Update relations with resolved entity IDs
    relations = resolve_relation_entities(relations, entities, merge_map)
    
    # Step 3: Link claims to entity IDs
    claims = link_claims_to_entities(claims, entities)
    
    # Step 4: Validate referential integrity
    integrity_warnings = validate_integrity(claims, entities, relations, state.chunks)

    state.log_action("kb_curator", "completed", {
        "entities_after_merge": len(entities),
        "merges_performed": len(merge_map),
        "integrity_warnings": len(integrity_warnings),
    })
    
    return {
        "entities": entities,
        "relations": relations,
        "claims": claims,
        "phase": "gap_scoring",
    }


def merge_duplicate_entities(entities: Dict[str, Entity]) -> tuple:
    """
    Merge duplicate entities based on name similarity.
    
    Returns:
        Tuple of (merged entities dict, merge map of old_id -> new_id)
    """
    merge_map = {}
    canonical_entities = {}
    
    # Group entities by type first
    by_type: Dict[EntityType, List[Entity]] = {}
    for entity in entities.values():
        if entity.entity_type not in by_type:
            by_type[entity.entity_type] = []
        by_type[entity.entity_type].append(entity)
    
    # Find duplicates within each type
    for entity_type, type_entities in by_type.items():
        processed = set()
        
        for entity in type_entities:
            if entity.entity_id in processed:
                continue
            
            # Find similar entities
            similar = find_similar_entities(entity, type_entities, processed)
            
            if similar:
                # Merge all similar into the first one
                canonical = entity
                for sim_entity in similar:
                    canonical.merge_with(sim_entity)
                    merge_map[sim_entity.entity_id] = canonical.entity_id
                    processed.add(sim_entity.entity_id)
            
            canonical_entities[entity.entity_id] = entity
            processed.add(entity.entity_id)
    
    return canonical_entities, merge_map


def find_similar_entities(entity: Entity, candidates: List[Entity], 
                          exclude: set, threshold: float = 0.85) -> List[Entity]:
    """Find entities with similar names."""
    similar = []
    
    entity_names = [entity.name.lower()] + [a.lower() for a in entity.aliases]
    
    for candidate in candidates:
        if candidate.entity_id == entity.entity_id:
            continue
        if candidate.entity_id in exclude:
            continue
        
        candidate_names = [candidate.name.lower()] + [a.lower() for a in candidate.aliases]
        
        # Check for name match
        for en in entity_names:
            for cn in candidate_names:
                # Exact match
                if en == cn:
                    similar.append(candidate)
                    break
                # Fuzzy match
                similarity = SequenceMatcher(None, en, cn).ratio()
                if similarity >= threshold:
                    similar.append(candidate)
                    break
            else:
                continue
            break
    
    return similar


def resolve_relation_entities(relations: Dict[str, Relation], 
                              entities: Dict[str, Entity],
                              merge_map: Dict[str, str]) -> Dict[str, Relation]:
    """Resolve relation entity references to actual entity IDs."""
    resolved_relations = {}
    
    for rel_id, relation in relations.items():
        # Resolve subject
        subject_id = resolve_entity_reference(
            relation.subject_entity_id, entities, merge_map
        )
        
        # Resolve object
        object_id = resolve_entity_reference(
            relation.object_entity_id, entities, merge_map
        )
        
        if subject_id and object_id:
            relation.subject_entity_id = subject_id
            relation.object_entity_id = object_id
            resolved_relations[rel_id] = relation
    
    return resolved_relations


def resolve_entity_reference(ref: str, entities: Dict[str, Entity],
                             merge_map: Dict[str, str]) -> Optional[str]:
    """Resolve an entity reference (ID or name) to a canonical entity ID."""
    # Check if it's already a valid ID
    if ref in entities:
        return ref
    
    # Check merge map
    if ref in merge_map:
        return merge_map[ref]
    
    # Try to match by name
    ref_lower = ref.lower()
    for entity in entities.values():
        if entity.name.lower() == ref_lower:
            return entity.entity_id
        if any(a.lower() == ref_lower for a in entity.aliases):
            return entity.entity_id
    
    return None


def link_claims_to_entities(claims: Dict[str, Claim],
                            entities: Dict[str, Entity]) -> Dict[str, Claim]:
    """Link claims to relevant entity IDs based on mentions."""
    for claim in claims.values():
        claim_text_lower = claim.text.lower()
        
        linked_entities = []
        for entity in entities.values():
            # Check if entity or aliases mentioned in claim
            if entity.name.lower() in claim_text_lower:
                linked_entities.append(entity.entity_id)
            elif any(alias.lower() in claim_text_lower for alias in entity.aliases):
                linked_entities.append(entity.entity_id)
        
        claim.entity_ids = list(set(linked_entities))
    
    return claims


def validate_integrity(claims: Dict[str, Claim], entities: Dict[str, Entity],
                      relations: Dict[str, Relation], chunks: Dict) -> List[str]:
    """Validate referential integrity of knowledge base.

    Returns a list of warning strings (empty when everything is consistent).
    """
    warnings: List[str] = []

    for claim_id, claim in claims.items():
        for evidence in claim.evidence:
            if evidence.chunk_id and evidence.chunk_id not in chunks:
                warnings.append(f"Claim {claim_id}: evidence chunk {evidence.chunk_id} missing")

    for rel_id, relation in relations.items():
        if relation.subject_entity_id not in entities:
            warnings.append(f"Relation {rel_id}: subject {relation.subject_entity_id} missing")
        if relation.object_entity_id not in entities:
            warnings.append(f"Relation {rel_id}: object {relation.object_entity_id} missing")

    return warnings
