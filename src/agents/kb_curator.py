"""KB Curator agent - maintains knowledge base consistency.

Merges duplicate entities, resolves relation endpoints, and links claims to
entities. Endpoint/claim resolution uses embedding similarity when available
(so paraphrased names still match), with a fuzzy-string fallback for offline
robustness.
"""
from typing import Dict, Any, List, Optional, Tuple
from difflib import SequenceMatcher

from ..models.state import ResearchState
from ..models.entity import Entity, EntityType
from ..models.relation import Relation
from ..models.claim import Claim
from ..config.settings import get_settings
from ..storage.vector_store import EmbeddingIndex
from .base import embed_texts


def kb_curator_node(state: ResearchState) -> Dict[str, Any]:
    """Ensure knowledge base consistency, then persist the cleaned graph."""
    kb = state.kb()
    entities = kb.entities_map()
    relations = kb.relations_map()
    claims = kb.claims_map()

    state.log_action("kb_curator", "starting", {
        "entities": len(entities), "relations": len(relations), "claims": len(claims),
    })

    entities, merge_map = merge_duplicate_entities(entities)
    resolver = _build_entity_resolver(entities)
    relations = resolve_relation_entities(relations, entities, merge_map, resolver)
    claims = link_claims_to_entities(claims, entities, resolver)

    kb.replace_entities(entities)
    kb.replace_relations(relations)
    kb.replace_claims(claims)

    state.log_action("kb_curator", "completed", {
        "entities_after_merge": len(entities),
        "merges_performed": len(merge_map),
        "relations_kept": len(relations),
    })

    return {"phase": "gap_scoring"}


# ---------------------------------------------------------------------------
# Entity resolver (embedding with fuzzy fallback)
# ---------------------------------------------------------------------------

class _EntityResolver:
    def __init__(self, entities: Dict[str, Entity]):
        self.entities = entities
        self.threshold = get_settings().entity_link_similarity
        self._exact: Dict[str, str] = {}
        for eid, e in entities.items():
            self._exact[e.name.lower().strip()] = eid
            for alias in e.aliases:
                self._exact.setdefault(alias.lower().strip(), eid)
        # Optional semantic index over entity names/aliases.
        self._index = EmbeddingIndex(embed_texts)
        items = []
        for eid, e in entities.items():
            surface = e.name + (" " + " ".join(e.aliases) if e.aliases else "")
            items.append((eid, surface, {}))
        self._index.add(items)

    def resolve(self, name: str) -> Optional[str]:
        if not name:
            return None
        key = name.lower().strip()
        if key in self._exact:
            return self._exact[key]
        # Semantic nearest
        hits = self._index.search(name, k=1)
        if hits and hits[0][1] >= self.threshold:
            return hits[0][0]
        # Fuzzy fallback
        best_id, best_ratio = None, 0.0
        for eid, e in self.entities.items():
            for surface in [e.name] + e.aliases:
                ratio = SequenceMatcher(None, key, surface.lower().strip()).ratio()
                if ratio > best_ratio:
                    best_ratio, best_id = ratio, eid
        if best_ratio >= 0.85:
            return best_id
        return None


def _build_entity_resolver(entities: Dict[str, Entity]) -> _EntityResolver:
    return _EntityResolver(entities)


# ---------------------------------------------------------------------------
# Entity de-duplication (within type)
# ---------------------------------------------------------------------------

def merge_duplicate_entities(entities: Dict[str, Entity]) -> Tuple[Dict[str, Entity], Dict[str, str]]:
    merge_map: Dict[str, str] = {}
    canonical_entities: Dict[str, Entity] = {}

    by_type: Dict[EntityType, List[Entity]] = {}
    for entity in entities.values():
        by_type.setdefault(entity.entity_type, []).append(entity)

    for _entity_type, type_entities in by_type.items():
        processed = set()
        for entity in type_entities:
            if entity.entity_id in processed:
                continue
            similar = find_similar_entities(entity, type_entities, processed)
            for sim_entity in similar:
                entity.merge_with(sim_entity)
                merge_map[sim_entity.entity_id] = entity.entity_id
                processed.add(sim_entity.entity_id)
            canonical_entities[entity.entity_id] = entity
            processed.add(entity.entity_id)

    return canonical_entities, merge_map


def find_similar_entities(entity: Entity, candidates: List[Entity],
                          exclude: set, threshold: float = 0.85) -> List[Entity]:
    similar = []
    entity_names = [entity.name.lower()] + [a.lower() for a in entity.aliases]
    for candidate in candidates:
        if candidate.entity_id == entity.entity_id or candidate.entity_id in exclude:
            continue
        candidate_names = [candidate.name.lower()] + [a.lower() for a in candidate.aliases]
        matched = False
        for en in entity_names:
            for cn in candidate_names:
                if en == cn or SequenceMatcher(None, en, cn).ratio() >= threshold:
                    similar.append(candidate)
                    matched = True
                    break
            if matched:
                break
    return similar


# ---------------------------------------------------------------------------
# Relation + claim resolution
# ---------------------------------------------------------------------------

def resolve_relation_entities(relations: Dict[str, Relation], entities: Dict[str, Entity],
                              merge_map: Dict[str, str], resolver: _EntityResolver) -> Dict[str, Relation]:
    resolved: Dict[str, Relation] = {}
    for rel_id, relation in relations.items():
        subject_id = _resolve_ref(relation.subject_entity_id, entities, merge_map, resolver)
        object_id = _resolve_ref(relation.object_entity_id, entities, merge_map, resolver)
        if subject_id and object_id and subject_id != object_id:
            relation.subject_entity_id = subject_id
            relation.object_entity_id = object_id
            resolved[rel_id] = relation
    return resolved


def _resolve_ref(ref: str, entities: Dict[str, Entity], merge_map: Dict[str, str],
                 resolver: _EntityResolver) -> Optional[str]:
    if ref in entities:
        return ref
    if ref in merge_map:
        return merge_map[ref]
    return resolver.resolve(ref)


def link_claims_to_entities(claims: Dict[str, Claim], entities: Dict[str, Entity],
                            resolver: _EntityResolver) -> Dict[str, Claim]:
    for claim in claims.values():
        claim_text_lower = claim.text.lower()
        linked = set()
        for entity in entities.values():
            names = [entity.name] + entity.aliases
            if any(n and n.lower() in claim_text_lower for n in names):
                linked.add(entity.entity_id)
        claim.entity_ids = list(linked)
    return claims
