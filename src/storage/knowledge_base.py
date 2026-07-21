"""Dynamic, persistent knowledge base for the research assistant.

This is the single source of truth at runtime. It holds papers, chunks, claims,
entities, relations and issues, plus two semantic indexes (chunks and claims)
used for RAG. It persists per corpus to a single SQLite file plus cached
embeddings, so re-runs of the same or overlapping topics accumulate knowledge
instead of starting from scratch, and no chunk is ever re-embedded or a paper
re-ingested unnecessarily.

Agents access the KB through ``storage.registry.get_knowledge_base(corpus_id)``
rather than holding it on the (checkpointed) ResearchState, which keeps state
small and gives us durable, queryable knowledge.
"""
from typing import List, Optional, Dict, Any, Tuple
import json
import os
import sqlite3

from ..models.paper import Paper
from ..models.chunk import Chunk
from ..models.claim import Claim
from ..models.entity import Entity, EntityType
from ..models.relation import Relation
from ..models.issue import Issue
from .vector_store import EmbeddingIndex


_TABLES = ["papers", "chunks", "claims", "entities", "relations", "issues"]

_INGESTION_COMPLETE = "complete"
_INGESTION_ABSTRACT_ONLY = "abstract_only"


class KnowledgeBase:
    """Runtime knowledge store with semantic retrieval and per-corpus persistence."""

    def __init__(self, corpus_id: str, persist_dir: Optional[str] = None,
                 enable_persistence: bool = True):
        from ..agents.base import embed_texts  # local import avoids cycle

        self.corpus_id = corpus_id
        self.enable_persistence = enable_persistence
        self.persist_dir = persist_dir

        # In-memory stores (fast path during a run)
        self.papers: Dict[str, Paper] = {}
        self.chunks: Dict[str, Chunk] = {}
        self.claims: Dict[str, Claim] = {}
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.issues: Dict[str, Issue] = {}
        self._extracted_chunk_ids: set = set()

        # Semantic indexes
        self.chunk_index = EmbeddingIndex(embed_texts)
        self.claim_index = EmbeddingIndex(embed_texts)

        self._conn: Optional[sqlite3.Connection] = None
        if self.enable_persistence and self.persist_dir:
            os.makedirs(self.persist_dir, exist_ok=True)
            self._conn = sqlite3.connect(os.path.join(self.persist_dir, "kb.sqlite"))
            self._init_schema()
            self._load()

    # ------------------------------------------------------------------ schema
    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        for table in _TABLES:
            cur.execute(f"CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, data TEXT)")
        cur.execute("CREATE TABLE IF NOT EXISTS embeddings (kind TEXT, id TEXT, vector TEXT, PRIMARY KEY (kind, id))")
        cur.execute("CREATE TABLE IF NOT EXISTS extracted_chunks (chunk_id TEXT PRIMARY KEY)")
        self._conn.commit()

    def _load(self) -> None:
        cur = self._conn.cursor()
        loaders = {
            "papers": (Paper, self.papers, "paper_id"),
            "chunks": (Chunk, self.chunks, "chunk_id"),
            "claims": (Claim, self.claims, "claim_id"),
            "entities": (Entity, self.entities, "entity_id"),
            "relations": (Relation, self.relations, "relation_id"),
            "issues": (Issue, self.issues, "issue_id"),
        }
        for table, (model, store, _id) in loaders.items():
            for (data,) in cur.execute(f"SELECT data FROM {table}"):
                try:
                    obj = model(**json.loads(data))
                    store[getattr(obj, _id)] = obj
                except Exception:
                    continue
        for (chunk_id,) in cur.execute("SELECT chunk_id FROM extracted_chunks"):
            self._extracted_chunk_ids.add(chunk_id)
        # Load cached embeddings into the indexes (no re-embedding).
        chunk_vecs, claim_vecs = {}, {}
        for kind, item_id, vector in cur.execute("SELECT kind, id, vector FROM embeddings"):
            try:
                vec = json.loads(vector)
            except Exception:
                continue
            if kind == "chunk":
                chunk_vecs[item_id] = vec
            elif kind == "claim":
                claim_vecs[item_id] = vec
        self.chunk_index.load_vectors(chunk_vecs, self._chunk_meta_map())
        self.claim_index.load_vectors(claim_vecs, self._claim_meta_map())

    # ------------------------------------------------------------- persistence
    def _upsert(self, table: str, item_id: str, obj) -> None:
        if not self._conn:
            return
        data = json.dumps(obj.model_dump(mode="json"))
        self._conn.execute(
            f"INSERT INTO {table} (id, data) VALUES (?, ?) "
            f"ON CONFLICT(id) DO UPDATE SET data=excluded.data",
            (item_id, data),
        )
        self._conn.commit()

    def _save_embeddings(self, kind: str, vectors: Dict[str, List[float]]) -> None:
        if not self._conn or not vectors:
            return
        self._conn.executemany(
            "INSERT INTO embeddings (kind, id, vector) VALUES (?, ?, ?) "
            "ON CONFLICT(kind, id) DO UPDATE SET vector=excluded.vector",
            [(kind, i, json.dumps(v)) for i, v in vectors.items()],
        )
        self._conn.commit()

    def persist(self) -> None:
        """Flush everything to disk (idempotent; also called incrementally)."""
        if not self._conn:
            return
        stores = {
            "papers": (self.papers, "paper_id"),
            "chunks": (self.chunks, "chunk_id"),
            "claims": (self.claims, "claim_id"),
            "entities": (self.entities, "entity_id"),
            "relations": (self.relations, "relation_id"),
            "issues": (self.issues, "issue_id"),
        }
        for table, (store, id_field) in stores.items():
            for item_id, obj in store.items():
                self._upsert(table, item_id, obj)
        self._conn.commit()

    # ------------------------------------------------------------------ papers
    def upsert_paper(self, paper: Paper) -> None:
        self.papers[paper.paper_id] = paper
        self._upsert("papers", paper.paper_id, paper)

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        return self.papers.get(paper_id)

    def all_papers(self) -> List[Paper]:
        return list(self.papers.values())

    def papers_map(self) -> Dict[str, Paper]:
        return dict(self.papers)

    def reviewed_papers(self) -> List[Paper]:
        """Papers with real full text extracted (excludes abstract-only)."""
        return [p for p in self.papers.values() if p.ingestion_status == _INGESTION_COMPLETE]

    def reviewed_count(self) -> int:
        return len(self.reviewed_papers())

    # ------------------------------------------------------------------ chunks
    def has_chunk(self, chunk_id: str) -> bool:
        return chunk_id in self.chunks

    def upsert_chunks(self, chunks: List[Chunk]) -> None:
        new_index_items = []
        for chunk in chunks:
            if chunk.chunk_id not in self.chunks:
                new_index_items.append((chunk.chunk_id, chunk.text, self._chunk_meta(chunk)))
            self.chunks[chunk.chunk_id] = chunk
            self._upsert("chunks", chunk.chunk_id, chunk)
        if new_index_items:
            embedded = self.chunk_index.add(new_index_items)
            self._save_embeddings("chunk", embedded)

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        return self.chunks.get(chunk_id)

    def all_chunks(self) -> List[Chunk]:
        return list(self.chunks.values())

    def chunks_map(self) -> Dict[str, Chunk]:
        return dict(self.chunks)

    def chunks_for_paper(self, paper_id: str) -> List[Chunk]:
        return [c for c in self.chunks.values() if c.paper_id == paper_id]

    def search_chunks(self, query: str, k: int = 25,
                      filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Chunk, float]]:
        hits = self.chunk_index.search(query, k=k, filters=filters)
        out = []
        for chunk_id, score in hits:
            chunk = self.chunks.get(chunk_id)
            if chunk:
                out.append((chunk, score))
        return out

    # ------------------------------------------------------------------ claims
    def upsert_claims(self, claims: List[Claim]) -> None:
        new_index_items = []
        for claim in claims:
            if claim.claim_id not in self.claims:
                new_index_items.append((claim.claim_id, claim.text, self._claim_meta(claim)))
            self.claims[claim.claim_id] = claim
            self._upsert("claims", claim.claim_id, claim)
        if new_index_items:
            embedded = self.claim_index.add(new_index_items)
            self._save_embeddings("claim", embedded)

    def replace_claims(self, claims: Dict[str, Claim]) -> None:
        """Overwrite claim records in place (used by KB curator after linking)."""
        self.claims = dict(claims)
        for cid, claim in self.claims.items():
            self._upsert("claims", cid, claim)

    def get_claim(self, claim_id: str) -> Optional[Claim]:
        return self.claims.get(claim_id)

    def all_claims(self) -> List[Claim]:
        return list(self.claims.values())

    def claims_map(self) -> Dict[str, Claim]:
        return dict(self.claims)

    def search_claims(self, query: str, k: int = 60,
                      filters: Optional[Dict[str, Any]] = None) -> List[Tuple[Claim, float]]:
        hits = self.claim_index.search(query, k=k, filters=filters)
        out = []
        for claim_id, score in hits:
            claim = self.claims.get(claim_id)
            if claim:
                out.append((claim, score))
        return out

    # --------------------------------------------------------- extraction bookkeeping
    def mark_chunks_extracted(self, chunk_ids: List[str]) -> None:
        for cid in chunk_ids:
            self._extracted_chunk_ids.add(cid)
        if self._conn and chunk_ids:
            self._conn.executemany(
                "INSERT OR IGNORE INTO extracted_chunks (chunk_id) VALUES (?)",
                [(cid,) for cid in chunk_ids],
            )
            self._conn.commit()

    def unextracted_chunks(self) -> List[Chunk]:
        return [c for c in self.chunks.values() if c.chunk_id not in self._extracted_chunk_ids]

    # ---------------------------------------------------------------- entities
    def upsert_entity(self, entity: Entity) -> None:
        self.entities[entity.entity_id] = entity
        self._upsert("entities", entity.entity_id, entity)

    def replace_entities(self, entities: Dict[str, Entity]) -> None:
        # Remove rows that no longer exist (post-merge) then rewrite the rest.
        if self._conn:
            self._conn.execute("DELETE FROM entities")
            self._conn.commit()
        self.entities = dict(entities)
        for eid, entity in self.entities.items():
            self._upsert("entities", eid, entity)

    def all_entities(self) -> List[Entity]:
        return list(self.entities.values())

    def entities_map(self) -> Dict[str, Entity]:
        return dict(self.entities)

    # --------------------------------------------------------------- relations
    def upsert_relation(self, relation: Relation) -> None:
        self.relations[relation.relation_id] = relation
        self._upsert("relations", relation.relation_id, relation)

    def replace_relations(self, relations: Dict[str, Relation]) -> None:
        if self._conn:
            self._conn.execute("DELETE FROM relations")
            self._conn.commit()
        self.relations = dict(relations)
        for rid, relation in self.relations.items():
            self._upsert("relations", rid, relation)

    def all_relations(self) -> List[Relation]:
        return list(self.relations.values())

    def relations_map(self) -> Dict[str, Relation]:
        return dict(self.relations)

    # ------------------------------------------------------------------ issues
    def upsert_issue(self, issue: Issue) -> None:
        self.issues[issue.issue_id] = issue
        self._upsert("issues", issue.issue_id, issue)

    # ------------------------------------------------------------------- stats
    def stats(self) -> Dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "papers_total": len(self.papers),
            "papers_reviewed": self.reviewed_count(),
            "chunks": len(self.chunks),
            "claims": len(self.claims),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "chunk_index": self.chunk_index.count(),
            "claim_index": self.claim_index.count(),
        }

    def export_graph(self) -> Dict[str, Any]:
        """Export the knowledge graph for inspection/visualisation."""
        return {
            "corpus_id": self.corpus_id,
            "papers": [p.model_dump(mode="json") for p in self.papers.values()],
            "entities": [e.model_dump(mode="json") for e in self.entities.values()],
            "relations": [r.model_dump(mode="json") for r in self.relations.values()],
            "claims": [c.model_dump(mode="json") for c in self.claims.values()],
        }

    # ------------------------------------------------------------------ helpers
    def _chunk_meta(self, chunk: Chunk) -> Dict[str, Any]:
        return {"paper_id": chunk.paper_id, "source_type": chunk.source_type.value,
                "language": chunk.metadata.language}

    def _claim_meta(self, claim: Claim) -> Dict[str, Any]:
        return {"paper_id": claim.paper_id, "claim_type": claim.claim_type.value}

    def _chunk_meta_map(self) -> Dict[str, Dict[str, Any]]:
        return {c.chunk_id: self._chunk_meta(c) for c in self.chunks.values()}

    def _claim_meta_map(self) -> Dict[str, Dict[str, Any]]:
        return {c.claim_id: self._claim_meta(c) for c in self.claims.values()}

    def clear(self) -> None:
        self.papers.clear(); self.chunks.clear(); self.claims.clear()
        self.entities.clear(); self.relations.clear(); self.issues.clear()
        self._extracted_chunk_ids.clear()
