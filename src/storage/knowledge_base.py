"""Knowledge base for structured storage of papers, claims, entities, relations."""
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import os

from ..models.paper import Paper
from ..models.chunk import Chunk
from ..models.claim import Claim, ClaimType
from ..models.entity import Entity, EntityType
from ..models.relation import Relation, RelationType
from ..models.issue import Issue, IssueStatus, IssueSeverity
from .base import InMemoryStorage
from .vector_store import VectorStore
from ..config.settings import get_settings


class KnowledgeBase:
    """
    Central knowledge base for the research assistant.
    
    Manages:
    - Papers: Academic papers that have been retrieved/ingested
    - Chunks: Text chunks for RAG retrieval
    - Claims: Structured assertions extracted from papers
    - Entities: Concepts in the knowledge graph
    - Relations: Relationships between entities
    - Issues: Problems driving iteration
    
    Provides unified interface for:
    - Storage and retrieval
    - Semantic search
    - Deduplication
    - Provenance tracking
    """
    
    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize the knowledge base.
        
        Args:
            persist_dir: Directory for persisting data
        """
        settings = get_settings()
        self.persist_dir = persist_dir or settings.data_dir
        
        # Initialize storage backends
        self._papers = InMemoryStorage[Paper]()
        self._papers.set_id_field("paper_id")
        
        self._claims = InMemoryStorage[Claim]()
        self._claims.set_id_field("claim_id")
        
        self._entities = InMemoryStorage[Entity]()
        self._entities.set_id_field("entity_id")
        
        self._relations = InMemoryStorage[Relation]()
        self._relations.set_id_field("relation_id")
        
        self._issues = InMemoryStorage[Issue]()
        self._issues.set_id_field("issue_id")
        
        # Vector store for chunks
        vector_path = os.path.join(self.persist_dir, "vector_store") if self.persist_dir else None
        self._vector_store = VectorStore(persist_path=vector_path)
        
        # Chunk metadata (in addition to vector store)
        self._chunks: Dict[str, Chunk] = {}
        
        # Counters for ID generation
        self._counters = {
            "claim": 0,
            "entity": 0,
            "relation": 0,
            "issue": 0,
        }
    
    # ==================== Paper Methods ====================
    
    def add_paper(self, paper: Paper) -> str:
        """Add a paper to the knowledge base."""
        # Check for duplicates
        existing = self.find_duplicate_paper(paper)
        if existing:
            return existing.paper_id
        return self._papers.save(paper)
    
    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a paper by ID."""
        return self._papers.get(paper_id)
    
    def get_all_papers(self) -> List[Paper]:
        """Get all papers."""
        return self._papers.get_all()
    
    def get_ingested_papers(self) -> List[Paper]:
        """Get all papers that have been fully ingested."""
        return self._papers.query({"is_ingested": True})
    
    def find_duplicate_paper(self, paper: Paper) -> Optional[Paper]:
        """Find if a paper already exists (by DOI, arXiv ID, or title similarity)."""
        for existing in self._papers.get_all():
            # Check DOI match
            if paper.doi and existing.doi and paper.doi == existing.doi:
                return existing
            # Check arXiv ID match
            if paper.arxiv_id and existing.arxiv_id:
                # Normalize arXiv IDs (remove version)
                p_base = paper.arxiv_id.split("v")[0]
                e_base = existing.arxiv_id.split("v")[0]
                if p_base == e_base:
                    return existing
            # Check title similarity (simple exact match for now)
            if paper.title and existing.title:
                if paper.title.lower().strip() == existing.title.lower().strip():
                    return existing
        return None
    
    def update_paper(self, paper_id: str, updates: Dict[str, Any]) -> Optional[Paper]:
        """Update a paper."""
        return self._papers.update(paper_id, updates)
    
    def mark_paper_ingested(self, paper_id: str) -> None:
        """Mark a paper as fully ingested."""
        self._papers.update(paper_id, {"is_ingested": True, "ingestion_status": "complete"})
    
    # ==================== Chunk Methods ====================
    
    def add_chunk(self, chunk: Chunk) -> str:
        """Add a chunk to the knowledge base."""
        self._chunks[chunk.chunk_id] = chunk
        self._vector_store.add_chunks([chunk])
        return chunk.chunk_id
    
    def add_chunks(self, chunks: List[Chunk]) -> List[str]:
        """Add multiple chunks."""
        ids = []
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            ids.append(chunk.chunk_id)
        self._vector_store.add_chunks(chunks)
        return ids
    
    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        """Get a chunk by ID."""
        return self._chunks.get(chunk_id)
    
    def get_chunks_for_paper(self, paper_id: str) -> List[Chunk]:
        """Get all chunks for a paper."""
        return [c for c in self._chunks.values() if c.paper_id == paper_id]
    
    def search_chunks(self, query: str, k: int = 5, 
                      paper_id: Optional[str] = None) -> List[Chunk]:
        """Search chunks semantically."""
        filters = {"paper_id": paper_id} if paper_id else None
        results = self._vector_store.search(query, k=k, filters=filters)
        return [chunk for chunk, score in results]
    
    # ==================== Claim Methods ====================
    
    def add_claim(self, claim: Claim) -> str:
        """Add a claim to the knowledge base."""
        return self._claims.save(claim)
    
    def get_claim(self, claim_id: str) -> Optional[Claim]:
        """Get a claim by ID."""
        return self._claims.get(claim_id)
    
    def get_all_claims(self) -> List[Claim]:
        """Get all claims."""
        return self._claims.get_all()
    
    def get_claims_for_paper(self, paper_id: str) -> List[Claim]:
        """Get all claims from a paper."""
        return self._claims.query({"paper_id": paper_id})
    
    def get_claims_by_type(self, claim_type: ClaimType) -> List[Claim]:
        """Get claims by type."""
        return self._claims.query({"claim_type": claim_type})
    
    def generate_claim_id(self) -> str:
        """Generate a unique claim ID."""
        self._counters["claim"] += 1
        return f"claim_{self._counters['claim']:05d}"
    
    def find_contradicting_claims(self, claim: Claim) -> List[Claim]:
        """Find claims that might contradict the given claim."""
        # Simple implementation: find claims with same entities but different results
        contradictions = []
        for existing in self._claims.get_all():
            if existing.claim_id == claim.claim_id:
                continue
            # Check for entity overlap
            common_entities = set(claim.entity_ids) & set(existing.entity_ids)
            if common_entities and existing.claim_type == claim.claim_type:
                # Potential contradiction - would need more sophisticated analysis
                contradictions.append(existing)
        return contradictions
    
    # ==================== Entity Methods ====================
    
    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the knowledge base."""
        # Check for duplicates/merges
        existing = self.find_matching_entity(entity.name, entity.entity_type)
        if existing:
            existing.merge_with(entity)
            return existing.entity_id
        return self._entities.save(entity)
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self._entities.get(entity_id)
    
    def get_all_entities(self) -> List[Entity]:
        """Get all entities."""
        return self._entities.get_all()
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get entities by type."""
        return self._entities.query({"entity_type": entity_type})
    
    def find_matching_entity(self, name: str, entity_type: EntityType) -> Optional[Entity]:
        """Find an existing entity by name and type."""
        for entity in self._entities.get_all():
            if entity.entity_type == entity_type and entity.matches_name(name):
                return entity
        return None
    
    def generate_entity_id(self) -> str:
        """Generate a unique entity ID."""
        self._counters["entity"] += 1
        return f"entity_{self._counters['entity']:05d}"
    
    # ==================== Relation Methods ====================
    
    def add_relation(self, relation: Relation) -> str:
        """Add a relation to the knowledge base."""
        return self._relations.save(relation)
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get a relation by ID."""
        return self._relations.get(relation_id)
    
    def get_all_relations(self) -> List[Relation]:
        """Get all relations."""
        return self._relations.get_all()
    
    def get_relations_for_entity(self, entity_id: str) -> List[Relation]:
        """Get all relations involving an entity."""
        relations = []
        for rel in self._relations.get_all():
            if rel.subject_entity_id == entity_id or rel.object_entity_id == entity_id:
                relations.append(rel)
        return relations
    
    def generate_relation_id(self) -> str:
        """Generate a unique relation ID."""
        self._counters["relation"] += 1
        return f"rel_{self._counters['relation']:05d}"
    
    # ==================== Issue Methods ====================
    
    def add_issue(self, issue: Issue) -> str:
        """Add an issue to the knowledge base."""
        return self._issues.save(issue)
    
    def get_issue(self, issue_id: str) -> Optional[Issue]:
        """Get an issue by ID."""
        return self._issues.get(issue_id)
    
    def get_all_issues(self) -> List[Issue]:
        """Get all issues."""
        return self._issues.get_all()
    
    def get_open_issues(self, severity: Optional[IssueSeverity] = None) -> List[Issue]:
        """Get open issues, optionally filtered by severity."""
        issues = self._issues.query({"status": IssueStatus.OPEN})
        if severity:
            issues = [i for i in issues if i.severity == severity]
        return issues
    
    def resolve_issue(self, issue_id: str, notes: str = "") -> None:
        """Resolve an issue."""
        issue = self._issues.get(issue_id)
        if issue:
            issue.resolve(notes)
    
    def generate_issue_id(self) -> str:
        """Generate a unique issue ID."""
        self._counters["issue"] += 1
        return f"issue_{self._counters['issue']:05d}"
    
    # ==================== Statistics ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        return {
            "papers_total": self._papers.count(),
            "papers_ingested": len(self.get_ingested_papers()),
            "chunks": len(self._chunks),
            "claims": self._claims.count(),
            "entities": self._entities.count(),
            "relations": self._relations.count(),
            "issues_total": self._issues.count(),
            "issues_open": len(self.get_open_issues()),
            "issues_blockers": len(self.get_open_issues(IssueSeverity.BLOCKER)),
        }
    
    # ==================== Persistence ====================
    
    def save(self) -> None:
        """Persist knowledge base to disk."""
        if not self.persist_dir:
            return
        
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Save vector store
        self._vector_store.save()
        
        # Save structured data as JSON
        data = {
            "papers": [p.model_dump() for p in self._papers.get_all()],
            "chunks": [c.model_dump() for c in self._chunks.values()],
            "claims": [c.model_dump() for c in self._claims.get_all()],
            "entities": [e.model_dump() for e in self._entities.get_all()],
            "relations": [r.model_dump() for r in self._relations.get_all()],
            "issues": [i.model_dump() for i in self._issues.get_all()],
            "counters": self._counters,
        }
        
        with open(os.path.join(self.persist_dir, "knowledge_base.json"), "w") as f:
            json.dump(data, f, indent=2, default=str)
    
    def load(self) -> None:
        """Load knowledge base from disk."""
        kb_path = os.path.join(self.persist_dir, "knowledge_base.json")
        if not os.path.exists(kb_path):
            return
        
        with open(kb_path, "r") as f:
            data = json.load(f)
        
        # Load papers
        for p_data in data.get("papers", []):
            paper = Paper(**p_data)
            self._papers.save(paper)
        
        # Load chunks
        for c_data in data.get("chunks", []):
            chunk = Chunk(**c_data)
            self._chunks[chunk.chunk_id] = chunk
        
        # Load claims
        for c_data in data.get("claims", []):
            claim = Claim(**c_data)
            self._claims.save(claim)
        
        # Load entities
        for e_data in data.get("entities", []):
            entity = Entity(**e_data)
            self._entities.save(entity)
        
        # Load relations
        for r_data in data.get("relations", []):
            relation = Relation(**r_data)
            self._relations.save(relation)
        
        # Load issues
        for i_data in data.get("issues", []):
            issue = Issue(**i_data)
            self._issues.save(issue)
        
        # Load counters
        self._counters = data.get("counters", self._counters)
    
    def clear(self) -> None:
        """Clear all data from knowledge base."""
        self._papers.clear()
        self._chunks.clear()
        self._claims.clear()
        self._entities.clear()
        self._relations.clear()
        self._issues.clear()
        self._vector_store.clear()
        self._counters = {"claim": 0, "entity": 0, "relation": 0, "issue": 0}
