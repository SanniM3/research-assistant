"""KB persistence round-trip and incremental-extraction bookkeeping."""
import os

from src.storage.knowledge_base import KnowledgeBase
from src.models.paper import Paper
from src.models.chunk import Chunk, SourceType
from src.models.claim import Claim, ClaimType, Evidence
from src.models.entity import Entity, EntityType
from src.models.relation import Relation, RelationType


def _paper(pid="arxiv:1234.5678"):
    return Paper(paper_id=pid, title="A Test Paper", authors=["Ada L."], year=2024,
                 ingestion_status="complete")


def _chunk(pid="arxiv:1234.5678", text="Transformers use self-attention."):
    return Chunk.create(paper_id=pid, text=text, source_type=SourceType.ARXIV_HTML,
                        section_path="Methods")


def test_persistence_round_trip(tmp_path):
    persist = os.path.join(tmp_path, "corpus")
    kb = KnowledgeBase("corpus-x", persist_dir=persist, enable_persistence=True)

    paper = _paper()
    chunk = _chunk()
    kb.upsert_paper(paper)
    kb.upsert_chunks([chunk])
    claim = Claim(claim_id="claim_1", claim_type=ClaimType.METHOD_SUMMARY,
                  text="Self-attention captures long-range dependencies.",
                  paper_id=paper.paper_id, evidence=[Evidence(chunk_id=chunk.chunk_id)])
    kb.upsert_claims([claim])
    kb.upsert_entity(Entity(entity_id="e1", entity_type=EntityType.METHOD, name="Transformer"))
    kb.upsert_relation(Relation(relation_id="r1", subject_entity_id="e1",
                                predicate=RelationType.USES, object_entity_id="e1"))
    kb.mark_chunks_extracted([chunk.chunk_id])
    kb.persist()

    # New handle on the same corpus dir should reload everything.
    kb2 = KnowledgeBase("corpus-x", persist_dir=persist, enable_persistence=True)
    assert kb2.reviewed_count() == 1
    assert len(kb2.all_chunks()) == 1
    assert len(kb2.all_claims()) == 1
    assert len(kb2.all_entities()) == 1
    assert len(kb2.all_relations()) == 1
    # Extracted chunk should not be re-processed.
    assert kb2.unextracted_chunks() == []


def test_abstract_only_not_counted_as_reviewed(tmp_path):
    kb = KnowledgeBase("corpus-y", persist_dir=os.path.join(tmp_path, "y"))
    kb.upsert_paper(_paper("arxiv:1"))  # complete
    abstract_only = _paper("arxiv:2")
    abstract_only.ingestion_status = "abstract_only"
    kb.upsert_paper(abstract_only)
    assert kb.reviewed_count() == 1
    assert len(kb.all_papers()) == 2


def test_chunk_dedup_by_id(tmp_path):
    kb = KnowledgeBase("corpus-z", persist_dir=os.path.join(tmp_path, "z"))
    c1 = _chunk(text="identical text")
    c2 = _chunk(text="identical text")  # same content+paper -> same chunk_id
    kb.upsert_chunks([c1, c2])
    assert len(kb.all_chunks()) == 1
