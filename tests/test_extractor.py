"""Extractor parses claims/entities/relations and never fabricates evidence ids."""
import json

from src.agents.extractor import extract_from_chunks
from src.models.chunk import Chunk, SourceType


class _Resp:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}


class _FakeLLM:
    def __init__(self, payload):
        self._payload = payload

    def invoke(self, messages):
        return _Resp(json.dumps(self._payload))


def _chunks():
    return [
        Chunk.create(paper_id="p1", text="BERT improves GLUE by 7 points.",
                     source_type=SourceType.ARXIV_HTML, section_path="Results"),
        Chunk.create(paper_id="p1", text="We use the Adam optimizer.",
                     source_type=SourceType.ARXIV_HTML, section_path="Methods"),
    ]


def test_extract_valid_and_invalid_evidence():
    chunks = _chunks()
    valid_id = chunks[0].chunk_id
    payload = {
        "claims": [
            {"type": "empirical_result", "text": "BERT improves GLUE by 7 points.",
             "evidence_chunk_id": valid_id, "confidence": "high"},
            {"type": "method_summary", "text": "Adam is used.",
             "evidence_chunk_id": "chunk:doesnotexist", "confidence": "medium"},
        ],
        "entities": [{"type": "model", "name": "BERT", "aliases": [], "description": "a model"}],
        "relations": [{"subject": "BERT", "predicate": "evaluated_on", "object": "GLUE",
                       "evidence_chunk_id": valid_id}],
    }
    claims, entities, relations = extract_from_chunks(chunks, "p1", "nlp", _FakeLLM(payload))

    assert len(claims) == 2
    # Valid evidence kept verbatim
    assert claims[0].evidence[0].chunk_id == valid_id
    # Invalid evidence id must fall back to a real chunk (never fabricated)
    assert claims[1].evidence[0].chunk_id == chunks[0].chunk_id
    assert claims[1].evidence[0].relevance_score == 0.5
    assert entities[0].name == "BERT"
    assert relations[0].predicate.value == "evaluated_on"


def test_extractor_ignores_empty_claims():
    chunks = _chunks()
    payload = {"claims": [{"type": "definition", "text": "  "}], "entities": [], "relations": []}
    claims, _, _ = extract_from_chunks(chunks, "p1", "nlp", _FakeLLM(payload))
    assert claims == []
