"""EmbeddingIndex ranks by cosine similarity and caches by id."""
import math

import pytest

np = pytest.importorskip("numpy")

from src.storage.vector_store import EmbeddingIndex


VOCAB = ["attention", "graph", "reinforcement", "protein", "diffusion"]


def _bow(text):
    text = text.lower()
    return [float(text.count(word)) for word in VOCAB]


def fake_embed(texts):
    return [_bow(t) for t in texts]


def test_ranks_relevant_first():
    idx = EmbeddingIndex(fake_embed)
    idx.add([
        ("c1", "attention attention transformers", {"paper_id": "p1"}),
        ("c2", "graph neural networks", {"paper_id": "p2"}),
        ("c3", "reinforcement learning policy", {"paper_id": "p3"}),
    ])
    hits = idx.search("attention mechanism", k=2)
    assert hits[0][0] == "c1"
    assert len(hits) == 2


def test_filter_by_metadata():
    idx = EmbeddingIndex(fake_embed)
    idx.add([
        ("c1", "graph graph", {"paper_id": "p1"}),
        ("c2", "graph graph", {"paper_id": "p2"}),
    ])
    hits = idx.search("graph", k=5, filters={"paper_id": "p2"})
    assert all(h[0] == "c2" for h in hits)


def test_dedup_and_cache():
    idx = EmbeddingIndex(fake_embed)
    first = idx.add([("c1", "diffusion models", {})])
    second = idx.add([("c1", "diffusion models", {})])  # same id
    assert "c1" in first
    assert second == {}  # already embedded, not re-added
    assert idx.count() == 1
