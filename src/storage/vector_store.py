"""Semantic embedding index for RAG retrieval over chunks and claims.

This is a lightweight, dependency-robust vector index: embeddings are computed
via ``agents.base.embed_texts`` (OpenAI) and similarity is brute-force cosine in
numpy.  At per-corpus scale (hundreds to a few thousand items) this is instant
and avoids FAISS (de)serialisation fragility, which matters for deployment.

Embeddings are cached by stable item id, so re-runs never re-embed material that
already exists in the corpus.  The index degrades gracefully: when embeddings
are unavailable it simply returns no results and callers fall back to keyword
scoring.
"""
from typing import List, Optional, Dict, Any, Tuple, Callable

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


class EmbeddingIndex:
    """In-memory cosine-similarity index with an external embed function."""

    def __init__(self, embed_fn: Callable[[List[str]], Optional[List[List[float]]]]):
        self._embed_fn = embed_fn
        self._ids: List[str] = []
        self._row_by_id: Dict[str, int] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._matrix = None  # np.ndarray (n, dim), L2-normalised rows

    # -- persistence helpers -------------------------------------------------

    def has(self, item_id: str) -> bool:
        return item_id in self._row_by_id

    def load_vectors(self, vectors: Dict[str, List[float]],
                     metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Populate the index directly from persisted vectors (no embedding)."""
        if np is None or not vectors:
            return
        rows = []
        for item_id, vec in vectors.items():
            if item_id in self._row_by_id:
                continue
            self._row_by_id[item_id] = len(self._ids)
            self._ids.append(item_id)
            if metadata and item_id in metadata:
                self._metadata[item_id] = metadata[item_id]
            rows.append(vec)
        if rows:
            new = self._normalise(np.array(rows, dtype="float32"))
            self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])

    def vectors(self) -> Dict[str, List[float]]:
        """Return all stored vectors keyed by id (for persistence)."""
        if np is None or self._matrix is None:
            return {}
        return {item_id: self._matrix[row].tolist() for item_id, row in self._row_by_id.items()}

    # -- indexing ------------------------------------------------------------

    def add(self, items: List[Tuple[str, str, Dict[str, Any]]]) -> Dict[str, List[float]]:
        """Add (id, text, metadata) items, embedding only new ids.

        Returns a mapping of newly-embedded id -> vector (so the caller can
        persist them). Returns {} when embeddings are unavailable.
        """
        if np is None:
            return {}
        new_items = [(i, t, m) for (i, t, m) in items if i not in self._row_by_id and t]
        if not new_items:
            return {}
        vectors = self._embed_fn([t for (_, t, _) in new_items])
        if not vectors:
            return {}
        rows = []
        embedded: Dict[str, List[float]] = {}
        for (item_id, _text, meta), vec in zip(new_items, vectors):
            self._row_by_id[item_id] = len(self._ids)
            self._ids.append(item_id)
            self._metadata[item_id] = meta or {}
            rows.append(vec)
            embedded[item_id] = vec
        new = self._normalise(np.array(rows, dtype="float32"))
        self._matrix = new if self._matrix is None else np.vstack([self._matrix, new])
        return embedded

    # -- search --------------------------------------------------------------

    def search(self, query: str, k: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[Tuple[str, float]]:
        """Return up to k (item_id, score) pairs most similar to the query."""
        if np is None or self._matrix is None or not self._ids:
            return []
        qvec = self._embed_fn([query])
        if not qvec:
            return []
        q = self._normalise(np.array(qvec, dtype="float32"))[0]
        scores = self._matrix @ q  # cosine (rows already normalised)
        order = np.argsort(-scores)
        results: List[Tuple[str, float]] = []
        for row in order:
            item_id = self._ids[int(row)]
            if filters:
                meta = self._metadata.get(item_id, {})
                if not all(meta.get(fk) == fv for fk, fv in filters.items()):
                    continue
            results.append((item_id, float(scores[int(row)])))
            if len(results) >= k:
                break
        return results

    @staticmethod
    def _normalise(matrix):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    def count(self) -> int:
        return len(self._ids)


# Backwards-compatible alias; the old FAISS-based VectorStore is superseded by
# EmbeddingIndex, which the KnowledgeBase now uses directly.
VectorStore = EmbeddingIndex
