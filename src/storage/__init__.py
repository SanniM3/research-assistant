"""Storage layer for the research assistant."""
from .base import StorageBackend, InMemoryStorage
from .vector_store import VectorStore, EmbeddingIndex
from .knowledge_base import KnowledgeBase
from .registry import get_knowledge_base, derive_corpus_id, reset_registry

__all__ = [
    "StorageBackend",
    "InMemoryStorage",
    "VectorStore",
    "EmbeddingIndex",
    "KnowledgeBase",
    "get_knowledge_base",
    "derive_corpus_id",
    "reset_registry",
]
