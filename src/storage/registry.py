"""Corpus identity and knowledge-base registry.

Agents obtain the (per-corpus) knowledge base through ``get_knowledge_base`` so
the heavy knowledge never has to live on the checkpointed ResearchState. The KB
is keyed by ``corpus_id`` derived from the topic, which is what makes the store
*dynamic*: re-running the same or an overlapping topic reuses and extends the
existing corpus.
"""
import hashlib
import os
import re
from typing import Dict, Optional

from .knowledge_base import KnowledgeBase
from ..config.settings import get_settings


_REGISTRY: Dict[str, KnowledgeBase] = {}


def derive_corpus_id(topic: str) -> str:
    """Derive a stable, filesystem-safe corpus id from a topic string."""
    normalized = re.sub(r"\s+", " ", (topic or "").strip().lower())
    slug = re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")[:48] or "corpus"
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:10]
    return f"{slug}-{digest}"


def get_knowledge_base(corpus_id: str) -> KnowledgeBase:
    """Return (creating if needed) the KnowledgeBase for a corpus id."""
    if corpus_id in _REGISTRY:
        return _REGISTRY[corpus_id]

    settings = get_settings()
    persist_dir = None
    if settings.enable_persistence:
        persist_dir = os.path.join(settings.corpus_dir, corpus_id)
    kb = KnowledgeBase(
        corpus_id=corpus_id,
        persist_dir=persist_dir,
        enable_persistence=settings.enable_persistence,
    )
    _REGISTRY[corpus_id] = kb
    return kb


def reset_registry(corpus_id: Optional[str] = None) -> None:
    """Drop cached KB handles (mainly for tests)."""
    if corpus_id is None:
        _REGISTRY.clear()
    else:
        _REGISTRY.pop(corpus_id, None)
