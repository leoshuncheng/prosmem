"""Embedding module using local BGE model via sentence-transformers."""

from __future__ import annotations

import numpy as np

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # BAAI/bge-small-en-v1.5: 384-dim, ~130MB, good quality/size tradeoff
        _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _model


def embed_text(text: str) -> np.ndarray:
    """Embed a single text string. Returns normalized vector."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Embed multiple texts in batch."""
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True)
    return [vecs[i] for i in range(len(texts))]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two (already normalized) vectors."""
    return float(np.dot(a, b))
