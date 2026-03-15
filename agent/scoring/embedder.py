"""Embedding backends. Model name always comes from config — never hardcoded."""
from abc import ABC, abstractmethod

import numpy as np


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts; returns parallel list of float vectors."""
        ...


class LocalSentenceTransformer(EmbeddingBackend):
    """Local CPU/GPU inference via sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        # Lazy import — sentence-transformers is large and slows startup
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))
