"""Dense retriever backed by a pretrained sentence-transformers model.

Each chunk (heading path + text) is encoded once into a unit-length vector
and cached under ``artifacts/``; a query is encoded the same way and scored
by cosine similarity (a dot product of normalised vectors).

The encoder is injectable so the retriever can be unit-tested with a fake
encoder and no model download.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from ..corpus import Chunk, corpus_fingerprint
from ..errors import ModelUnavailableError
from .base import RetrievedChunk


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:  # (n, dim), rows L2-normalised
        ...


class SentenceTransformerEncoder:
    """Thin wrapper so the heavy import happens only when needed."""

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ModelUnavailableError(
                "sentence-transformers is not installed. Run `pip install -r requirements.txt`."
            ) from exc
        try:
            self.model = SentenceTransformer(model_name, device=device)
        except Exception as exc:  # noqa: BLE001 - surface any load failure with a clear message
            raise ModelUnavailableError(
                f"Could not load embedding model '{model_name}': {exc}. "
                "Check the model id and your network connection (first run downloads the weights)."
            ) from exc
        self.model_name = model_name
        self.batch_size = batch_size
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=len(texts) > 256,
        )
        return np.asarray(vectors, dtype=np.float32)


def _safe_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)


class DenseRetriever:
    name = "dense"

    def __init__(self, chunks: list[Chunk], encoder: Encoder, *, model_name: str,
                 artifacts_dir: Optional[Path] = None, use_cache: bool = True):
        if not chunks:
            raise ValueError("DenseRetriever needs at least one chunk")
        self.chunks = chunks
        self.encoder = encoder
        self.model_name = model_name
        self.fingerprint = corpus_fingerprint(chunks)
        self.cache_path: Optional[Path] = None
        if artifacts_dir is not None and use_cache:
            artifacts_dir = Path(artifacts_dir)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path = artifacts_dir / f"dense_{_safe_name(model_name)}_{self.fingerprint}.npy"
        self.index_seconds = 0.0
        self.embeddings = self._load_or_build()
        self.dim = int(self.embeddings.shape[1])

    def _load_or_build(self) -> np.ndarray:
        if self.cache_path is not None and self.cache_path.exists():
            emb = np.load(self.cache_path)
            if emb.shape[0] == len(self.chunks):
                return emb
        t0 = time.perf_counter()
        emb = self.encoder.encode([c.index_text() for c in self.chunks])
        self.index_seconds = time.perf_counter() - t0
        if self.cache_path is not None:
            np.save(self.cache_path, emb)
        return emb

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        k = max(0, min(k, len(self.chunks)))
        if k == 0:
            return []
        q = self.encoder.encode([query])[0]
        scores = self.embeddings @ q
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]


__all__ = ["DenseRetriever", "SentenceTransformerEncoder", "Encoder"]
