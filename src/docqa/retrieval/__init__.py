"""Retrieval: a lexical BM25 baseline and a dense Transformer retriever."""

from __future__ import annotations

from typing import Optional

from ..config import Config
from ..corpus import Chunk
from .base import RetrievedChunk, Retriever
from .bm25 import BM25Retriever, tokenize
from .dense import DenseRetriever, Encoder, SentenceTransformerEncoder
from .filters import SourceDiverseRetriever, limit_per_source
from .hybrid import HybridRetriever

_ENCODER_CACHE: dict[str, SentenceTransformerEncoder] = {}


def get_encoder(model_name: str) -> SentenceTransformerEncoder:
    """Load each embedding model once per process."""
    if model_name not in _ENCODER_CACHE:
        _ENCODER_CACHE[model_name] = SentenceTransformerEncoder(model_name)
    return _ENCODER_CACHE[model_name]


def build_retriever(cfg: Config, chunks: list[Chunk], method: Optional[str] = None,
                    encoder: Optional[Encoder] = None, max_per_source: Optional[int] = None) -> Retriever:
    """Construct the retriever named in the config (or ``method``)."""
    r = cfg.retrieval
    method = method or r.method
    if method == "bm25":
        base: Retriever = BM25Retriever(chunks, k1=r.bm25_k1, b=r.bm25_b)
    elif method == "dense":
        enc = encoder or get_encoder(r.dense_model)
        base = DenseRetriever(chunks, enc, model_name=r.dense_model,
                              artifacts_dir=cfg.resolve(r.artifacts_dir))
    elif method == "hybrid":
        enc = encoder or get_encoder(r.dense_model)
        base = HybridRetriever(
            BM25Retriever(chunks, k1=r.bm25_k1, b=r.bm25_b),
            DenseRetriever(chunks, enc, model_name=r.dense_model, artifacts_dir=cfg.resolve(r.artifacts_dir)),
        )
    else:
        raise ValueError(f"Unknown retrieval method '{method}' (expected 'bm25', 'dense' or 'hybrid')")
    limit = r.max_per_source if max_per_source is None else max_per_source
    if limit:
        return SourceDiverseRetriever(base, int(limit))
    return base


__all__ = [
    "Retriever",
    "RetrievedChunk",
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "SentenceTransformerEncoder",
    "SourceDiverseRetriever",
    "build_retriever",
    "get_encoder",
    "limit_per_source",
    "tokenize",
]
