"""BM25 lexical retriever: the non-Transformer baseline.

Implemented from scratch (about 60 lines) so every step can be explained:

    score(q, d) = sum over query terms t of
                  idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

    idf(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

where N is the number of chunks, df(t) the number of chunks containing t,
|d| the chunk length in tokens and avgdl the average chunk length.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

from ..corpus import Chunk
from .base import RetrievedChunk

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens; underscores are kept so ``response_model`` stays one token."""
    return TOKEN_RE.findall(text.lower())


class BM25Retriever:
    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        doc_tokens = [tokenize(c.index_text()) for c in chunks]
        self.doc_len = np.array([len(t) for t in doc_tokens], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(doc_tokens) else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_idx, tokens in enumerate(doc_tokens):
            for term, tf in Counter(tokens).items():
                self.postings[term].append((doc_idx, tf))
        n_docs = len(chunks)
        self.idf = {
            term: math.log(1.0 + (n_docs - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        for term in tokenize(query):
            if term not in self.postings:
                continue
            idf = self.idf[term]
            for doc_idx, tf in self.postings[term]:
                norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl)
                scores[doc_idx] += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return scores

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        scores = self.scores(query)
        k = max(0, min(k, len(self.chunks)))
        if k == 0:
            return []
        # Stable ordering: highest score first, ties broken by chunk position.
        order = np.lexsort((np.arange(len(scores)), -scores))[:k]
        return [
            RetrievedChunk(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order, start=1)
        ]


__all__ = ["BM25Retriever", "tokenize"]
