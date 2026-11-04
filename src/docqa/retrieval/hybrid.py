"""Hybrid retriever: reciprocal rank fusion (RRF) of BM25 and dense rankings.

    rrf(chunk) = sum over retrievers r of 1 / (c + rank_r(chunk))     (c = 60)

A chunk that both retrievers rank highly gets the highest fused score; a
chunk found by only one retriever still gets credit from that ranking.
This is included only as a validation experiment (E5); it is adopted in the
final configuration only if it beats both single retrievers on validation.
"""

from __future__ import annotations

from .base import RetrievedChunk, Retriever


class HybridRetriever:
    name = "hybrid"

    def __init__(self, lexical: Retriever, dense: Retriever, c: int = 60, overfetch: int = 4):
        self.lexical = lexical
        self.dense = dense
        self.c = c
        self.overfetch = overfetch

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        n = max(k * self.overfetch, k)
        fused: dict[str, float] = {}
        chunks = {}
        for retriever in (self.lexical, self.dense):
            for item in retriever.search(query, n):
                chunks[item.chunk.id] = item.chunk
                fused[item.chunk.id] = fused.get(item.chunk.id, 0.0) + 1.0 / (self.c + item.rank)
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [RetrievedChunk(chunk=chunks[cid], score=score, rank=rank)
                for rank, (cid, score) in enumerate(ordered, start=1)]


__all__ = ["HybridRetriever"]
