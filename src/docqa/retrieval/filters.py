"""Post-retrieval filters.

``SourceDiverseRetriever`` wraps any retriever and limits how many chunks
from the same source file may appear in the top-k, so that one long page
cannot crowd out the others. It over-fetches from the inner retriever and
re-ranks the survivors.
"""

from __future__ import annotations

from .base import RetrievedChunk, Retriever


def limit_per_source(results: list[RetrievedChunk], max_per_source: int, k: int) -> list[RetrievedChunk]:
    if max_per_source < 1:
        raise ValueError("max_per_source must be >= 1")
    counts: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    for item in results:
        n = counts.get(item.chunk.source, 0)
        if n >= max_per_source:
            continue
        counts[item.chunk.source] = n + 1
        kept.append(RetrievedChunk(chunk=item.chunk, score=item.score, rank=len(kept) + 1))
        if len(kept) == k:
            break
    return kept


class SourceDiverseRetriever:
    def __init__(self, inner: Retriever, max_per_source: int, overfetch: int = 4):
        self.inner = inner
        self.max_per_source = max_per_source
        self.overfetch = overfetch
        self.name = f"{inner.name}+max{max_per_source}/source"

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        candidates = self.inner.search(query, k * self.overfetch)
        return limit_per_source(candidates, self.max_per_source, k)


__all__ = ["SourceDiverseRetriever", "limit_per_source"]
