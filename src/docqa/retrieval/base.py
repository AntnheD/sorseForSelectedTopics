"""Shared retrieval interface.

Every retriever answers the same call, ``search(query, k)``, and returns the
same record type, so the lexical baseline and the dense Transformer retriever
can be swapped and evaluated identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..corpus import Chunk


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int  # 1-based position in the result list

    def to_dict(self, include_text: bool = True) -> dict:
        d = {
            "rank": self.rank,
            "score": round(float(self.score), 4),
            "chunk_id": self.chunk.id,
            "source": self.chunk.source,
            "section": self.chunk.section,
            "section_anchor": self.chunk.section_anchor,
        }
        if include_text:
            d["text"] = self.chunk.text
        return d


@runtime_checkable
class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int) -> list[RetrievedChunk]:
        ...


__all__ = ["RetrievedChunk", "Retriever"]
