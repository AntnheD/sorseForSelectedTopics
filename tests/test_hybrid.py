"""Reciprocal rank fusion checked against a hand-computed example."""

from __future__ import annotations

import pytest

from docqa.retrieval.base import RetrievedChunk
from docqa.retrieval.hybrid import HybridRetriever
from tests.conftest import make_chunk


class FixedRetriever:
    def __init__(self, name, ranking):
        self.name = name
        self.ranking = ranking

    def search(self, query, k):
        return [RetrievedChunk(chunk=c, score=1.0, rank=i + 1) for i, c in enumerate(self.ranking[:k])]


def test_rrf_scores_and_order():
    a, b, c = (make_chunk(0, "a.md", "a"), make_chunk(0, "b.md", "b"), make_chunk(0, "c.md", "c"))
    lexical = FixedRetriever("bm25", [a, b, c])
    dense = FixedRetriever("dense", [b, c, a])
    hybrid = HybridRetriever(lexical, dense, c=60)
    results = hybrid.search("q", k=3)
    # a: 1/61 + 1/63 ; b: 1/62 + 1/61 ; c: 1/63 + 1/62
    expected = {"a.md": 1 / 61 + 1 / 63, "b.md": 1 / 62 + 1 / 61, "c.md": 1 / 63 + 1 / 62}
    assert [r.chunk.source for r in results] == ["b.md", "a.md", "c.md"]
    for r in results:
        assert r.score == pytest.approx(expected[r.chunk.source])
    assert [r.rank for r in results] == [1, 2, 3]


def test_chunk_found_by_one_retriever_only_still_appears():
    a, b = make_chunk(0, "a.md", "a"), make_chunk(0, "b.md", "b")
    hybrid = HybridRetriever(FixedRetriever("l", [a]), FixedRetriever("d", [b]), c=60)
    sources = [r.chunk.source for r in hybrid.search("q", k=2)]
    assert sorted(sources) == ["a.md", "b.md"]
