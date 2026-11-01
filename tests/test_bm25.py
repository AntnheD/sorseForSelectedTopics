"""BM25 baseline checked against a hand-computed example."""

from __future__ import annotations

import math

import pytest

from docqa.retrieval.bm25 import BM25Retriever, tokenize
from tests.conftest import make_chunk


def test_tokenizer_keeps_identifiers_and_lowercases():
    assert tokenize("Use response_model=Item in @app.get()") == ["use", "response_model", "item", "in", "app", "get"]


def test_bm25_scores_match_hand_calculation():
    # Three "documents" (heading_path + text are indexed; keep heading tokens distinct and query-free).
    chunks = [
        make_chunk(0, "a.md", "apple apple banana", section="S1", anchor="s1"),
        make_chunk(0, "b.md", "banana cherry", section="S2", anchor="s2"),
        make_chunk(0, "c.md", "cherry cherry cherry date", section="S3", anchor="s3"),
    ]
    r = BM25Retriever(chunks, k1=1.5, b=0.75)
    # index_text() = "<source> > <section>\n<text>"; tokens per doc:
    # a: [a, md, s1, apple, apple, banana] -> 6 ; b: [b, md, s2, banana, cherry] -> 5 ; c: [c, md, s3, cherry x3, date] -> 7
    assert list(r.doc_len) == [6, 5, 7]
    avgdl = 6.0
    n = 3
    df_apple, df_banana = 1, 2
    idf_apple = math.log(1 + (n - df_apple + 0.5) / (df_apple + 0.5))
    idf_banana = math.log(1 + (n - df_banana + 0.5) / (df_banana + 0.5))

    def term(tf, dl, idf):
        return idf * tf * 2.5 / (tf + 1.5 * (1 - 0.75 + 0.75 * dl / avgdl))

    expected_a = term(2, 6, idf_apple) + term(1, 6, idf_banana)
    expected_b = term(1, 5, idf_banana)
    scores = r.scores("apple banana")
    assert scores[0] == pytest.approx(expected_a)
    assert scores[1] == pytest.approx(expected_b)
    assert scores[2] == 0.0

    results = r.search("apple banana", k=2)
    assert [x.chunk.source for x in results] == ["a.md", "b.md"]
    assert [x.rank for x in results] == [1, 2]


def test_unknown_terms_and_k_bounds():
    chunks = [make_chunk(0, "a.md", "alpha beta")]
    r = BM25Retriever(chunks)
    assert r.search("zzz", k=5)[0].score == 0.0
    assert r.search("alpha", k=0) == []
    assert len(r.search("alpha", k=50)) == 1
