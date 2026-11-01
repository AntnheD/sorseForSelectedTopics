"""Retrieval metrics verified against hand-calculated values."""

from __future__ import annotations

import pytest

from docqa.metrics import (
    evaluate_rankings,
    first_relevant_rank,
    hit_rate_at_k,
    matches,
    mrr_at_k,
    recall_at_k,
)


def test_matching_rules_at_source_and_section_level():
    assert matches("a.md", "a.md#x", "source")              # anchor ignored at source level
    assert matches("a.md#x", "a.md#x", "section")
    assert not matches("a.md#y", "a.md#x", "section")
    assert matches("a.md#y", "a.md", "section")             # label without anchor = any section
    assert not matches("b.md", "a.md", "source")


def test_hand_calculated_example():
    # Ranked sources for one question; relevant = {b.md, d.md}
    keys = ["a.md", "b.md", "a.md", "c.md", "d.md", "e.md"]
    labels = ["b.md", "d.md#sec"]
    assert first_relevant_rank(keys, labels, "source") == 2
    assert hit_rate_at_k(keys, labels, "source", 1) == 0.0
    assert hit_rate_at_k(keys, labels, "source", 5) == 1.0
    assert recall_at_k(keys, labels, "source", 3) == pytest.approx(0.5)   # only b.md within top 3
    assert recall_at_k(keys, labels, "source", 5) == pytest.approx(1.0)   # b.md and d.md
    assert mrr_at_k(keys, labels, "source", 5) == pytest.approx(0.5)      # first relevant at rank 2
    assert mrr_at_k(keys, labels, "source", 1) == 0.0                     # rank 2 is outside k=1


def test_no_relevant_result_gives_zero():
    assert first_relevant_rank(["x.md"], ["y.md"], "source") is None
    assert mrr_at_k(["x.md"], ["y.md"], "source", 5) == 0.0
    assert recall_at_k(["x.md"], ["y.md"], "source", 5) == 0.0


def test_aggregate_means():
    ranked = {"q1": ["a.md", "b.md"], "q2": ["c.md", "a.md"], "q3": ["z.md"]}
    labels = {"q1": ["a.md"], "q2": ["a.md"], "q3": ["a.md"]}
    out = evaluate_rankings(ranked, labels, k=2, level="source")
    s = out["summary"]
    assert s["n_questions"] == 3
    tol = 1e-4  # summary values are rounded to 4 decimals
    assert s["hit_rate@2"] == pytest.approx(2 / 3, abs=tol)
    assert s["recall@2"] == pytest.approx(2 / 3, abs=tol)
    assert s["mrr@2"] == pytest.approx((1 + 0.5 + 0) / 3, abs=tol)
    assert s["top1_accuracy"] == pytest.approx(1 / 3, abs=tol)
    assert out["per_question"]["q2"]["first_relevant_rank"] == 2
