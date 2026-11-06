"""Automatic answer-evaluation proxies and the review template."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from docqa.evaluate_answers import (
    citation_precision,
    keyword_coverage,
    keyword_present,
    score_record,
    summarise,
    write_review_template,
)
from docqa.pipeline import ABSTAIN_MESSAGE, AnswerRecord
from docqa.questions import Question


def _question(**over) -> Question:
    base = dict(id="q", question="How?", category="direct", split="val", answerability="answerable",
                expected_answer_points=["None default"], answer_keywords=[["None"], ["default"]],
                relevant_sources=["tutorial/query-params.md#optional-parameters"])
    base.update(over)
    return Question(**base)


def _record(answer: str, abstained: bool = False, citations=None, sentences=None, passages=None) -> AnswerRecord:
    return AnswerRecord(
        question="How?", answer=answer, abstained=abstained, abstain_reason="x" if abstained else None,
        raw_answer=answer, citations=citations or [], sentences=sentences or [], passages=passages or [],
    )


def test_keyword_coverage_counts_groups_case_insensitively():
    assert keyword_coverage("Set the DEFAULT to none.", [["None"], ["default"]]) == 1.0
    assert keyword_coverage("Set it to None.", [["None"], ["default"]]) == 0.5
    assert keyword_coverage("whatever", [["xyz", "qrs"]]) == 0.0
    assert keyword_coverage("whatever", []) is None


def test_keyword_matching_respects_token_boundaries():
    assert keyword_present("use app.get() here", "ge") is False      # 'ge' is not the token 'get'
    assert keyword_present("declare ge=1 with Path", "ge") is True
    assert keyword_present("use {file_path:path} in the route", ":path") is True
    assert keyword_present("listens on 0.0.0.0 by default", "0.0.0.0") is True
    assert keyword_present("response_model takes priority", "response_model") is True


def test_citation_precision_uses_relevance_labels():
    q = _question()
    rec = _record("x", citations=[
        {"source": "tutorial/query-params.md", "section_anchor": "optional-parameters"},
        {"source": "deployment/docker.md", "section_anchor": "dockerfile"},
    ])
    assert citation_precision(rec, q, "source") == 0.5
    assert citation_precision(_record("x"), q) is None


def test_score_record_for_answerable_and_insufficient_questions():
    q = _question()
    rec = _record("Give it a default of None.",
                  citations=[{"source": "tutorial/query-params.md", "section_anchor": "optional-parameters"}],
                  sentences=[{"supported": True}, {"supported": False}],
                  passages=[{"source": "tutorial/query-params.md", "section_anchor": "optional-parameters"}])
    m = score_record(rec, q, "source")
    assert m["keyword_coverage"] == 1.0
    assert m["citation_precision"] == 1.0
    assert m["support_rate"] == 0.5
    assert m["abstention_correct"] is True and m["retrieval_hit"] is True

    ie = _question(answerability="insufficient_evidence", relevant_sources=[], answer_keywords=[])
    abstained = _record(ABSTAIN_MESSAGE, abstained=True)
    m2 = score_record(abstained, ie, "source")
    assert m2["abstention_correct"] is True
    assert m2["keyword_coverage"] is None and m2["retrieval_hit"] is None
    assert score_record(_record("made up"), ie, "source")["abstention_correct"] is False


def test_summarise_and_review_template(tmp_path: Path):
    rows = [
        {"id": "a", "answerability": "answerable", "expected_answer_points": ["p"], "relevant_sources": [], "seconds": 1.0,
         "metrics": {"keyword_coverage": 1.0, "citation_precision": 1.0, "support_rate": 1.0,
                     "abstained": False, "abstention_correct": True, "retrieval_hit": True},
         "prediction": {"question": "q", "answer": "ans", "citations": [{"source": "s.md", "section_anchor": "x"}]}},
        {"id": "b", "answerability": "answerable", "expected_answer_points": ["p"], "relevant_sources": [], "seconds": 1.0,
         "metrics": {"keyword_coverage": None, "citation_precision": None, "support_rate": None,
                     "abstained": True, "abstention_correct": False, "retrieval_hit": False},
         "prediction": {"question": "q", "answer": ABSTAIN_MESSAGE, "citations": []}},
        {"id": "c", "answerability": "insufficient_evidence", "expected_answer_points": [], "relevant_sources": [], "seconds": 1.0,
         "metrics": {"keyword_coverage": None, "citation_precision": None, "support_rate": None,
                     "abstained": True, "abstention_correct": True, "retrieval_hit": None},
         "prediction": {"question": "q", "answer": ABSTAIN_MESSAGE, "citations": []}},
    ]
    s = summarise(rows)
    assert s["n_answerable"] == 2 and s["n_insufficient_evidence"] == 1
    assert s["answer_rate_on_answerable"] == 0.5
    assert s["false_abstention_rate"] == 0.5
    assert s["correct_abstention_rate"] == 1.0
    assert s["abstention_accuracy"] == pytest.approx(2 / 3, abs=1e-4)
    assert s["keyword_coverage_full_match_rate_over_answerable"] == 0.5
    path = tmp_path / "review.csv"
    write_review_template(rows, path)
    with open(path, newline="", encoding="utf-8") as fh:
        reader = list(csv.DictReader(fh))
    assert [r["id"] for r in reader] == ["a", "b", "c"]
    assert reader[0]["citations"] == "s.md#x" and reader[0]["correct"] == ""
