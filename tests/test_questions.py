"""Question-set schema checks, and consistency of the real set with the real corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docqa.config import Config
from docqa.corpus import load_chunks
from docqa.errors import MalformedDataError
from docqa.questions import by_split, load_questions, split_counts, validate_against_corpus
from tests.conftest import make_chunk


def _q(**over):
    base = dict(id="q1", question="How?", category="direct", split="dev", answerability="answerable",
                expected_answer_points=["x"], answer_keywords=[["x"]], relevant_sources=["a.md#s"])
    base.update(over)
    return base


def _write(tmp_path: Path, questions: list[dict]) -> Path:
    p = tmp_path / "q.json"
    p.write_text(json.dumps({"version": "t", "questions": questions}), encoding="utf-8")
    return p


def test_schema_violations_are_reported(tmp_path: Path):
    with pytest.raises(MalformedDataError, match="duplicate id"):
        load_questions(_write(tmp_path, [_q(), _q()]))
    with pytest.raises(MalformedDataError, match="split must be"):
        load_questions(_write(tmp_path, [_q(split="train")]))
    with pytest.raises(MalformedDataError, match="relevant source"):
        load_questions(_write(tmp_path, [_q(relevant_sources=[])]))
    with pytest.raises(MalformedDataError, match="must not list"):
        load_questions(_write(tmp_path, [_q(answerability="insufficient_evidence")]))
    with pytest.raises(MalformedDataError, match="answer_keywords"):
        load_questions(_write(tmp_path, [_q(answer_keywords=[])]))


def test_validation_against_corpus_flags_unknown_sources_and_duplicates(tmp_path: Path):
    chunks = [make_chunk(0, "a.md", "text", anchor="s")]
    meta, qs = load_questions(_write(tmp_path, [
        _q(id="q1", question="How do I declare a query parameter?"),
        _q(id="q2", split="test", question="How do I declare a query parameter"),
        _q(id="q3", split="val", question="Other", relevant_sources=["a.md#missing", "b.md"]),
    ]))
    issues = validate_against_corpus(qs, chunks)
    assert any("near-duplicates" in i for i in issues)
    assert any("a.md#missing" in i for i in issues)
    assert any("'b.md' is not in the corpus" in i for i in issues)
    assert split_counts(qs)["dev"]["answerable"] == 1
    assert [q.id for q in by_split(qs, "test")] == ["q2"]


def test_real_question_set_is_consistent_with_real_corpus():
    cfg = Config.load("configs/default.json")
    qpath = cfg.resolve(cfg.evaluation.questions_path)
    cpath = cfg.resolve(cfg.corpus.chunks_path)
    if not qpath.exists() or not cpath.exists():
        pytest.skip("real question set or processed corpus not present")
    meta, questions = load_questions(qpath)
    chunks = load_chunks(cpath)
    issues = validate_against_corpus(questions, chunks)
    assert issues == [], "\n".join(issues)
    counts = split_counts(questions)
    assert counts["test"]["answerable"] >= 10, "the test split must be big enough to report on"
    assert counts["val"]["answerable"] >= 5
