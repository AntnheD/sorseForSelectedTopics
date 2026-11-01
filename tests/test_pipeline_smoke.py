"""End-to-end smoke test with no model download: BM25 + extractive generator."""

from __future__ import annotations

import pytest

from docqa.config import Config
from docqa.errors import EmptyQuestionError
from docqa.generation.generator import ExtractiveGenerator
from docqa.pipeline import ABSTAIN_MESSAGE, DocQAPipeline
from docqa.retrieval import build_retriever


@pytest.fixture
def pipeline(tiny_chunks) -> DocQAPipeline:
    cfg = Config.load(overrides={
        "retrieval.method": "bm25", "retrieval.top_k": 3,
        "generation.generator": "extractive", "generation.min_retrieval_score": 0.5,
        "generation.min_sentence_support": 0.35, "generation.min_supported_fraction": 0.5,
    })
    retriever = build_retriever(cfg, tiny_chunks)
    return DocQAPipeline(cfg, tiny_chunks, retriever, ExtractiveGenerator())


def test_answer_has_valid_citations(pipeline, tiny_chunks):
    record = pipeline.answer("How do I make a query parameter optional?")
    assert not record.abstained
    assert record.citations, record.to_dict()
    corpus_ids = {c.id for c in tiny_chunks}
    retrieved_ids = {p["chunk_id"] for p in record.passages}
    for c in record.citations:
        assert c["chunk_id"] in retrieved_ids <= corpus_ids
        assert c["source"] == "tutorial/query-params.md"
    assert record.timings["retrieval"] >= 0 and record.timings["generation"] >= 0
    assert "None" in record.answer


def test_abstains_when_retrieval_is_weak(pipeline):
    record = pipeline.answer("What is the capital of Ethiopia?")
    assert record.abstained
    assert record.answer == ABSTAIN_MESSAGE
    assert record.citations == []
    assert record.abstain_reason


def test_empty_question_is_rejected(pipeline):
    with pytest.raises(EmptyQuestionError):
        pipeline.answer("   ")


def test_record_serialises_and_formats(pipeline):
    record = pipeline.answer("How are path parameters declared?")
    d = record.to_dict()
    assert set(d) >= {"question", "answer", "abstained", "citations", "passages", "sentences", "timings"}
    text = record.format(debug=True)
    assert "Retrieved passages:" in text and "Sentence support:" in text
