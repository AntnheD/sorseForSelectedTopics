"""Citation mapping and abstention logic."""

from __future__ import annotations

import pytest

from docqa.generation.grounding import (
    ground_answer,
    should_abstain_after,
    should_abstain_before,
    split_sentences,
    strip_citation_marks,
    support_score,
)
from docqa.retrieval.base import RetrievedChunk
from tests.conftest import make_chunk


def _passages(chunks):
    return [RetrievedChunk(chunk=c, score=1.0 - 0.1 * i, rank=i + 1) for i, c in enumerate(chunks)]


def test_sentence_splitting_handles_code_dots_and_newlines():
    text = "Use app.get() to declare it. Then call it twice.\nSecond line here."
    assert split_sentences(text) == ["Use app.get() to declare it.", "Then call it twice.", "Second line here."]
    assert split_sentences("") == []


def test_code_blocks_and_trivial_lead_ins_are_not_claims():
    text = "Declare a default value. For example:\n```python\nq: str | None = None\napp = FastAPI()\n```\nReturns the item."
    assert split_sentences(text) == ["Declare a default value.", "Returns the item."]
    # an unterminated code block (truncated generation) is dropped too
    assert split_sentences("Use Query for validation.\n```python\nx = 1\ny = 2") == ["Use Query for validation."]
    # a very short answer is still one claim, not "no sentences"
    assert split_sentences("307") == ["307"]
    assert split_sentences("response_model_exclude_unset") == ["response_model_exclude_unset"]


def test_support_score_is_content_word_overlap():
    passage = "Query parameters with a default value of None are optional."
    assert support_score("Optional query parameters default to None.", passage) == pytest.approx(1.0)
    assert support_score("Path parameters are required.", passage) == pytest.approx(1 / 3)
    assert support_score("the and of", passage) == 0.0  # only stop words


def test_citations_point_only_at_retrieved_passages(tiny_chunks):
    passages = _passages(tiny_chunks[:3])
    answer = ("Query parameters are optional when you give them a default value such as None. "
              "Unicorns are purple animals.")
    supports, citations = ground_answer(answer, passages, min_sentence_support=0.35)
    assert [s.supported for s in supports] == [True, False]
    assert len(citations) == 1
    assert citations[0].chunk_id == "tutorial/query-params.md::000"
    assert citations[0].source == "tutorial/query-params.md"
    allowed = {p.chunk.id for p in passages}
    assert all(c.chunk_id in allowed for c in citations)


def test_explicit_markers_are_validated(tiny_chunks):
    passages = _passages(tiny_chunks[:2])
    supports, citations = ground_answer("Path operations exist [2]. Nonsense here [9].", passages, 0.9)
    refs = [c.ref for c in citations]
    assert refs == [2]          # [9] does not exist and is ignored
    assert strip_citation_marks("Text [1] more [12].") == "Text more."


def test_abstain_before_generation():
    assert should_abstain_before([], 0.3) == "no passages retrieved"
    low = _passages([make_chunk(0, "a.md", "x")])
    low[0].score = 0.1
    assert "below the threshold" in should_abstain_before(low, 0.3)
    low[0].score = 0.5
    assert should_abstain_before(low, 0.3) is None


def test_abstain_after_generation(tiny_chunks):
    passages = _passages(tiny_chunks[:1])
    supports, _ = ground_answer("Unrelated sentence about cats. Another about dogs.", passages, 0.35)
    reason = should_abstain_after("Unrelated sentence about cats. Another about dogs.", supports, 0.5)
    assert "0/2 answer sentences are supported" in reason
    assert "do not contain" in should_abstain_after("The excerpts do not contain this.", supports, 0.0)
    assert should_abstain_after("I don't know.", supports, 0.0) is not None
    assert should_abstain_after("   ", supports, 0.0) == "the model produced an empty answer"
    good, _ = ground_answer("Query parameters are optional with a default value such as None.", passages, 0.35)
    assert should_abstain_after("Query parameters are optional with a default value such as None.", good, 0.5) is None
