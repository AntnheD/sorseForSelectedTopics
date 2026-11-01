"""Grounding: citation mapping and abstention decisions.

The model is never trusted to cite correctly. Instead every answer sentence
is compared with the retrieved passages by content-word overlap:

    support(sentence, passage) = |words(sentence) ∩ words(passage)| / |words(sentence)|

(stop words removed). A sentence is *supported* when its best passage reaches
``min_sentence_support``; that passage becomes its citation. Citations can
therefore only ever point at passages that were actually retrieved, which a
unit test enforces.

Two abstention gates:
* before generation, when the best retrieval score is below a threshold;
* after generation, when the model says it does not know, or when too few
  sentences are supported by the passages.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Optional

from ..retrieval.base import RetrievedChunk

WORD_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = frozenset(
    """a an the and or but of to in on for with is are was were be been being it its this that these those
    as by at from into than then so such you your yours we our they them their he she his her i me my
    can could will would should may might must shall do does did doing done have has had having not no nor
    if else when where which what who whom whose how why there here also just only very more most some any
    each other same own over under about above below between through during before after again further
    up down out off all both few many much s t re ve ll d m use used using uses e g i.e eg ie etc""".split()
)
CITATION_MARK_RE = re.compile(r"\[(\d{1,2})\]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[^a-z])|\n+")
ABSTAIN_RE = re.compile(
    r"\b(i don'?t know|i do not know|don'?t have enough|not enough (?:information|evidence|context|details?)|"
    r"insufficient (?:information|evidence|context)|cannot (?:be )?(?:answer|determin)|"
    r"(?:does|do) not (?:mention|contain|provide|say|specify|cover|include)|no information|"
    r"not (?:mentioned|provided|specified|covered|included|stated) in)\b",
    re.IGNORECASE,
)


def content_tokens(text: str) -> set[str]:
    return {t for t in WORD_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1}


CODE_BLOCK_RE = re.compile(r"```.*?(```|$)", re.DOTALL)


def split_sentences(text: str, min_content_tokens: int = 2) -> list[str]:
    """Prose sentences of an answer. Fenced code blocks are not claims and are
    dropped; so are lead-ins with fewer than ``min_content_tokens`` content
    words (e.g. "For example:"), which would otherwise count as unsupported."""
    text = CODE_BLOCK_RE.sub(" ", text or "").strip()
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text)]
    kept = [p for p in parts if p and len(content_tokens(p)) >= min_content_tokens]
    if not kept and text:
        # A very short answer such as "307" or "response_model_exclude_unset" is still a claim.
        return [text]
    return kept


def support_score(sentence: str, passage_text: str) -> float:
    s = content_tokens(CITATION_MARK_RE.sub("", sentence))
    if not s:
        return 0.0
    return len(s & content_tokens(passage_text)) / len(s)


@dataclass
class SentenceSupport:
    sentence: str
    best_chunk_id: Optional[str]
    support: float
    supported: bool

    def to_dict(self) -> dict:
        d = asdict(self)
        d["support"] = round(self.support, 3)
        return d


@dataclass
class Citation:
    ref: int                # 1-based index into the passages handed to the generator
    chunk_id: str
    source: str
    section: str
    section_anchor: str
    support: float          # best sentence support that led to this citation

    def to_dict(self) -> dict:
        d = asdict(self)
        d["support"] = round(self.support, 3)
        return d


def ground_answer(answer: str, passages: list[RetrievedChunk], min_sentence_support: float
                  ) -> tuple[list[SentenceSupport], list[Citation]]:
    """Map each answer sentence to its best supporting passage."""
    supports: list[SentenceSupport] = []
    citations: dict[int, Citation] = {}

    def cite(ref: int, support: float) -> None:
        p = passages[ref - 1]
        existing = citations.get(ref)
        if existing is None or support > existing.support:
            citations[ref] = Citation(ref=ref, chunk_id=p.chunk.id, source=p.chunk.source,
                                      section=p.chunk.section, section_anchor=p.chunk.section_anchor,
                                      support=support)

    for sentence in split_sentences(answer):
        best_ref, best_score = None, 0.0
        for ref, p in enumerate(passages, start=1):
            score = support_score(sentence, p.chunk.text)
            if score > best_score:
                best_ref, best_score = ref, score
        supported = best_ref is not None and best_score >= min_sentence_support
        supports.append(SentenceSupport(
            sentence=sentence,
            best_chunk_id=passages[best_ref - 1].chunk.id if best_ref else None,
            support=best_score,
            supported=supported,
        ))
        if supported:
            cite(best_ref, best_score)
        # Explicit "[n]" markers written by the model are honoured only when n is a real passage.
        for mark in CITATION_MARK_RE.findall(sentence):
            ref = int(mark)
            if 1 <= ref <= len(passages):
                cite(ref, support_score(sentence, passages[ref - 1].chunk.text))
    return supports, [citations[r] for r in sorted(citations)]


def should_abstain_before(results: list[RetrievedChunk], min_retrieval_score: float) -> Optional[str]:
    if not results:
        return "no passages retrieved"
    top = results[0].score
    if top < min_retrieval_score:
        return f"top retrieval score {top:.3f} is below the threshold {min_retrieval_score:.3f}"
    return None


def should_abstain_after(answer: str, supports: list[SentenceSupport], min_supported_fraction: float
                         ) -> Optional[str]:
    if not answer or not answer.strip():
        return "the model produced an empty answer"
    if ABSTAIN_RE.search(answer):
        return "the model reported that the excerpts do not contain the answer"
    if not supports:
        return "the answer contains no sentences"
    n_supported = sum(1 for s in supports if s.supported)
    fraction = n_supported / len(supports)
    if fraction < min_supported_fraction:
        return (f"only {n_supported}/{len(supports)} answer sentences are supported by the retrieved passages "
                f"(minimum fraction {min_supported_fraction:.2f})")
    return None


def strip_citation_marks(text: str) -> str:
    return re.sub(r"\s*\[\d{1,2}\]", "", text).strip()


__all__ = [
    "Citation",
    "SentenceSupport",
    "content_tokens",
    "split_sentences",
    "support_score",
    "ground_answer",
    "should_abstain_before",
    "should_abstain_after",
    "strip_citation_marks",
    "ABSTAIN_RE",
]
