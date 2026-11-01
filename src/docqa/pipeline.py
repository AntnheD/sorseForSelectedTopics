"""End-to-end pipeline: question -> retrieval -> generation -> grounded answer.

The returned ``AnswerRecord`` keeps everything needed to inspect a decision:
the retrieved passages with scores, the raw model output, per-sentence
support, the citations and the reason for any abstention.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .config import Config
from .corpus import Chunk, load_chunks
from .errors import EmptyQuestionError
from .generation.generator import Generator, build_generator
from .generation.grounding import (
    ground_answer,
    should_abstain_after,
    should_abstain_before,
    strip_citation_marks,
)
from .retrieval import Retriever, build_retriever
from .retrieval.base import RetrievedChunk

ABSTAIN_MESSAGE = "I could not find enough evidence in the FastAPI documentation to answer this question."


@dataclass
class AnswerRecord:
    question: str
    answer: str
    abstained: bool
    abstain_reason: Optional[str]
    raw_answer: str
    citations: list[dict] = field(default_factory=list)
    sentences: list[dict] = field(default_factory=list)
    passages: list[dict] = field(default_factory=list)
    retriever: str = ""
    generator: str = ""
    model: str = ""
    timings: dict = field(default_factory=dict)
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "raw_answer": self.raw_answer,
            "citations": self.citations,
            "sentences": self.sentences,
            "passages": self.passages,
            "retriever": self.retriever,
            "generator": self.generator,
            "model": self.model,
            "timings": self.timings,
            "settings": self.settings,
        }

    def format(self, debug: bool = False) -> str:
        lines = [f"Q: {self.question}", "", f"A: {self.answer}"]
        if self.abstained:
            lines.append(f"   (abstained: {self.abstain_reason})")
        if self.citations:
            lines.append("")
            lines.append("Sources:")
            for c in self.citations:
                lines.append(f"  [{c['ref']}] {c['source']} § {c['section']}  (support {c['support']:.2f})")
        elif self.passages:
            lines.append("")
            lines.append("Closest passages (not cited):")
            for p in self.passages[:3]:
                lines.append(f"  - {p['source']} § {p['section']}  (score {p['score']:.3f})")
        if debug:
            lines.append("")
            lines.append(f"Retriever: {self.retriever} | Generator: {self.generator} ({self.model})")
            lines.append(f"Timings (s): {', '.join(f'{k}={v:.2f}' for k, v in self.timings.items())}")
            lines.append(f"Raw model output: {self.raw_answer!r}")
            lines.append("Sentence support:")
            for s in self.sentences:
                flag = "ok " if s["supported"] else "UNSUPPORTED"
                lines.append(f"  [{flag}] {s['support']:.2f} <- {s['best_chunk_id']}: {s['sentence']}")
            lines.append("Retrieved passages:")
            for p in self.passages:
                lines.append(f"  #{p['rank']} {p['score']:.3f} {p['chunk_id']} ({p['heading_path']})")
                snippet = p["text"][:300].replace("\n", " ")
                lines.append(f"      {snippet}{'...' if len(p['text']) > 300 else ''}")
        return "\n".join(lines)


class DocQAPipeline:
    def __init__(self, cfg: Config, chunks: list[Chunk], retriever: Retriever, generator: Generator):
        self.cfg = cfg
        self.chunks = chunks
        self.retriever = retriever
        self.generator = generator

    @classmethod
    def from_config(cls, cfg: Config, chunks: Optional[list[Chunk]] = None,
                    retriever: Optional[Retriever] = None, generator: Optional[Generator] = None
                    ) -> "DocQAPipeline":
        chunks = chunks or load_chunks(cfg.resolve(cfg.corpus.chunks_path))
        retriever = retriever or build_retriever(cfg, chunks)
        generator = generator or build_generator(cfg)
        return cls(cfg, chunks, retriever, generator)

    # ------------------------------------------------------------------ steps
    def retrieve(self, question: str, k: Optional[int] = None) -> list[RetrievedChunk]:
        question = self._validate(question)
        return self.retriever.search(question, k or self.cfg.retrieval.top_k)

    def answer(self, question: str) -> AnswerRecord:
        question = self._validate(question)
        g = self.cfg.generation
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        results = self.retriever.search(question, self.cfg.retrieval.top_k)
        timings["retrieval"] = time.perf_counter() - t0
        passages = results[: g.passages_for_generation]
        base = dict(
            question=question,
            passages=[r.to_dict() | {"heading_path": r.chunk.heading_path} for r in results],
            retriever=self.retriever.name,
            generator=self.generator.name,
            model=self.generator.model_name,
            settings={
                "top_k": self.cfg.retrieval.top_k,
                "passages_for_generation": g.passages_for_generation,
                "min_retrieval_score": g.min_retrieval_score,
                "min_sentence_support": g.min_sentence_support,
                "min_supported_fraction": g.min_supported_fraction,
            },
        )

        reason = should_abstain_before(results, g.min_retrieval_score)
        if reason:
            timings["generation"] = 0.0
            return AnswerRecord(answer=ABSTAIN_MESSAGE, abstained=True, abstain_reason=reason,
                                raw_answer="", timings=timings, **base)

        t0 = time.perf_counter()
        raw = self.generator.generate(question, passages)
        timings["generation"] = time.perf_counter() - t0

        supports, citations = ground_answer(raw, passages, g.min_sentence_support)
        reason = should_abstain_after(raw, supports, g.min_supported_fraction)
        record = AnswerRecord(
            answer=ABSTAIN_MESSAGE if reason else strip_citation_marks(raw),
            abstained=bool(reason),
            abstain_reason=reason,
            raw_answer=raw,
            citations=[] if reason else [c.to_dict() for c in citations],
            sentences=[s.to_dict() for s in supports],
            timings=timings,
            **base,
        )
        return record

    @staticmethod
    def _validate(question: str) -> str:
        if question is None or not str(question).strip():
            raise EmptyQuestionError("The question is empty. Please type a question about FastAPI.")
        return " ".join(str(question).split())


__all__ = ["DocQAPipeline", "AnswerRecord", "ABSTAIN_MESSAGE"]
