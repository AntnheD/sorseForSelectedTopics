"""Evaluation question set: schema, loader and consistency checks.

File format (``data/evaluation/questions.json``)::

    {
      "version": "1.0",
      "corpus_commit": "<git commit of the FastAPI docs used for labelling>",
      "questions": [
        {
          "id": "q001",
          "question": "How do I make a query parameter optional?",
          "category": "direct",            # direct | multi-step | comparison | troubleshooting | insufficient-evidence
          "split": "dev",                  # dev | val | test
          "answerability": "answerable",   # answerable | insufficient_evidence
          "expected_answer_points": ["..."],            # facts a correct answer must contain
          "answer_keywords": [["None"], ["default"]],  # one group per fact; any alternative in a group counts
          "relevant_sources": ["tutorial/query-params.md#optional-parameters"],
          "notes": "where in the docs the label was verified"
        }
      ]
    }
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .corpus import Chunk
from .errors import MalformedDataError

SPLITS = ("dev", "val", "test")
ANSWERABILITY = ("answerable", "insufficient_evidence")
CATEGORIES = ("direct", "multi-step", "comparison", "troubleshooting", "insufficient-evidence")
_WORD_RE = re.compile(r"[a-z0-9_]+")


@dataclass
class Question:
    id: str
    question: str
    category: str
    split: str
    answerability: str
    expected_answer_points: list[str] = field(default_factory=list)
    answer_keywords: list[list[str]] = field(default_factory=list)
    relevant_sources: list[str] = field(default_factory=list)
    notes: str = ""

    @property
    def answerable(self) -> bool:
        return self.answerability == "answerable"

    def to_dict(self) -> dict:
        return asdict(self)


def load_questions(path: str | Path) -> tuple[dict, list[Question]]:
    path = Path(path)
    if not path.exists():
        raise MalformedDataError(f"Questions file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise MalformedDataError(f"{path}: invalid JSON ({exc.msg} at line {exc.lineno})") from exc
    if not isinstance(data, dict) or "questions" not in data:
        raise MalformedDataError(f"{path}: expected an object with a 'questions' list")
    meta = {k: v for k, v in data.items() if k != "questions"}
    questions: list[Question] = []
    seen: set[str] = set()
    for i, raw in enumerate(data["questions"]):
        where = f"{path}: question #{i}"
        for required in ("id", "question", "category", "split", "answerability"):
            if required not in raw:
                raise MalformedDataError(f"{where}: missing '{required}'")
        q = Question(**{k: raw[k] for k in Question.__dataclass_fields__ if k in raw})
        if q.id in seen:
            raise MalformedDataError(f"{where}: duplicate id '{q.id}'")
        seen.add(q.id)
        if q.split not in SPLITS:
            raise MalformedDataError(f"{where}: split must be one of {SPLITS}, got '{q.split}'")
        if q.answerability not in ANSWERABILITY:
            raise MalformedDataError(f"{where}: answerability must be one of {ANSWERABILITY}")
        if q.category not in CATEGORIES:
            raise MalformedDataError(f"{where}: category must be one of {CATEGORIES}")
        if not q.question.strip():
            raise MalformedDataError(f"{where}: empty question text")
        if q.answerable and not q.relevant_sources:
            raise MalformedDataError(f"{where}: answerable questions need at least one relevant source")
        if not q.answerable and q.relevant_sources:
            raise MalformedDataError(f"{where}: insufficient-evidence questions must not list relevant sources")
        if q.answerable and not q.answer_keywords:
            raise MalformedDataError(f"{where}: answerable questions need answer_keywords for automatic checks")
        questions.append(q)
    return meta, questions


def by_split(questions: Iterable[Question], split: str) -> list[Question]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    return [q for q in questions if q.split == split]


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def validate_against_corpus(questions: list[Question], chunks: list[Chunk],
                            near_duplicate_threshold: float = 0.8) -> list[str]:
    """Return human-readable issues: unknown sources/anchors, near-duplicate
    questions across splits. An empty list means the set is consistent."""
    sources = {c.source for c in chunks}
    sections = {f"{c.source}#{c.section_anchor}" for c in chunks}
    issues: list[str] = []
    for q in questions:
        for label in q.relevant_sources:
            src = label.split("#", 1)[0]
            if src not in sources:
                issues.append(f"{q.id}: relevant source '{src}' is not in the corpus")
            elif "#" in label and label not in sections:
                issues.append(f"{q.id}: section anchor '{label}' is not in the corpus")
    for i, a in enumerate(questions):
        ta = _tokens(a.question)
        for b in questions[i + 1:]:
            if a.split == b.split:
                continue
            tb = _tokens(b.question)
            if not ta or not tb:
                continue
            jaccard = len(ta & tb) / len(ta | tb)
            if jaccard >= near_duplicate_threshold:
                issues.append(f"{a.id} ({a.split}) and {b.id} ({b.split}) are near-duplicates (jaccard={jaccard:.2f})")
    return issues


def split_counts(questions: Iterable[Question]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {s: {"answerable": 0, "insufficient_evidence": 0} for s in SPLITS}
    for q in questions:
        out[q.split][q.answerability] += 1
    return out


__all__ = ["Question", "load_questions", "by_split", "validate_against_corpus", "split_counts",
           "SPLITS", "ANSWERABILITY", "CATEGORIES"]
