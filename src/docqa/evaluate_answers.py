"""Answer evaluation: automatic proxies plus a manual-review template.

Automatic signals (all defined here, all inspectable in the predictions file):

* ``keyword_coverage``   share of the question's ``answer_keywords`` groups
                         whose alternative appears in the answer (case-insensitive
                         substring). A proxy for correctness/completeness.
* ``citation_precision`` share of cited sources that are labelled relevant.
* ``support_rate``       share of answer sentences supported by a passage.
* ``abstention_correct`` answerable -> answered; insufficient evidence -> abstained.
* ``retrieval_hit``      a relevant source was among the retrieved passages.

These proxies are *not* ground truth for correctness. The manual review CSV
written next to the predictions is where a person records correctness,
completeness, evidence support and citation validity per question.
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Optional

from .config import Config
from .metrics import matches
from .pipeline import AnswerRecord, DocQAPipeline
from .questions import Question, by_split, load_questions


_WORDISH = re.compile(r"^[a-z0-9_]+$")


def keyword_present(answer: str, keyword: str) -> bool:
    """Case-insensitive match. Alphanumeric keywords must match whole tokens
    (so ``ge`` does not match ``get``); other keywords (``:path``, ``==``,
    ``0.0.0.0``) are matched as substrings."""
    kw = keyword.lower()
    text = answer.lower()
    if _WORDISH.match(kw):
        return re.search(rf"(?<![a-z0-9_]){re.escape(kw)}(?![a-z0-9_])", text) is not None
    return kw in text


def keyword_coverage(answer: str, answer_keywords: list[list[str]]) -> Optional[float]:
    if not answer_keywords:
        return None
    hits = sum(1 for group in answer_keywords if any(keyword_present(answer, alt) for alt in group))
    return hits / len(answer_keywords)


def citation_precision(record: AnswerRecord, question: Question, level: str = "source") -> Optional[float]:
    if not record.citations:
        return None
    ok = 0
    for c in record.citations:
        key = c["source"] if level == "source" else f"{c['source']}#{c['section_anchor']}"
        if any(matches(key, lab, level) for lab in question.relevant_sources):
            ok += 1
    return ok / len(record.citations)


def score_record(record: AnswerRecord, question: Question, level: str) -> dict:
    supported = [s for s in record.sentences if s["supported"]]
    retrieved_sources = [p["source"] if level == "source" else f"{p['source']}#{p['section_anchor']}"
                         for p in record.passages]
    retrieval_hit = any(matches(k, lab, level) for k in retrieved_sources for lab in question.relevant_sources)
    return {
        "keyword_coverage": None if record.abstained or not question.answerable
        else keyword_coverage(record.answer, question.answer_keywords),
        "citation_precision": citation_precision(record, question, level) if question.answerable else None,
        "support_rate": (len(supported) / len(record.sentences)) if record.sentences else None,
        "abstained": record.abstained,
        "abstention_correct": (not record.abstained) if question.answerable else record.abstained,
        "retrieval_hit": retrieval_hit if question.answerable else None,
    }


def summarise(rows: list[dict]) -> dict:
    def mean(values: list) -> Optional[float]:
        vals = [v for v in values if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    answerable = [r for r in rows if r["answerability"] == "answerable"]
    insufficient = [r for r in rows if r["answerability"] != "answerable"]
    answered = [r for r in answerable if not r["metrics"]["abstained"]]
    return {
        "n_questions": len(rows),
        "n_answerable": len(answerable),
        "n_insufficient_evidence": len(insufficient),
        "answer_rate_on_answerable": round(len(answered) / len(answerable), 4) if answerable else None,
        "false_abstention_rate": round(1 - len(answered) / len(answerable), 4) if answerable else None,
        "correct_abstention_rate": mean([r["metrics"]["abstained"] for r in insufficient]),
        "abstention_accuracy": mean([r["metrics"]["abstention_correct"] for r in rows]),
        "keyword_coverage_mean_over_answered": mean([r["metrics"]["keyword_coverage"] for r in answered]),
        "keyword_coverage_full_match_rate_over_answerable": round(
            sum(1 for r in answerable if (r["metrics"]["keyword_coverage"] or 0.0) >= 0.999) / len(answerable), 4
        ) if answerable else None,
        "citation_precision_mean": mean([r["metrics"]["citation_precision"] for r in answerable]),
        "support_rate_mean": mean([r["metrics"]["support_rate"] for r in answered]),
        "retrieval_hit_rate": mean([r["metrics"]["retrieval_hit"] for r in answerable]),
        "mean_seconds_per_question": mean([r["seconds"] for r in rows]),
    }


def run(cfg: Config, split: str, *, pipeline: Optional[DocQAPipeline] = None,
        out_dir: Optional[Path] = None, tag: Optional[str] = None, write: bool = True) -> dict:
    level = cfg.evaluation.relevance_level
    meta, questions = load_questions(cfg.resolve(cfg.evaluation.questions_path))
    subset = by_split(questions, split)
    if not subset:
        raise ValueError(f"No questions in split '{split}'")
    pipeline = pipeline or DocQAPipeline.from_config(cfg)
    rows: list[dict] = []
    for q in subset:
        t0 = time.perf_counter()
        record = pipeline.answer(q.question)
        seconds = time.perf_counter() - t0
        rows.append({
            "id": q.id,
            "category": q.category,
            "answerability": q.answerability,
            "expected_answer_points": q.expected_answer_points,
            "relevant_sources": q.relevant_sources,
            "seconds": round(seconds, 3),
            "metrics": score_record(record, q, level),
            "prediction": record.to_dict(),
        })
    summary = summarise(rows)
    summary.update({
        "split": split,
        "retriever": pipeline.retriever.name,
        "generator": pipeline.generator.name,
        "model": pipeline.generator.model_name,
        "relevance_level": level,
        "questions_version": meta.get("version"),
        "settings": cfg.to_dict(),
    })
    out = {"summary": summary, "rows": rows}
    if write:
        out_dir = out_dir or cfg.resolve(cfg.evaluation.predictions_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = tag or f"{split}_{pipeline.retriever.name.replace('/', '-')}_{pipeline.generator.name}"
        pred_path = out_dir / f"{tag}.jsonl"
        with open(pred_path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary_path = out_dir / f"{tag}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        review_path = out_dir / f"{tag}_manual_review.csv"
        write_review_template(rows, review_path)
        summary["predictions_path"] = str(pred_path)
        summary["summary_path"] = str(summary_path)
        summary["manual_review_template"] = str(review_path)
    return out


REVIEW_COLUMNS = [
    "id", "answerability", "question", "expected_answer_points", "answer", "citations",
    "correct", "complete", "evidence_supported", "citations_valid", "abstention_ok", "error_type", "notes",
]


def write_review_template(rows: list[dict], path: Path) -> None:
    """CSV for a person to fill in: correct/complete/supported/citations_valid as 1, 0.5 or 0."""
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        for row in rows:
            pred = row["prediction"]
            writer.writerow({
                "id": row["id"],
                "answerability": row["answerability"],
                "question": pred["question"],
                "expected_answer_points": " | ".join(row["expected_answer_points"]),
                "answer": pred["answer"],
                "citations": "; ".join(f"{c['source']}#{c['section_anchor']}" for c in pred["citations"]),
                "correct": "", "complete": "", "evidence_supported": "", "citations_valid": "",
                "abstention_ok": "", "error_type": "", "notes": "",
            })


def format_summary(summary: dict) -> str:
    keys = [
        "n_questions", "n_answerable", "n_insufficient_evidence", "answer_rate_on_answerable",
        "false_abstention_rate", "correct_abstention_rate", "abstention_accuracy",
        "keyword_coverage_mean_over_answered", "keyword_coverage_full_match_rate_over_answerable",
        "citation_precision_mean", "support_rate_mean", "retrieval_hit_rate", "mean_seconds_per_question",
    ]
    width = max(len(k) for k in keys)
    return "\n".join(f"{k:<{width}}  {summary.get(k)}" for k in keys)


__all__ = ["run", "score_record", "summarise", "keyword_coverage", "citation_precision",
           "write_review_template", "format_summary", "REVIEW_COLUMNS"]
