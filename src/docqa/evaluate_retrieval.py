"""Retrieval evaluation shared by the lexical baseline and the dense retriever.

Both retrievers see the same chunks, the same questions, the same labels and
the same metric definitions (see ``metrics.py``). Only answerable questions
count; the number excluded is reported.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

from .config import Config
from .corpus import Chunk, load_chunks
from .metrics import evaluate_rankings, matches, relevant_labels
from .questions import Question, by_split, load_questions
from .retrieval import Retriever, build_retriever


def evaluate_retriever(retriever: Retriever, questions: list[Question], k: int, level: str) -> dict:
    answerable = [q for q in questions if q.answerable]
    ranked: dict[str, list[str]] = {}
    labels: dict[str, list[str]] = {}
    details: dict[str, dict] = {}
    query_seconds: list[float] = []
    for q in answerable:
        t0 = time.perf_counter()
        results = retriever.search(q.question, k)
        query_seconds.append(time.perf_counter() - t0)
        keys = [r.chunk.relevance_key(level) for r in results]
        ranked[q.id] = keys
        labels[q.id] = q.relevant_sources
        rel = relevant_labels(q.relevant_sources, level)
        details[q.id] = {
            "question": q.question,
            "category": q.category,
            "relevant": rel,
            "top": [
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk.id,
                    "source": r.chunk.source,
                    "section_anchor": r.chunk.section_anchor,
                    "score": round(float(r.score), 4),
                    "relevant": any(matches(r.chunk.relevance_key(level), lab, level) for lab in rel),
                }
                for r in results
            ],
        }
    evaluation = evaluate_rankings(ranked, labels, k, level)
    for qid, m in evaluation["per_question"].items():
        details[qid].update(m)
    summary = evaluation["summary"]
    summary["retriever"] = retriever.name
    summary["n_excluded_insufficient_evidence"] = len(questions) - len(answerable)
    summary["mean_query_seconds"] = round(sum(query_seconds) / len(query_seconds), 4) if query_seconds else 0.0
    summary["sources_with_multiple_hits"] = sum(
        1 for d in details.values() if len({t["source"] for t in d["top"]}) < len(d["top"])
    )
    return {"summary": summary, "per_question": details}


def run(cfg: Config, split: str, methods: list[str], *, k: Optional[int] = None,
        level: Optional[str] = None, chunks: Optional[list[Chunk]] = None,
        out_dir: Optional[Path] = None, write: bool = True) -> dict[str, dict]:
    k = k or cfg.evaluation.k
    level = level or cfg.evaluation.relevance_level
    chunks = chunks or load_chunks(cfg.resolve(cfg.corpus.chunks_path))
    meta, questions = load_questions(cfg.resolve(cfg.evaluation.questions_path))
    subset = by_split(questions, split)
    if not subset:
        raise ValueError(f"No questions in split '{split}'")
    out_dir = out_dir or cfg.resolve(cfg.evaluation.reports_dir) / "retrieval"
    results: dict[str, dict] = {}
    for method in methods:
        retriever = build_retriever(cfg, chunks, method=method)
        result = evaluate_retriever(retriever, subset, k, level)
        result["summary"]["split"] = split
        result["summary"]["questions_version"] = meta.get("version")
        result["summary"]["chunk_size_words"] = cfg.corpus.chunk_size_words
        result["summary"]["chunk_overlap_words"] = cfg.corpus.chunk_overlap_words
        result["summary"]["max_per_source"] = cfg.retrieval.max_per_source
        results[method] = result
        if write:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{split}_{method}_k{k}_{level}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2, ensure_ascii=False)
            result["summary"]["report_path"] = str(path)
    return results


def format_table(results: dict[str, dict]) -> str:
    rows = []
    header = None
    for method, result in results.items():
        s = result["summary"]
        k = s["k"]
        if header is None:
            header = f"{'retriever':<22} {'n':>3} {f'Recall@{k}':>9} {f'HitRate@{k}':>10} {f'MRR@{k}':>7} {'Top-1':>6} {'s/query':>8}"
            rows.append(header)
            rows.append("-" * len(header))
        rows.append(
            f"{s['retriever']:<22} {s['n_questions']:>3} {s[f'recall@{k}']:>9.3f} {s[f'hit_rate@{k}']:>10.3f} "
            f"{s[f'mrr@{k}']:>7.3f} {s['top1_accuracy']:>6.3f} {s['mean_query_seconds']:>8.3f}"
        )
    return "\n".join(rows)


__all__ = ["evaluate_retriever", "run", "format_table"]
