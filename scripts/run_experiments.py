"""Controlled retrieval experiments on the VALIDATION split only.

Usage (from the repository root, inside the virtual environment):

    python scripts/run_experiments.py [--split val] [--out reports/experiments]

One factor changes at a time, starting from the configuration in
configs/default.json:

  E1  retriever          bm25 vs dense (default chunking, k = 5)
  E2  chunk size/overlap (120/30), (180/40), (260/60) for both retrievers
  E3  top-k              3, 5, 8 for both retrievers (default chunking)
  E4  max_per_source     none, 2, 1 for both retrievers (default chunking, k = 5)

Each row records the configuration, the metrics at source and section level,
index build time and mean query time. Results are written to
<out>/experiments.json and <out>/experiments.md. The test split is never used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docqa.config import Config  # noqa: E402
from docqa.corpus import load_chunks  # noqa: E402
from docqa.evaluate_retrieval import evaluate_retriever  # noqa: E402
from docqa.preprocess import preprocess_corpus  # noqa: E402
from docqa.questions import by_split, load_questions  # noqa: E402
from docqa.retrieval import build_retriever  # noqa: E402

CHUNKINGS = [(120, 30), (180, 40), (260, 60)]
TOP_KS = [3, 5, 8]
MAX_PER_SOURCE = [None, 2, 1]


def chunks_for(cfg: Config, size: int, overlap: int, cache: dict) -> list:
    key = (size, overlap)
    if key in cache:
        return cache[key]
    if (size, overlap) == (cfg.corpus.chunk_size_words, cfg.corpus.chunk_overlap_words):
        chunks = load_chunks(cfg.resolve(cfg.corpus.chunks_path))
    else:
        exp_cfg = cfg.copy()
        exp_cfg.corpus.chunk_size_words = size
        exp_cfg.corpus.chunk_overlap_words = overlap
        out_dir = cfg.resolve("data/processed/experiments")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"chunks_{size}_{overlap}.jsonl"
        preprocess_corpus(exp_cfg, chunks_path=path, report_path=out_dir / f"report_{size}_{overlap}.json")
        chunks = load_chunks(path)
    cache[key] = chunks
    return chunks


def run_one(cfg: Config, chunks: list, questions: list, *, method: str, k: int, max_per_source, label: str,
            size: int, overlap: int) -> dict:
    t0 = time.perf_counter()
    retriever = build_retriever(cfg, chunks, method=method, max_per_source=max_per_source or 0)
    build_seconds = time.perf_counter() - t0
    inner = getattr(retriever, "inner", retriever)
    row = {
        "experiment": label, "retriever": method, "chunk_size_words": size, "chunk_overlap_words": overlap,
        "k": k, "max_per_source": max_per_source, "n_chunks": len(chunks),
        "index_build_seconds": round(build_seconds, 2),
        "embedding_cached": bool(getattr(inner, "cache_path", None)) and getattr(inner, "index_seconds", 0.0) == 0.0,
    }
    for level in ("source", "section"):
        res = evaluate_retriever(retriever, questions, k, level)
        s = res["summary"]
        row[f"{level}_recall@{k}"] = s[f"recall@{k}"]
        row[f"{level}_hit_rate@{k}"] = s[f"hit_rate@{k}"]
        row[f"{level}_mrr@{k}"] = s[f"mrr@{k}"]
        row[f"{level}_top1"] = s["top1_accuracy"]
        row["mean_query_seconds"] = s["mean_query_seconds"]
        row["n_questions"] = s["n_questions"]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--split", default="val", choices=["dev", "val"], help="never 'test'")
    ap.add_argument("--out", default="reports/experiments")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    meta, questions = load_questions(cfg.resolve(cfg.evaluation.questions_path))
    qs = by_split(questions, args.split)
    cache: dict = {}
    rows: list[dict] = []
    d_size, d_over = cfg.corpus.chunk_size_words, cfg.corpus.chunk_overlap_words
    default_chunks = chunks_for(cfg, d_size, d_over, cache)

    for method in ("bm25", "dense"):  # E1 + E2
        for size, overlap in CHUNKINGS:
            chunks = chunks_for(cfg, size, overlap, cache)
            label = "E1_retriever" if (size, overlap) == (d_size, d_over) else "E2_chunking"
            rows.append(run_one(cfg, chunks, qs, method=method, k=cfg.evaluation.k, max_per_source=None,
                                label=label, size=size, overlap=overlap))
            print(json.dumps(rows[-1]), flush=True)
    for method in ("bm25", "dense"):  # E3
        for k in TOP_KS:
            if k == cfg.evaluation.k:
                continue
            rows.append(run_one(cfg, default_chunks, qs, method=method, k=k, max_per_source=None,
                                label="E3_top_k", size=d_size, overlap=d_over))
            print(json.dumps(rows[-1]), flush=True)
    for method in ("bm25", "dense"):  # E4
        for mps in MAX_PER_SOURCE:
            if mps is None:
                continue
            rows.append(run_one(cfg, default_chunks, qs, method=method, k=cfg.evaluation.k, max_per_source=mps,
                                label="E4_max_per_source", size=d_size, overlap=d_over))
            print(json.dumps(rows[-1]), flush=True)
    for k in TOP_KS:  # E5: reciprocal-rank fusion of both retrievers
        rows.append(run_one(cfg, default_chunks, qs, method="hybrid", k=k, max_per_source=None,
                            label="E5_hybrid_rrf", size=d_size, overlap=d_over))
        print(json.dumps(rows[-1]), flush=True)

    out_dir = cfg.resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"split": args.split, "n_questions": rows[0]["n_questions"] if rows else 0,
               "questions_version": meta.get("version"), "config": cfg.to_dict(), "rows": rows}
    (out_dir / "experiments.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "experiments.md").write_text(to_markdown(payload), encoding="utf-8")
    print(f"wrote {out_dir / 'experiments.json'} and experiments.md")
    return 0


def to_markdown(payload: dict) -> str:
    lines = [f"# Validation experiments (split = {payload['split']}, n = {payload['n_questions']} answerable questions)", ""]
    lines.append("Metrics at source level (relevance = correct file) and section level (relevance = correct file#section).")
    lines.append("")
    header = ("| exp | retriever | chunk | k | max/src | chunks | Recall@k src | MRR@k src | Top-1 src | "
              "Recall@k sec | MRR@k sec | Top-1 sec | build s | query s |")
    lines.append(header)
    lines.append("|" + "---|" * 14)
    for r in payload["rows"]:
        k = r["k"]
        lines.append(
            f"| {r['experiment']} | {r['retriever']} | {r['chunk_size_words']}/{r['chunk_overlap_words']} | {k} | "
            f"{r['max_per_source'] or '-'} | {r['n_chunks']} | {r[f'source_recall@{k}']:.3f} | {r[f'source_mrr@{k}']:.3f} | "
            f"{r['source_top1']:.3f} | {r[f'section_recall@{k}']:.3f} | {r[f'section_mrr@{k}']:.3f} | {r['section_top1']:.3f} | "
            f"{r['index_build_seconds']:.1f} | {r['mean_query_seconds']:.3f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
