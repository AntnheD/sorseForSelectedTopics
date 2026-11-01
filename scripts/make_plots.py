"""Regenerate the figures under reports/figures from the saved JSON reports.

    python scripts/make_plots.py

Figures (each is skipped if its input is missing):
  chunk_lengths.png          histogram of chunk lengths (data/processed/corpus_report.json)
  retrieval_<split>.png      BM25 vs dense retrieval metrics (reports/retrieval/<split>_*.json)
  experiments_chunking.png   E2: MRR@5 by chunk size for both retrievers (reports/experiments/experiments.json)
  experiments_topk.png       E3: Recall@k and MRR@k vs k
  answers_<tag>.png          answer-quality proxies (reports/predictions/<tag>_summary.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COLORS = {"bm25": "#8a8f98", "dense": "#1a73e8"}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def chunk_lengths() -> None:
    report = load(ROOT / "data/processed/corpus_report.json")
    if not report:
        return
    hist = report["chunk_length_words"]["histogram_by_40_words"]
    labels, values = list(hist.keys()), list(hist.values())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.bar(labels, values, color=COLORS["dense"])
    ax.set_xlabel("chunk length (words)")
    ax.set_ylabel("chunks")
    ax.set_title(f"Chunk length distribution ({report['chunks_written']} chunks, "
                 f"{report['chunk_size_words']}/{report['chunk_overlap_words']} words)")
    fig.tight_layout()
    fig.savefig(FIG / "chunk_lengths.png", dpi=150)
    plt.close(fig)


def retrieval(split: str) -> None:
    results = {}
    for method in ("bm25", "dense"):
        matches = sorted((ROOT / "reports/retrieval").glob(f"{split}_{method}_k*_source.json"))
        if matches:
            results[method] = load(matches[-1])["summary"]
    if len(results) < 2:
        return
    k = results["bm25"]["k"]
    metrics = [f"recall@{k}", f"hit_rate@{k}", f"mrr@{k}", "top1_accuracy"]
    names = [f"Recall@{k}", f"HitRate@{k}", f"MRR@{k}", "Top-1"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.38
    for i, method in enumerate(("bm25", "dense")):
        vals = [results[method][m] for m in metrics]
        xs = [x + (i - 0.5) * width for x in range(len(metrics))]
        bars = ax.bar(xs, vals, width, label=method, color=COLORS[method])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
    ax.set_xticks(range(len(metrics)), names)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Retrieval on the {split} split (n = {results['bm25']['n_questions']}, source level)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / f"retrieval_{split}.png", dpi=150)
    plt.close(fig)


def experiments() -> None:
    data = load(ROOT / "reports/experiments/experiments.json")
    if not data:
        return
    rows = data["rows"]
    k = data["config"]["evaluation"]["k"]
    # E1 + E2: chunking
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for method in ("bm25", "dense"):
        pts = sorted((r["chunk_size_words"], r[f"source_mrr@{k}"], r[f"section_mrr@{k}"]) for r in rows
                     if r["retriever"] == method and r["experiment"] in ("E1_retriever", "E2_chunking"))
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=COLORS[method], label=f"{method} (source)")
            ax.plot([p[0] for p in pts], [p[2] for p in pts], "s--", color=COLORS[method], label=f"{method} (section)")
    ax.set_xlabel("chunk size (words)")
    ax.set_ylabel(f"MRR@{k}")
    ax.set_ylim(0, 1)
    ax.set_title(f"E2: chunk size vs MRR@{k} on {data['split']} (n = {data['n_questions']})")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "experiments_chunking.png", dpi=150)
    plt.close(fig)
    # E3: top-k
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for method in ("bm25", "dense"):
        pts = {}
        for r in rows:
            if r["retriever"] == method and r["experiment"] in ("E1_retriever", "E3_top_k") \
                    and r["chunk_size_words"] == data["config"]["corpus"]["chunk_size_words"]:
                kk = r["k"]
                pts[kk] = (r[f"source_recall@{kk}"], r[f"source_mrr@{kk}"])
        if pts:
            ks = sorted(pts)
            ax.plot(ks, [pts[x][0] for x in ks], "o-", color=COLORS[method], label=f"{method} Recall@k")
            ax.plot(ks, [pts[x][1] for x in ks], "s--", color=COLORS[method], label=f"{method} MRR@k")
    ax.set_xlabel("k")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"E3: top-k on {data['split']} (source level)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "experiments_topk.png", dpi=150)
    plt.close(fig)


def answers() -> None:
    for path in sorted((ROOT / "reports/predictions").glob("*_summary.json")):
        s = load(path)
        tag = path.name.replace("_summary.json", "")
        keys = [("keyword_coverage_mean_over_answered", "keyword\ncoverage"),
                ("citation_precision_mean", "citation\nprecision"),
                ("support_rate_mean", "support\nrate"),
                ("retrieval_hit_rate", "retrieval\nhit rate"),
                ("answer_rate_on_answerable", "answer rate\n(answerable)"),
                ("correct_abstention_rate", "abstained\n(insufficient)")]
        vals = [s.get(k) or 0.0 for k, _ in keys]
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        bars = ax.bar([lbl for _, lbl in keys], vals, color=COLORS["dense"])
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"Answer quality proxies: {tag} (n = {s['n_questions']}, {s['model']})", fontsize=10)
        fig.tight_layout()
        fig.savefig(FIG / f"answers_{tag}.png", dpi=150)
        plt.close(fig)


def main() -> int:
    chunk_lengths()
    for split in ("dev", "val", "test"):
        retrieval(split)
    experiments()
    answers()
    print(f"figures written to {FIG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
