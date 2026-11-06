"""Choose the pre-generation abstention threshold on the VALIDATION split.

    python scripts/tune_abstention.py [--method bm25|dense|hybrid] [--split val]

For every question in the split (answerable and insufficient-evidence) the
script prints the top-1 retrieval score, then sweeps thresholds and reports
the abstention accuracy of the rule "abstain if top score < t". The chosen
threshold is written into the final configuration by hand (configs/final.json)
and justified in reports/experiments.md. The test split is never used here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docqa.config import Config  # noqa: E402
from docqa.corpus import load_chunks  # noqa: E402
from docqa.questions import by_split, load_questions  # noqa: E402
from docqa.retrieval import build_retriever  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--method", default=None, help="bm25 | dense | hybrid (default: config)")
    ap.add_argument("--split", default="val", choices=["dev", "val"], help="never 'test'")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    chunks = load_chunks(cfg.resolve(cfg.corpus.chunks_path))
    retriever = build_retriever(cfg, chunks, method=args.method)
    _, questions = load_questions(cfg.resolve(cfg.evaluation.questions_path))
    qs = by_split(questions, args.split)

    rows = []
    for q in qs:
        results = retriever.search(q.question, cfg.retrieval.top_k)
        top = results[0].score if results else 0.0
        rows.append((top, q.answerable, q.id, results[0].chunk.source if results else "-"))
    rows.sort()
    print(f"retriever={retriever.name} split={args.split} n={len(rows)}")
    print(f"{'top-1 score':>12}  {'answerable':>10}  id        top source")
    for top, ans, qid, src in rows:
        print(f"{top:>12.4f}  {str(ans):>10}  {qid:<9} {src}")

    scores = sorted({r[0] for r in rows})
    candidates = [0.0] + [(a + b) / 2 for a, b in zip(scores, scores[1:])] + [scores[-1] + 1e-6]
    print("\nthreshold sweep (abstain if top-1 score < t):")
    print(f"{'t':>10}  {'accuracy':>8}  {'false abstain':>13}  {'missed abstain':>14}")
    best = None
    for t in candidates:
        correct = sum(1 for top, ans, _, _ in rows if (top < t) != ans)
        false_abstain = sum(1 for top, ans, _, _ in rows if ans and top < t)
        missed = sum(1 for top, ans, _, _ in rows if not ans and top >= t)
        acc = correct / len(rows)
        print(f"{t:>10.4f}  {acc:>8.3f}  {false_abstain:>13}  {missed:>14}")
        # prefer the lowest false-abstention count, then accuracy
        key = (-acc, false_abstain)
        if best is None or key < best[0]:
            best = (key, t)
    print(f"\nbest threshold on {args.split}: t = {best[1]:.4f} (accuracy {-best[0][0]:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
