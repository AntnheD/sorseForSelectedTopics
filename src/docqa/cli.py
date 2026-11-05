"""Command-line interface: ``python -m docqa <command> [options]``.

Commands
--------
preprocess        Markdown docs -> data/processed/chunks.jsonl + corpus report
index             Build (and cache) the dense embedding index
ask               Answer one question, optionally with full debug output
eval-retrieval    Compare retrievers on a question split
eval-answers      Run the full pipeline on a split and write predictions
check-questions   Validate the question set against the corpus
info              Print the resolved configuration and corpus status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .errors import DocQAError


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default="configs/default.json",
                        help="JSON config file, relative to the repository root (default: configs/default.json)")
    parser.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="override a config value, e.g. --set retrieval.top_k=8 (repeatable)")


def _load_config(args: argparse.Namespace) -> Config:
    path = args.config
    if path and not Path(Config.load().resolve(path)).exists():
        raise DocQAError(f"Config file not found: {path}")
    return Config.load(path, overrides=args.set)


def cmd_preprocess(args: argparse.Namespace) -> int:
    from .preprocess import preprocess_corpus

    cfg = _load_config(args)
    report = preprocess_corpus(cfg)
    print(f"files included: {report['files_included']}  skipped: {report['files_skipped']}")
    print(f"chunks written: {report['chunks_written']}  duplicates: {report['duplicate_chunks']}")
    print(f"code includes resolved/unresolved: {report['resolved_code_includes']}/{report['unresolved_code_includes']}")
    lengths = report["chunk_length_words"]
    print(f"chunk words min/median/max: {lengths['min']}/{lengths['median']}/{lengths['max']}")
    print(f"chunks -> {report['chunks_path']}")
    print(f"report -> {cfg.resolve(cfg.corpus.report_path)}")
    return 0


def cmd_index(args: argparse.Namespace) -> int:
    from .corpus import load_chunks
    from .retrieval import build_retriever

    cfg = _load_config(args)
    chunks = load_chunks(cfg.resolve(cfg.corpus.chunks_path))
    retriever = build_retriever(cfg, chunks, method="dense")
    inner = getattr(retriever, "inner", retriever)
    print(f"model: {inner.model_name}  dim: {inner.dim}  chunks: {len(chunks)}")
    print(f"cache: {inner.cache_path}  (built in {inner.index_seconds:.1f}s, 0 = loaded from cache)")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .pipeline import DocQAPipeline

    cfg = _load_config(args)
    pipeline = DocQAPipeline.from_config(cfg)
    record = pipeline.answer(args.question)
    if args.json:
        print(json.dumps(record.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(record.format(debug=args.debug))
    return 0


def cmd_eval_retrieval(args: argparse.Namespace) -> int:
    from .evaluate_retrieval import format_table, run

    cfg = _load_config(args)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    results = run(cfg, args.split, methods, k=args.k, level=args.level)
    print(format_table(results))
    for method, result in results.items():
        print(f"{method}: report -> {result['summary'].get('report_path')}")
    return 0


def cmd_eval_answers(args: argparse.Namespace) -> int:
    from .evaluate_answers import format_summary, run

    cfg = _load_config(args)
    out = run(cfg, args.split, tag=args.tag)
    print(format_summary(out["summary"]))
    print(f"predictions -> {out['summary']['predictions_path']}")
    print(f"manual review template -> {out['summary']['manual_review_template']}")
    return 0


def cmd_check_questions(args: argparse.Namespace) -> int:
    from .corpus import load_chunks
    from .questions import load_questions, split_counts, validate_against_corpus

    cfg = _load_config(args)
    meta, questions = load_questions(cfg.resolve(cfg.evaluation.questions_path))
    chunks = load_chunks(cfg.resolve(cfg.corpus.chunks_path))
    issues = validate_against_corpus(questions, chunks)
    print(f"questions: {len(questions)}  version: {meta.get('version')}  corpus_commit: {meta.get('corpus_commit')}")
    for split, counts in split_counts(questions).items():
        print(f"  {split:<5} answerable={counts['answerable']:>3}  insufficient_evidence={counts['insufficient_evidence']:>2}")
    if issues:
        print("ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    print("no issues found")
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    cfg = _load_config(args)
    print(f"docqa {__version__}")
    print(f"repository root: {cfg.root}")
    print(f"config file: {cfg.source_file}")
    print(json.dumps(cfg.to_dict(), indent=2))
    chunks_path = cfg.resolve(cfg.corpus.chunks_path)
    print(f"chunks file: {chunks_path} ({'present' if chunks_path.exists() else 'MISSING - run preprocess'})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docqa", description="Grounded Q&A over FastAPI documentation.")
    parser.add_argument("--version", action="version", version=f"docqa {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="build chunks.jsonl from the Markdown docs")
    _add_common(p)
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("index", help="build/cache the dense embedding index")
    _add_common(p)
    p.set_defaults(func=cmd_index)

    p = sub.add_parser("ask", help="answer one question")
    _add_common(p)
    p.add_argument("question")
    p.add_argument("--debug", action="store_true", help="show passages, raw output and sentence support")
    p.add_argument("--json", action="store_true", help="print the full record as JSON")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("eval-retrieval", help="evaluate retrievers on a split")
    _add_common(p)
    p.add_argument("--split", default="val", choices=["dev", "val", "test"])
    p.add_argument("--methods", default="bm25,dense", help="comma-separated: bm25,dense")
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--level", default=None, choices=["source", "section"])
    p.set_defaults(func=cmd_eval_retrieval)

    p = sub.add_parser("eval-answers", help="run the full pipeline on a split")
    _add_common(p)
    p.add_argument("--split", default="val", choices=["dev", "val", "test"])
    p.add_argument("--tag", default=None, help="file name tag for the prediction files")
    p.set_defaults(func=cmd_eval_answers)

    p = sub.add_parser("check-questions", help="validate the question set against the corpus")
    _add_common(p)
    p.set_defaults(func=cmd_check_questions)

    p = sub.add_parser("info", help="show the resolved configuration")
    _add_common(p)
    p.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except DocQAError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
