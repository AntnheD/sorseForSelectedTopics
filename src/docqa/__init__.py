"""docqa: retrieval-augmented question answering over FastAPI documentation.

Package layout
--------------
config.py        Typed configuration loaded from configs/*.json with CLI overrides.
errors.py        Exception types with clear messages for the known failure modes.
corpus.py        Chunk data model and JSONL loader with validation.
preprocess.py    Markdown -> chunks.jsonl (deterministic, section-aware chunking).
retrieval/       BM25 lexical baseline and dense (sentence-transformers) retriever.
generation/      Prompt building, HF generator, extractive fallback, grounding.
pipeline.py      End-to-end question -> grounded answer with citations.
metrics.py       Recall@k, HitRate@k, MRR@k with explicit definitions.
evaluate_*.py    Retrieval and answer evaluation over labelled question splits.
cli.py           `python -m docqa <command>` entry point.
"""

__version__ = "0.1.0"
