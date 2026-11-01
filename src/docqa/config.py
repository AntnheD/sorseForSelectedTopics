"""Typed configuration.

Defaults live in the dataclasses below; ``configs/default.json`` overrides
them, and ``--set section.key=value`` on the command line overrides both.
Every relative path in the configuration is resolved against the repository
root, so commands behave the same from any working directory.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CorpusConfig:
    documents_dir: str = "data/documents/fastapi"
    docs_subdir: str = "docs"
    docs_src_subdir: str = "docs_src"
    chunks_path: str = "data/processed/chunks.jsonl"
    report_path: str = "data/processed/corpus_report.json"
    chunk_size_words: int = 180
    chunk_overlap_words: int = 40
    exclude_files: list[str] = field(default_factory=lambda: ["_llm-test.md"])
    doc_version: str = "unknown"


@dataclass
class RetrievalConfig:
    method: str = "dense"  # "bm25" | "dense"
    top_k: int = 5
    max_per_source: Optional[int] = None  # None = no source-diversity filter
    dense_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    artifacts_dir: str = "artifacts"
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass
class GenerationConfig:
    generator: str = "hf"  # "hf" | "extractive"
    model: str = "google/flan-t5-base"
    dtype: str = "float32"  # "float32" | "bfloat16" (halves memory for decoder models on CPU)
    max_new_tokens: int = 128
    passages_for_generation: int = 3
    include_code_in_prompt: bool = True  # False replaces fenced code blocks with a marker in the prompt only
    min_retrieval_score: float = 0.30  # abstain before generation if top score is below this
    min_sentence_support: float = 0.35  # a sentence is "supported" if this share of its content words is in a passage
    min_supported_fraction: float = 0.5  # abstain after generation if fewer sentences are supported


@dataclass
class EvaluationConfig:
    questions_path: str = "data/evaluation/questions.json"
    k: int = 5
    relevance_level: str = "source"  # "source" | "section"
    reports_dir: str = "reports"
    predictions_dir: str = "reports/predictions"


@dataclass
class Config:
    corpus: CorpusConfig = field(default_factory=CorpusConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    root: Path = field(default_factory=lambda: REPO_ROOT)
    source_file: Optional[str] = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Optional[str | Path] = None, overrides: Optional[dict[str, Any] | list[str]] = None,
             root: Optional[Path] = None) -> "Config":
        """Build a config from defaults, then a JSON file, then overrides.

        ``overrides`` may be a dict ``{"retrieval.top_k": 8}`` or a list of
        ``"retrieval.top_k=8"`` strings as typed on the command line.
        """
        cfg = cls(root=Path(root) if root else REPO_ROOT)
        if path is not None:
            file_path = Path(path)
            if not file_path.is_absolute():
                file_path = cfg.root / file_path
            with open(file_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            cfg._apply_nested(data)
            cfg.source_file = str(file_path)
        if overrides:
            if isinstance(overrides, list):
                overrides = parse_overrides(overrides)
            for dotted, value in overrides.items():
                cfg.set(dotted, value)
        return cfg

    def _apply_nested(self, data: dict[str, Any]) -> None:
        for section_name, section_values in data.items():
            if section_name not in {"corpus", "retrieval", "generation", "evaluation"}:
                raise ValueError(f"Unknown config section '{section_name}'")
            section = getattr(self, section_name)
            valid = {f.name for f in fields(section)}
            for key, value in section_values.items():
                if key not in valid:
                    raise ValueError(f"Unknown config key '{section_name}.{key}'")
                setattr(section, key, value)

    def set(self, dotted: str, value: Any) -> None:
        section_name, _, key = dotted.partition(".")
        if not key:
            raise ValueError(f"Override '{dotted}' must look like section.key")
        self._apply_nested({section_name: {key: value}})

    # ------------------------------------------------------------------ helpers
    def resolve(self, relative: str | Path) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else self.root / p

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in ("corpus", "retrieval", "generation", "evaluation"):
            out[name] = asdict(getattr(self, name))
        return out

    def copy(self) -> "Config":
        return copy.deepcopy(self)

    def dump(self, path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
            fh.write("\n")


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """Parse ``["retrieval.top_k=8", "generation.generator=extractive"]``.

    Values are parsed as JSON when possible (numbers, booleans, null, lists),
    otherwise kept as strings.
    """
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like section.key=value")
        dotted, raw = item.split("=", 1)
        try:
            value: Any = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        out[dotted.strip()] = value
    return out


__all__ = [
    "Config",
    "CorpusConfig",
    "RetrievalConfig",
    "GenerationConfig",
    "EvaluationConfig",
    "REPO_ROOT",
    "parse_overrides",
]
