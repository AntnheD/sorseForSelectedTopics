"""Chunk data model and the validated JSONL loader.

A *chunk* is a passage of documentation with enough metadata to cite it:
the source file, the section heading it came from, and a stable id.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .errors import CorpusNotFoundError, MalformedDataError

REQUIRED_FIELDS = (
    "id",
    "source",
    "section",
    "section_anchor",
    "heading_path",
    "chunk_index",
    "text",
    "n_words",
    "doc_version",
)


@dataclass(frozen=True)
class Chunk:
    id: str                 # "<source>::<chunk_index>", stable across runs
    source: str             # path relative to the docs root, e.g. "tutorial/query-params.md"
    section: str            # heading text of the section the chunk belongs to
    section_anchor: str     # heading anchor, e.g. "optional-parameters"
    heading_path: str       # "Doc title > Section > Subsection"
    chunk_index: int        # position of the chunk within its document
    text: str               # the passage text shown to the user and the model
    n_words: int
    doc_version: str        # documentation commit/date recorded at preprocessing time

    def index_text(self) -> str:
        """Text used for indexing: heading context plus the passage."""
        return f"{self.heading_path}\n{self.text}"

    def relevance_key(self, level: str = "source") -> str:
        if level == "source":
            return self.source
        if level == "section":
            return f"{self.source}#{self.section_anchor}"
        raise ValueError(f"Unknown relevance level '{level}'")

    def to_dict(self) -> dict:
        return asdict(self)


def write_chunks(chunks: Iterable[Chunk], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        for chunk in chunks:
            fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False, sort_keys=True))
            fh.write("\n")
            n += 1
    return n


def load_chunks(path: str | Path) -> list[Chunk]:
    """Load chunks from JSONL, failing loudly on malformed or inconsistent data."""
    path = Path(path)
    if not path.exists():
        raise CorpusNotFoundError(
            f"Chunk file not found: {path}. Run `python -m docqa preprocess` first."
        )
    chunks: list[Chunk] = []
    seen_ids: set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MalformedDataError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
            if not isinstance(record, dict):
                raise MalformedDataError(f"{path}:{line_no}: expected a JSON object")
            missing = [f for f in REQUIRED_FIELDS if f not in record]
            if missing:
                raise MalformedDataError(f"{path}:{line_no}: missing fields {missing}")
            if not record["source"]:
                raise MalformedDataError(f"{path}:{line_no}: empty 'source' metadata")
            if record["id"] in seen_ids:
                raise MalformedDataError(f"{path}:{line_no}: duplicate chunk id '{record['id']}'")
            seen_ids.add(record["id"])
            chunks.append(Chunk(**{f: record[f] for f in REQUIRED_FIELDS}))
    if not chunks:
        raise CorpusNotFoundError(f"Chunk file {path} contains no chunks.")
    return chunks


def corpus_fingerprint(chunks: Iterable[Chunk]) -> str:
    """Short hash of ids and texts; changes whenever the corpus content changes."""
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk.id.encode("utf-8"))
        h.update(b"\0")
        h.update(chunk.index_text().encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def chunks_by_id(chunks: Iterable[Chunk]) -> dict[str, Chunk]:
    return {c.id: c for c in chunks}


__all__ = ["Chunk", "load_chunks", "write_chunks", "corpus_fingerprint", "chunks_by_id", "REQUIRED_FIELDS"]
