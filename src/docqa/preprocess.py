"""Turn FastAPI Markdown documentation into retrieval chunks.

Pipeline for one document
-------------------------
1. Normalise line endings and strip a YAML frontmatter block if present.
2. Resolve ``{* ../../docs_src/x.py *}`` code includes into fenced code
   blocks (the FastAPI docs keep examples in separate files).
3. Rewrite MkDocs admonitions (``/// tip``) and tabs into plain prose labels,
   drop links' URLs, images and a few HTML tags.
4. Split the document into sections at Markdown headings (ignoring ``#``
   lines inside code fences). Headings carry ``{ #anchor }`` ids.
5. Chunk each section into windows of ``chunk_size_words`` words with
   ``chunk_overlap_words`` of overlap. A trailing window shorter than the
   overlap is merged into the previous chunk so no text is lost and no
   near-duplicate chunk is produced.

Everything is deterministic: the same input and settings produce the same
``chunks.jsonl`` byte for byte.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import Config
from .corpus import Chunk, write_chunks
from .errors import CorpusNotFoundError

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*(?:\{\s*#([^}\s]+)\s*\})?\s*$")
INCLUDE_RE = re.compile(r"\{\*\s*(\S+?)(?:\s+[^*}]*)?\s*\*\}")
OLD_INCLUDE_RE = re.compile(r"\{!>?\s*(\S+?)\s*!\}")
ADMONITION_OPEN_RE = re.compile(r"^(/{3,4})\s*(\w+)(?:\s*\|\s*(.*?))?\s*$")
ADMONITION_CLOSE_RE = re.compile(r"^/{3,4}\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
ABBR_RE = re.compile(r"<abbr[^>]*>(.*?)</abbr>", re.DOTALL)
ANCHOR_TAG_RE = re.compile(r"<a\b[^>]*>(.*?)</a>", re.DOTALL)
DROP_TAG_RE = re.compile(
    r"</?(?:img|br|hr|div|span|details|summary|small|b|i|em|strong|p|iframe|video|source|"
    r"table|thead|tbody|tr|th|td|sup|sub|center|font)\b[^>]*>",
    re.IGNORECASE,
)
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
ATTR_LIST_RE = re.compile(r"\{[^{}]*(?:internal-link|external-link|target=_blank)[^{}]*\}")
MULTI_BLANK_RE = re.compile(r"\n{3,}")
NON_SLUG_RE = re.compile(r"[^a-z0-9]+")


# ---------------------------------------------------------------------------
# Document-level helpers (each is small and individually testable)
# ---------------------------------------------------------------------------

def normalise_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if text.startswith("﻿"):
        text = text[1:]
    return text


def strip_frontmatter(text: str) -> str:
    """Remove a leading YAML frontmatter block (``---`` ... ``---``)."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---", 4)
    if end == -1:
        return text
    rest = text[end + 4:]
    if rest and not rest.startswith("\n"):
        return text  # not a frontmatter terminator, e.g. "---foo"
    return rest.lstrip("\n")


def slugify(heading: str) -> str:
    slug = NON_SLUG_RE.sub("-", heading.lower()).strip("-")
    return slug or "section"


def resolve_includes(text: str, docs_src_dir: Optional[Path]) -> tuple[str, int, int]:
    """Replace include directives with the referenced code as a fenced block.

    Returns the new text, the number of resolved includes and the number of
    unresolved ones (left as a short note so the reader knows code was there).
    """
    resolved = 0
    unresolved = 0

    def load(ref: str) -> Optional[str]:
        if docs_src_dir is None:
            return None
        # Directives are relative to docs/en/, e.g. "../../docs_src/x.py".
        parts = [p for p in ref.replace("\\", "/").split("/") if p not in ("..", ".", "")]
        if not parts or parts[0] != docs_src_dir.name:
            return None
        candidate = docs_src_dir.joinpath(*parts[1:])
        if not candidate.is_file():
            return None
        return normalise_newlines(candidate.read_text(encoding="utf-8")).strip("\n")

    def replace(match: re.Match) -> str:
        nonlocal resolved, unresolved
        ref = match.group(1)
        code = load(ref)
        if code is None:
            unresolved += 1
            return f"(code example: {ref})"
        resolved += 1
        lang = "python" if ref.endswith(".py") else ""
        return f"```{lang}\n{code}\n```"

    text = INCLUDE_RE.sub(replace, text)
    text = OLD_INCLUDE_RE.sub(replace, text)
    return text, resolved, unresolved


def clean_markup(line: str) -> str:
    """Rewrite one prose line: drop URLs, images, HTML tags and attribute lists."""
    line = HTML_COMMENT_RE.sub("", line)
    line = IMAGE_RE.sub("", line)
    line = ABBR_RE.sub(r"\1", line)
    line = ANCHOR_TAG_RE.sub(r"\1", line)
    line = DROP_TAG_RE.sub("", line)
    line = LINK_RE.sub(r"\1", line)
    line = ATTR_LIST_RE.sub("", line)
    return line.rstrip()


def rewrite_admonition(line: str) -> Optional[str]:
    """Turn ``/// tip | Title`` into ``Tip (Title):`` and ``///`` into ``""``."""
    if ADMONITION_CLOSE_RE.match(line):
        return ""
    m = ADMONITION_OPEN_RE.match(line)
    if not m:
        return None
    kind, title = m.group(2), m.group(3)
    if kind.lower() == "tab":
        return f"{title.strip()}:" if title else ""
    label = kind.capitalize()
    return f"{label} ({title.strip()}):" if title else f"{label}:"


@dataclass
class Section:
    heading: str
    anchor: str
    heading_path: str
    lines: list[str] = field(default_factory=list)

    def text(self) -> str:
        body = "\n".join(self.lines)
        body = MULTI_BLANK_RE.sub("\n\n", body)
        return body.strip()


def split_sections(text: str, doc_title_fallback: str) -> list[Section]:
    """Split Markdown into heading-delimited sections, code-fence aware."""
    sections: list[Section] = []
    stack: list[tuple[int, str]] = []  # (level, heading text)
    current = Section(heading=doc_title_fallback, anchor=slugify(doc_title_fallback),
                      heading_path=doc_title_fallback)
    in_fence = False
    fence_marker = ""
    for raw_line in text.split("\n"):
        fence = FENCE_RE.match(raw_line)
        if fence and (not in_fence or raw_line.strip().startswith(fence_marker)):
            if in_fence:
                in_fence = False
            else:
                in_fence = True
                fence_marker = fence.group(1)
            current.lines.append(raw_line.rstrip())
            continue
        if in_fence:
            current.lines.append(raw_line.rstrip())
            continue

        heading = HEADING_RE.match(raw_line)
        if heading:
            level = len(heading.group(1))
            title = clean_markup(heading.group(2)).strip()
            anchor = heading.group(3) or slugify(title)
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(current)
            current = Section(heading=title, anchor=anchor,
                              heading_path=" > ".join(t for _, t in stack))
            continue

        rewritten = rewrite_admonition(raw_line)
        if rewritten is not None:
            current.lines.append(rewritten)
            continue
        current.lines.append(clean_markup(raw_line))
    sections.append(current)
    return [s for s in sections if s.text()]


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into word windows, keeping the original whitespace inside each chunk."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and smaller than the chunk size")
    text = text.strip()
    if not text:
        return []
    parts = re.split(r"(\s+)", text)  # [word, space, word, space, ..., word]
    n_words = (len(parts) + 1) // 2

    def window(a: int, b: int) -> str:  # words a .. b-1 inclusive, original spacing
        return "".join(parts[2 * a: 2 * (b - 1) + 1]).strip()

    if n_words <= size:
        return [window(0, n_words)]

    stride = size - overlap
    starts = list(range(0, n_words - overlap, stride))
    chunks: list[str] = []
    for i, start in enumerate(starts):
        end = min(start + size, n_words)
        is_last = i == len(starts) - 1
        if is_last:
            end = n_words
        new_words = end - (starts[i - 1] + size) if i > 0 else end
        if is_last and i > 0 and new_words < overlap:
            # Trailing window would add fewer than `overlap` new words: merge it
            # into the previous chunk instead of emitting a near-duplicate.
            chunks[-1] = window(starts[i - 1], n_words)
            break
        chunks.append(window(start, end))
    return chunks


def process_document(text: str, source: str, *, chunk_size: int, chunk_overlap: int,
                     doc_version: str, docs_src_dir: Optional[Path] = None) -> tuple[list[Chunk], dict]:
    """Convert one Markdown document to chunks. Returns (chunks, stats)."""
    text = normalise_newlines(text)
    text = strip_frontmatter(text)
    if not text.strip():
        return [], {"resolved_includes": 0, "unresolved_includes": 0, "sections": 0}
    text, resolved, unresolved = resolve_includes(text, docs_src_dir)
    fallback_title = Path(source).stem.replace("-", " ").replace("_", " ").title()
    sections = split_sections(text, fallback_title)

    chunks: list[Chunk] = []
    index = 0
    for section in sections:
        for piece in chunk_words(section.text(), chunk_size, chunk_overlap):
            chunks.append(Chunk(
                id=f"{source}::{index:03d}",
                source=source,
                section=section.heading,
                section_anchor=section.anchor,
                heading_path=section.heading_path,
                chunk_index=index,
                text=piece,
                n_words=len(piece.split()),
                doc_version=doc_version,
            ))
            index += 1
    stats = {"resolved_includes": resolved, "unresolved_includes": unresolved, "sections": len(sections)}
    return chunks, stats


# ---------------------------------------------------------------------------
# Corpus-level driver
# ---------------------------------------------------------------------------

def iter_source_files(docs_dir: Path, exclude_files: list[str]) -> tuple[list[Path], list[dict]]:
    """All ``*.md`` files under docs_dir, sorted, minus explicit exclusions."""
    if not docs_dir.is_dir():
        raise CorpusNotFoundError(
            f"Documentation directory not found: {docs_dir}. Run scripts/fetch_docs.sh first."
        )
    included: list[Path] = []
    skipped: list[dict] = []
    excluded = set(exclude_files)
    for path in sorted(docs_dir.rglob("*.md")):
        rel = path.relative_to(docs_dir).as_posix()
        if path.name in excluded or rel in excluded:
            skipped.append({"source": rel, "reason": "listed in corpus.exclude_files"})
            continue
        included.append(path)
    return included, skipped


def length_summary(lengths: list[int]) -> dict:
    if not lengths:
        return {"count": 0}
    ordered = sorted(lengths)
    q = statistics.quantiles(ordered, n=4) if len(ordered) >= 2 else [ordered[0]] * 3
    buckets = Counter()
    for n in lengths:
        buckets[f"{(n // 40) * 40}-{(n // 40) * 40 + 39}"] += 1
    return {
        "count": len(lengths),
        "min": ordered[0],
        "p25": round(q[0], 1),
        "median": statistics.median(ordered),
        "p75": round(q[2], 1),
        "max": ordered[-1],
        "mean": round(statistics.fmean(ordered), 1),
        "histogram_by_40_words": dict(sorted(buckets.items(), key=lambda kv: int(kv[0].split("-")[0]))),
    }


def preprocess_corpus(cfg: Config, *, chunks_path: Optional[Path] = None,
                      report_path: Optional[Path] = None) -> dict:
    """Run preprocessing for the whole corpus and write chunks + report."""
    c = cfg.corpus
    documents_dir = cfg.resolve(c.documents_dir)
    docs_dir = documents_dir / c.docs_subdir
    docs_src_dir = documents_dir / c.docs_src_subdir
    if not docs_src_dir.is_dir():
        docs_src_dir = None
    chunks_path = chunks_path or cfg.resolve(c.chunks_path)
    report_path = report_path or cfg.resolve(c.report_path)

    files, skipped = iter_source_files(docs_dir, c.exclude_files)
    all_chunks: list[Chunk] = []
    per_file: dict[str, int] = {}
    empty_files: list[str] = []
    resolved = unresolved = 0
    for path in files:
        source = path.relative_to(docs_dir).as_posix()
        text = path.read_text(encoding="utf-8")
        chunks, stats = process_document(
            text, source, chunk_size=c.chunk_size_words, chunk_overlap=c.chunk_overlap_words,
            doc_version=c.doc_version, docs_src_dir=docs_src_dir,
        )
        resolved += stats["resolved_includes"]
        unresolved += stats["unresolved_includes"]
        if not chunks:
            empty_files.append(source)
            skipped.append({"source": source, "reason": "no text after cleaning"})
            continue
        per_file[source] = len(chunks)
        all_chunks.extend(chunks)

    text_counts = Counter(" ".join(ch.text.split()) for ch in all_chunks)
    duplicates = sum(n - 1 for n in text_counts.values() if n > 1)
    n_written = write_chunks(all_chunks, chunks_path)

    by_dir = Counter(ch.source.split("/")[0] if "/" in ch.source else "(root)" for ch in all_chunks)
    report = {
        "documents_dir": str(documents_dir),
        "doc_version": c.doc_version,
        "chunk_size_words": c.chunk_size_words,
        "chunk_overlap_words": c.chunk_overlap_words,
        "files_seen": len(files) + sum(1 for s in skipped if s["reason"].startswith("listed")),
        "files_included": len(per_file),
        "files_skipped": len(skipped),
        "skipped": skipped,
        "chunks_written": n_written,
        "duplicate_chunks": duplicates,
        "resolved_code_includes": resolved,
        "unresolved_code_includes": unresolved,
        "chunks_by_top_level_dir": dict(sorted(by_dir.items())),
        "chunk_length_words": length_summary([ch.n_words for ch in all_chunks]),
        "chunks_per_file": per_file,
        "chunks_path": str(chunks_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return report


__all__ = [
    "preprocess_corpus",
    "process_document",
    "chunk_words",
    "split_sections",
    "resolve_includes",
    "strip_frontmatter",
    "normalise_newlines",
    "rewrite_admonition",
    "clean_markup",
    "iter_source_files",
]
