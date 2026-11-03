"""Preprocessing: cleaning, sectioning, chunking, determinism and the loader's checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docqa.config import Config
from docqa.corpus import load_chunks
from docqa.errors import CorpusNotFoundError, MalformedDataError
from docqa.preprocess import (
    chunk_words,
    preprocess_corpus,
    process_document,
    resolve_includes,
    rewrite_admonition,
    split_sections,
    strip_frontmatter,
)

KW = dict(chunk_size=180, chunk_overlap=40, doc_version="t")


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


# ----------------------------------------------------------------- documents
def test_empty_and_whitespace_documents_yield_no_chunks():
    assert process_document("", "a.md", **KW)[0] == []
    assert process_document("   \n\n\t \r\n", "a.md", **KW)[0] == []


def test_frontmatter_is_removed_and_horizontal_rules_kept():
    text = "---\ntitle: X\nhide: [nav]\n---\n\n# Title\n\nBody text.\n\n---\n\nMore."
    assert strip_frontmatter(text).startswith("# Title")
    chunks, _ = process_document(text, "a.md", **KW)
    assert len(chunks) == 1
    assert "title: X" not in chunks[0].text
    assert "---" in chunks[0].text  # the rule inside the body survives


def test_crlf_and_bom_are_normalised():
    text = "﻿# Title { #title }\r\n\r\nLine one.\r\nLine two.\r\n"
    chunks, _ = process_document(text, "a.md", **KW)
    assert chunks[0].text == "Line one.\nLine two."
    assert chunks[0].section_anchor == "title"


def test_headings_define_sections_and_heading_path():
    text = "# Doc { #doc }\n\nIntro.\n\n## Sub { #sub }\n\nSub body.\n\n### Deep\n\nDeep body."
    chunks, stats = process_document(text, "x/y.md", **KW)
    assert stats["sections"] == 3
    assert [c.section for c in chunks] == ["Doc", "Sub", "Deep"]
    assert chunks[1].heading_path == "Doc > Sub"
    assert chunks[2].heading_path == "Doc > Sub > Deep"
    assert chunks[2].section_anchor == "deep"  # slugified when no explicit anchor
    assert [c.id for c in chunks] == ["x/y.md::000", "x/y.md::001", "x/y.md::002"]


def test_hash_lines_inside_code_fences_are_not_headings():
    text = "# Doc\n\nText.\n\n```python\n# not a heading\nx = 1\n```\n\nAfter."
    sections = split_sections(text, "Doc")
    assert len(sections) == 1
    assert "# not a heading" in sections[0].text()


def test_admonitions_and_tabs_become_labels():
    assert rewrite_admonition("/// tip") == "Tip:"
    assert rewrite_admonition("/// note | Technical Details") == "Note (Technical Details):"
    assert rewrite_admonition("//// tab | Python 3.10+") == "Python 3.10+:"
    assert rewrite_admonition("///") == ""
    assert rewrite_admonition("////") == ""
    assert rewrite_admonition("regular line") is None


def test_links_images_and_html_are_cleaned():
    text = "# T\n\nSee [the tutorial](../tutorial/index.md){.internal-link target=_blank} and <abbr title=\"x\">API</abbr>.\n\n![img](a.png)\n\n<img src=\"a.png\">"
    chunks, _ = process_document(text, "a.md", **KW)
    assert chunks[0].text == "See the tutorial and API."


def test_includes_are_resolved_or_marked(tmp_path: Path):
    docs_src = tmp_path / "docs_src"
    (docs_src / "demo").mkdir(parents=True)
    (docs_src / "demo" / "ex.py").write_text("print('hi')\n", encoding="utf-8")
    text = "{* ../../docs_src/demo/ex.py hl[1] *}\n\n{!> ../../docs_src/demo/missing.py!}"
    out, resolved, unresolved = resolve_includes(text, docs_src)
    assert resolved == 1 and unresolved == 1
    assert "```python\nprint('hi')\n```" in out
    assert "(code example: ../../docs_src/demo/missing.py)" in out
    assert resolve_includes(text, None)[2] == 2


# ----------------------------------------------------------------- chunking
def test_short_text_is_one_chunk():
    assert chunk_words(words(50), 180, 40) == [words(50)]


def test_chunk_boundaries_and_overlap_are_exact():
    text = words(500)
    chunks = chunk_words(text, 180, 40)
    assert [len(c.split()) for c in chunks] == [180, 180, 180, 80]
    first, second = chunks[0].split(), chunks[1].split()
    assert first[140:] == second[:40]      # 40 words shared
    assert second[0] == "w140"
    assert chunks[-1].split()[-1] == "w499"  # nothing lost
    # Every word index appears in at least one chunk.
    seen = {w for c in chunks for w in c.split()}
    assert len(seen) == 500


def test_tiny_trailing_window_is_merged_into_previous_chunk():
    chunks = chunk_words(words(181), 180, 40)
    assert len(chunks) == 1 and len(chunks[0].split()) == 181
    chunks = chunk_words(words(220), 180, 40)
    assert [len(c.split()) for c in chunks] == [180, 80]


def test_chunking_keeps_newlines_inside_chunk():
    text = "a b\nc d\n\ne f"
    assert chunk_words(text, 10, 2) == [text]


def test_chunk_parameter_validation():
    with pytest.raises(ValueError):
        chunk_words("a b", 0, 0)
    with pytest.raises(ValueError):
        chunk_words("a b", 10, 10)


# ----------------------------------------------------------------- corpus run
def _write_corpus(root: Path) -> Config:
    docs = root / "data" / "documents" / "fastapi" / "docs"
    (docs / "tutorial").mkdir(parents=True)
    (docs / "tutorial" / "a.md").write_text("# A { #a }\n\n" + words(400, "a"), encoding="utf-8")
    (docs / "tutorial" / "b.md").write_text("# B\n\nShort body.\n\n## B2\n\nSecond.", encoding="utf-8")
    (docs / "empty.md").write_text("\n\n", encoding="utf-8")
    (docs / "_llm-test.md").write_text("# LLM test\n\nignore me", encoding="utf-8")
    cfg = Config.load(root=root)
    cfg.corpus.doc_version = "test-version"
    return cfg


def test_preprocess_corpus_writes_chunks_and_report(tmp_path: Path):
    cfg = _write_corpus(tmp_path)
    report = preprocess_corpus(cfg)
    assert report["files_seen"] == 4
    assert report["files_included"] == 2
    assert {s["source"] for s in report["skipped"]} == {"_llm-test.md", "empty.md"}
    assert report["chunks_written"] == 3 + 2
    assert report["duplicate_chunks"] == 0
    assert report["chunk_length_words"]["max"] == 180
    chunks = load_chunks(cfg.resolve(cfg.corpus.chunks_path))
    assert all(c.doc_version == "test-version" for c in chunks)
    assert all(c.source for c in chunks)


def test_preprocess_is_deterministic(tmp_path: Path):
    cfg = _write_corpus(tmp_path)
    preprocess_corpus(cfg)
    first = cfg.resolve(cfg.corpus.chunks_path).read_bytes()
    preprocess_corpus(cfg)
    assert cfg.resolve(cfg.corpus.chunks_path).read_bytes() == first


def test_missing_docs_dir_raises(tmp_path: Path):
    with pytest.raises(CorpusNotFoundError):
        preprocess_corpus(Config.load(root=tmp_path))


# ----------------------------------------------------------------- loader
def _record(**over):
    base = dict(id="a.md::000", source="a.md", section="S", section_anchor="s", heading_path="A > S",
                chunk_index=0, text="t", n_words=1, doc_version="v")
    base.update(over)
    return base


def test_loader_rejects_malformed_jsonl(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(_record()) + "\n{not json\n", encoding="utf-8")
    with pytest.raises(MalformedDataError, match=":2: invalid JSON"):
        load_chunks(p)


def test_loader_rejects_duplicate_ids_and_missing_metadata(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps(_record()) + "\n" + json.dumps(_record()) + "\n", encoding="utf-8")
    with pytest.raises(MalformedDataError, match="duplicate chunk id"):
        load_chunks(p)
    p.write_text(json.dumps(_record(source="")) + "\n", encoding="utf-8")
    with pytest.raises(MalformedDataError, match="empty 'source'"):
        load_chunks(p)
    rec = _record()
    del rec["section"]
    p.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    with pytest.raises(MalformedDataError, match="missing fields"):
        load_chunks(p)


def test_loader_missing_file_message(tmp_path: Path):
    with pytest.raises(CorpusNotFoundError, match="preprocess"):
        load_chunks(tmp_path / "nope.jsonl")
