"""Shared fixtures: a tiny in-memory corpus and a deterministic fake encoder."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from docqa.corpus import Chunk
from docqa.retrieval.bm25 import tokenize


def make_chunk(idx: int, source: str, text: str, section: str = "Intro", anchor: str = "intro") -> Chunk:
    return Chunk(
        id=f"{source}::{idx:03d}", source=source, section=section, section_anchor=anchor,
        heading_path=f"{source} > {section}", chunk_index=idx, text=text,
        n_words=len(text.split()), doc_version="test",
    )


@pytest.fixture
def tiny_chunks() -> list[Chunk]:
    return [
        make_chunk(0, "tutorial/query-params.md",
                   "Query parameters that are not part of the path are optional when you give them a default "
                   "value such as None. Declare them as function parameters.",
                   "Optional parameters", "optional-parameters"),
        make_chunk(1, "tutorial/query-params.md",
                   "When you declare a query parameter without a default value it is required. "
                   "FastAPI returns an error if it is missing.",
                   "Required query parameters", "required-query-parameters"),
        make_chunk(0, "tutorial/path-params.md",
                   "Path parameters are declared with Python format string syntax in the path. "
                   "The value is passed to the function as an argument.",
                   "Path parameters", "path-parameters"),
        make_chunk(0, "deployment/docker.md",
                   "A Dockerfile copies the requirements file, installs the dependencies and runs "
                   "the fastapi run command to start the server.",
                   "Dockerfile", "dockerfile"),
    ]


class HashingEncoder:
    """Deterministic bag-of-words hashing encoder (no model download)."""

    dim = 64

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim
                out[i, h] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


@pytest.fixture
def fake_encoder() -> HashingEncoder:
    return HashingEncoder()
