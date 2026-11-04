"""Dense retriever (with an injected fake encoder), caching and the source filter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from docqa.config import Config
from docqa.retrieval import build_retriever
from docqa.retrieval.base import RetrievedChunk
from docqa.retrieval.dense import DenseRetriever
from docqa.retrieval.filters import SourceDiverseRetriever, limit_per_source
from tests.conftest import make_chunk


def test_dense_retriever_returns_ranked_metadata(tiny_chunks, fake_encoder, tmp_path: Path):
    r = DenseRetriever(tiny_chunks, fake_encoder, model_name="fake/model", artifacts_dir=tmp_path)
    assert r.embeddings.shape == (4, fake_encoder.dim)
    assert np.allclose(np.linalg.norm(r.embeddings, axis=1), 1.0)
    results = r.search("Dockerfile requirements fastapi run command", k=2)
    assert results[0].chunk.source == "deployment/docker.md"
    assert [x.rank for x in results] == [1, 2]
    assert results[0].score >= results[1].score
    assert results[0].chunk.section_anchor == "dockerfile"


def test_dense_embeddings_are_cached_by_corpus_fingerprint(tiny_chunks, fake_encoder, tmp_path: Path):
    r1 = DenseRetriever(tiny_chunks, fake_encoder, model_name="fake/model", artifacts_dir=tmp_path)
    assert r1.cache_path.exists() and r1.index_seconds > 0
    r2 = DenseRetriever(tiny_chunks, fake_encoder, model_name="fake/model", artifacts_dir=tmp_path)
    assert r2.index_seconds == 0.0  # loaded from cache
    assert np.array_equal(r1.embeddings, r2.embeddings)
    changed = tiny_chunks[:-1] + [make_chunk(9, "deployment/docker.md", "different text")]
    r3 = DenseRetriever(changed, fake_encoder, model_name="fake/model", artifacts_dir=tmp_path)
    assert r3.cache_path != r1.cache_path


def test_limit_per_source_reranks(tiny_chunks):
    results = [RetrievedChunk(chunk=c, score=1.0 - 0.1 * i, rank=i + 1) for i, c in enumerate(tiny_chunks)]
    kept = limit_per_source(results, max_per_source=1, k=3)
    assert [x.chunk.source for x in kept] == ["tutorial/query-params.md", "tutorial/path-params.md", "deployment/docker.md"]
    assert [x.rank for x in kept] == [1, 2, 3]


def test_source_diverse_wrapper_over_bm25(tiny_chunks):
    cfg = Config.load()
    base = build_retriever(cfg, tiny_chunks, method="bm25")
    wrapped = SourceDiverseRetriever(base, max_per_source=1)
    results = wrapped.search("query parameter default value", k=3)
    assert len({x.chunk.source for x in results}) == len(results)
    assert wrapped.name == "bm25+max1/source"


def test_build_retriever_dispatch(tiny_chunks, fake_encoder, tmp_path: Path):
    cfg = Config.load(overrides={"retrieval.artifacts_dir": str(tmp_path), "retrieval.max_per_source": 2})
    r = build_retriever(cfg, tiny_chunks, method="dense", encoder=fake_encoder)
    assert isinstance(r, SourceDiverseRetriever) and r.inner.name == "dense"
    assert build_retriever(cfg, tiny_chunks, method="bm25", max_per_source=0).name == "bm25"
