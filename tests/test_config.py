"""Configuration loading and overrides."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from docqa.config import REPO_ROOT, Config, parse_overrides


def test_defaults_and_root_resolution():
    cfg = Config.load()
    assert cfg.root == REPO_ROOT
    assert cfg.resolve("data/x").is_absolute()
    assert cfg.corpus.chunk_size_words == 180


def test_json_file_then_overrides(tmp_path: Path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"retrieval": {"top_k": 7}, "generation": {"generator": "extractive"}}), encoding="utf-8")
    cfg = Config.load(p, overrides=["retrieval.top_k=9", "retrieval.max_per_source=2", "generation.model=x/y"])
    assert cfg.retrieval.top_k == 9
    assert cfg.retrieval.max_per_source == 2
    assert cfg.generation.generator == "extractive"
    assert cfg.generation.model == "x/y"
    assert cfg.source_file == str(p)


def test_parse_overrides_types():
    out = parse_overrides(["a.b=1", "a.c=true", "a.d=null", "a.e=hello", "a.f=[1,2]"])
    assert out == {"a.b": 1, "a.c": True, "a.d": None, "a.e": "hello", "a.f": [1, 2]}
    with pytest.raises(ValueError):
        parse_overrides(["novalue"])


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="Unknown config key"):
        Config.load(overrides={"retrieval.nope": 1})
    with pytest.raises(ValueError, match="Unknown config section"):
        Config.load(overrides={"nope.x": 1})


def test_default_config_file_is_valid():
    cfg = Config.load("configs/default.json")
    assert cfg.retrieval.method in ("bm25", "dense")
    assert cfg.to_dict()["generation"]["model"]
