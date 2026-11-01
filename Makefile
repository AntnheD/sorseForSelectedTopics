# Convenience targets. Every target has a plain command equivalent in README.md.
PY ?= python

.PHONY: help data preprocess index test smoke eval-retrieval eval-answers experiments plots serve clean

help:
	@echo "make data            fetch the FastAPI docs at the pinned commit"
	@echo "make preprocess      build data/processed/chunks.jsonl"
	@echo "make index           build/cache the dense embedding index"
	@echo "make test            run the unit tests (no model downloads)"
	@echo "make smoke           end-to-end run with BM25 + extractive generator (no downloads)"
	@echo "make eval-retrieval  compare BM25 and dense retrieval on the validation split"
	@echo "make experiments     run the validation experiment matrix"
	@echo "make eval-answers    run the frozen final pipeline on the test split"
	@echo "make plots           regenerate figures under reports/figures"
	@echo "make serve           start the optional API + demo page on :8000"

data:
	bash scripts/fetch_docs.sh

preprocess:
	$(PY) -m docqa preprocess

index:
	$(PY) -m docqa index

test:
	$(PY) -m pytest

smoke:
	$(PY) -m docqa ask "How do I make a query parameter optional?" --debug \
	    --set retrieval.method=bm25 --set generation.generator=extractive --set generation.min_retrieval_score=0

eval-retrieval:
	$(PY) -m docqa eval-retrieval --split val --methods bm25,dense

experiments:
	$(PY) scripts/run_experiments.py --split val

eval-answers:
	$(PY) -m docqa eval-answers --split test --config configs/final.json --tag test_final

plots:
	$(PY) scripts/make_plots.py

serve:
	uvicorn server.app:app --port 8000

clean:
	rm -rf artifacts data/processed/experiments .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
