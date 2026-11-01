# Transformer-Powered Software Documentation Q&A

Grounded question answering over the **FastAPI documentation**: a developer asks a question,
the system retrieves the relevant documentation passages, a pretrained Transformer writes a
concise answer, and every sentence of that answer is checked against the retrieved passages
before it is cited. When the evidence is weak, the system says so instead of guessing.

*Addis Ababa University · Selected Topics in Software Engineering · Project 1 (Transformers)*

> Demo video: **not yet recorded** (see `reports/demo_outline.md`). The link will be added here.

## 1. Problem and user

| | |
|---|---|
| **User** | A developer building or maintaining a FastAPI service who needs a quick, trustworthy answer from the official docs. |
| **Input** | One natural-language question, e.g. *"How do I make a query parameter optional?"* |
| **Output** | A one-to-three sentence answer with citations (`tutorial/query-params.md § Optional parameters`), **or** an explicit insufficient-evidence response with the closest passages. |
| **Why a Transformer** | Dense retrieval (sentence embeddings) and answer generation both need language understanding beyond keyword matching. A keyword baseline (BM25) is kept to measure what the Transformer components add. |

The full task statement, correctness/citation definitions, corpus provenance and split rules are
in [`reports/task_definition.md`](reports/task_definition.md).

## 2. Architecture

```
question ──► Retriever ──► top-k passages ──► Generator ──► raw answer
             BM25 (baseline)                 Transformer      │
             dense MiniLM (Transformer)                       ▼
             hybrid (rank fusion, experiment)          Grounding check
                                                       sentence ↔ passage overlap
                                                              │
                                        ┌─────────────────────┴───────────────────┐
                                        ▼                                         ▼
                              answer + citations                     abstain ("insufficient evidence")
                              (only retrieved passages                + reason + closest passages
                               can be cited)
```

| Component | Module | What it does |
|---|---|---|
| Preprocessing | `src/docqa/preprocess.py` | Markdown → section-aware word chunks (180 words, 40 overlap) with file/section metadata; inlines the docs' `{* docs_src/... *}` code examples; deterministic. |
| Lexical baseline | `src/docqa/retrieval/bm25.py` | BM25 implemented from scratch (≈60 lines), unit-tested against a hand computation. |
| Dense retriever | `src/docqa/retrieval/dense.py` | `sentence-transformers/all-MiniLM-L6-v2` (22 M params, 384-d), cosine similarity, embeddings cached under `artifacts/`. |
| Hybrid | `src/docqa/retrieval/hybrid.py` | Reciprocal rank fusion of both rankings (validation experiment E5). |
| Generator | `src/docqa/generation/generator.py` | Any Hugging Face seq2seq or instruction-tuned decoder model (`google/flan-t5-base`, `Qwen/Qwen2.5-0.5B-Instruct`); plus an **extractive** no-model fallback used by the smoke test. |
| Grounding | `src/docqa/generation/grounding.py` | Maps each answer sentence to its best-supporting passage by content-word overlap; citations can only point at retrieved passages (enforced by a test); two abstention gates (weak retrieval score; "I don't know"/unsupported sentences). |
| Pipeline | `src/docqa/pipeline.py` | `question → AnswerRecord` with passages, raw output, sentence support, citations, timings. |
| Evaluation | `src/docqa/metrics.py`, `evaluate_retrieval.py`, `evaluate_answers.py` | Recall@k / HitRate@k / MRR@k with explicit definitions; answer proxies + manual review CSV. |

## 3. Setup

Requirements: Python 3.10–3.12, ~3 GB disk for models, CPU only. Tested on Python 3.12 (WSL Ubuntu).

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # CPU wheel, avoids the CUDA download
pip install -r requirements.txt
pip install -e .                                          # makes `python -m docqa` / `docqa` available
```

(With [uv](https://docs.astral.sh/uv/): `uv venv --python 3.12 && uv pip install --index-url https://download.pytorch.org/whl/cpu torch && uv pip install -r requirements.txt -e .`)

The documentation corpus is committed under `data/documents/fastapi/`. To re-fetch it at the
pinned commit: `bash scripts/fetch_docs.sh`.

## 4. Run

```bash
python -m docqa preprocess            # data/processed/chunks.jsonl + corpus_report.json  (~2 s)
python -m docqa index                 # dense embeddings, first run downloads MiniLM (~90 MB), ~1 min on CPU
python -m docqa ask "How do I make a query parameter optional?" --debug
python -m docqa ask "..." --config configs/final.json     # the frozen, evaluated configuration
```

No-download smoke run (BM25 + extractive generator):

```bash
python -m docqa ask "How do I make a query parameter optional?" --set retrieval.method=bm25 --set generation.generator=extractive --set generation.min_retrieval_score=0
```

Any setting can be overridden with `--set section.key=value`; see `configs/default.json`.

Optional local API + demo page (not needed for evaluation):

```bash
uvicorn server.app:app --port 8000      # then open http://127.0.0.1:8000/
```

## 5. Test

```bash
python -m pytest            # 57 tests, < 1 s, no model downloads
```

Covered: preprocessing (empty/whitespace docs, frontmatter, CRLF, headings/anchors, code-fence
awareness, include resolution, chunk boundaries and overlap, determinism, duplicate ids,
malformed JSONL, missing metadata), BM25 against a hand computation, dense retrieval with an
injected fake encoder (caching by corpus fingerprint), source filter, rank fusion, metrics
against hand-calculated values, grounding/citation validity, abstention gates, config
overrides, question-set consistency with the real corpus, and an end-to-end smoke path.

## 6. Evaluate

```bash
python -m docqa check-questions                              # labels exist in the corpus, splits ok
python -m docqa eval-retrieval --split val --methods bm25,dense,hybrid
python scripts/run_experiments.py --split val                # experiment matrix → reports/experiments/
python scripts/tune_abstention.py --method bm25 --split val  # threshold sweep (validation only)
python -m docqa eval-answers --split test --config configs/final.json --tag test_final
python scripts/make_plots.py                                 # figures → reports/figures/
```

Metric definitions (denominators, relevance rule) are in `src/docqa/metrics.py` and
`reports/task_definition.md`.

## 7. Results

See [`reports/results.md`](reports/results.md) for the frozen final-test numbers,
[`reports/experiments/experiments.md`](reports/experiments/experiments.md) for the validation
experiments, and [`reports/failure_analysis.md`](reports/failure_analysis.md) for real failures.

**Held-out test split, frozen configuration (`configs/final.json`), run once.**

Retrieval, 31 answerable questions, relevance = labelled documentation file (source) or file#section:

| retriever | MRR@5 (source) | Top-1 (source) | Recall@5 (section) | MRR@5 (section) | s/query |
|---|---|---|---|---|---|
| BM25 (non-Transformer baseline) | 0.888 | 0.806 | 0.919 | 0.764 | 0.005 |
| dense MiniLM-L6-v2 (Transformer) | 0.935 | 0.871 | 0.919 | 0.748 | 0.11 |
| **hybrid rank fusion (final)** | **0.984** | **0.968** | **0.952** | **0.903** | 0.13 |

Recall@5 at source level is 1.000 for all three. The hybrid was selected on validation
(`reports/experiments.md`, E5) and its lead holds on test.

Answers, 36 test questions (31 answerable + 5 insufficient evidence), generator Qwen2.5-0.5B-Instruct:

| | automatic proxy | manual review (every question read) |
|---|---|---|
| answered (of 31 answerable) | 27 (87%) | 12 fully correct, 7 partly, 8 wrong, 4 abstained |
| correctness | keyword coverage 0.77 over answered | **0.57 over answered, 0.50 over all answerable** |
| citations | precision 0.84 (cited file is relevant) | 0.72 (cited passage supports the claim) |
| abstention on insufficient-evidence questions | 2 of 5 | 2 of 5 (3 answers were fabricated) |
| latency (CPU) | 18.5 s per question (0.13 s retrieval) | |

The keyword proxy overstates correctness (it accepted four wrong answers that contain the
right words, e.g. 401 and 403 swapped); the manual numbers are the ones to quote. One test
question is about 3 points, so read these as "about half of the answers are fully correct".

Most important experiment: fusing BM25 and dense rankings (E5) raised validation MRR@5 from
0.76/0.68 to 0.82 and confirmed on test (0.98); the generator, not retrieval, is the
bottleneck. Figures: `reports/figures/`.

## 8. Limitations

* **Grounding checks words, not facts.** A sentence that reuses the passage's vocabulary passes
  even when it inverts the relation (401/403 swapped) or adds a number the passage never
  states ("5 MB"). Three of five off-topic test questions were answered with invented facts.
* **Small generator.** Qwen2.5-0.5B fits a 7.6 GB laptop in bfloat16 but gets about half of the
  answers fully right; it paraphrases identifiers (`ge` → `gte`), which the gate then rejects.
* **Small benchmark.** 58 questions written and labelled by the team; 31 answerable test
  questions (±3 points per question); only 2 insufficient-evidence questions on validation to
  tune abstention. Conclusions are directional, not precise.
* **One documentation version, English only.** Corpus pinned to FastAPI commit `5cb6272e`;
  version-specific answers (e.g. the 0.122.0 status-code change) may not hold elsewhere.
* **Latency.** ~18 s per answer on CPU, dominated by generation.
* **Section ranking.** The right page is almost always rank 1, but the right *section* of a long
  page is missed in the top 3 passages for a few questions (test-02, test-29, test-34).
  Full failure analysis: `reports/failure_analysis.md`.

## 9. Repository layout

```
configs/            default.json (development), final.json (frozen for the test run)
data/documents/     FastAPI docs subset + referenced code examples + SOURCE.md (provenance)
data/evaluation/    questions.json (58 hand-verified questions, dev/val/test) + README
data/processed/     chunks.jsonl + corpus_report.json (generated by `preprocess`)
artifacts/          cached embeddings (gitignored, rebuilt by `index`)
src/docqa/          the package (see Architecture)
scripts/            fetch_docs.sh, run_experiments.py, tune_abstention.py, make_plots.py
tests/              pytest suite
reports/            task_definition, experiments, results, failure_analysis, demo_outline, predictions, figures
server/, client/    optional FastAPI endpoint + static demo page
```

## 10. Team, roles and AI-assistance disclosure

| member | role | contributions (fill in before submission) |
|---|---|---|
| _name 1_ | data & evaluation | corpus fetch/preprocessing, question labelling, metrics, manual review |
| _name 2_ | retrieval & generation | BM25/dense/hybrid retrievers, generator and grounding, experiments |
| _name 3_ (optional) | engineering & demo | tests, CLI, server/client, README, video |

All members appear in the demo video and can explain every module (course rule).

**AI assistance.** The code structure, implementation, tests and report drafts in this
repository were produced with Claude (Anthropic) as a coding assistant, working from the course
brief; the team reviewed, ran and is able to explain every module. All reported numbers come
from commands in this README executed on the committed code; nothing was edited by hand.
Reused third-party work: FastAPI documentation (MIT), `sentence-transformers/all-MiniLM-L6-v2`
(Apache-2.0), `google/flan-t5-base` (Apache-2.0), `Qwen/Qwen2.5-0.5B-Instruct` (Apache-2.0),
BM25 (Robertson & Zaragoza), reciprocal rank fusion (Cormack et al., 2009).
