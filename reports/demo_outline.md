# Demo video outline (5–8 minutes)

Status: **not recorded yet.** Record with the commands below so that every number and output
on screen is the real thing. Every team member presents at least one section (course rule).
Upload unlisted (YouTube or Google Drive) and paste the link at the top of `README.md`.

Before recording: `source .venv/bin/activate`, run `python -m docqa index` once (so the demo
does not wait for embeddings), and keep `reports/results.md` open in a second window.

| # | Section | Time | What to show and say |
|---|---|---|---|
| 1 | Problem and user | 0:30–0:45 | Developer needs a trustworthy answer from the FastAPI docs. Show the docs tree (115 pages) and one question. State the output contract: concise answer + citations, or an explicit "insufficient evidence". |
| 2 | System flow | 1:00–1:30 | Architecture diagram from `README.md` §2. Walk through: Markdown → 1217 section-aware chunks (`data/processed/corpus_report.json`), BM25 baseline (own 60-line implementation) + MiniLM dense retriever + rank fusion, Qwen2.5-0.5B generator, grounding check that maps each sentence to a passage and can only cite retrieved passages. Mention everything runs on a laptop CPU, no paid services. |
| 3 | A correct, cited answer | 1:00 | `python -m docqa ask "Which attributes does an UploadFile object expose?" --config configs/final.json --debug` → answer lists `filename`, `content_type`, `file`; citation `tutorial/request-files.md § UploadFile`; show the sentence-support lines and the three retrieved passages. (test-17, fully correct in the manual review.) |
| 4 | A hard / insufficient-evidence question | 0:45–1:00 | `python -m docqa ask "How do I cache responses in memcached with FastAPI?" --config configs/final.json --debug` → abstains before generation ("retrievers disagree"). Then `python -m docqa ask "What is the default maximum request body size that FastAPI accepts?" --config configs/final.json --debug` → the model invents "5 MB" and the lexical gate does **not** catch it. Say plainly: this is a real failure (test-ie-01) and why word overlap cannot detect a fabricated number. |
| 5 | Baseline comparison and results | 1:00–1:30 | Show `reports/figures/retrieval_test.png` and the table in `reports/results.md` §1: BM25 vs dense vs hybrid (MRR@5 0.89 / 0.94 / 0.98 at page level; 0.76 / 0.75 / 0.90 at section level, n = 31). Then §2.2: manual review 12 fully correct, 7 partial, 8 wrong, 4 abstained out of 31; keyword proxy 0.77 vs manual correctness 0.57, and why the proxy is optimistic (test-27 swapped 401/403). Show `reports/experiments.md` E5 to explain the hybrid was chosen on validation, not test. |
| 6 | One real failure and the lesson | 0:45–1:00 | test-27 from `reports/predictions/test_final.jsonl`: the passage says 401 now / 403 before, the model swapped them, every keyword matched, and the citation is valid. Lesson: lexical support checks verify *vocabulary*, not *facts*; a 0.5 B model needs either a larger model or an entailment/NLI check, and automatic proxies must be validated by reading outputs. |
| 7 | Reproduce, limits, conclusion | 0:30–0:45 | Show the four commands in `README.md` §4–§6 (`preprocess`, `index`, `pytest` → 57 passed, `eval-answers --config configs/final.json`). Limits: 36 test questions (one question ≈ 3 points), single docs version, English only, 18 s per answer on CPU, 3 of 5 off-topic questions answered. Conclusion: retrieval is reliable; generation and fabricated-fact detection are the next work (Project 2 will learn when to abstain from feedback). |

Total: about 6–7 minutes.

Commands used on screen (copy-paste):

```bash
python -m docqa ask "Which attributes does an UploadFile object expose?" --config configs/final.json --debug
python -m docqa ask "How do I cache responses in memcached with FastAPI?" --config configs/final.json --debug
python -m docqa ask "What is the default maximum request body size that FastAPI accepts?" --config configs/final.json --debug
python -m pytest
```
