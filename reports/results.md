# Final results (frozen configuration, test split)

**Frozen on 2026-11-03** in `configs/final.json`: hybrid retrieval (BM25 + MiniLM, reciprocal
rank fusion), k = 5, 3 passages to the generator, generator `Qwen/Qwen2.5-0.5B-Instruct`
(bfloat16, greedy, 128 new tokens, code blocks removed from the prompt), abstention gates
`min_retrieval_score = 0.0229`, `min_sentence_support = 0.35`, `min_supported_fraction = 0.5`.
Every choice was made on the validation split (`reports/experiments.md`). The test split
(31 answerable + 5 insufficient-evidence questions, `data/evaluation/questions.json` v1.0)
was run **once** with this configuration. No test result was used to change anything.

Git state: the repository had not been initialised in Git at the time of the run (the team
initialises it before submission); corpus commit `5cb6272e`, question set v1.0.

Commands (CPU, WSL Ubuntu, 12 cores, 7.6 GB RAM):

```bash
python -m docqa eval-retrieval --split test --methods bm25,dense,hybrid --config configs/final.json
python -m docqa eval-retrieval --split test --methods bm25,dense,hybrid --level section --config configs/final.json
python -m docqa eval-answers   --split test --config configs/final.json --tag test_final
python scripts/apply_manual_review.py
```

## 1. Retrieval (n = 31 answerable test questions)

Relevance = the labelled documentation file (source level) or file#section (section level).
Definitions in `src/docqa/metrics.py`.

| retriever | Recall@5 src | HitRate@5 src | MRR@5 src | Top-1 src | Recall@5 sec | MRR@5 sec | Top-1 sec | s/query |
|---|---|---|---|---|---|---|---|---|
| BM25 (non-Transformer baseline) | 1.000 | 1.000 | 0.888 | 0.806 | 0.919 | 0.764 | 0.613 | 0.005 |
| dense MiniLM-L6-v2 (Transformer) | 1.000 | 1.000 | 0.935 | 0.871 | 0.919 | 0.748 | 0.645 | 0.11 |
| **hybrid (final)** | **1.000** | **1.000** | **0.984** | **0.968** | **0.952** | **0.903** | **0.839** | 0.13 |

Reading: the right page is in the top 5 for every test question with every retriever; the
hybrid puts it at rank 1 for 30 of 31 questions and the right *section* at rank 1 for 26 of 31.
On the test split the dense Transformer retriever beats BM25 at source level (the opposite of
validation, where BM25 led by one question), which confirms that differences of one or two
questions between the single retrievers are noise; the hybrid's lead (+0.05 to +0.14 MRR over
both) is consistent across validation and test.

## 2. Answer quality (n = 36 test questions)

### 2.1 Automatic proxies (`reports/predictions/test_final_summary.json`)

| proxy | value | definition |
|---|---|---|
| answer rate on answerable | 0.871 (27/31) | did not abstain |
| false abstention rate | 0.129 (4/31) | abstained on an answerable question |
| correct abstention rate | 0.400 (2/5) | abstained on an insufficient-evidence question |
| keyword coverage (mean over answered) | 0.765 | share of required-fact keyword groups present in the answer |
| keyword coverage, all facts present | 0.548 (17/31) | over all answerable questions (abstentions count 0) |
| citation precision (mean) | 0.840 | cited files that are labelled relevant |
| support rate (mean over answered) | 0.881 | answer sentences with ≥ 0.35 content-word overlap with a passage |
| retrieval hit rate | 1.000 | a relevant file among the 5 retrieved passages |
| mean seconds per question | 18.5 | CPU; ~0.13 s retrieval, the rest generation |

### 2.2 Manual review (every test question, `reports/predictions/test_final_manual_review.csv`)

Rubric from `reports/task_definition.md`; 1 / 0.5 / 0 per criterion.

| criterion | over the 27 answered | over all 31 answerable (abstention = 0) |
|---|---|---|
| correct | 0.574 | 0.500 |
| complete | 0.537 | – |
| evidence supported | 0.630 | – |
| citations valid | 0.722 | – |

| outcome on the 31 answerable questions | count |
|---|---|
| fully correct | 12 |
| partly correct (right parameter, plus an invented detail or a missing point) | 7 |
| answered but wrong | 8 |
| abstained (false abstention; in all 4 cases the model had hallucinated and the support gate caught it) | 4 |

Insufficient-evidence questions: 2 of 5 abstained (one by the retrieval gate, one by the
support gate); 3 were answered with invented facts ("5 MB limit", "HTTP/3 since 0.128.0",
Prometheus steps built from the OpenTelemetry page).

**The automatic keyword proxy overstates correctness.** It scored 1.0 on four answers that a
reader marks wrong: test-27 (the 401/403 codes are both present but swapped), test-23 ("Yes you
can put a model in JSONResponse" when the docs say the opposite), test-16 and test-03 (the
right identifiers inside a wrong explanation). Keyword coverage 0.77 vs manual correctness
0.57 is the honest gap; only the manual numbers are claimed as correctness.

## 3. Generator comparison on validation (n = 14), for context

| generator | answer rate (answerable) | keyword coverage (answered) | all facts | citation precision | correct abstention | s/question |
|---|---|---|---|---|---|---|
| extractive (no Transformer generator) | 1.00 | 0.43 | 0.33 | 0.67 | 0.5 | 0.2 |
| google/flan-t5-base (248 M) | 0.58 | 0.64 | 0.33 | 0.86 | 1.0 | 3.2 |
| **Qwen/Qwen2.5-0.5B-Instruct (494 M, bf16)** | **0.92** | **0.79** | **0.67** | 0.73 | 1.0 | 20.5 |

Flan-T5 answered "I don't know" on 5 of 12 answerable questions and its SentencePiece
vocabulary has no `{ } < >` or backticks, so it mangles code; Qwen was chosen despite being
6× slower.

## 4. What the evidence establishes

1. **Retrieval is solved for this corpus and question style**: hybrid rank fusion gives
   MRR@5 0.98 at page level and 0.90 at section level on held-out questions, better than
   either the keyword baseline or the dense Transformer alone on both validation and test.
2. **The generator is the bottleneck.** With the right passage in front of it, the 0.5 B
   model still answers wrongly or incompletely about half the time, and writes confident
   invented steps for topics the documentation does not cover.
3. **Lexical grounding is a useful but coarse safety net.** It caught 4 hallucinated
   answers and 1 off-topic one, but let 3 fabricated answers through because they reuse
   words from the retrieved passages (e.g. "request body", "metrics", "FastAPI version").
4. **Sample size.** With 31 answerable questions one question is 3.2 points; the manual
   correctness of 0.50 ± ~0.09 (binomial s.e.) should be read as "about half".

## 5. Changes after seeing test results

None. The failure analysis (`reports/failure_analysis.md`) proposes fixes; the one that was
tried (min_supported_fraction 0.5 → 0.75) was evaluated on validation only, and its outcome is
reported there, not here.
