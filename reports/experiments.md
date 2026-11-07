# Validation experiments

All experiments use the **validation split only** (12 answerable + 2 insufficient-evidence
questions). The test split was not touched until `configs/final.json` was frozen. With 12
questions, one question moves MRR by up to 0.08 and a hit-rate by 0.083, so differences below
that are noise and are described as such.

Commands: `python scripts/run_experiments.py --split val` (table in
`reports/experiments/experiments.md`), `python scripts/tune_abstention.py --method <m> --split val`,
and `python -m docqa eval-answers --split val --tag ...` for the generator rows. Starting
point for every row: `configs/default.json` (chunks of 180 words / 40 overlap, k = 5).

## E1 Lexical baseline vs dense Transformer retrieval

| retriever | Recall@5 (source) | MRR@5 (source) | Top-1 (source) | Recall@5 (section) | MRR@5 (section) | s/query |
|---|---|---|---|---|---|---|
| BM25 (baseline) | 1.000 | 0.757 | 0.583 | 0.750 | 0.531 | 0.004 |
| dense MiniLM-L6-v2 | 1.000 | 0.682 | 0.583 | 0.583 | 0.318 | 0.10 |

**Outcome.** On this corpus the keyword baseline is at least as good as the 22 M-parameter
sentence-embedding model, and clearly better at section level. Explanation from inspecting the
per-question reports (`reports/retrieval/val_*.json`): the questions reuse the documentation's
own identifiers (`dependency_overrides`, `response_model_exclude_unset`, `--root-path`), which
BM25 matches exactly, while MiniLM treats sibling sections that share wording ("Optional query
/ cookie / header parameter models") as near-identical. The dense model does win on
paraphrased questions (dev-06 "enable CORS", val-04 "status code of RedirectResponse" rank 1
for dense, rank 3 and 2 for BM25).

## E2 Chunk size

| chunk words / overlap | chunks | BM25 MRR@5 src / sec | dense MRR@5 src / sec |
|---|---|---|---|
| 120 / 30 | 1492 | 0.736 / 0.521 | 0.732 / 0.389 |
| **180 / 40** (default) | 1217 | **0.757 / 0.531** | 0.682 / 0.318 |
| 260 / 60 | 1099 | 0.743 / 0.535 | 0.628 / 0.312 |

**Outcome.** No setting beats the default by more than one question's worth; 180/40 is kept.
Dense retrieval prefers shorter chunks (0.732 vs 0.682 at source level), consistent with
MiniLM's 256-token input limit truncating long chunks with code.

## E3 Top-k

| k | BM25 Recall@k / MRR@k | dense Recall@k / MRR@k | hybrid Recall@k / MRR@k |
|---|---|---|---|
| 3 | 0.917 / 0.736 | 0.667 / 0.611 | 0.833 / 0.778 |
| 5 | 1.000 / 0.757 | 1.000 / 0.682 | 1.000 / 0.822 |
| 8 | 1.000 / 0.757 | 1.000 / 0.682 | 0.917 / 0.806 |

**Outcome.** k = 5 reaches full source recall for every retriever; the generator only receives
the top 3 passages (prompt length), so k = 5 is kept for retrieval and 3 for generation. (The
hybrid list at k = 8 is fused from larger candidate lists, which is why its recall can differ
from k = 5.)

## E4 Limiting chunks per source file

| max chunks per source | BM25 MRR@5 src / sec | dense MRR@5 src / sec |
|---|---|---|---|
| none (default) | 0.757 / 0.531 | 0.682 / 0.318 |
| 2 | 0.757 / 0.535 | 0.693 / 0.274 |
| 1 | 0.757 / 0.444 | 0.722 / 0.243 |

**Outcome.** The filter never helps at source level and hurts at section level (it throws
away the second-best section of the right page). Not adopted.

## E5 Hybrid: reciprocal rank fusion of BM25 and dense

| retriever | Recall@5 src | MRR@5 src | Top-1 src | Recall@5 sec | MRR@5 sec | s/query |
|---|---|---|---|---|---|---|
| BM25 | 1.000 | 0.757 | 0.583 | 0.750 | 0.531 | 0.004 |
| dense | 1.000 | 0.682 | 0.583 | 0.583 | 0.318 | 0.10 |
| **hybrid (RRF, c = 60)** | **1.000** | **0.822** | **0.750** | **0.792** | 0.531 | 0.09 |

**Outcome.** Fusing the two rankings puts the right page at rank 1 for 9 of 12 questions
(vs 7 for either retriever alone) and gives the best section recall. The gain is two
questions, which is at the edge of what 12 questions can show, but it is consistent with the
complementary error patterns seen in E1 (exact identifiers vs paraphrases). **Adopted** as the
final retriever; the ~30-line fusion is the only retrieval machinery added beyond the two
basic retrievers.

## Abstention threshold (pre-generation gate)

`scripts/tune_abstention.py` prints the top-1 score for every validation question and sweeps
the rule "abstain if top score < t".

| retriever | best t on val | false abstentions (dev+val answerable, n = 20) | insufficient-evidence caught (n = 2) |
|---|---|---|---|
| BM25 | 15.5 | 1 (dev-06 scores 13.7) → unusable; t = 13.0 gives 0 | 1 of 2 |
| dense | 0.55 | 1 (val-12 scores 0.43) | 2 of 2 |
| **hybrid** | **0.0229** | **0** | 1 of 2 |

With RRF scores, t = 0.0229 has a plain meaning: a chunk scores above it only if **both**
retrievers rank it (one retriever alone can contribute at most 1/61 = 0.0164). The rule
"abstain unless BM25 and the dense model agree on the top passage" is adopted. It is a weak
gate (only 2 negative examples exist on validation); the post-generation gates (model says it
does not know; fewer than half of the sentences supported) carry the rest of the abstention
behaviour.

## Generator comparison

Hybrid retrieval, threshold 0.0229, code blocks removed from the prompt, validation split
(12 answerable + 2 insufficient). `python -m docqa eval-answers --split val --tag val_hybrid_*`.

| generator | answer rate (answerable) | false abstention | keyword coverage (answered) | all facts present | citation precision | correct abstention (n = 2) | s/question |
|---|---|---|---|---|---|---|---|
| extractive (top overlapping passage sentences, no model) | 1.00 | 0.00 | 0.43 | 0.33 | 0.67 | 0.5 | 0.2 |
| google/flan-t5-base (248 M, fp32) | 0.58 | 0.42 | 0.64 | 0.33 | 0.86 | 1.0 | 3.2 |
| **Qwen/Qwen2.5-0.5B-Instruct (494 M, bf16)** | **0.92** | **0.08** | **0.79** | **0.67** | 0.73 | 1.0 | 20.5 |

**Outcome.** Flan-T5 replied "I don't know." to 5 of 12 answerable questions and, when it
did answer, copied passage text; its T5 vocabulary cannot even emit `{`, `}` or backticks, so
code-heavy passages come out garbled (seen on dev-01). Qwen2.5-0.5B-Instruct answers almost
everything in prose and doubles the all-facts rate, at 6× the latency (still fine on a laptop
CPU). Adopted. A first Qwen trial with code blocks *in* the prompt reproduced whole examples
and invented an "empty string" rule (dev-01), which is why `include_code_in_prompt=false` was
set before the comparison.

## Grounding rule fixes found on dev/val (before freezing)

* Code lines in an answer were counted as "sentences" with support 1.0, inflating the support
  rate; fenced code is now removed before sentence splitting.
* One-token answers such as `307` were dropped as "no sentences" and caused abstentions
  (Flan-T5 val-03, val-04); a short answer is now kept as a single claim.
* `exclude_unset` was not matched inside `response_model_exclude_unset` by the token-boundary
  keyword rule; the val-03 keyword list was extended. No test labels were changed.

## Post-hoc check proposed by the failure analysis (validation only)

See `reports/failure_analysis.md`, section "Fixes tried".
