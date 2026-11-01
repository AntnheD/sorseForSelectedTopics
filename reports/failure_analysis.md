# Failure analysis

All cases are real outputs of the frozen final run (`reports/predictions/test_final.jsonl`,
scores in `reports/predictions/test_final_manual_review.csv`). Each entry separates what the
logs show from what is inferred.

Error-type counts over the 36 test questions (manual review): incorrect claim 8, unsupported
claim 6, false abstention after a caught hallucination 4, failure to abstain 3, incomplete 3,
citation error 2, non-answer 1, ranking error contributing 2, fully correct 12.

## F1. Retrieval ranking error (section level): test-34

* **Question.** "How do I apply a partial update with PATCH so that fields the client did not
  send keep their stored values?"
* **Expected.** `item.model_dump(exclude_unset=True)`; sources
  `tutorial/body-updates.md#using-pydantics-exclude-unset-parameter` (also `#partial-updates-with-patch`).
* **Retrieved (hybrid, top 3).** 1 `body-updates.md#partial-updates-with-patch`,
  2 `#partial-updates-recap`, 3 `#partial-updates-recap` (second chunk of the same section). The
  section that names `exclude_unset` was not among the 3 passages handed to the generator.
* **System answer.** Six generic steps ("retrieve the stored data … update the data using the
  dict without default values …"), never naming `exclude_unset`. Cited the recap section.
* **Error type.** Ranking error (right page, wrong sections in the top 3) → incomplete answer.
* **Evidence vs hypothesis.** Evidence: the page is correct and all three passages are from it;
  the recap section repeats the same vocabulary ("partial update", "PATCH", "stored") and
  outranks the specific section. Hypothesis: the question wording matches the recap's summary
  sentences more than the specific section's "exclude_unset" sentence; a source-diversity filter
  would not help (same source), a section-level diversity filter might.
* **Proposed fix.** Pass 5 passages instead of 3 to the generator when they come from the same
  page, or dedupe by section before truncating to 3. Not implemented; would be evaluated on
  validation (E4 showed per-source limits do not help; a per-section limit is the untested
  variant).
* **Same pattern.** test-29 (`async.md#in-a-hurry` ranked below three other `async.md` sections)
  and test-02 (`#path-convertor` below `#path-parameters-containing-paths`).

## F2. Unsupported claim that passed the support gate: test-27

* **Question.** "Since FastAPI 0.122.0, which status code do the built-in security utilities
  return after a failed authentication, and what did they return before?"
* **Expected.** 401 Unauthorized now, 403 Forbidden before.
* **Retrieved.** Rank 1 is exactly the right page and section; its text states both facts.
* **System answer.** "Since FastAPI 0.122.0, the built-in security utilities return an HTTP
  status code 403 (Forbidden) … Before … they used 401 (Unauthorized)." Both sentences have
  support ≥ 0.35 and were cited. The keyword proxy scored 1.0.
* **Error type.** Incorrect claim (facts swapped) with a valid citation.
* **Evidence vs hypothesis.** Evidence: the model had the correct passage and reversed the
  mapping; the grounding check measures word overlap, so a sentence containing the right words
  in the wrong relation scores as supported. Hypothesis: a 0.5 B model confuses "before/after"
  relations in a sentence with two numbers; larger instruction models usually do not.
* **Proposed fix.** Replace or augment lexical support with an entailment check (a small NLI
  cross-encoder scoring passage → sentence), or a larger generator (1.5 B–3 B) if the laptop
  budget allows. Not implemented in Project 1; the NLI check is the natural "reward" candidate
  for Project 2.

## F3. Failure to abstain: test-ie-01

* **Question.** "What is the default maximum request body size that FastAPI accepts?"
  (labelled insufficient evidence; `grep` finds no body-size limit in the corpus).
* **Retrieved.** `tutorial/body.md#request-body-path-query-parameters` (0.0318, both retrievers
  rank it) → the retrieval gate does not fire; `handling-errors.md#use-the-requestvalidationerror-body`,
  `advanced/strict-content-type.md`.
* **System answer.** "FastAPI accepts a maximum request body size of 5MB." cited to the
  RequestValidationError-body passage.
* **Error type.** Failure to abstain + fabricated fact + citation that does not support the claim.
* **Evidence vs hypothesis.** Evidence: support score of the sentence is high because "request",
  "body", "FastAPI", "accepts" all appear in the passage; "5MB" is a single unmatched token and
  the rule tolerates it. Hypothesis: the model fills the pattern "default maximum X is N" from
  pre-training. Same mechanism in test-ie-02 ("HTTP/3 since 0.128.0") and test-ie-03
  (Prometheus steps assembled from the OpenTelemetry page).
* **Proposed fix.** (a) Require that numbers, version strings and identifiers in an answer
  sentence appear in the supporting passage (a cheap "entity support" rule); (b) raise
  `min_supported_fraction`. Fix (b) was tried on validation, see "Fixes tried".

## F4. False abstention after a caught hallucination: test-07

* **Question.** "How do I require a path parameter item_id to be an integer greater than or
  equal to 1?" Expected `Path(ge=1)`; the right section was rank 1.
* **System answer (raw).** "… you can use the `gte` numeric validation constraint …" → 0/2
  sentences supported → abstained with the reason shown to the user.
* **Error type.** From the user's view a false abstention; from the system's view the gate
  worked: the answer was wrong (`gte` does not exist in FastAPI). The same happened for
  test-02 (`{file_path}` without `:path`), test-13 (invented denial-of-service reasoning) and
  test-18 (nonsense about immutable strings).
* **Evidence vs hypothesis.** Evidence: in all four cases the raw answer was factually wrong
  and the abstention message plus the "closest passages" list still pointed the user at the
  correct section. Hypothesis: the 0.5 B model paraphrases identifiers (`ge` → `gte`), which
  the overlap rule rightly rejects.
* **Proposed fix.** Show the top passage verbatim when abstaining after generation (the right
  answer was on screen in all four cases); try constrained prompting ("copy parameter names
  exactly as written"). Not evaluated.

## F5. Citation mapped to the wrong passage: test-09

* **Question.** return type vs `response_model` priority. Answer correct ("FastAPI uses the
  response_model parameter when both … exist"), but the citation went to
  `advanced/dataclasses.md#dataclasses-in-nested-data-structures` (rank 2) instead of
  `tutorial/response-model.md#response-model-priority` (rank 1).
* **Evidence.** Both passages contain "response_model", "return", "type", "annotation"; the
  dataclasses passage is longer and matched one more content word. **Hypothesis.** Overlap
  normalised by sentence length only; a tie-break by retrieval rank, or overlap normalised by
  passage length, would have chosen rank 1.
* **Proposed fix.** Break ties (and near-ties, within 0.1) in favour of the higher-ranked
  passage. Small change; not applied after the freeze.

## Fixes tried (validation only, after the test run; the test numbers above are unchanged)

* **Stricter support gate** `min_supported_fraction` 0.5 → 0.75 with the final configuration
  on the validation split (`python -m docqa eval-answers --split val --config configs/final.json
  --set generation.min_supported_fraction=0.75 --tag val_final_supported075`): both
  insufficient-evidence questions are still abstained (2/2, same as before), but the false
  abstention rate on answerable questions rises from 0.08 to **0.50** (6 of 12 answered) and the
  all-facts rate drops from 0.67 to 0.33. **Rejected**: the stricter gate removes correct
  answers faster than fabricated ones, because fabricated answers reuse passage vocabulary just
  as well as correct ones. This supports the conclusion that an entailment-style check, not a
  higher overlap threshold, is needed.

## Unresolved limitations

1. The lexical support check verifies vocabulary, not facts (F2, F3). An NLI-based check is the
   intended next step and is deferred to Project 2 where it can be learned from feedback.
2. The generator is small; about half of the answered questions are fully correct. Larger free
   models exist but exceed the 7.6 GB laptop used here when loaded with the rest of the stack.
3. The abstention gates were tuned on only 2 insufficient-evidence validation questions.
4. The benchmark has 36 test questions written by the team, so topic coverage reflects the
   team's reading of the docs, not real user traffic; Project 2 should log real questions.
