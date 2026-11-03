# Evaluation question set

`questions.json` holds 58 hand-written, hand-verified questions over the FastAPI documentation
at commit `5cb6272e` (see `../documents/fastapi/SOURCE.md`).

| split | answerable | insufficient evidence | purpose |
|---|---|---|---|
| dev | 8 | 0 | developing and debugging code |
| val | 12 | 2 | choosing retriever, chunking, top-k, thresholds |
| test | 31 | 5 | reported once with the frozen configuration |

Schema, labelling protocol, split rules and metric definitions are documented in
`../../reports/task_definition.md`. Consistency with the processed corpus (every
`file#anchor` label exists, no near-duplicate questions across splits) is checked by:

```bash
python -m docqa check-questions
```

The same check runs as a unit test (`tests/test_questions.py`). Do not edit test questions
after reading test results; add a new, untouched test set instead and say so in
`reports/results.md`.
