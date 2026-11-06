"""Fill the manual-review CSV for the final test run and print the manual aggregates.

The scores below were assigned by a person reading every prediction in
reports/predictions/test_final.jsonl against the cited documentation sections
(rubric in reports/task_definition.md): 1 = yes, 0.5 = partly, 0 = no, blank = not applicable.

    python scripts/apply_manual_review.py            # writes reports/predictions/test_final_manual_review.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "reports/predictions/test_final.jsonl"
OUT = ROOT / "reports/predictions/test_final_manual_review.csv"

# id: (correct, complete, evidence_supported, citations_valid, abstention_ok, error_type, notes)
SCORES = {
    "test-01": (0.5, 1, 0.5, 1, 1, "unsupported claim", "adds 'or specify a valid value', which the docs do not say"),
    "test-02": (0, 0, 0, "", 0, "false abstention (hallucination caught)", "model omitted the :path convertor and wrote {file_path}; support gate abstained"),
    "test-03": (0, 0, 0.5, 0.5, 1, "incorrect claim", "says the alias is item_query; docs: alias='item-query' is the external name"),
    "test-04": (0, 0, 1, 1, 1, "non-answer", "restates the question; never says 'list' or 'Query'"),
    "test-05": (1, 1, 1, 0.5, 1, "citation error", "answer right; cites deprecate-a-path-operation instead of the Query section"),
    "test-06": (0.5, 0.5, 0.5, 0.5, 1, "unsupported claim", "== is right; invents 'fastapi==latest_version' steps; two extra citations"),
    "test-07": (0, 0, 0, "", 0, "false abstention (hallucination caught)", "model invented 'gte' (docs: ge); support gate abstained"),
    "test-08": (0.5, 0.5, 0.5, 1, 1, "unsupported claim", "names Body, then invents Depends-based code"),
    "test-09": (1, 1, 1, 0, 1, "citation error", "correct answer mapped to the dataclasses passage (lexical overlap)"),
    "test-10": (1, 1, 0.5, 1, 1, "-", "second sentence is filler"),
    "test-12": (0, 0, 0, 0, 1, "incorrect claim", "HTTPException instead of a RequestValidationError handler; cited section does not support it"),
    "test-13": (0, 0, 0, "", 0, "false abstention (hallucination caught)", "invented denial-of-service reasoning; support gate abstained"),
    "test-14": (1, 0.5, 1, 1, 1, "incomplete", "omits keyword arguments"),
    "test-15": (0.5, 0.5, 0.5, 1, 1, "unsupported claim", "first sentence right; the APIRouter sentence is invented"),
    "test-16": (0, 0, 0, 0, 1, "incorrect claim", "invents a FastAPIMiddleware base class; keyword proxy false positive"),
    "test-17": (1, 1, 1, 1, 1, "-", ""),
    "test-18": (0, 0, 0, "", 0, "false abstention (hallucination caught)", "nonsense about immutable strings; support gate abstained"),
    "test-20": (1, 1, 0.5, 1, 1, "-", "second sentence vague but harmless"),
    "test-22": (1, 1, 0.5, 0.5, 1, "unsupported claim", "parameter right; invents 'based on the request method'; second citation irrelevant"),
    "test-23": (0, 0, 0.5, 0.5, 1, "incorrect claim", "says 'Yes' where the docs say you cannot; keyword proxy false positive"),
    "test-24": (0, 0, 0, 0, 1, "incorrect claim", "invents a header-setting middleware; never names GZipMiddleware"),
    "test-25": (0.5, 1, 0.5, 1, 1, "unsupported claim", "include_in_schema=False right; invented parameter name around it"),
    "test-26": (0.5, 0.5, 0.5, 0.5, 1, "incorrect claim", "docs_url right; redoc_url=None wrong; CDN steps invented"),
    "test-27": (0, 0, 0.5, 1, 1, "incorrect claim", "swaps 401 and 403; keyword proxy false positive (both numbers present)"),
    "test-28": (1, 0.5, 1, 1, 1, "incomplete", "omits reading openapi_url from Pydantic settings"),
    "test-29": (0, 0, 0.5, 1, 1, "incorrect claim + ranking", "'prefer async def' contradicts the TL;DR; the in-a-hurry section was not in the top 3"),
    "test-30": (1, 1, 1, 1, 1, "-", ""),
    "test-31": (1, 1, 1, 1, 1, "-", ""),
    "test-32": (1, 0.5, 1, 1, 1, "incomplete", "omits the deprecation note (minor)"),
    "test-33": (1, 1, 1, 1, 1, "-", ""),
    "test-34": (0.5, 0, 0.5, 0.5, 1, "ranking error + incomplete", "exclude_unset section not in the top 3; generic steps instead"),
    "test-ie-01": (0, 0, 0, 0, 0, "failure to abstain", "invents a 5 MB limit and cites an unrelated passage"),
    "test-ie-02": (0, 0, 0, 0, 0, "failure to abstain", "invents HTTP/3 support 'in 0.128.0'"),
    "test-ie-03": (0, 0, 0, 0, 0, "failure to abstain", "invents OTLP/Prometheus steps from the OpenTelemetry page"),
    "test-ie-04": ("", "", "", "", 1, "-", "abstained: 0/6 sentences supported"),
    "test-ie-05": ("", "", "", "", 1, "-", "abstained: retrievers disagree on the top passage"),
}
COLUMNS = ["id", "answerability", "question", "expected_answer_points", "answer", "citations",
           "correct", "complete", "evidence_supported", "citations_valid", "abstention_ok", "error_type", "notes"]


def main() -> int:
    rows = [json.loads(l) for l in open(PRED, encoding="utf-8")]
    out = []
    for r in rows:
        p = r["prediction"]
        s = SCORES[r["id"]]
        out.append({
            "id": r["id"], "answerability": r["answerability"], "question": p["question"],
            "expected_answer_points": " | ".join(r["expected_answer_points"]), "answer": p["answer"],
            "citations": "; ".join(f"{c['source']}#{c['section_anchor']}" for c in p["citations"]),
            "correct": s[0], "complete": s[1], "evidence_supported": s[2], "citations_valid": s[3],
            "abstention_ok": s[4], "error_type": s[5], "notes": s[6],
        })
    with open(OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(out)

    def mean(col, subset):
        vals = [float(x[col]) for x in subset if x[col] != ""]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    answerable = [x for x in out if x["answerability"] == "answerable"]
    answered = [x for x in answerable if not x["error_type"].startswith("false abstention")]
    insufficient = [x for x in out if x["answerability"] != "answerable"]
    print(f"manual review written to {OUT}")
    print(f"answerable n={len(answerable)}  answered n={len(answered)}  insufficient n={len(insufficient)}")
    for col in ("correct", "complete", "evidence_supported", "citations_valid"):
        m, n = mean(col, answered)
        print(f"  {col:<20} mean over answered = {m:.3f} (n={n})")
    m, n = mean("correct", answerable)
    print(f"  correct mean over all answerable (abstentions count 0) = {m:.3f} (n={n})")
    print(f"  fully correct (1.0): {sum(1 for x in answerable if x['correct'] == 1)}/{len(answerable)}")
    print(f"  partly correct (0.5): {sum(1 for x in answerable if x['correct'] == 0.5)}/{len(answerable)}")
    print(f"  wrong but answered (0): {sum(1 for x in answered if x['correct'] == 0)}/{len(answerable)}")
    print(f"  false abstentions: {len(answerable) - len(answered)}/{len(answerable)}")
    m, n = mean("abstention_ok", insufficient)
    print(f"  insufficient-evidence questions abstained: {m:.2f} ({int(m * n)}/{n})")
    from collections import Counter
    print("  error types:", dict(Counter(x["error_type"] for x in out if x["error_type"] != "-")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
