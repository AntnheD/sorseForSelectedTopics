"""Retrieval metrics with explicit definitions.

Relevance labels are strings of the form ``"tutorial/query-params.md"`` or
``"tutorial/query-params.md#optional-parameters"``. A retrieved chunk is
described by its *relevance key* at the chosen level:

* level ``"source"``: key = source path; a label matches when its source
  path equals the key (its ``#anchor``, if any, is ignored).
* level ``"section"``: key = ``source#anchor``; a label with an anchor must
  match exactly, a label without an anchor matches any section of that file.

For one question with ranked keys ``r_1 .. r_n`` and relevant labels ``L``:

* ``HitRate@k`` = 1 if any of ``r_1 .. r_k`` matches a label, else 0.
  ``Top-1 accuracy`` is ``HitRate@1``.
* ``Recall@k``  = |labels in L matched by some r_i, i <= k| / |L|.
* ``MRR@k``     = 1 / (rank of the first matching r_i) if that rank <= k, else 0.

All three are averaged over the evaluated questions. Questions labelled
``insufficient_evidence`` have no relevant labels and are excluded from
retrieval metrics (the denominator is the number of answerable questions).
"""

from __future__ import annotations

from typing import Optional

LEVELS = ("source", "section")


def normalise_label(label: str, level: str) -> str:
    if level == "source":
        return label.split("#", 1)[0]
    if level == "section":
        return label
    raise ValueError(f"Unknown relevance level '{level}'")


def matches(key: str, label: str, level: str) -> bool:
    label = normalise_label(label, level)
    if level == "source":
        return key.split("#", 1)[0] == label
    if "#" in label:
        return key == label
    return key.split("#", 1)[0] == label


def relevant_labels(labels: list[str], level: str) -> list[str]:
    seen: list[str] = []
    for label in labels:
        norm = normalise_label(label, level)
        if norm not in seen:
            seen.append(norm)
    return seen


def first_relevant_rank(keys: list[str], labels: list[str], level: str) -> Optional[int]:
    rel = relevant_labels(labels, level)
    for rank, key in enumerate(keys, start=1):
        if any(matches(key, label, level) for label in rel):
            return rank
    return None


def hit_rate_at_k(keys: list[str], labels: list[str], level: str, k: int) -> float:
    rank = first_relevant_rank(keys, labels, level)
    return 1.0 if rank is not None and rank <= k else 0.0


def recall_at_k(keys: list[str], labels: list[str], level: str, k: int) -> float:
    rel = relevant_labels(labels, level)
    if not rel:
        return 0.0
    found = {label for label in rel if any(matches(key, label, level) for key in keys[:k])}
    return len(found) / len(rel)


def mrr_at_k(keys: list[str], labels: list[str], level: str, k: int) -> float:
    rank = first_relevant_rank(keys, labels, level)
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def evaluate_rankings(ranked_keys: dict[str, list[str]], labels: dict[str, list[str]],
                      k: int, level: str) -> dict:
    """Aggregate metrics over questions. ``ranked_keys[qid]`` is the list of
    relevance keys of the retrieved chunks in rank order."""
    per_question = {}
    for qid, keys in ranked_keys.items():
        q_labels = labels[qid]
        per_question[qid] = {
            "first_relevant_rank": first_relevant_rank(keys, q_labels, level),
            f"hit_rate@{k}": hit_rate_at_k(keys, q_labels, level, k),
            f"recall@{k}": recall_at_k(keys, q_labels, level, k),
            f"mrr@{k}": mrr_at_k(keys, q_labels, level, k),
            "top1": hit_rate_at_k(keys, q_labels, level, 1),
        }
    n = len(per_question)

    def mean(field: str) -> float:
        return round(sum(v[field] for v in per_question.values()) / n, 4) if n else 0.0

    summary = {
        "n_questions": n,
        "k": k,
        "relevance_level": level,
        f"recall@{k}": mean(f"recall@{k}"),
        f"hit_rate@{k}": mean(f"hit_rate@{k}"),
        f"mrr@{k}": mean(f"mrr@{k}"),
        "top1_accuracy": mean("top1"),
    }
    return {"summary": summary, "per_question": per_question}


__all__ = [
    "LEVELS",
    "matches",
    "normalise_label",
    "relevant_labels",
    "first_relevant_rank",
    "hit_rate_at_k",
    "recall_at_k",
    "mrr_at_k",
    "evaluate_rankings",
]
