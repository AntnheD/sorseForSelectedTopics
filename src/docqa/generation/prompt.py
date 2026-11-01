"""Prompt construction for grounded answer generation.

The instructions tell the model to use only the supplied excerpts and to
say "I don't know" otherwise. Prompt wording alone is not treated as proof of
grounding: ``grounding.py`` checks every answer sentence against the
passages afterwards.
"""

from __future__ import annotations

import re

from ..retrieval.base import RetrievedChunk

INSTRUCTIONS = (
    "Answer the question using only the documentation excerpts below. "
    "Be concise: one to three sentences. "
    "Do not invent function names, parameters, versions or behaviour that the excerpts do not state. "
    "If the excerpts do not contain the answer, reply exactly: I don't know."
)


CODE_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)


def truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def strip_code_blocks(text: str) -> str:
    """Replace fenced code blocks with a marker (prompt only; citations keep the full passage)."""
    return CODE_BLOCK_RE.sub("[code example omitted]", text)


def format_passages(passages: list[RetrievedChunk], max_words_per_passage: int = 220,
                    include_code: bool = True) -> str:
    blocks = []
    for i, p in enumerate(passages, start=1):
        header = f"[{i}] Source: {p.chunk.source} | {p.chunk.heading_path}"
        text = p.chunk.text if include_code else strip_code_blocks(p.chunk.text)
        blocks.append(f"{header}\n{truncate_words(text, max_words_per_passage)}")
    return "\n\n".join(blocks)


def build_prompt(question: str, passages: list[RetrievedChunk], include_code: bool = True) -> str:
    """Single-string prompt for encoder-decoder models such as Flan-T5."""
    return (
        f"{INSTRUCTIONS}\n\n"
        f"Documentation excerpts:\n{format_passages(passages, include_code=include_code)}\n\n"
        f"Question: {question.strip()}\nAnswer:"
    )


def build_chat_messages(question: str, passages: list[RetrievedChunk], include_code: bool = True) -> list[dict]:
    """Chat-format prompt for instruction-tuned decoder models."""
    return [
        {"role": "system", "content": INSTRUCTIONS},
        {
            "role": "user",
            "content": (f"Documentation excerpts:\n{format_passages(passages, include_code=include_code)}\n\n"
                        f"Question: {question.strip()}"),
        },
    ]


__all__ = ["INSTRUCTIONS", "build_prompt", "build_chat_messages", "format_passages", "truncate_words",
           "strip_code_blocks"]
