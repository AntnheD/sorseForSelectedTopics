"""Generation: prompts, Transformer generator, extractive fallback and grounding."""

from .generator import ExtractiveGenerator, Generator, HFGenerator, build_generator
from .grounding import (
    Citation,
    SentenceSupport,
    ground_answer,
    should_abstain_after,
    should_abstain_before,
    split_sentences,
    support_score,
)
from .prompt import build_chat_messages, build_prompt, format_passages

__all__ = [
    "Generator",
    "ExtractiveGenerator",
    "HFGenerator",
    "build_generator",
    "Citation",
    "SentenceSupport",
    "ground_answer",
    "should_abstain_after",
    "should_abstain_before",
    "split_sentences",
    "support_score",
    "build_prompt",
    "build_chat_messages",
    "format_passages",
]
