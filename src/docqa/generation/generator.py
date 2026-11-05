"""Answer generators.

``HFGenerator`` wraps a pretrained Hugging Face Transformer (encoder-decoder
such as Flan-T5, or an instruction-tuned decoder such as Qwen2.5-Instruct).
``ExtractiveGenerator`` needs no model at all: it returns the passage
sentences that overlap most with the question. It serves as the smoke-test
generator and as a cheap "no Transformer generator" reference point.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import Config
from ..errors import ModelUnavailableError
from ..retrieval.base import RetrievedChunk
from .grounding import content_tokens, split_sentences
from .prompt import build_chat_messages, build_prompt


@runtime_checkable
class Generator(Protocol):
    name: str
    model_name: str

    def generate(self, question: str, passages: list[RetrievedChunk]) -> str:
        ...


class ExtractiveGenerator:
    name = "extractive"
    model_name = "none (extractive)"

    def __init__(self, max_sentences: int = 2, min_words: int = 4):
        self.max_sentences = max_sentences
        self.min_words = min_words

    def generate(self, question: str, passages: list[RetrievedChunk]) -> str:
        q = content_tokens(question)
        if not q:
            return "I don't know."
        scored: list[tuple[float, int, str]] = []
        order = 0
        for p in passages:
            for sentence in split_sentences(p.chunk.text):
                order += 1
                if len(sentence.split()) < self.min_words or sentence.startswith("```"):
                    continue
                overlap = len(q & content_tokens(sentence)) / len(q)
                if overlap > 0:
                    scored.append((overlap, order, sentence))
        if not scored:
            return "I don't know."
        best = sorted(scored, key=lambda t: (-t[0], t[1]))[: self.max_sentences]
        best.sort(key=lambda t: t[1])
        return " ".join(s for _, _, s in best)


class HFGenerator:
    name = "hf"

    def __init__(self, model_name: str, max_new_tokens: int = 128, device: str = "cpu",
                 max_input_tokens: int = 1536, dtype: str = "float32", include_code: bool = True):
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ModelUnavailableError(
                "transformers/torch are not installed. Run `pip install -r requirements.txt`."
            ) from exc
        if dtype not in ("float32", "bfloat16"):
            raise ValueError("generation.dtype must be 'float32' or 'bfloat16'")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_input_tokens = max_input_tokens
        self.device = device
        self.dtype = dtype
        self.include_code = include_code
        self._torch = torch
        try:
            config = AutoConfig.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
            loader = AutoModelForSeq2SeqLM if self.is_encoder_decoder else AutoModelForCausalLM
            self.model = loader.from_pretrained(model_name, dtype=getattr(torch, dtype))
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"Could not load generator model '{model_name}': {exc}. "
                "Check the model id and network access, or set generation.generator=extractive."
            ) from exc
        self.model.to(device)
        self.model.eval()
        n_params = sum(p.numel() for p in self.model.parameters())
        self.n_parameters = int(n_params)

    def _prompt_ids(self, question: str, passages: list[RetrievedChunk]):
        tok = self.tokenizer
        if self.is_encoder_decoder or not getattr(tok, "chat_template", None):
            text = build_prompt(question, passages, include_code=self.include_code)
        else:
            text = tok.apply_chat_template(build_chat_messages(question, passages, include_code=self.include_code),
                                           tokenize=False, add_generation_prompt=True)
        return tok(text, return_tensors="pt", truncation=True, max_length=self.max_input_tokens)

    def generate(self, question: str, passages: list[RetrievedChunk]) -> str:
        torch = self._torch
        inputs = self._prompt_ids(question, passages).to(self.device)
        gen_kwargs = dict(max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1)
        if not self.is_encoder_decoder and self.tokenizer.pad_token_id is None:
            gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        with torch.no_grad():
            output = self.model.generate(**inputs, **gen_kwargs)
        if self.is_encoder_decoder:
            ids = output[0]
        else:
            ids = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()


def build_generator(cfg: Config) -> Generator:
    g = cfg.generation
    if g.generator == "extractive":
        return ExtractiveGenerator()
    if g.generator == "hf":
        return HFGenerator(g.model, max_new_tokens=g.max_new_tokens, dtype=g.dtype,
                           include_code=g.include_code_in_prompt)
    raise ValueError(f"Unknown generator '{g.generator}' (expected 'hf' or 'extractive')")


__all__ = ["Generator", "ExtractiveGenerator", "HFGenerator", "build_generator"]
