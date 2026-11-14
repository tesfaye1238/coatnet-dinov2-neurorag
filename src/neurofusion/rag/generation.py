"""Generators. ``QwenGenerator`` is the configurable LLM backend (candidate checkpoint Qwen/Qwen2.5-7B-Instruct,
NOT confirmed as the manuscript's checkpoint). ``ExtractiveGenerator`` is a deterministic no-LLM baseline."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Protocol

from neurofusion.rag.retrieval import DISPLAY, Retrieved


class GeneratorUnavailable(RuntimeError):
    pass


@dataclass
class GenerationConfig:
    model_id: str = "Qwen/Qwen2.5-7B-Instruct"  # candidate checkpoint, unconfirmed
    revision: str | None = None
    device: str = "auto"
    dtype: str = "bfloat16"
    quantization: str | None = None  # None | "bnb-4bit" | "bnb-8bit" (documented when used)
    do_sample: bool = False
    max_new_tokens: int = 512
    context_token_budget: int = 3000

    def as_dict(self) -> dict:
        return asdict(self)


class Generator(Protocol):
    def generate(self, system: str, user: str) -> str: ...
    def describe(self) -> dict: ...


class QwenGenerator:
    def __init__(self, cfg: GenerationConfig | None = None):
        self.cfg = cfg or GenerationConfig()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise GeneratorUnavailable("transformers is required: pip install 'neurofusion[rag]'") from exc
        kw: dict = {"revision": self.cfg.revision}
        if self.cfg.quantization:
            from transformers import BitsAndBytesConfig

            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=self.cfg.quantization == "bnb-4bit", load_in_8bit=self.cfg.quantization == "bnb-8bit")
        else:
            kw["torch_dtype"] = getattr(torch, self.cfg.dtype)
        try:
            self.tok = AutoTokenizer.from_pretrained(self.cfg.model_id, revision=self.cfg.revision)
            self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, device_map=self.cfg.device, **kw)
        except Exception as exc:  # noqa: BLE001
            raise GeneratorUnavailable(f"could not load {self.cfg.model_id}: {exc}") from exc

    def generate(self, system: str, user: str) -> str:
        import torch

        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        if ids.shape[1] > self.cfg.context_token_budget:
            raise GeneratorUnavailable(f"prompt has {ids.shape[1]} tokens > context_token_budget {self.cfg.context_token_budget}; reduce passages (no silent truncation)")
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=self.cfg.max_new_tokens, do_sample=self.cfg.do_sample)
        return self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def describe(self) -> dict:
        return {"generator": "qwen", **self.cfg.as_dict(), "revision_note": "unpinned" if not self.cfg.revision else "pinned"}


def first_sentence(text: str) -> str:
    s = re.split(r"(?<=[.!?])\s+", text.strip())[0]
    return s


class ExtractiveGenerator:
    """Deterministic, extractive: each summary statement is the first sentence of a retrieved passage, with its id.
    It contains no model knowledge and cannot hallucinate beyond the passages; it is a software baseline."""

    def __init__(self, category: str):
        self.category = category

    def generate(self, system: str, user: str) -> str:
        ids = re.findall(r'<passage id="([^"]+)">\n(.*?)\n</passage>', user, re.S)
        disp = DISPLAY.get(self.category, self.category)
        if not ids:
            return json.dumps({"predicted_category": disp, "general_background": "", "contextual_summary": [],
                               "evidence_limitations": "No reference passages were available; evidence is insufficient.",
                               "references_used": []})
        summary = [{"statement": first_sentence(t), "passage_ids": [i]} for i, t in ids]
        return json.dumps({
            "predicted_category": disp,
            "general_background": f"The image classifier assigned the category '{disp}'. This is a model output, not a confirmed diagnosis.",
            "contextual_summary": summary,
            "evidence_limitations": "Statements are extracted from retrieved passages; support and clinical appropriateness are not verified.",
            "references_used": [i for i, _ in ids],
        })

    def describe(self) -> dict:
        return {"generator": "extractive-template (no LLM)"}


def no_retrieval_prompt(category: str, confidence: float | None) -> str:
    """User prompt for the closed-book comparison (same system prompt, zero passages)."""
    conf = "not supplied" if confidence is None else f"{confidence:.3f}"
    return f"Predicted image category: {DISPLAY.get(category, category)}\nClassifier confidence: {conf}\n\n(no passages were retrieved)\n"


def retrieved_ids(passages: list[Retrieved]) -> list[str]:
    return [p.passage_id for p in passages]
