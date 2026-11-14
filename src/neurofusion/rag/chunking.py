"""Tokenizer-aware chunking. Passages are never silently truncated."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


class ChunkLengthError(ValueError):
    pass


class Tokenizer(Protocol):
    max_model_tokens: int
    num_special_tokens: int

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]: ...


class WhitespaceTokenizer:
    """Word-level tokenizer for tests and the demo. Not a model tokenizer."""

    def __init__(self, max_model_tokens: int = 100_000, num_special_tokens: int = 0):
        self.max_model_tokens, self.num_special_tokens = max_model_tokens, num_special_tokens

    def encode_with_offsets(self, text: str):
        offs = [m.span() for m in re.finditer(r"\S+", text)]
        return list(range(len(offs))), offs


class HFTokenizerAdapter:
    """Adapter for a HuggingFace *fast* tokenizer (offset mapping required)."""

    def __init__(self, tokenizer, max_model_tokens: int | None = None):
        if not getattr(tokenizer, "is_fast", False):
            raise ValueError("a fast tokenizer (with offset mappings) is required")
        self.tok = tokenizer
        self.max_model_tokens = max_model_tokens or int(tokenizer.model_max_length)
        self.num_special_tokens = len(tokenizer.build_inputs_with_special_tokens([]))

    def encode_with_offsets(self, text: str):
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    index: int
    text: str
    n_tokens: int
    n_tokens_with_special: int
    char_start: int
    char_end: int


def chunk_document(doc_id: str, text: str, tok: Tokenizer, max_tokens: int = 512, overlap: int = 64) -> list[Chunk]:
    """Windows of at most ``max_tokens`` model tokens INCLUDING special tokens, with ``overlap`` tokens shared."""
    if max_tokens > tok.max_model_tokens:
        raise ChunkLengthError(f"max_tokens {max_tokens} exceeds the model limit {tok.max_model_tokens}")
    window = max_tokens - tok.num_special_tokens
    if window <= 0 or overlap < 0 or overlap >= window:
        raise ChunkLengthError(f"overlap {overlap} must satisfy 0 <= overlap < window ({window})")
    ids, offs = tok.encode_with_offsets(text)
    n = len(ids)
    chunks: list[Chunk] = []
    if n == 0:
        return chunks
    stride = window - overlap
    start, k = 0, 0
    while True:
        end = min(start + window, n)
        cs, ce = offs[start][0], offs[end - 1][1]
        c = Chunk(f"{doc_id}::c{k:04d}", doc_id, k, text[cs:ce], end - start, end - start + tok.num_special_tokens, cs, ce)
        if c.n_tokens_with_special > tok.max_model_tokens:  # defensive; cannot trigger given the window
            raise ChunkLengthError(f"{c.chunk_id} has {c.n_tokens_with_special} tokens > {tok.max_model_tokens}")
        chunks.append(c)
        if end >= n:
            break
        start += stride
        k += 1
    return chunks
