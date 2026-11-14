"""Embedding backends. ``HashingEmbedder`` is a deterministic, NON-SEMANTIC stand-in for tests and demos."""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "  # BGE v1.5 model card


class Embedder(Protocol):
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_queries(self, texts: list[str]) -> np.ndarray: ...
    def describe(self) -> dict: ...


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


class HashingEmbedder:
    """Signed feature hashing of unigrams+bigrams. Lexical overlap only - it has no semantic knowledge."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        toks = re.findall(r"[a-z0-9]+", text.lower())
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _l2(np.stack([self._vec(t) for t in texts])) if texts else np.zeros((0, self.dim), np.float32)

    embed_queries = embed_documents

    def describe(self) -> dict:
        return {"name": "hashing-embedder", "revision": "n/a", "dim": self.dim, "normalized": True, "semantic": False,
                "query_instruction": "", "note": "deterministic lexical baseline for tests/demo; not a semantic model"}


class SentenceTransformerEmbedder:
    """sentence-transformers wrapper. Records the model id and requested revision."""

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", revision: str | None = None,
                 query_instruction: str = BGE_QUERY_INSTRUCTION, device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError("sentence-transformers is required: pip install 'neurofusion[rag]'") from exc
        self.model_name, self.revision, self.query_instruction = model_name, revision, query_instruction
        self.model = SentenceTransformer(model_name, revision=revision, device=device)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.max_seq_length = int(self.model.max_seq_length)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents([self.query_instruction + t for t in texts])

    def describe(self) -> dict:
        return {"name": self.model_name, "revision": self.revision or "unpinned (latest at download time)", "dim": self.dim,
                "normalized": True, "semantic": True, "query_instruction": self.query_instruction,
                "max_seq_length": self.max_seq_length}
