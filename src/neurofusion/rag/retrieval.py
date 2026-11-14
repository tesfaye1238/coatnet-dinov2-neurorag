"""Index building and retrieval.

Candidate settings (NOT verified historical settings): cosine similarity, top-10 candidates, top-5 passages to
generation, optional category metadata filter, no reranker. The retrieval configuration is recorded with every run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from neurofusion.rag.chunking import Tokenizer, chunk_document
from neurofusion.rag.embedding import Embedder
from neurofusion.rag.sources import SourceDocument, read_document_text

DISPLAY = {"glioma": "glioma", "meningioma": "meningioma", "notumor": "no tumor", "pituitary": "pituitary tumor"}


@dataclass
class RetrievalConfig:
    top_k_candidates: int = 10
    top_k_context: int = 5
    category_filter: bool = True
    include_confidence_in_query: bool = False
    min_score: float | None = None  # below this the passage is dropped (None = keep all)
    reranker: str = "none"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Retrieved:
    passage_id: str
    doc_id: str
    text: str
    score: float
    rank: int
    category_tags: list[str]


def build_query(category: str, confidence: float | None = None, include_confidence: bool = False) -> str:
    q = f"Brain MRI category {DISPLAY.get(category, category)}: clinical background, imaging characteristics and general management considerations."
    if include_confidence and confidence is not None:
        q += f" Classifier confidence {confidence:.2f}."
    return q


def build_index(docs: list[SourceDocument], base_dir, tokenizer: Tokenizer, embedder: Embedder, store,
                max_tokens: int = 512, overlap: int = 64) -> dict:
    """Chunk, embed and upsert documents. Returns counts (source documents vs chunks) and effective chunk lengths."""
    from pathlib import Path

    n_chunks, lens = 0, []
    for d in docs:
        text = read_document_text(Path(base_dir) / d.path)
        chunks = chunk_document(d.doc_id, text, tokenizer, max_tokens, overlap)
        if not chunks:
            continue
        vecs = embedder.embed_documents([c.text for c in chunks])
        store.upsert(
            [c.chunk_id for c in chunks], vecs,
            [{"doc_id": d.doc_id, "text": c.text, "category_tags": list(d.category_tags), "title": d.title,
              "version_date": d.version_date, "n_tokens": c.n_tokens, "review_status": d.review_status} for c in chunks],
        )
        n_chunks += len(chunks)
        lens += [c.n_tokens_with_special for c in chunks]
    return {"source_documents": len(docs), "chunks": n_chunks, "chunk_tokens_min": min(lens) if lens else 0,
            "chunk_tokens_max": max(lens) if lens else 0, "max_tokens": max_tokens, "overlap": overlap,
            "embedder": embedder.describe(), "indexed_in_store": store.count()}


class Retriever:
    def __init__(self, store, embedder: Embedder, cfg: RetrievalConfig | None = None):
        self.store, self.embedder, self.cfg = store, embedder, cfg or RetrievalConfig()

    def retrieve(self, category: str, confidence: float | None = None) -> tuple[str, list[Retrieved]]:
        query = build_query(category, confidence, self.cfg.include_confidence_in_query)
        q = self.embedder.embed_queries([query])[0]
        hits = self.store.search(q, self.cfg.top_k_candidates, category if self.cfg.category_filter else None)
        out = []
        for h in hits:
            if self.cfg.min_score is not None and h.score < self.cfg.min_score:
                continue
            out.append(Retrieved(h.chunk_id, h.payload["doc_id"], h.payload["text"], h.score, len(out) + 1, h.payload.get("category_tags", [])))
            if len(out) >= self.cfg.top_k_context:
                break
        return query, out
