"""Vector stores: a dependency-free NumPy store (tests, small demos) and a Qdrant store."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NS = uuid.UUID("6f1a0a58-6a6e-4a7e-9d33-5c1c6f3a9a11")


@dataclass
class Hit:
    chunk_id: str
    score: float
    payload: dict


class NumpyStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.ids: list[str] = []
        self.payloads: list[dict] = []
        self.mat = np.zeros((0, dim), dtype=np.float32)

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        if vectors.shape[1] != self.dim:
            raise ValueError(f"vector dim {vectors.shape[1]} != store dim {self.dim}")
        for i, cid in enumerate(ids):
            if cid in self.ids:
                j = self.ids.index(cid)
                self.mat[j], self.payloads[j] = vectors[i], payloads[i]
            else:
                self.ids.append(cid)
                self.payloads.append(payloads[i])
                self.mat = np.vstack([self.mat, vectors[i : i + 1]])

    def count(self) -> int:
        return len(self.ids)

    def search(self, q: np.ndarray, top_k: int, category: str | None = None) -> list[Hit]:
        if not self.ids:
            return []
        scores = self.mat @ q.astype(np.float32)  # cosine when both sides are L2-normalised
        mask = np.ones(len(self.ids), bool)
        if category:
            mask = np.array([category in p.get("category_tags", []) for p in self.payloads])
        idx = [i for i in np.argsort(-scores) if mask[i]][:top_k]
        return [Hit(self.ids[i], float(scores[i]), self.payloads[i]) for i in idx]

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.mat)
        (d / "meta.json").write_text(json.dumps({"dim": self.dim, "ids": self.ids, "payloads": self.payloads}))

    @classmethod
    def load(cls, directory: str | Path) -> NumpyStore:
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        s = cls(meta["dim"])
        s.ids, s.payloads, s.mat = meta["ids"], meta["payloads"], np.load(d / "vectors.npy")
        return s


class QdrantStore:
    """Qdrant wrapper. ``location`` = ':memory:', a local path, or an http(s) URL."""

    def __init__(self, dim: int, location: str = ":memory:", collection: str = "neurorag", recreate: bool = True):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
        except ImportError as exc:  # pragma: no cover
            raise ImportError("qdrant-client is required: pip install 'neurofusion[rag]'") from exc
        self.dim, self.collection = dim, collection
        if location == ":memory:":
            self.client = QdrantClient(":memory:")
        elif location.startswith("http"):
            self.client = QdrantClient(url=location)
        else:
            self.client = QdrantClient(path=location)
        exists = self.client.collection_exists(collection)
        if exists and recreate:
            self.client.delete_collection(collection)
            exists = False
        if not exists:
            self.client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def upsert(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        pts = [
            PointStruct(id=str(uuid.uuid5(_NS, cid)), vector=vectors[i].tolist(), payload={**payloads[i], "chunk_id": cid})
            for i, cid in enumerate(ids)
        ]
        self.client.upsert(self.collection, points=pts)

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def search(self, q: np.ndarray, top_k: int, category: str | None = None) -> list[Hit]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        flt = Filter(must=[FieldCondition(key="category_tags", match=MatchAny(any=[category]))]) if category else None
        res = self.client.query_points(self.collection, query=q.tolist(), limit=top_k, query_filter=flt, with_payload=True)
        return [Hit(p.payload["chunk_id"], float(p.score), p.payload) for p in res.points]
