"""Hashing and fingerprint helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def md5_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()  # noqa: S324 - used only to verify third-party published md5 sums
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode())[:16]


def split_fingerprint(df: pd.DataFrame) -> str:
    """Hash of the (image_id, split) assignment, independent of row order."""
    rows = sorted(zip(df["image_id"].astype(str), df["split"].astype(str)))
    return sha256_bytes(canonical_json(rows).encode())[:16]


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Hash of image identity and labels (file hashes), independent of row order and split."""
    rows = sorted(zip(df["image_id"].astype(str), df["file_sha256"].astype(str), df["label"].astype(str)))
    return sha256_bytes(canonical_json(rows).encode())[:16]
