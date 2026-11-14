"""Duplicate screening.

Exact file hash and decoded-pixel hash identify duplicates. Perceptual similarity only produces *candidate*
pairs for manual review; it does not prove common patient identity. No file is ever deleted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(path: str | Path, size: int = 8) -> np.ndarray:
    """Difference hash as a packed uint8 array (size*size bits)."""
    with Image.open(path) as im:
        g = im.convert("L").resize((size + 1, size), Image.LANCZOS)
        a = np.asarray(g, dtype=np.float32)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return np.packbits(bits)


def phash(path: str | Path, size: int = 8, highfreq: int = 4) -> np.ndarray:
    """DCT perceptual hash as a packed uint8 array."""
    n = size * highfreq
    with Image.open(path) as im:
        g = im.convert("L").resize((n, n), Image.LANCZOS)
        a = np.asarray(g, dtype=np.float32)
    d = dctn(a, norm="ortho")[:size, :size]
    bits = (d > np.median(d)).flatten()
    return np.packbits(bits)


HASHERS = {"dhash": dhash, "phash": phash}


def hamming_matrix_block(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Hamming distances between packed-bit rows of ``a`` and ``b``."""
    x = np.bitwise_xor(a[:, None, :], b[None, :, :])
    return _POPCOUNT[x].sum(axis=2)


def exact_duplicate_groups(df: pd.DataFrame, column: str = "file_sha256") -> pd.DataFrame:
    """Rows sharing an identical hash (file or decoded pixel). Returns a long table with a group number."""
    dup = df[df.duplicated(column, keep=False)].copy()
    if dup.empty:
        return pd.DataFrame(columns=["dup_group", "kind", "image_id", "label", "source_id"])
    dup["dup_group"] = dup.groupby(column).ngroup()
    dup["kind"] = "exact_file" if column == "file_sha256" else "exact_decoded"
    return dup[["dup_group", "kind", "image_id", "label", "source_id"]].sort_values(["dup_group", "image_id"])


def perceptual_candidate_pairs(
    df: pd.DataFrame,
    root: str | Path,
    method: str = "dhash",
    max_hamming: int = 6,
    block: int = 512,
) -> pd.DataFrame:
    """Candidate near-duplicate pairs (for manual review). O(n^2) in blocks; fine up to ~20k images."""
    root = Path(root)
    hasher = HASHERS[method]
    ids = df["image_id"].tolist()
    hashes = np.stack([hasher(root / rp) for rp in df["relative_path"]])
    meta = df.set_index("image_id")
    rows = []
    for i0 in range(0, len(ids), block):
        a = hashes[i0 : i0 + block]
        dist = hamming_matrix_block(a, hashes)
        for ii in range(a.shape[0]):
            gi = i0 + ii
            js = np.where(dist[ii] <= max_hamming)[0]
            for j in js:
                if j <= gi:
                    continue
                ia, ib = ids[gi], ids[j]
                ra, rb = meta.loc[ia], meta.loc[ib]
                rows.append(
                    {
                        "kind": f"perceptual_{method}",
                        "image_id_a": ia,
                        "image_id_b": ib,
                        "hamming": int(dist[ii, j]),
                        "label_a": ra["label"],
                        "label_b": rb["label"],
                        "source_a": ra["source_id"],
                        "source_b": rb["source_id"],
                        "cross_source": ra["source_id"] != rb["source_id"],
                        "label_conflict": ra["label"] != rb["label"],
                        "decision": "pending",
                    }
                )
    return pd.DataFrame(
        rows,
        columns=[
            "kind", "image_id_a", "image_id_b", "hamming", "label_a", "label_b", "source_a", "source_b",
            "cross_source", "label_conflict", "decision",
        ],
    )


def export_candidates(pairs: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(path, index=False)
    return path


def resolve_exact_duplicates(df: pd.DataFrame, column: str = "file_sha256") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the lexicographically first image_id of each exact-duplicate set.

    Returns ``(retained_df, decision_log)``. The log records every removed image and the retained one, so the
    decision is auditable. The input files are untouched.
    """
    ordered = df.sort_values("image_id")
    keep = ordered.drop_duplicates(column, keep="first")
    first = ordered.groupby(column)["image_id"].first()
    removed = ordered[~ordered["image_id"].isin(keep["image_id"])]
    log = pd.DataFrame(
        {
            "removed_image_id": removed["image_id"].values,
            "retained_image_id": removed[column].map(first).values,
            "reason": f"exact_{column}",
            "label_conflict": [
                df.set_index("image_id").loc[r, "label"] != df.set_index("image_id").loc[k, "label"]
                for r, k in zip(removed["image_id"], removed[column].map(first))
            ],
        }
    )
    return keep.sort_index(), log


def apply_pair_decisions(df: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply reviewed decisions: ``keep_a`` | ``keep_b`` | ``keep_both`` | ``pending``. ``pending`` keeps both."""
    drop: dict[str, str] = {}
    for r in decisions.itertuples():
        if r.decision == "keep_a":
            drop[r.image_id_b] = r.image_id_a
        elif r.decision == "keep_b":
            drop[r.image_id_a] = r.image_id_b
    log = pd.DataFrame({"removed_image_id": list(drop), "retained_image_id": list(drop.values()), "reason": "reviewed_pair"})
    return df[~df["image_id"].isin(drop)].copy(), log
