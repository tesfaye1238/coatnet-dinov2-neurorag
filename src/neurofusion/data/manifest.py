"""Image manifest: build, validate, save and load.

Missing patient identifiers stay missing. ``patient_id_verified`` is True only when a patient identifier
is present *and* was supplied by the source (never synthesised).
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from neurofusion.constants import CLASS_NAMES, IMAGE_EXTENSIONS, MANIFEST_COLUMNS, SPLIT_ALIASES, SPLITS
from neurofusion.data.labels import normalize_label
from neurofusion.utils.hashing import sha256_file


class ManifestError(ValueError):
    """Raised when a manifest violates an integrity rule."""


def decoded_image_hash(path: str | Path) -> tuple[str, int, int]:
    """SHA-256 of the decoded RGB pixel array (plus shape). Detects re-encoded exact pixel duplicates."""
    with Image.open(path) as im:
        rgb = im.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest(), arr.shape[1], arr.shape[0]


def iter_image_files(root: str | Path):
    root = Path(root)
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def build_manifest_from_folder(
    root: str | Path,
    source_id: str,
    source_release: str = "unspecified",
    label_fn: Callable[[Path], str] | None = None,
    patient_id_fn: Callable[[Path], str | None] | None = None,
    patient_ids_verified: bool = False,
) -> pd.DataFrame:
    """Scan ``root`` (class sub-folders by default) and build a manifest. Raw files are only read.

    ``patient_id_fn`` may return a patient identifier supplied by the source. Only set
    ``patient_ids_verified=True`` when the source documents those identifiers as real patient IDs.
    """
    root = Path(root)
    if not root.is_dir():
        raise ManifestError(f"Image root {root} is not a directory")
    rows = []
    for p in iter_image_files(root):
        rel = p.relative_to(root).as_posix()
        label = normalize_label(label_fn(p) if label_fn else p.parent.name)
        pid = patient_id_fn(p) if patient_id_fn else None
        pid = "" if pid is None else str(pid)
        dec_hash, w, h = decoded_image_hash(p)
        rows.append(
            {
                "image_id": f"{source_id}/{rel}",
                "relative_path": rel,
                "source_id": source_id,
                "source_release": source_release,
                "label": label,
                "patient_id": pid,
                "patient_id_verified": bool(pid) and patient_ids_verified,
                "group_id": f"img:{source_id}/{rel}",
                "grouping_method": "singleton",
                "file_sha256": sha256_file(p),
                "decoded_image_hash": dec_hash,
                "width": w,
                "height": h,
                "split": "",
            }
        )
    if not rows:
        raise ManifestError(f"No images with extensions {IMAGE_EXTENSIONS} found under {root}")
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))


def normalise_split_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["split"] = out["split"].fillna("").astype(str).str.strip().str.lower().replace(SPLIT_ALIASES)
    return out


def validate_manifest(df: pd.DataFrame, require_split: bool = False) -> None:
    """Raise :class:`ManifestError` on any integrity violation."""
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ManifestError(f"Manifest missing columns: {missing}")
    if df["image_id"].duplicated().any():
        dups = df.loc[df["image_id"].duplicated(), "image_id"].head(5).tolist()
        raise ManifestError(f"Duplicate image_id values, e.g. {dups}")
    bad = sorted(set(df["label"].astype(str)) - set(CLASS_NAMES))
    if bad:
        raise ManifestError(f"Labels outside {CLASS_NAMES}: {bad}")
    verified = df["patient_id_verified"].astype(bool)
    empty_pid = df["patient_id"].fillna("").astype(str).str.strip() == ""
    if (verified & empty_pid).any():
        raise ManifestError("patient_id_verified is True for rows without a patient_id (identities must not be invented)")
    df = normalise_split_column(df)
    bad_split = sorted(set(df["split"]) - set(SPLITS) - {""})
    if bad_split:
        raise ManifestError(f"Invalid split values: {bad_split}")
    if require_split and (df["split"] == "").any():
        raise ManifestError("Rows without a split assignment")
    assigned = df[df["split"] != ""]
    if len(assigned):
        n_splits = assigned.groupby("group_id")["split"].nunique()
        leaking = n_splits[n_splits > 1]
        if len(leaking):
            raise ManifestError(
                f"{len(leaking)} group(s) span more than one split, e.g. {leaking.index[:3].tolist()} "
                "(group integrity violated)"
            )
        vpat = assigned[assigned["patient_id_verified"].astype(bool)]
        if len(vpat):
            key = vpat["source_id"].astype(str) + "|" + vpat["patient_id"].astype(str)
            n = vpat.assign(_k=key).groupby("_k")["split"].nunique()
            if (n > 1).any():
                raise ManifestError("A verified patient appears in more than one split")


def save_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)
    return path


def load_manifest(path: str | Path, validate: bool = True) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, keep_default_na=False)
    df["patient_id_verified"] = df["patient_id_verified"].astype(str).str.lower().isin({"true", "1"})
    df["patient_id"] = df["patient_id"].astype(str).replace({"nan": ""})
    df = normalise_split_column(df)
    if validate:
        validate_manifest(df)
    return df
