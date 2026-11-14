"""Leakage-aware splitting. Group integrity always outranks hitting exact target counts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from neurofusion.constants import CLASS_NAMES, SPLITS
from neurofusion.data.manifest import normalise_split_column, validate_manifest


class SplitError(ValueError):
    pass


def _group_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("group_id").agg(label=("label", lambda s: sorted(set(s))), n=("image_id", "size"))
    mixed = g[g["label"].map(len) > 1]
    if len(mixed):
        raise SplitError(
            f"{len(mixed)} group(s) contain more than one label, e.g. {mixed.index[:3].tolist()}. "
            "Resolve the label conflict (it may indicate a mislabelled duplicate) before splitting."
        )
    g["label"] = g["label"].map(lambda x: x[0])
    return g.reset_index()


def group_stratified_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    per_class_counts: dict[str, int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign whole groups to train/val/test, approximately stratified by class.

    ``per_class_counts`` (e.g. ``{"val": 160, "test": 400}``) are *targets*; with multi-image groups the
    realised counts can differ, and the realised counts are what ``audit_split`` reports.
    """
    rng = np.random.default_rng(seed)
    gt = _group_table(df)
    split_of: dict[str, str] = {}
    for cls, sub in gt.groupby("label"):
        total = int(sub["n"].sum())
        if per_class_counts:
            tv = int(per_class_counts.get("val", 0))
            tt = int(per_class_counts.get("test", 0))
        else:
            tv, tt = round(total * val_fraction), round(total * test_fraction)
        targets = {"train": max(total - tv - tt, 0), "val": tv, "test": tt}
        current = dict.fromkeys(SPLITS, 0)
        order = rng.permutation(len(sub))
        for idx in order:
            row = sub.iloc[idx]

            def deficit(s: str, _targets=targets, _current=current) -> float:
                return (_targets[s] - _current[s]) / max(_targets[s], 1)

            best = max(SPLITS, key=deficit)
            split_of[row["group_id"]] = best
            current[best] += int(row["n"])
    out = df.copy()
    out["split"] = out["group_id"].map(split_of)
    validate_manifest(out, require_split=True)
    return out


def kfold_splits(
    df: pd.DataFrame, k: int = 5, val_fraction_of_pool: float = 0.1, seed: int = 0
) -> list[pd.DataFrame]:
    """k test folds (grouped, stratified); validation is drawn only from each fold's training pool."""
    _group_table(df)  # raises on label conflict inside a group
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for fold, (pool_idx, test_idx) in enumerate(sgkf.split(df, df["label"], df["group_id"])):
        pool = df.iloc[pool_idx].copy()
        test = df.iloc[test_idx].copy()
        pool = group_stratified_split(pool, val_fraction=val_fraction_of_pool, test_fraction=0.0, seed=seed + fold)
        test["split"] = "test"
        f = pd.concat([pool, test]).sort_index()
        f["fold"] = fold
        validate_manifest(f.drop(columns="fold"), require_split=True)
        folds.append(f)
    return folds


def check_source_compatibility(df: pd.DataFrame, test_source: str) -> dict:
    """Class availability audit for a leave-one-source-out experiment."""
    if test_source not in set(df["source_id"]):
        raise SplitError(f"test source {test_source!r} not present in manifest")
    test = df[df["source_id"] == test_source]
    train = df[df["source_id"] != test_source]
    rep = {
        "test_source": test_source,
        "train_sources": sorted(set(train["source_id"])),
        "train_class_counts": train["label"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict(),
        "test_class_counts": test["label"].value_counts().reindex(CLASS_NAMES, fill_value=0).to_dict(),
    }
    rep["classes_missing_in_train_present_in_test"] = [
        c for c in CLASS_NAMES if rep["train_class_counts"][c] == 0 and rep["test_class_counts"][c] > 0
    ]
    rep["classes_absent_in_test"] = [c for c in CLASS_NAMES if rep["test_class_counts"][c] == 0]
    rep["cross_source_exact_overlap"] = cross_source_overlap(df)
    return rep


def cross_source_overlap(df: pd.DataFrame) -> dict:
    """Exact (file or decoded-pixel) duplicates that span more than one source."""
    out = {}
    for col in ("file_sha256", "decoded_image_hash"):
        n_src = df.groupby(col)["source_id"].nunique()
        out[col] = int((n_src > 1).sum())
    return out


def source_held_out_split(df: pd.DataFrame, test_source: str, val_fraction: float = 0.1, seed: int = 0) -> pd.DataFrame:
    rep = check_source_compatibility(df, test_source)
    if rep["classes_missing_in_train_present_in_test"]:
        raise SplitError(
            f"Classes {rep['classes_missing_in_train_present_in_test']} occur in test source {test_source!r} "
            "but not in the training sources."
        )
    test = df[df["source_id"] == test_source].copy()
    test["split"] = "test"
    pool = group_stratified_split(df[df["source_id"] != test_source], val_fraction=val_fraction, test_fraction=0.0, seed=seed)
    return pd.concat([pool, test]).sort_index()


def import_split_manifest(df: pd.DataFrame, split_csv: str | Path) -> pd.DataFrame:
    """Apply a supplied split file (columns image_id, split) without modification."""
    sp = pd.read_csv(split_csv, keep_default_na=False)
    if not {"image_id", "split"} <= set(sp.columns):
        raise SplitError("split file needs columns image_id and split")
    sp = normalise_split_column(sp)
    missing = set(df["image_id"]) - set(sp["image_id"])
    if missing:
        raise SplitError(f"{len(missing)} manifest image(s) have no split in the supplied file, e.g. {sorted(missing)[:3]}")
    out = df.drop(columns="split").merge(sp[["image_id", "split"]], on="image_id", how="left")
    validate_manifest(out, require_split=True)
    return out


def audit_split(df: pd.DataFrame, expected: dict | None = None) -> dict:
    """Realised counts, group/patient integrity and (optional) differences from expected counts."""
    d = normalise_split_column(df)
    counts = d.pivot_table(index="label", columns="split", values="image_id", aggfunc="count", fill_value=0)
    counts = counts.reindex(index=CLASS_NAMES, columns=list(SPLITS), fill_value=0)
    counts["total"] = counts.sum(axis=1)
    rep: dict = {
        "counts": counts.to_dict(orient="index"),
        "totals": {s: int(counts[s].sum()) for s in list(SPLITS) + ["total"]},
        "n_groups": int(d["group_id"].nunique()),
        "grouping_methods": d["grouping_method"].value_counts().to_dict(),
        "verified_patient_fraction": float(d["patient_id_verified"].astype(bool).mean()),
        "groups_spanning_multiple_splits": int((d.groupby("group_id")["split"].nunique() > 1).sum()),
        "source_by_split": d.pivot_table(index="source_id", columns="split", values="image_id", aggfunc="count", fill_value=0).to_dict(orient="index"),
    }
    vp = d[d["patient_id_verified"].astype(bool)]
    rep["verified_patients_spanning_multiple_splits"] = (
        int((vp.assign(_k=vp["source_id"] + "|" + vp["patient_id"]).groupby("_k")["split"].nunique() > 1).sum()) if len(vp) else 0
    )
    rep["patient_level_independence_established"] = bool(
        rep["verified_patient_fraction"] == 1.0 and rep["verified_patients_spanning_multiple_splits"] == 0
    )
    if expected:
        diffs = {}
        for cls, exp in expected.items():
            for s in SPLITS:
                if s in exp:
                    diffs[f"{cls}/{s}"] = {"expected": exp[s], "actual": int(counts.loc[cls, s]), "diff": int(counts.loc[cls, s]) - exp[s]}
        rep["difference_from_expected"] = diffs
    return rep
