"""Resampling uncertainty.

* Unit of resampling: patient/group clusters when ``cluster_col`` is given (use only verified patient ids),
  otherwise individual images. Image-level intervals are conditional on treating images as independent.
* Percentile intervals; ``n_boot`` and ``seed`` are returned so the procedure is reproducible.
* These intervals describe TEST-SET sampling uncertainty for fixed trained models. They do not capture
  variation across training seeds.
* Paired differences use the SAME resampled units for both models.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from neurofusion.evaluation.paired import align_predictions


def _cluster_index(n: int, clusters: np.ndarray | None) -> list[np.ndarray]:
    if clusters is None:
        return [np.array([i]) for i in range(n)]
    out = []
    for g in pd.unique(clusters):
        out.append(np.where(clusters == g)[0])
    return out


def _draw(rng: np.random.Generator, units: list[np.ndarray]) -> np.ndarray:
    pick = rng.integers(0, len(units), size=len(units))
    return np.concatenate([units[i] for i in pick])


def bootstrap_ci(
    df: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    cluster_col: str | None = None,
) -> dict:
    clusters = df[cluster_col].to_numpy() if cluster_col else None
    units = _cluster_index(len(df), clusters)
    rng = np.random.default_rng(seed)
    point = float(metric(df))
    vals = []
    for _ in range(n_boot):
        idx = _draw(rng, units)
        try:
            vals.append(float(metric(df.iloc[idx])))
        except ValueError:
            continue
    arr = np.array(vals)
    arr = arr[~np.isnan(arr)]
    return {
        "point": point,
        "lo": float(np.quantile(arr, alpha / 2)),
        "hi": float(np.quantile(arr, 1 - alpha / 2)),
        "n_boot": int(len(arr)),
        "n_boot_requested": n_boot,
        "seed": seed,
        "alpha": alpha,
        "method": "percentile",
        "unit": f"cluster:{cluster_col}" if cluster_col else "image (assumes independent images)",
        "conditional_on": "fixed trained model; test-set sampling only",
    }


def paired_bootstrap_difference(
    a: pd.DataFrame,
    b: pd.DataFrame,
    metric: Callable[[pd.DataFrame], float],
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    clusters: np.ndarray | None = None,
) -> dict:
    """metric(A) - metric(B) with shared resampling. ``clusters`` aligned to sorted image_id."""
    a2, b2 = align_predictions(a, b)
    units = _cluster_index(len(a2), clusters)
    rng = np.random.default_rng(seed)
    point = float(metric(a2) - metric(b2))
    diffs = []
    for _ in range(n_boot):
        idx = _draw(rng, units)
        try:
            diffs.append(float(metric(a2.iloc[idx]) - metric(b2.iloc[idx])))
        except ValueError:
            continue
    d = np.array(diffs)
    d = d[~np.isnan(d)]
    # two-sided bootstrap p: share of resamples on the opposite side of zero from the point estimate, doubled
    p = float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean())))
    return {
        "difference": point, "lo": float(np.quantile(d, alpha / 2)), "hi": float(np.quantile(d, 1 - alpha / 2)),
        "p_bootstrap_two_sided": p, "n_boot": int(len(d)), "seed": seed, "alpha": alpha,
        "method": "paired percentile bootstrap (shared resampling units)",
        "unit": "cluster" if clusters is not None else "image (assumes independent images)",
    }
