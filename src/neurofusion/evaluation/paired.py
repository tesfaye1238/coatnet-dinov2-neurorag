"""Paired comparisons between models evaluated on the same cases.

Predictions are joined on stable ``image_id`` and the ground truth must match. Paired significance is never
derived from aggregate accuracies. Multiple-comparison families must be declared by the caller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from neurofusion.constants import CLASS_NAMES


class AlignmentError(ValueError):
    pass


def align_predictions(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if set(a["image_id"]) != set(b["image_id"]):
        only_a = len(set(a["image_id"]) - set(b["image_id"]))
        only_b = len(set(b["image_id"]) - set(a["image_id"]))
        raise AlignmentError(f"image_id sets differ ({only_a} only in A, {only_b} only in B)")
    a2 = a.sort_values("image_id").reset_index(drop=True)
    b2 = b.sort_values("image_id").reset_index(drop=True)
    if (a2["y_true"].to_numpy() != b2["y_true"].to_numpy()).any():
        raise AlignmentError("ground-truth labels differ between the two prediction files")
    return a2, b2


def mcnemar(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """McNemar on paired correctness. ``b_only``: A wrong & B right is counted in ``n01``; A right & B wrong in ``n10``."""
    a2, b2 = align_predictions(a, b)
    ca = a2["y_pred"].to_numpy() == a2["y_true"].to_numpy()
    cb = b2["y_pred"].to_numpy() == b2["y_true"].to_numpy()
    n10 = int((ca & ~cb).sum())  # A correct, B wrong
    n01 = int((~ca & cb).sum())  # A wrong, B correct
    m = n10 + n01
    out = {"n": len(a2), "n10_a_right_b_wrong": n10, "n01_a_wrong_b_right": n01, "discordant": m,
           "acc_a": float(ca.mean()), "acc_b": float(cb.mean())}
    if m == 0:
        out.update(chi2_uncorrected=float("nan"), p_chi2_uncorrected=1.0, chi2_corrected=float("nan"), p_chi2_corrected=1.0, p_exact=1.0)
    else:
        chi_u = (n10 - n01) ** 2 / m
        chi_c = (abs(n10 - n01) - 1) ** 2 / m
        out.update(
            chi2_uncorrected=chi_u, p_chi2_uncorrected=float(stats.chi2.sf(chi_u, 1)),
            chi2_corrected=chi_c, p_chi2_corrected=float(stats.chi2.sf(chi_c, 1)),
            p_exact=float(stats.binomtest(min(n10, n01), m, 0.5).pvalue),
        )
    out["recommended"] = "p_exact"
    return out


def exact_mcnemar_p(n10: int, n01: int) -> float:
    m = n10 + n01
    return 1.0 if m == 0 else float(stats.binomtest(min(n10, n01), m, 0.5).pvalue)


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (family = the list passed in)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj.tolist()


def bonferroni(pvals: list[float]) -> list[float]:
    m = len(pvals)
    return [min(p * m, 1.0) for p in pvals]


def compare_to_reference(reference: pd.DataFrame, others: dict[str, pd.DataFrame], family_name: str) -> pd.DataFrame:
    """McNemar of ``reference`` vs each other model, with Holm and Bonferroni over THIS family."""
    rows = []
    for name, df in others.items():
        r = mcnemar(reference, df)
        r["comparison"] = name
        rows.append(r)
    out = pd.DataFrame(rows)
    out["family"] = family_name
    out["family_size"] = len(out)
    out["p_exact_holm"] = holm(out["p_exact"].tolist())
    out["p_exact_bonferroni"] = bonferroni(out["p_exact"].tolist())
    return out


def predicted_classes_agree(a: pd.DataFrame, b: pd.DataFrame, class_names=CLASS_NAMES) -> float:
    a2, b2 = align_predictions(a, b)
    return float((a2["y_pred"].to_numpy() == b2["y_pred"].to_numpy()).mean())
