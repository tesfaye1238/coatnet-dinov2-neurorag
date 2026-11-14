"""Calibration and selective-prediction measures with explicit definitions.

* ECE / MCE: top-label confidence, ``n_bins`` equal-width bins on [0,1]; per bin |accuracy - mean confidence|,
  ECE = sum_b (n_b/N) * gap_b, MCE = max_b gap_b over non-empty bins.
* Brier: mean over samples of sum_c (p_c - 1[y=c])^2 (multiclass, range [0, 2]).
* NLL: mean -log p_y with p clipped at 1e-12.
* Risk-coverage: samples sorted by decreasing confidence; at coverage j/N the risk is the error rate among the
  j most confident samples. AURC = mean of the risk over j = 1..N (the area under the empirical curve).
* coverage_at_risk(r): the largest coverage whose risk is <= r (0 if none). This is sensitive to ties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from neurofusion.constants import CLASS_NAMES
from neurofusion.evaluation.predict import prob_columns


def _arrays(df: pd.DataFrame, class_names=CLASS_NAMES):
    return df[prob_columns(class_names)].to_numpy(dtype=float), df["y_true"].to_numpy()


def reliability_bins(P: np.ndarray, y: np.ndarray, n_bins: int = 15) -> pd.DataFrame:
    conf = P.max(1)
    correct = (P.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        rows.append({"bin": b, "lo": edges[b], "hi": edges[b + 1], "n": n,
                     "accuracy": float(correct[m].mean()) if n else np.nan,
                     "confidence": float(conf[m].mean()) if n else np.nan})
    out = pd.DataFrame(rows)
    out["gap"] = (out["accuracy"] - out["confidence"]).abs()
    return out


def ece_mce(df: pd.DataFrame, n_bins: int = 15, class_names=CLASS_NAMES) -> dict:
    P, y = _arrays(df, class_names)
    bins = reliability_bins(P, y, n_bins)
    ok = bins[bins["n"] > 0]
    ece = float((ok["n"] / ok["n"].sum() * ok["gap"]).sum())
    return {"ece": ece, "mce": float(ok["gap"].max()), "n_bins": n_bins}


def brier(df: pd.DataFrame, class_names=CLASS_NAMES) -> float:
    P, y = _arrays(df, class_names)
    onehot = np.eye(P.shape[1])[y]
    return float(((P - onehot) ** 2).sum(1).mean())


def nll(df: pd.DataFrame, class_names=CLASS_NAMES) -> float:
    P, y = _arrays(df, class_names)
    return float(-np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None)).mean())


def risk_coverage(df: pd.DataFrame, class_names=CLASS_NAMES) -> pd.DataFrame:
    P, y = _arrays(df, class_names)
    conf = P.max(1)
    err = (P.argmax(1) != y).astype(float)
    order = np.argsort(-conf, kind="stable")
    cum_err = np.cumsum(err[order])
    j = np.arange(1, len(y) + 1)
    return pd.DataFrame({"coverage": j / len(y), "risk": cum_err / j, "threshold": conf[order]})


def aurc(df: pd.DataFrame, class_names=CLASS_NAMES) -> float:
    return float(risk_coverage(df, class_names)["risk"].mean())


def coverage_at_risk(df: pd.DataFrame, max_risk: float = 0.01, class_names=CLASS_NAMES) -> float:
    rc = risk_coverage(df, class_names)
    ok = rc[rc["risk"] <= max_risk]
    return float(ok["coverage"].max()) if len(ok) else 0.0


def calibration_report(df: pd.DataFrame, n_bins: int = 15, class_names=CLASS_NAMES) -> dict:
    P, _ = _arrays(df, class_names)
    ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(1)
    return {
        **ece_mce(df, n_bins, class_names),
        "brier": brier(df, class_names),
        "nll": nll(df, class_names),
        "aurc": aurc(df, class_names),
        "coverage_at_risk_1pct": coverage_at_risk(df, 0.01, class_names),
        "mean_entropy": float(ent.mean()),
        "sd_entropy": float(ent.std(ddof=1)) if len(ent) > 1 else float("nan"),
    }
