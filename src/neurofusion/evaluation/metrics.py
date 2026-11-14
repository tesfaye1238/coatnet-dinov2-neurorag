"""Classification metrics from sample-level predictions, with transparent handling of undefined values.

Definitions
-----------
* precision_c = TP/(TP+FP) (NaN, flagged, when class c is never predicted)
* recall_c = TP/(TP+FN) (NaN when class c has no samples)
* specificity_c, MCC_c: one-versus-rest. ``ovr_mcc`` is NOT the multiclass MCC (``multiclass_mcc``).
* macro-averages use only classes for which the metric is defined; the classes used are reported.
* ROC-AUC and average precision (AP) are one-versus-rest from the saved probabilities. AP is the
  step-wise average precision of sklearn, not the trapezoidal area under the PR curve.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score

from neurofusion.constants import CLASS_NAMES
from neurofusion.evaluation.predict import prob_columns


def confusion_matrix_from(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _div(a: float, b: float) -> float:
    return a / b if b else float("nan")


def _nanmean(vals: list[float]) -> tuple[float, int]:
    arr = np.array(vals, dtype=float)
    ok = ~np.isnan(arr)
    return (float(arr[ok].mean()) if ok.any() else float("nan")), int(ok.sum())


def classification_report(df: pd.DataFrame, class_names: tuple[str, ...] = CLASS_NAMES) -> dict:
    k = len(class_names)
    y, yp = df["y_true"].to_numpy(), df["y_pred"].to_numpy()
    P = df[prob_columns(class_names)].to_numpy()
    cm = confusion_matrix_from(y, yp, k)
    n = int(cm.sum())
    correct = int(np.trace(cm))
    per, undefined = {}, []
    for i, c in enumerate(class_names):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        tn = n - tp - fp - fn
        prec, rec = _div(tp, tp + fp), _div(tp, tp + fn)
        f1 = _div(2 * tp, 2 * tp + fp + fn)
        spec = _div(tn, tn + fp)
        denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = _div(tp * tn - fp * fn, denom)
        pos = (y == i)
        if pos.any() and (~pos).any():
            auc = float(roc_auc_score(pos, P[:, i]))
            ap = float(average_precision_score(pos, P[:, i]))
        else:
            auc = ap = float("nan")
        per[c] = {"support": int(pos.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
                  "f1": f1, "specificity": spec, "ovr_mcc": mcc, "auc": auc, "ap": ap}
        for m, v in (("precision", prec), ("recall", rec), ("f1", f1), ("specificity", spec), ("ovr_mcc", mcc), ("auc", auc), ("ap", ap)):
            if isinstance(v, float) and math.isnan(v):
                undefined.append(f"{c}:{m}")
    macro = {}
    for m in ("precision", "recall", "f1", "specificity", "ovr_mcc", "auc", "ap"):
        val, used = _nanmean([per[c][m] for c in class_names])
        macro[m] = val
        macro[f"{m}_n_classes"] = used
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else float("nan"),
        "confusion_matrix": cm.tolist(),
        "class_names": list(class_names),
        "per_class": per,
        "macro": macro,
        "multiclass_mcc": float(matthews_corrcoef(y, yp)) if n else float("nan"),
        "undefined": undefined,
        "classes_absent": [c for c in class_names if per[c]["support"] == 0],
    }


def accuracy(df: pd.DataFrame) -> float:
    return float((df["y_true"].to_numpy() == df["y_pred"].to_numpy()).mean())


def macro_auc(df: pd.DataFrame, class_names: tuple[str, ...] = CLASS_NAMES) -> float:
    return classification_report(df, class_names)["macro"]["auc"]


def macro_f1(df: pd.DataFrame, class_names: tuple[str, ...] = CLASS_NAMES) -> float:
    return classification_report(df, class_names)["macro"]["f1"]
