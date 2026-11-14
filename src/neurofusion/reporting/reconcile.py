"""Reconcile independently recomputed results with paper-reported values. Differences are reported, never hidden
and never used to adjust anything."""
from __future__ import annotations

import pandas as pd

from neurofusion.evaluation.metrics import classification_report
from neurofusion.evaluation.predict import load_predictions
from neurofusion.reporting.numeric_audit import load_reference


def reconcile_accuracy(pred_files: dict[str, str], ref: dict | None = None) -> pd.DataFrame:
    """Compare recomputed test accuracy / macro metrics with Table 4 (matching on model name)."""
    ref = ref or load_reference()
    t4 = ref["table4"]["rows"]
    rows = []
    for name, path in pred_files.items():
        rep = classification_report(load_predictions(path))
        key = "Proposed" if name.lower() in {"proposed", "proposed_model"} else name
        rpt = t4.get(key)
        row = {"model": name, "n": rep["n"], "recomputed_acc_%": 100 * rep["accuracy"],
               "recomputed_macro_f1_%": 100 * rep["macro"]["f1"]}
        if rpt:
            row |= {"reported_acc_%": rpt[0], "diff_acc_pp": 100 * rep["accuracy"] - rpt[0], "reported_macro_f1_%": rpt[3],
                    "diff_f1_pp": 100 * rep["macro"]["f1"] - rpt[3]}
        else:
            row["note"] = "no manuscript-reported counterpart"
        row["status"] = "NEW EXPERIMENT; a difference is expected until the original data/config are verified"
        rows.append(row)
    return pd.DataFrame(rows)
