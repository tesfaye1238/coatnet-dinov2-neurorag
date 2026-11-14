"""Result tables and figures generated ONLY from saved evaluation artifacts (prediction files)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from neurofusion.constants import CLASS_NAMES
from neurofusion.evaluation.bootstrap import bootstrap_ci
from neurofusion.evaluation.calibration import calibration_report
from neurofusion.evaluation.metrics import accuracy, classification_report, macro_auc
from neurofusion.evaluation.predict import load_predictions


def results_table(pred_files: dict[str, str | Path], n_boot: int = 1000, seed: int = 0, cluster_col: str | None = None) -> pd.DataFrame:
    """One row per model. Intervals are image-level bootstrap (conditional on a fixed model, images treated as
    independent) unless clusters are supplied through ``cluster_col`` in the prediction file."""
    rows = []
    for name, p in pred_files.items():
        df = load_predictions(p)
        rep = classification_report(df)
        ci = bootstrap_ci(df, accuracy, n_boot=n_boot, seed=seed, cluster_col=cluster_col if cluster_col in df.columns else None)
        auc_ci = bootstrap_ci(df, macro_auc, n_boot=min(n_boot, 500), seed=seed)
        cal = calibration_report(df)
        rows.append({
            "model": name, "n": rep["n"], "correct": rep["correct"], "accuracy_%": 100 * rep["accuracy"],
            "acc_ci_lo_%": 100 * ci["lo"], "acc_ci_hi_%": 100 * ci["hi"],
            "macro_precision_%": 100 * rep["macro"]["precision"], "macro_recall_%": 100 * rep["macro"]["recall"],
            "macro_f1_%": 100 * rep["macro"]["f1"], "macro_auc": rep["macro"]["auc"], "auc_ci_lo": auc_ci["lo"], "auc_ci_hi": auc_ci["hi"],
            "macro_ap": rep["macro"]["ap"], "multiclass_mcc": rep["multiclass_mcc"], "ece": cal["ece"], "brier": cal["brier"], "nll": cal["nll"],
            "interval_unit": ci["unit"],
        })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(f"{v:{floatfmt}}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, out_dir: str | Path, stem: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"{stem}.csv", index=False)
    (out / f"{stem}.md").write_text(to_markdown(df) + "\n", encoding="utf-8")


def per_class_table(pred_file: str | Path) -> pd.DataFrame:
    rep = classification_report(load_predictions(pred_file))
    return pd.DataFrame(rep["per_class"]).T.rename_axis("class").reset_index()[["class", "support", "precision", "recall", "f1", "specificity", "ovr_mcc", "auc", "ap"]]


def plot_confusion(pred_file: str | Path, out: str | Path, class_names=CLASS_NAMES) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    cm = np.array(classification_report(load_predictions(pred_file))["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
