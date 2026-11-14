"""Sample-level predictions: every reported metric is computed from these saved artifacts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from neurofusion.constants import CLASS_NAMES


def prob_columns(class_names: tuple[str, ...] = CLASS_NAMES) -> list[str]:
    return [f"p_{c}" for c in class_names]


@torch.no_grad()
def predict_loader(model: nn.Module, loader: DataLoader, device: str | torch.device = "cpu", class_names: tuple[str, ...] = CLASS_NAMES) -> pd.DataFrame:
    """Run ``model`` (returning logits) and return image_id, y_true, y_pred, p_<class>."""
    model.eval().to(device)
    ids, ys, probs = [], [], []
    for x, y, iid in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        ys.append(y.numpy())
        ids.extend(list(iid))
    p = np.concatenate(probs)
    df = pd.DataFrame(p, columns=prob_columns(class_names))
    df.insert(0, "y_pred", p.argmax(1))
    df.insert(0, "y_true", np.concatenate(ys))
    df.insert(0, "image_id", ids)
    return df


def save_predictions(df: pd.DataFrame, path: str | Path, meta: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    if meta is not None:
        import json

        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return path


class PredictionError(ValueError):
    pass


def load_predictions(path: str | Path, class_names: tuple[str, ...] = CLASS_NAMES, tol: float = 1e-3) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = ["image_id", "y_true", "y_pred", *prob_columns(class_names)]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise PredictionError(f"{path}: missing columns {missing}")
    p = df[prob_columns(class_names)].to_numpy()
    if not np.allclose(p.sum(1), 1.0, atol=tol):
        raise PredictionError(f"{path}: probability rows do not sum to 1")
    if (df["y_pred"].to_numpy() != p.argmax(1)).any():
        raise PredictionError(f"{path}: y_pred disagrees with argmax of the probabilities")
    if df["image_id"].duplicated().any():
        raise PredictionError(f"{path}: duplicate image_id")
    return df
