"""Shared fixtures. All data here is SYNTHETIC and exists only to test software behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from neurofusion.constants import CLASS_NAMES
from neurofusion.data.grouping import assign_groups
from neurofusion.data.manifest import build_manifest_from_folder


def _write_img(path, rng, label_idx: int, size: int = 32) -> None:
    """Class-dependent bright blob position + noise; purely synthetic."""
    a = rng.normal(40, 10, (size, size)).clip(0, 255)
    cy, cx = [(8, 8), (8, 24), (24, 8), (24, 24)][label_idx]
    yy, xx = np.ogrid[:size, :size]
    a += 150 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / 30.0))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a.clip(0, 255).astype(np.uint8)).save(path)


@pytest.fixture
def synth_root(tmp_path):
    """4 classes x 12 images, filenames like pat{N}_slice{M}.png (heuristic patient pattern)."""
    rng = np.random.default_rng(0)
    root = tmp_path / "images"
    for ci, c in enumerate(CLASS_NAMES):
        for i in range(12):
            _write_img(root / c / f"pat{ci * 100 + i // 2}_s{i % 2}.png", rng, ci)
    return root


@pytest.fixture
def synth_manifest(synth_root) -> pd.DataFrame:
    df = build_manifest_from_folder(synth_root, "synthetic_src", "synthetic")
    return assign_groups(df, r"(pat\d+)_s\d")


class TinyBranch(nn.Module):
    """Stand-in backbone: conv stages + pooled features. Exposes ``stages`` like timm CNNs."""

    def __init__(self, dim: int):
        super().__init__()
        self.stages = nn.ModuleList([nn.Sequential(nn.Conv2d(3, 8, 3, 2, 1), nn.ReLU()), nn.Sequential(nn.Conv2d(8, dim, 3, 2, 1), nn.ReLU())])
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        for s in self.stages:
            x = s(x)
        return self.pool(x).flatten(1)


class TinyViT(nn.Module):
    """Stand-in transformer: tokens = cls + 4x4 patches; exposes ``blocks``."""

    def __init__(self, dim: int):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 8, 8)  # 32px -> 4x4
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        # pre-norm blocks (norm_first) like timm ViTs: each layer exposes norm1 feeding the attention
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(dim, 2, dim * 2, batch_first=True, dropout=0.0, norm_first=True)])

    def forward(self, x):
        t = self.patch(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1)
        for b in self.blocks:
            t = b(t)
        return t[:, 0]


@pytest.fixture
def tiny_fusion():
    from neurofusion.models.classifier import FusionClassifier, ModelConfig

    torch.manual_seed(0)
    return FusionClassifier(ModelConfig(pretrained=False), TinyBranch(24), TinyViT(16), 24, 16)


def synthetic_predictions(n=200, k=4, acc=0.9, seed=0, ids=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, k, n)
    logits = rng.normal(0, 1, (n, k))
    correct = rng.random(n) < acc
    logits[np.arange(n), np.where(correct, y, (y + 1) % k)] += 4
    p = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    df = pd.DataFrame(p, columns=[f"p_{c}" for c in CLASS_NAMES])
    df.insert(0, "y_pred", p.argmax(1))
    df.insert(0, "y_true", y)
    df.insert(0, "image_id", ids if ids is not None else [f"img{i:04d}" for i in range(n)])
    return df
