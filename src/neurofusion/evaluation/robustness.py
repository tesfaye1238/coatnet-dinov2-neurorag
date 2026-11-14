"""Image-domain perturbations applied to exported 224x224 images.

These are image-space approximations. They do NOT simulate MRI acquisition physics, scanner differences or
protocol changes. Severity levels follow the values listed in the manuscript's Supplementary Table S2
("documented"); the functional forms (Rician construction, bias-field model, motion kernel) are NOT given in
the manuscript and are new implementation decisions.
"""
from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from neurofusion.constants import CLASS_TO_INDEX, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

Perturb = Callable[[np.ndarray, float, np.random.Generator], np.ndarray]


def gaussian_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return np.clip(x + rng.normal(0, sigma, x.shape), 0, 1)


def rician_noise(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Magnitude of (x + n1) + i*n2 with n1, n2 ~ N(0, sigma^2). Applied to a displayed image, not raw k-space."""
    return np.clip(np.sqrt((x + rng.normal(0, sigma, x.shape)) ** 2 + rng.normal(0, sigma, x.shape) ** 2), 0, 1)


def gaussian_blur(x: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return cv2.GaussianBlur(x, (0, 0), sigmaX=sigma, sigmaY=sigma)


def motion_blur(x: np.ndarray, kernel: float, rng: np.random.Generator) -> np.ndarray:
    k = int(kernel)
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1.0 / k  # horizontal line kernel (direction fixed: a new decision)
    return cv2.filter2D(x, -1, ker, borderType=cv2.BORDER_REFLECT)


def bias_field(x: np.ndarray, coeff: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative smooth field 1 + coeff * (random low-order polynomial normalised to [-1, 1])."""
    h, w = x.shape[:2]
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    c = rng.uniform(-1, 1, 6)
    poly = c[0] * xx + c[1] * yy + c[2] * xx * yy + c[3] * xx**2 + c[4] * yy**2 + c[5]
    poly = poly / max(np.abs(poly).max(), 1e-8)
    field = (1.0 + coeff * poly).astype(np.float32)
    return np.clip(x * field[..., None], 0, 1)


def contrast_reduction(x: np.ndarray, factor: float, rng: np.random.Generator) -> np.ndarray:
    m = x.mean()
    return np.clip(m + factor * (x - m), 0, 1)


def jpeg_compression(x: np.ndarray, quality: float, rng: np.random.Generator) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray((x * 255).round().astype(np.uint8)).save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0


PERTURBATIONS: dict[str, Perturb] = {
    "gaussian_noise": gaussian_noise, "rician_noise": rician_noise, "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur, "bias_field": bias_field, "contrast_reduction": contrast_reduction,
    "jpeg_compression": jpeg_compression,
}

# Severity levels 1..5 as listed in manuscript Supplementary Table S2 (documented settings).
SEVERITIES: dict[str, tuple[float, ...]] = {
    "gaussian_noise": (0.01, 0.02, 0.04, 0.06, 0.08),
    "rician_noise": (0.01, 0.02, 0.04, 0.06, 0.08),
    "gaussian_blur": (0.5, 1.0, 1.5, 2.0, 3.0),
    "motion_blur": (3, 5, 7, 9, 13),
    "bias_field": (0.2, 0.4, 0.6, 0.8, 1.0),
    "contrast_reduction": (0.9, 0.8, 0.7, 0.6, 0.5),
    "jpeg_compression": (70, 50, 30, 20, 10),
}


class PerturbedDataset(Dataset):
    """Loads an image, resizes (bilinear, 224), perturbs in [0,1] RGB space, then ImageNet-normalises."""

    def __init__(self, df: pd.DataFrame, root: str | Path, name: str | None, level: float | None, seed: int = 0):
        self.df, self.root, self.name, self.level, self.seed = df.reset_index(drop=True), Path(root), name, level, seed
        self.resize = T.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR, antialias=True)
        self.norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        with Image.open(self.root / r["relative_path"]) as im:
            img = self.resize(im.convert("RGB"))
        x = np.asarray(img, dtype=np.float32) / 255.0
        if self.name is not None:
            rng = np.random.default_rng(self.seed * 1_000_003 + i)
            x = PERTURBATIONS[self.name](x, self.level, rng).astype(np.float32)
        t = self.norm(torch.from_numpy(x).permute(2, 0, 1).contiguous())
        return t, CLASS_TO_INDEX[r["label"]], r["image_id"]


def evaluate_robustness(model: nn.Module, df: pd.DataFrame, root: str | Path, perturbations: list[str] | None = None,
                        batch_size: int = 32, device: str = "cpu", seed: int = 0) -> pd.DataFrame:
    """Accuracy per (perturbation, severity) plus the clean reference, computed on the supplied test rows."""
    model.eval().to(device)
    rows = []
    cases = [("clean", 0, None)]
    for name in perturbations or list(PERTURBATIONS):
        cases += [(name, s + 1, lvl) for s, lvl in enumerate(SEVERITIES[name])]
    for name, sev, lvl in cases:
        ds = PerturbedDataset(df, root, None if name == "clean" else name, lvl, seed)
        correct = n = 0
        with torch.no_grad():
            for x, y, _ in DataLoader(ds, batch_size=batch_size):
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
                n += len(y)
        rows.append({"perturbation": name, "severity": sev, "level": lvl, "accuracy": correct / n, "n": n})
    return pd.DataFrame(rows)
