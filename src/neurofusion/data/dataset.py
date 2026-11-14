"""Torch dataset over a manifest."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from neurofusion.constants import CLASS_TO_INDEX
from neurofusion.utils.seed import make_generator, worker_init_fn


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, root: str | Path, transform: Callable | None = None, check_files: bool = True):
        self.df = df.reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform
        if check_files:
            missing = [p for p in self.df["relative_path"] if not (self.root / p).is_file()]
            if missing:
                raise FileNotFoundError(f"{len(missing)} image file(s) missing under {self.root}, e.g. {missing[:3]}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        with Image.open(self.root / r["relative_path"]) as im:
            im.load()
            img = im.copy()
        x = self.transform(img) if self.transform else img
        return x, CLASS_TO_INDEX[r["label"]], r["image_id"]


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int = 0, num_workers: int = 0) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn if num_workers else None,
        generator=make_generator(seed) if shuffle else None,
        pin_memory=torch.cuda.is_available(),
    )
