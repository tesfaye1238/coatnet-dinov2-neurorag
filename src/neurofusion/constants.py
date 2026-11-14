"""Shared constants. Values marked MANUSCRIPT are reported values, used as audit expectations only."""
from __future__ import annotations

# Alphabetical order, identical to the class order of the manuscript figures.
CLASS_NAMES: tuple[str, ...] = ("glioma", "meningioma", "notumor", "pituitary")
CLASS_TO_INDEX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)  # MANUSCRIPT 3.2.2
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)  # MANUSCRIPT 3.2.2
IMAGE_SIZE = 224  # MANUSCRIPT 3.2.1

SPLITS: tuple[str, ...] = ("train", "val", "test")
SPLIT_ALIASES = {"validation": "val", "valid": "val", "training": "train", "testing": "test"}

MANIFEST_COLUMNS: tuple[str, ...] = (
    "image_id",
    "relative_path",
    "source_id",
    "source_release",
    "label",
    "patient_id",
    "patient_id_verified",
    "group_id",
    "grouping_method",
    "file_sha256",
    "decoded_image_hash",
    "width",
    "height",
    "split",
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# MANUSCRIPT Table 1: audit expectation only. Never used to force a split.
MANUSCRIPT_TABLE1 = {
    "glioma": {"raw": 1812, "exact": 24, "near": 33, "final": 1755, "train": 1195, "val": 160, "test": 400},
    "meningioma": {"raw": 1808, "exact": 21, "near": 31, "final": 1756, "train": 1196, "val": 160, "test": 400},
    "notumor": {"raw": 1811, "exact": 19, "near": 38, "final": 1754, "train": 1194, "val": 160, "test": 400},
    "pituitary": {"raw": 1809, "exact": 16, "near": 36, "final": 1757, "train": 1197, "val": 160, "test": 400},
}
