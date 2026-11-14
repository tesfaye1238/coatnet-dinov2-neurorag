"""Preprocessing and augmentation.

Reported (manuscript 3.2): RGB, bilinear resize to 224, [0,1] scaling, ImageNet normalisation; training
augmentation with the listed parameters. Everything else here (order, interpolation, fill, erasing ranges,
noise domain) is NOT reported and is exposed through :class:`AugmentationConfig` as a reconstruction choice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from neurofusion.constants import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

UNSPECIFIED_AUGMENTATION_DETAILS = (
    "order of transformations",
    "per-transform application probability for rotation, zoom and brightness (assumed 1.0)",
    "interpolation mode for rotation/zoom (assumed bilinear)",
    "fill value for rotation/zoom borders (assumed 0 = black)",
    "whether zoom is isotropic (assumed isotropic)",
    "random-erasing area range, aspect-ratio range and fill value",
    "intensity domain of the Gaussian noise (assumed [0,1] before normalisation) and clipping",
    "whether brightness is multiplicative (assumed multiplicative factor in [0.9, 1.1])",
)


@dataclass(frozen=True)
class AugmentationConfig:
    # REPORTED
    hflip_p: float = 0.50
    vflip_p: float = 0.50
    rotation_deg: float = 15.0
    zoom: tuple[float, float] = (0.90, 1.10)
    brightness: float = 0.10
    noise_std: float = 0.01
    erase_p: float = 0.10
    # RECONSTRUCTION ASSUMPTIONS (unreported)
    erase_scale: tuple[float, float] = (0.02, 0.33)
    erase_ratio: tuple[float, float] = (0.3, 3.3)
    erase_value: float = 0.0
    fill: float = 0.0
    noise_clip: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


class AddGaussianNoise:
    """Additive Gaussian noise on a [0,1] tensor."""

    def __init__(self, std: float, clip: bool = True):
        self.std, self.clip = std, clip

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return x
        y = x + torch.randn_like(x) * self.std
        return y.clamp(0.0, 1.0) if self.clip else y


class ToRGB:
    def __call__(self, img):
        return img.convert("RGB")


def build_eval_transform(image_size: int = IMAGE_SIZE) -> T.Compose:
    """Deterministic evaluation preprocessing (also used for validation)."""
    return T.Compose(
        [
            ToRGB(),
            T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_train_transform(cfg: AugmentationConfig | None = None, image_size: int = IMAGE_SIZE) -> T.Compose:
    cfg = cfg or AugmentationConfig()
    ops = [
        ToRGB(),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR, antialias=True),
        T.RandomHorizontalFlip(cfg.hflip_p),
        T.RandomVerticalFlip(cfg.vflip_p),
        T.RandomAffine(
            degrees=cfg.rotation_deg,
            scale=cfg.zoom,
            interpolation=InterpolationMode.BILINEAR,
            fill=cfg.fill,
        ),
        T.ColorJitter(brightness=cfg.brightness),
        T.ToTensor(),
        AddGaussianNoise(cfg.noise_std, cfg.noise_clip),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=cfg.erase_p, scale=cfg.erase_scale, ratio=cfg.erase_ratio, value=cfg.erase_value),
    ]
    return T.Compose(ops)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)
