"""Grad-CAM and Grad-CAM++ on a chosen internal layer of the fused classifier.

* The class score is the fused model's LOGIT for the target class; gradients flow through the whole head,
  gate and fusion into the selected branch layer. A map from one branch is therefore a branch-specific view of
  the fused decision, not a complete explanation of it.
* CNN stage outputs are used as [B, C, H, W] maps. Transformer outputs are [B, 1+N, C]; the class token is
  dropped and the N patch tokens are reshaped to a sqrt(N) x sqrt(N) grid (16x16 for ViT-S/14 at 224 px).
* Maps are upsampled bilinearly to the input size and min-max normalised per sample for display; the raw
  (un-normalised) upsampled map is returned and saved too.
* Attribution maps are not segmentation masks and carry no localisation accuracy unless compared with
  expert annotations.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GradFlowError(RuntimeError):
    pass


def vit_tokens_to_map(t: torch.Tensor, has_cls: bool = True) -> torch.Tensor:
    if t.ndim != 3:
        raise ValueError(f"expected [B, tokens, C], got {tuple(t.shape)}")
    if has_cls:
        t = t[:, 1:, :]
    n = t.shape[1]
    s = int(math.isqrt(n))
    if s * s != n:
        raise ValueError(f"{n} patch tokens is not a square grid (register/extra tokens present?)")
    return t.transpose(1, 2).reshape(t.shape[0], t.shape[2], s, s)


class CAMExtractor:
    def __init__(self, model: nn.Module, layer: nn.Module, method: str = "gradcam", reshape: Callable | None = None):
        if method not in {"gradcam", "gradcampp"}:
            raise ValueError(method)
        self.model, self.layer, self.method, self.reshape = model, layer, method, reshape
        self._act: torch.Tensor | None = None
        self._grad: torch.Tensor | None = None
        self._h = layer.register_forward_hook(self._fwd)

    def _fwd(self, _m, _i, out):
        out = out[0] if isinstance(out, tuple) else out
        self._act = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, "_grad", g))

    def close(self) -> None:
        self._h.remove()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __call__(self, x: torch.Tensor, class_idx: int | torch.Tensor | None = None):
        """Returns ``(cam_raw [B,H,W], logits, class_idx)`` with ``cam_raw`` at the input resolution."""
        self.model.eval()
        x = x.clone().requires_grad_(True)
        self._act = self._grad = None
        with torch.enable_grad():
            logits = self.model(x)
            if class_idx is None:
                idx = logits.argmax(1)
            elif isinstance(class_idx, int):
                idx = torch.full((x.shape[0],), class_idx, dtype=torch.long, device=x.device)
            else:
                idx = class_idx
            self.model.zero_grad(set_to_none=True)
            logits.gather(1, idx[:, None]).sum().backward()
        if self._act is None or self._grad is None:
            raise GradFlowError("no gradient reached the selected layer; check that it lies on the path to the logits")
        if float(self._grad.abs().sum()) == 0.0:
            raise GradFlowError("gradient at the selected layer is identically zero")
        act, grad = self._act.detach(), self._grad.detach()
        if self.reshape is not None:
            act, grad = self.reshape(act), self.reshape(grad)
        if self.method == "gradcam":
            w = grad.mean(dim=(2, 3), keepdim=True)
        else:
            g2, g3 = grad**2, grad**3
            denom = 2 * g2 + act.sum(dim=(2, 3), keepdim=True) * g3
            alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
            w = (alpha * F.relu(grad)).sum(dim=(2, 3), keepdim=True)
        cam = F.relu((w * act).sum(1, keepdim=True))
        if float(cam.abs().sum()) == 0.0:
            raise GradFlowError(
                "attribution map is identically zero. For transformers pooled by the class token, the OUTPUT of the last block "
                "carries no patch-token gradient; choose an earlier layer (e.g. blocks[-1].norm1)."
            )
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        return cam, logits.detach(), idx


def normalise_cam(cam: torch.Tensor) -> torch.Tensor:
    flat = cam.flatten(1)
    lo, hi = flat.min(1)[0][:, None, None], flat.max(1)[0][:, None, None]
    return (cam - lo) / (hi - lo).clamp_min(1e-12)


def target_layer(model: nn.Module, branch: str) -> tuple[nn.Module, Callable | None, str]:
    """Default layers: CoAtNet last stage ('a'), DINOv2 last block with token reshape ('b')."""
    if branch == "a":
        return model.branch_a.stages[-1], None, "branch_a.stages[-1]"
    if branch == "b":
        # NOT blocks[-1] itself: only the class token is pooled, so the OUTPUT of the last block has zero gradient on
        # every patch token (verified in tests). The input of the last attention (norm1 output) still lets patch tokens
        # influence the class token, which gives a non-degenerate map.
        return model.branch_b.blocks[-1].norm1, vit_tokens_to_map, "branch_b.blocks[-1].norm1 (input to the last attention; class token dropped, patch grid reshaped)"
    raise ValueError("branch must be 'a' (CoAtNet) or 'b' (DINOv2)")


def export_attribution(out_dir: str | Path, name: str, image01: np.ndarray, cam_raw: np.ndarray, settings: dict) -> dict:
    """Save raw array (.npy), a rendered overlay (.png) and the settings (.json). ``image01``: HxWx3 in [0,1]."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{name}.cam_raw.npy", cam_raw)
    rng = cam_raw.max() - cam_raw.min()
    cam01 = (cam_raw - cam_raw.min()) / rng if rng > 0 else np.zeros_like(cam_raw)
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(image01)
    ax[0].set_title("input")
    ax[1].imshow(cam01, cmap="jet", vmin=0, vmax=1)
    ax[1].set_title("attribution (min-max)")
    ax[2].imshow(image01)
    ax[2].imshow(cam01, cmap="jet", alpha=0.4, vmin=0, vmax=1)
    ax[2].set_title("overlay")
    for a in ax:
        a.axis("off")
    fig.savefig(out / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    meta = {**settings, "render": {"colormap": "jet", "overlay_alpha": 0.4, "normalisation": "per-image min-max (display only)"},
            "note": "attribution of the fused model's logit through one branch layer; not a segmentation mask"}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta
