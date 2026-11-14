"""Logic behind the local research demonstration (UI-free, testable).

Principles: explicit checkpoint only; no invented predictions when weights are missing; uploads are validated and
held in memory only (never written to disk); no images or clinical information leave the machine; every output is
labelled experimental and a model score is never presented as diagnostic certainty.
"""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from neurofusion.constants import CLASS_NAMES
from neurofusion.data.transforms import build_eval_transform
from neurofusion.rag.retrieval import DISPLAY

ALLOWED_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_SIDE, MAX_SIDE = 32, 8192

EXPERIMENTAL_BANNER = (
    "EXPERIMENTAL RESEARCH PROTOTYPE. The output is an image-category prediction from a model trained on a public benchmark. "
    "It is not a diagnosis, not clinically validated, and a model score is not diagnostic certainty. "
    "A 'no tumor' result does not exclude other abnormalities."
)
PRIVACY_NOTE = (
    "Uploaded images are processed in memory on this machine and are not saved. Nothing is sent to an external service unless "
    "you enable the optional language model, which downloads model weights (no image is transmitted)."
)


class UploadError(ValueError):
    pass


class WeightsMissing(RuntimeError):
    pass


def validate_upload(data: bytes, filename: str) -> Image.Image:
    """Return a loaded RGB PIL image or raise :class:`UploadError`. Nothing is written to disk."""
    if not data:
        raise UploadError("empty file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"file larger than {MAX_UPLOAD_BYTES // 2**20} MB")
    if Path(filename).suffix.lower() not in ALLOWED_SUFFIXES:
        raise UploadError(f"unsupported file type; allowed: {', '.join(ALLOWED_SUFFIXES)}")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        im = Image.open(io.BytesIO(data))
        im.load()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise UploadError(f"not a readable image: {exc}") from exc
    w, h = im.size
    if min(w, h) < MIN_SIDE or max(w, h) > MAX_SIDE:
        raise UploadError(f"image side must be between {MIN_SIDE} and {MAX_SIDE} px (got {w}x{h})")
    return im.convert("RGB")


@dataclass
class LoadedModel:
    model: torch.nn.Module
    checkpoint: str
    config_hash: str | None
    model_name: str


def load_checkpoint_model(path: str | Path, model_name: str = "proposed", builder=None) -> LoadedModel:
    """Load weights from an EXPLICIT checkpoint path. Missing file -> :class:`WeightsMissing` (no fake prediction)."""
    from neurofusion.training.checkpoint import CheckpointError, load_checkpoint

    p = Path(str(path)).expanduser()
    if not str(path).strip() or not p.is_file():
        raise WeightsMissing(f"checkpoint not found: {path!s}. Train a model or provide a checkpoint; no prediction will be invented.")
    try:
        ck = load_checkpoint(p)
    except CheckpointError as exc:
        raise WeightsMissing(str(exc)) from exc
    if builder is None:
        from neurofusion.models.factory import build_model

        model = build_model(model_name, pretrained=False)
    else:
        model = builder()
    model.load_state_dict(ck["model_state"])
    return LoadedModel(model.eval(), str(p), ck.get("config_hash"), model_name)


def predict_image(lm: LoadedModel, image: Image.Image) -> dict:
    x = build_eval_transform()(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(lm.model(x), 1)[0].numpy()
    top = int(np.argmax(probs))
    return {
        "predicted_category": CLASS_NAMES[top],
        "predicted_display": DISPLAY[CLASS_NAMES[top]],
        "model_score": float(probs[top]),
        "probabilities": {c: float(p) for c, p in zip(CLASS_NAMES, probs)},
        "checkpoint": lm.checkpoint,
        "config_hash": lm.config_hash,
        "label": "model score (not diagnostic certainty)",
    }
