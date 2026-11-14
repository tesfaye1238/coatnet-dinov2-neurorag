"""Label normalisation. Unknown or binary labels are rejected, never guessed."""
from __future__ import annotations

from neurofusion.constants import CLASS_NAMES


class LabelError(ValueError):
    """Raised when a label cannot be mapped to the four-class task without inventing information."""


_ALIASES = {
    "glioma": "glioma",
    "glioma_tumor": "glioma",
    "glioma tumor": "glioma",
    "glioma-tumor": "glioma",
    "meningioma": "meningioma",
    "meningioma_tumor": "meningioma",
    "meningioma tumor": "meningioma",
    "meningioma-tumor": "meningioma",
    "notumor": "notumor",
    "no_tumor": "notumor",
    "no tumor": "notumor",
    "no-tumor": "notumor",
    "pituitary": "pituitary",
    "pituitary_tumor": "pituitary",
    "pituitary tumor": "pituitary",
    "pituitary-tumor": "pituitary",
}

# Labels that carry only tumour present/absent information. They cannot define a 4-class label.
_BINARY = {"yes", "no", "tumor", "tumour", "normal", "abnormal", "positive", "negative", "pathological"}

# figshare 'cjdata.label' (README 2024): 1 meningioma, 2 glioma, 3 pituitary tumor
FIGSHARE_LABEL_MAP = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def normalize_label(raw: object) -> str:
    key = str(raw).strip().lower()
    if key in _ALIASES:
        return _ALIASES[key]
    if key in _BINARY:
        raise LabelError(
            f"Label {raw!r} is a binary tumour/no-tumour label. It cannot be converted to glioma, "
            "meningioma or pituitary without supporting annotations. Run a separately defined binary analysis."
        )
    raise LabelError(f"Unknown label {raw!r}; expected one of {CLASS_NAMES} (or a documented alias).")


def figshare_label(code: int) -> str:
    try:
        return FIGSHARE_LABEL_MAP[int(code)]
    except (KeyError, ValueError) as exc:
        raise LabelError(f"figshare label code {code!r} not in {sorted(FIGSHARE_LABEL_MAP)}") from exc
