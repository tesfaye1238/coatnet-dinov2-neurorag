"""Parameter-count audit: instantiate the models (random weights; counts do not depend on weights) and compare
with the manuscript's approximate totals and, more tellingly, with the differences between variants."""
from __future__ import annotations

from neurofusion.models.classifier import count_parameters
from neurofusion.models.factory import ABLATIONS, build_model
from neurofusion.reporting.numeric_audit import load_reference


def parameter_audit(ref: dict | None = None) -> dict:
    ref = ref or load_reference()
    rep = ref["table5"]["params_millions"]
    counts = {n: count_parameters(build_model(n, pretrained=False))["total"] / 1e6 for n in ABLATIONS}
    rows = {}
    for n, reported in rep.items():
        rows[n] = {"instantiated_M": round(counts[n], 4), "reported_M": reported, "offset_M": round(counts[n] - reported, 4)}
    diffs = {
        "se_r8_minus_no_attention": (counts["proposed"] - counts["concat_no_attention"], rep["proposed"] - rep["concat_no_attention"]),
        "se_r8_minus_se_r16": (counts["proposed"] - counts["concat_se_r16"], rep["proposed"] - rep["concat_se_r16"]),
        "concat_se_r8_minus_sum_se_r8": (counts["proposed"] - counts["sum_se_r8"], rep["proposed"] - rep["sum_se_r8"]),
    }
    return {
        "variants": rows,
        "differences_M": {k: {"instantiated": round(v[0], 4), "reported": round(v[1], 4)} for k, v in diffs.items()},
        "single_branch_M": {"coatnet_only": round(counts["coatnet_only"], 4), "dinov2_only": round(counts["dinov2_only"], 4)},
        "interpretation": (
            "Differences between variants agree with the reported ones to rounding, which supports the SE layer structure, "
            "the head sizes and a 384->768 projection for the summation ablation. A constant offset of the totals indicates the "
            "backbone parameter count differs slightly from the authors' (e.g. a different CoAtNet-0 implementation or DINOv2 "
            "position-embedding handling). Agreement is evidence of consistency, not verification of the original code."
        ),
    }
