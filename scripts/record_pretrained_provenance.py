"""Download the pretrained backbones, verify pooled dimensions and record hub identifiers + commit SHAs.

Needs network. Writes docs/pretrained_provenance.json. Run:  python scripts/record_pretrained_provenance.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from neurofusion.models.backbones import (  # noqa: E402
    COATNET0,
    DINOV2_S14,
    create_backbone,
    pretrained_provenance,
)


def hub_sha(repo_id: str) -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(repo_id).sha
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc}"


def main() -> None:
    out = {}
    for spec in (COATNET0, DINOV2_S14):
        m = create_backbone(spec, pretrained=True)
        info = pretrained_provenance(spec)
        with torch.no_grad():
            dim = m.eval()(torch.zeros(1, 3, 224, 224)).shape[1]
        info.update(
            verified_pooled_dim=int(dim),
            parameters_millions=round(sum(p.numel() for p in m.parameters()) / 1e6, 4),
            hub_commit_sha_at_download=hub_sha(info["hf_hub_id"]) if info.get("hf_hub_id") else None,
        )
        out[spec.name] = info
    path = Path(__file__).resolve().parents[1] / "docs" / "pretrained_provenance.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
