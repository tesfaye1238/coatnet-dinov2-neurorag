"""Quick environment check: versions, hardware, and whether the optional components are importable."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurofusion.utils.env import environment_info  # noqa: E402

OPTIONAL = {"h5py": "mat", "qdrant_client": "rag", "sentence_transformers": "rag", "transformers": "rag", "pypdf": "rag", "bert_score": "rag-eval", "streamlit": "app"}


def main() -> int:
    info = environment_info()
    info["optional_components"] = {m: {"extra": e, "available": importlib.util.find_spec(m) is not None} for m, e in OPTIONAL.items()}
    print(json.dumps(info, indent=2, default=str))
    if not info["hardware"]["cuda_available"]:
        print("\nNOTE: no CUDA device visible to PyTorch. Full training will be very slow on CPU.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
