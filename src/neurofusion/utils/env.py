"""Environment and provenance capture."""
from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
import sys
from pathlib import Path

TRACKED_PACKAGES = (
    "torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn", "scipy", "pillow", "pyyaml",
    "matplotlib", "opencv-python-headless", "h5py", "qdrant-client", "sentence-transformers",
    "transformers", "pypdf", "streamlit", "bert-score",
)


def package_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def git_commit(repo: str | Path | None = None) -> str | None:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo) if repo else None,
            capture_output=True, text=True, timeout=10, check=False,
        )
        return res.stdout.strip() or None if res.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def git_dirty(repo: str | Path | None = None) -> bool | None:
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"], cwd=str(repo) if repo else None,
            capture_output=True, text=True, timeout=10, check=False,
        )
        return bool(res.stdout.strip()) if res.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_info() -> dict:
    import torch

    info: dict = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "torch_threads": torch.get_num_threads(),
        "cpu": platform.processor() or platform.machine(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_memory_mb"] = round(props.total_memory / 2**20)
    return info


def environment_info(repo: str | Path | None = None) -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
        "hardware": hardware_info(),
        "git_commit": git_commit(repo),
        "git_dirty": git_dirty(repo),
    }
