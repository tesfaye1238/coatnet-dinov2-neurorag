"""Pre-publication checks: secret scan, forbidden/large files, staged-file review. Read-only."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

SECRET_PATTERNS = {
    "github_pat_classic": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "github_pat_fine": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "hf_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "generic_bearer": re.compile(r"(?i)\bauthorization:\s*bearer\s+[A-Za-z0-9._\-]{20,}"),
    "url_with_credentials": re.compile(r"https?://[^/\s:@]+:[^/\s@]{8,}@"),
}

FORBIDDEN_SUFFIXES = {
    ".pdf", ".docx", ".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".mat", ".nii", ".nii.gz", ".dcm",
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".zip", ".tar", ".gz", ".npz",
}
ALLOWED_BINARY_DIRS = ("docs/figures/",)  # figures generated from recorded artifacts may be added explicitly
MAX_BYTES = 2 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".yml", ".yaml", ".json", ".csv", ".cfg", ".ini", ".env", ".example", ""}


def scan_text_for_secrets(text: str) -> list[str]:
    return [name for name, rx in SECRET_PATTERNS.items() if rx.search(text)]


def scan_files(root: str | Path, files: list[str]) -> list[dict]:
    root = Path(root)
    problems = []
    for rel in files:
        p = root / rel
        if not p.is_file():
            continue
        suf = "".join(p.suffixes[-2:]).lower() if p.name.endswith(".nii.gz") else p.suffix.lower()
        if suf in FORBIDDEN_SUFFIXES and not rel.replace("\\", "/").startswith(ALLOWED_BINARY_DIRS):
            problems.append({"file": rel, "problem": f"forbidden file type {suf} (data, weights, documents or archives are not committed)"})
        if p.stat().st_size > MAX_BYTES:
            problems.append({"file": rel, "problem": f"larger than {MAX_BYTES} bytes"})
        if p.name == ".env":
            problems.append({"file": rel, "problem": ".env must not be committed"})
        if p.suffix.lower() in TEXT_SUFFIXES or p.name.startswith(".env"):
            try:
                hits = scan_text_for_secrets(p.read_text(encoding="utf-8", errors="ignore"))
            except OSError:
                hits = []
            for h in hits:
                problems.append({"file": rel, "problem": f"possible secret ({h})"})
    return problems


def git_files(root: str | Path, staged_only: bool = True) -> list[str]:
    args = ["git", "diff", "--cached", "--name-only"] if staged_only else ["git", "ls-files"]
    res = subprocess.run(args, cwd=str(root), capture_output=True, text=True, check=False)
    return [line for line in res.stdout.splitlines() if line.strip()]
