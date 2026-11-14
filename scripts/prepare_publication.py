"""Review what would be published: list files, flag secrets / data / weights / large files. Read-only.

    python scripts/prepare_publication.py            # staged files
    python scripts/prepare_publication.py --tracked  # all tracked files
Exit status 1 when any problem is found.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neurofusion.utils.publication import git_files, scan_files  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracked", action="store_true")
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    files = git_files(a.root, staged_only=not a.tracked)
    print(f"{len(files)} file(s) under review")
    for f in files:
        print("  ", f)
    problems = scan_files(a.root, files)
    for p in problems:
        print("PROBLEM:", p["file"], "-", p["problem"])
    print("OK: no problems found" if not problems else f"{len(problems)} problem(s) found")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
