# Dependency and secret checks

Run on 2026-11-14 in the tested environment (`pip-audit -r requirements-lock.txt --no-deps`, advisory database as of that date).

| Package | Tested pin | Result | Action |
|---|---|---|---|
| pypdf | 7.1.0 | Initially 6.21.2 had 7 advisories (fixed in 6.21.3 / 6.21.4). Upgraded; clean. | `pypdf>=6.21.4` in `pyproject.toml` |
| pillow | 12.3.0 | Advisories fixed only in **13.0.0, which requires Python >= 3.11**. No fixed release exists for Python 3.10. | `pillow>=13.0` is required on Python >= 3.11; the Python 3.10 environment used here is **known-vulnerable** and should not be used to open untrusted images |
| torch, torchvision | 2.13.0+cpu, 0.28.0+cpu | Local build tags cannot be audited against PyPI | none |

Practical consequence: the Streamlit demo and every image loader decode user-supplied images with Pillow. Use Python 3.11+ with Pillow >= 13
for anything beyond local synthetic testing. The demo additionally limits upload size (20 MB), side length (32 to 8192 px) and rejects
decompression bombs, and never writes uploads to disk.

`scripts/prepare_publication.py` scans staged files for GitHub, AWS, Hugging Face and private-key patterns and for forbidden file types
(data, weights, archives, documents) and files over 2 MB. It is also run in CI.
