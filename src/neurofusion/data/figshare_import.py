"""Import figshare ``cjdata`` .mat (MATLAB v7.3 / HDF5) files into PNG images plus a manifest.

Conversion follows the source README: per-image min-max scaling to uint8. That scaling discards the original
dynamic range, which is a preprocessing decision recorded here, not a property of the source images.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from neurofusion.constants import MANIFEST_COLUMNS
from neurofusion.data.labels import figshare_label
from neurofusion.data.manifest import decoded_image_hash
from neurofusion.utils.hashing import sha256_file


def read_cjdata(path: str | Path) -> dict:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("h5py is required: pip install 'neurofusion[mat]'") from exc
    with h5py.File(path, "r") as f:
        g = f["cjdata"]
        label = int(np.array(g["label"]).squeeze())
        pid_arr = np.array(g["PID"]).flatten()
        pid = "".join(chr(int(c)) for c in pid_arr if int(c) != 0)
        image = np.array(g["image"]).T  # MATLAB stores column-major
    return {"label": label, "pid": pid, "image": image}


def minmax_to_uint8(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float64)
    lo, hi = a.min(), a.max()
    if hi == lo:
        return np.zeros(a.shape, dtype=np.uint8)
    return np.rint(255.0 * (a - lo) / (hi - lo)).astype(np.uint8)


def import_figshare_mats(
    mat_dir: str | Path, out_dir: str | Path, source_id: str = "figshare_cheng_v8", release: str = "v8"
) -> pd.DataFrame:
    mat_dir, out_dir = Path(mat_dir), Path(out_dir)
    files = sorted(mat_dir.rglob("*.mat"), key=lambda p: (len(p.stem), p.stem))
    files = [p for p in files if p.name != "cvind.mat"]
    if not files:
        raise FileNotFoundError(f"No cjdata .mat files under {mat_dir}")
    rows = []
    for p in files:
        rec = read_cjdata(p)
        label = figshare_label(rec["label"])
        dest = out_dir / label / f"{p.stem}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(minmax_to_uint8(rec["image"])).save(dest)
        rel = dest.relative_to(out_dir).as_posix()
        dec, w, h = decoded_image_hash(dest)
        rows.append(
            {
                "image_id": f"{source_id}/{rel}",
                "relative_path": rel,
                "source_id": source_id,
                "source_release": release,
                "label": label,
                "patient_id": rec["pid"],
                "patient_id_verified": bool(rec["pid"]),  # source-supplied (README: cjdata.PID = patient ID)
                "group_id": f"patient:{source_id}:{rec['pid']}" if rec["pid"] else f"img:{source_id}/{rel}",
                "grouping_method": "verified_patient_id" if rec["pid"] else "singleton",
                "file_sha256": sha256_file(dest),
                "decoded_image_hash": dec,
                "width": w,
                "height": h,
                "split": "",
            }
        )
    return pd.DataFrame(rows, columns=list(MANIFEST_COLUMNS))
