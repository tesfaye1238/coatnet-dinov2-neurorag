"""Source registry and verified download adapters.

Entries were checked against the public record APIs on the access date given. Nothing here is inferred from
an aggregator's folder names. Sources that require authentication have no download adapter by design.
"""
from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from neurofusion.utils.hashing import md5_file


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteFile:
    name: str
    url: str
    size_bytes: int
    md5: str


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    title: str
    kind: str  # "original_collection" | "aggregator_dataset" | "model_record_not_dataset"
    original_urls: tuple[str, ...]
    redistribution_urls: tuple[str, ...]
    release: str
    access_date: str
    license: str
    verification: str  # how the entry was verified
    label_definitions: str
    patient_metadata: str
    acquisition_metadata: str
    requires_authentication: bool
    contains_images: bool
    files: tuple[RemoteFile, ...] = field(default_factory=tuple)
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


ACCESS_DATE = "2026-11-14"

FIGSHARE = SourceRecord(
    source_id="figshare_cheng_v8",
    title="brain tumor dataset (Cheng, Jun)",
    kind="original_collection",
    original_urls=("https://doi.org/10.6084/m9.figshare.1512427.v8", "https://figshare.com/articles/dataset/brain_tumor_dataset/1512427"),
    redistribution_urls=("https://api.figshare.com/v2/articles/1512427",),
    release="v8 (README 2024)",
    access_date=ACCESS_DATE,
    license="CC BY 4.0 (figshare API license field)",
    verification="figshare public API v2, queried on access_date; md5 sums copied from the API 'computed_md5' field",
    label_definitions="cjdata.label: 1 meningioma, 2 glioma, 3 pituitary tumor (README 2024). No no-tumor class.",
    patient_metadata="cjdata.PID: patient ID supplied by the source (233 patients; 3064 slices)",
    acquisition_metadata=(
        "T1-weighted contrast-enhanced; Nanfang Hospital (Guangzhou) and General Hospital, Tianjin Medical University, "
        "2005.9-2010.10; 512x512 in-plane, 0.49x0.49 mm, slice thickness 6 mm, gap 1 mm, Gd 0.1 mmol/kg at 2 ml/s. "
        "README: 'No further details are available'."
    ),
    requires_authentication=False,
    contains_images=True,
    files=(
        RemoteFile("brainTumorDataPublic_1-766.zip", "https://ndownloader.figshare.com/files/3381290", 214401279, "74b949ad33f042e6e103523091cd1428"),
        RemoteFile("brainTumorDataPublic_767-1532.zip", "https://ndownloader.figshare.com/files/3381296", 217848429, "7e8a875500d2c8a346f270538e29890e"),
        RemoteFile("brainTumorDataPublic_1533-2298.zip", "https://ndownloader.figshare.com/files/3381293", 215563856, "8227bf6080cb71f15a88be8d25c79ae7"),
        RemoteFile("brainTumorDataPublic_2299-3064.zip", "https://ndownloader.figshare.com/files/3381302", 231679762, "b378a80d6174e5317d59eb28430c6652"),
        RemoteFile("cvind.mat", "https://ndownloader.figshare.com/files/7005344", 5736, "5ea82eb3212c9fcb84290ccf0beb486d"),
        RemoteFile("README_2024.txt", "https://ndownloader.figshare.com/files/51340418", 3303, "4188f7c30957e9673c3611613071bd6e"),
    ),
)

ZENODO_7786009 = SourceRecord(
    source_id="zenodo_7786009_model_record",
    title="20: Brain Tumor MRI Dataset (Mayer, Hartmann, Auer, Krafft)",
    kind="model_record_not_dataset",
    original_urls=("https://doi.org/10.5281/zenodo.7786009",),
    redistribution_urls=("https://zenodo.org/api/records/7786009",),
    release="2023-03-30",
    access_date=ACCESS_DATE,
    license="CC BY 4.0 (Zenodo API license field)",
    verification=(
        "Zenodo records API queried on access_date. The record lists 4 files: 63f293_model.h5, "
        "cbbeb9_model_converted.zip, ec7f46_metrics-javascript.json, 579c53_metrics-python.json. NONE is an image archive. "
        "The description states the model was trained on a combination of figshare, SARTAJ and Br35H (7022 images, 4 classes) "
        "and that the no-tumor class came from Br35H."
    ),
    label_definitions="glioma, meningioma, no tumor, pituitary (record description)",
    patient_metadata="none in the record",
    acquisition_metadata="none in the record",
    requires_authentication=False,
    contains_images=False,
    files=(
        RemoteFile("579c53_metrics-python.json", "https://zenodo.org/api/records/7786009/files/579c53_metrics-python.json/content", 49682, "7f831ef631e3a39b5dd17c0d89ee62ef"),
    ),
    notes=(
        "The manuscript's Data Availability statement cites this record as the dataset. It is a trained-model record. "
        "Its metrics-python.json reports a held-out distribution of glioma 324, meningioma 329, notumor 400, pituitary 351 (n=1404)."
    ),
)

SARTAJ = SourceRecord(
    source_id="sartaj_kaggle",
    title="Brain Tumor Classification (MRI) - Sartaj Bhuvaji et al.",
    kind="aggregator_dataset",
    original_urls=("https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri",),
    redistribution_urls=(),
    release="unverified",
    access_date=ACCESS_DATE,
    license="unverified (check the Kaggle dataset page before redistribution)",
    verification="URL taken from the Zenodo 7786009 description; the Kaggle page itself was not opened (login required)",
    label_definitions="unverified; class folders on Kaggle",
    patient_metadata="unverified (assume none)",
    acquisition_metadata="unverified",
    requires_authentication=True,
    contains_images=True,
    notes="No download adapter: use `import-folder` on a copy you obtained under the Kaggle terms.",
)

BR35H = SourceRecord(
    source_id="br35h_kaggle",
    title="Brain Tumor Detection (Br35H) - Ahmed Hamada",
    kind="aggregator_dataset",
    original_urls=("https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection",),
    redistribution_urls=(),
    release="unverified",
    access_date=ACCESS_DATE,
    license="unverified (check the Kaggle dataset page before redistribution)",
    verification="URL taken from the Zenodo 7786009 description; the Kaggle page itself was not opened (login required)",
    label_definitions="unverified; the Zenodo description uses only its 'no' folder. Binary (yes/no) labels cannot define glioma/meningioma/pituitary.",
    patient_metadata="unverified (assume none)",
    acquisition_metadata="unverified",
    requires_authentication=True,
    contains_images=True,
    notes="No download adapter. Binary labels are rejected by the label mapper.",
)

REGISTRY: dict[str, SourceRecord] = {r.source_id: r for r in (FIGSHARE, ZENODO_7786009, SARTAJ, BR35H)}


def get_source(source_id: str) -> SourceRecord:
    try:
        return REGISTRY[source_id]
    except KeyError as exc:
        raise SourceError(f"Unknown source {source_id!r}; known: {sorted(REGISTRY)}") from exc


def _download(url: str, dest: Path, timeout: int = 60) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "neurofusion-research-client"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as out:  # noqa: S310 - https URLs from the registry
        while block := r.read(1 << 20):
            out.write(block)


def safe_extract_zip(zip_path: Path, dest: Path) -> list[Path]:
    """Extract a zip, rejecting entries that would escape ``dest`` (zip-slip)."""
    dest = dest.resolve()
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = (dest / info.filename).resolve()
            if dest not in target.parents and target != dest:
                raise SourceError(f"Unsafe path in archive {zip_path.name}: {info.filename}")
            zf.extract(info, dest)
            out.append(target)
    return out


def download_figshare(dest: str | Path, only: list[str] | None = None, extract: bool = True) -> list[Path]:
    """Download the verified figshare files, checking each md5 against the registry. Never overwrites a
    verified file; a checksum mismatch raises and the partial file is removed."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    for rf in FIGSHARE.files:
        if only and rf.name not in only:
            continue
        target = dest / rf.name
        if not (target.exists() and md5_file(target) == rf.md5):
            _download(rf.url, target)
            if md5_file(target) != rf.md5:
                target.unlink(missing_ok=True)
                raise SourceError(f"md5 mismatch for {rf.name}; expected {rf.md5}")
        got.append(target)
        if extract and rf.name.endswith(".zip"):
            safe_extract_zip(target, dest / "mat")
    return got
