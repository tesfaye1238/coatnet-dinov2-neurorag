"""Reference-collection source manifest and document readers.

The number of indexed entries always comes from the actual collection. Nothing here invents documents.
"""
from __future__ import annotations

import html.parser
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from neurofusion.constants import CLASS_NAMES
from neurofusion.utils.hashing import sha256_file

REQUIRED_FIELDS = (
    "doc_id", "title", "issuer", "url_or_doi", "version_date", "access_date", "license", "sha256",
    "category_tags", "review_status", "path",
)
REVIEW_STATUSES = ("approved", "pending", "rejected", "demo_synthetic")


class SourceManifestError(ValueError):
    pass


@dataclass(frozen=True)
class SourceDocument:
    doc_id: str
    title: str
    issuer: str
    url_or_doi: str
    version_date: str
    access_date: str
    license: str
    sha256: str
    category_tags: tuple[str, ...]
    review_status: str
    path: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["category_tags"] = ";".join(self.category_tags)
        return d


def load_source_manifest(path: str | Path) -> list[SourceDocument]:
    path = Path(path)
    df = pd.read_json(path) if path.suffix == ".json" else pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        raise SourceManifestError(f"source manifest missing columns {missing}")
    docs = []
    for r in df.to_dict("records"):
        tags = tuple(t.strip() for t in str(r["category_tags"]).split(";") if t.strip())
        bad = [t for t in tags if t not in CLASS_NAMES and t != "general"]
        if bad:
            raise SourceManifestError(f"{r['doc_id']}: unknown category tags {bad}")
        if r["review_status"] not in REVIEW_STATUSES:
            raise SourceManifestError(f"{r['doc_id']}: review_status must be one of {REVIEW_STATUSES}")
        docs.append(SourceDocument(**{**{k: str(r[k]) for k in REQUIRED_FIELDS if k != "category_tags"}, "category_tags": tags}))
    ids = [d.doc_id for d in docs]
    if len(set(ids)) != len(ids):
        raise SourceManifestError("duplicate doc_id in source manifest")
    return docs


def verify_documents(docs: list[SourceDocument], base_dir: str | Path) -> list[str]:
    """Return a list of problems (missing files, checksum mismatches). Empty list = verified."""
    problems = []
    for d in docs:
        p = Path(base_dir) / d.path
        if not p.is_file():
            problems.append(f"{d.doc_id}: file missing ({p})")
        elif d.sha256 and sha256_file(p) != d.sha256:
            problems.append(f"{d.doc_id}: sha256 mismatch")
    return problems


def indexable(docs: list[SourceDocument], allow_demo: bool = False, allow_pending: bool = False) -> list[SourceDocument]:
    ok = {"approved"} | ({"demo_synthetic"} if allow_demo else set()) | ({"pending"} if allow_pending else set())
    return [d for d in docs if d.review_status in ok]


class _Text(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def read_document_text(path: str | Path) -> str:
    p = Path(path)
    suf = p.suffix.lower()
    if suf in {".txt", ".md"}:
        return p.read_text(encoding="utf-8", errors="replace")
    if suf in {".html", ".htm"}:
        t = _Text()
        t.feed(p.read_text(encoding="utf-8", errors="replace"))
        return " ".join(" ".join(t.parts).split())
    if suf == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pypdf is required for PDF sources: pip install 'neurofusion[rag]'") from exc
        return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)
    if suf == ".json":
        return json.dumps(json.loads(p.read_text(encoding="utf-8")))
    raise SourceManifestError(f"unsupported document format {suf!r}")
