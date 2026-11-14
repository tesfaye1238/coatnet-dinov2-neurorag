# Data provenance

Checked on **2026-11-14** against public record APIs. Machine-readable form: `neurofusion registry`.

| Source | Kind | Images | Licence | Auth | Verified how |
|---|---|---|---|---|---|
| `figshare_cheng_v8` | original collection | 3,064 T1-CE slices, 233 patients, 3 tumour classes (no no-tumor) | CC BY 4.0 | none | figshare API v2: files, sizes and md5 sums copied into the registry; README 2024 read |
| `zenodo_7786009_model_record` | **trained-model record, not a dataset** | 0 | CC BY 4.0 | none | Zenodo API: 4 files, none an image archive; the description names the three sources and says no-tumor came from Br35H |
| `sartaj_kaggle` | aggregator | unverified | **unverified** | Kaggle login | URL only from the Zenodo description; page not opened |
| `br35h_kaggle` | aggregator | unverified | **unverified** | Kaggle login | URL only from the Zenodo description; binary yes/no labels |

## What this means

* The manuscript's Data Availability statement points to a model record. The benchmark must be rebuilt from the three originals, and the
  result cannot be assumed identical to the authors' 7,022 images.
* Only figshare supplies patient ids (`cjdata.PID`). For the other sources patient identity is missing and stays missing; any grouping there is
  heuristic (`grouping_method`) and does not establish patient-level independence.
* Br35H has binary tumour/no-tumour labels. It can contribute a no-tumor class only through its documented "no" folder and cannot define
  glioma, meningioma or pituitary. `normalize_label` rejects `yes`, `no` and `tumor`.
* figshare `.mat` images are converted to 8-bit PNG by per-image min-max scaling (the README procedure), which discards dynamic range.
* SARTAJ and Br35H are redistributions with their own terms. This repository downloads nothing from Kaggle and commits no images.

## Manifest

Columns: `image_id, relative_path, source_id, source_release, label, patient_id, patient_id_verified, group_id, grouping_method, file_sha256,
decoded_image_hash, width, height, split`. `patient_id_verified` may be true only when a source-supplied id is present (validated). Raw files
are never modified or deleted; duplicate decisions are written to logs.

## Duplicate screening

1. Exact file hash. 2. Decoded-pixel hash (catches re-encoded copies). 3. Perceptual dHash/pHash **candidate** pairs exported for manual review.
Perceptual similarity does not prove a common patient. `merge_groups_by_pairs` keeps reviewed related images in one group.

## Splits

Fixed group-stratified split; five-fold grouped stratified folds with validation drawn from each fold's training pool; source-held-out with a
class-availability and cross-source-overlap audit; import of supplied split files (never modified). Group integrity outranks target counts;
`audit-manifest --expected` prints realised minus Table 1 counts.
