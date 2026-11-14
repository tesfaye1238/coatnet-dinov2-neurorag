# Reproducibility

## Environment actually tested

Windows 11, Python 3.10.11, CPU-only PyTorch 2.13.0 (no CUDA build although an RTX 3050 6 GB exists), torchvision 0.28.0, timm 1.0.28,
numpy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, scipy 1.15.3, qdrant-client 1.20.0, streamlit 1.65.0, h5py 3.12.1, pypdf 6.21.2, ruff 0.16.10.
`requirements-lock.txt` is the `pip freeze` of that environment (it pins what was tested). CI installs from the `pyproject.toml` ranges on Linux
(Python 3.10 to 3.12, CPU torch). It has not run on GitHub because publication was blocked (see README).

## What was and was not executed

Executed: the unit and integration test suite on synthetic data; the numerical audit; the parameter audit; instantiation of every reported
model; download of the two pretrained backbones with dimension checks.

**Not executed:** any training on real MRI, any evaluation on real data, cross-validation, source-held-out, attribution figures, robustness
runs, efficiency tables, Qwen generation, BERTScore, the 200-query RAG study and clinician review. No result in this repository is a measurement
on the manuscript's data.

## Determinism

`set_seed` seeds Python, NumPy and PyTorch (and CUDA), requests deterministic kernels (warnings where unavailable) and disables cuDNN
benchmarking; DataLoader generators are seeded. Bitwise identity across hardware, library versions and GPU models is not guaranteed.

Every run writes `run_record.json`: resolved train, augmentation and model config with a hash, class weights, dataset and split fingerprints,
the split audit, the environment (packages, hardware, git commit and dirty flag), pretrained provenance, seed, history, best epoch and a status
line ("NEW EXPERIMENT ... not verified to reproduce the manuscript's original run"). Checkpoints hold model, optimiser, scheduler and scaler
state, the class mapping and the config hash, and are resumable (`last.pt`).

## Commands

```
neurofusion env
neurofusion registry [--source ID]
neurofusion download-figshare --dest data/figshare
neurofusion import-figshare --mat-dir data/figshare/mat --out-dir data/images/figshare --manifest data/manifest_figshare.csv
neurofusion import-folder --root data/images/other --source-id sartaj_kaggle --manifest data/manifest_other.csv
neurofusion dedup --manifest M.csv --root IMAGES --out reports/dedup
neurofusion split --manifest M.csv --mode fixed|cv|source|import --out splits [--test-source ID] [--split-csv F]
neurofusion train --manifest splits/manifest_fixed.csv --root IMAGES --out runs/proposed --config configs/proposed.yaml
neurofusion seeds --manifest ... --root ... --out runs/seeds --seeds 0,1,2
neurofusion cv --manifests splits/manifest_fold0.csv splits/manifest_fold1.csv ... --root IMAGES --out runs/cv
neurofusion predict --checkpoint runs/proposed/best.pt --manifest ... --root ... --out preds/proposed.csv
neurofusion evaluate --predictions preds/proposed.csv
neurofusion compare --reference preds/proposed.csv --others preds/a.csv preds/b.csv --family baselines
neurofusion gradcam --checkpoint ... --image IMG --out figs/cam --branch a --method gradcampp
neurofusion robustness --checkpoint ... --manifest ... --root ... --out reports/robustness.csv
neurofusion efficiency --model proposed --out reports/efficiency.json
neurofusion report-tables --predictions proposed=preds/proposed.csv ... --out reports/tables
neurofusion figures --predictions preds/proposed.csv --out figs
neurofusion audit-manuscript --out docs/numerical_audit.md
neurofusion audit-parameters --out docs/parameter_audit.json
neurofusion rag-demo-collection --out rag_sources/demo
neurofusion rag-build --sources rag_sources/demo/source_manifest.csv --allow-demo --out rag_index
neurofusion rag-generate --index rag_index --category glioma --confidence 0.93
neurofusion rag-eval --run run_and_qrels.json
neurofusion claims --annotations annotated_claims.csv
```

`--max-epochs N` is a smoke-test limit. A run it truncates is flagged `truncated_for_smoke_test`.
