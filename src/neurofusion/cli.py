"""Command-line interface. Every command shown in the README is implemented here."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


def _jprint(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _load_cfg(path: str | None) -> dict:
    return yaml.safe_load(Path(path).read_text()) if path else {}


# ---------------------------------------------------------------- data
def cmd_env(a):
    from neurofusion.utils.env import environment_info

    _jprint(environment_info())


def cmd_registry(a):
    from neurofusion.data.sources import REGISTRY

    _jprint({k: v.as_dict() for k, v in REGISTRY.items()} if not a.source else REGISTRY[a.source].as_dict())


def cmd_download_figshare(a):
    from neurofusion.data.sources import download_figshare

    print("\n".join(str(p) for p in download_figshare(a.dest, extract=not a.no_extract)))


def cmd_import_figshare(a):
    from neurofusion.data.figshare_import import import_figshare_mats
    from neurofusion.data.manifest import save_manifest

    df = import_figshare_mats(a.mat_dir, a.out_dir)
    save_manifest(df, a.manifest)
    print(f"{len(df)} images, {df['patient_id'].nunique()} patients -> {a.manifest}")


def cmd_import_folder(a):
    from neurofusion.data.manifest import build_manifest_from_folder, save_manifest

    df = build_manifest_from_folder(a.root, a.source_id, a.release)
    save_manifest(df, a.manifest)
    print(f"{len(df)} images -> {a.manifest} (patient ids missing: labels only; patient-level independence NOT established)")


def cmd_audit_manifest(a):
    from neurofusion.constants import MANUSCRIPT_TABLE1
    from neurofusion.data.manifest import load_manifest
    from neurofusion.data.splits import audit_split

    df = load_manifest(a.manifest)
    rep = {"n_images": len(df), "label_counts": df["label"].value_counts().to_dict(),
           "source_counts": df["source_id"].value_counts().to_dict(),
           "exact_file_duplicates": int(df["file_sha256"].duplicated().sum()),
           "exact_decoded_duplicates": int(df["decoded_image_hash"].duplicated().sum())}
    if (df["split"] != "").any():
        rep["split_audit"] = audit_split(df, MANUSCRIPT_TABLE1 if a.expected else None)
    _jprint(rep)


def cmd_dedup(a):
    from neurofusion.data.dedup import exact_duplicate_groups, export_candidates, perceptual_candidate_pairs
    from neurofusion.data.manifest import load_manifest

    df = load_manifest(a.manifest)
    exact = exact_duplicate_groups(df)
    pairs = perceptual_candidate_pairs(df, a.root, a.method, a.max_hamming)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    exact.to_csv(Path(a.out) / "exact_duplicates.csv", index=False)
    export_candidates(pairs, Path(a.out) / "perceptual_candidates.csv")
    print(f"exact duplicate rows: {len(exact)}; perceptual candidate pairs (for MANUAL review): {len(pairs)}")


def cmd_split(a):
    from neurofusion.data import splits as S
    from neurofusion.data.grouping import assign_groups
    from neurofusion.data.manifest import load_manifest, save_manifest

    df = load_manifest(a.manifest, validate=False)
    if a.filename_pattern:
        df = assign_groups(df, a.filename_pattern)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "fixed":
        res = S.group_stratified_split(df, a.val_fraction, a.test_fraction, seed=a.seed)
        save_manifest(res, out / "manifest_fixed.csv")
        _jprint(S.audit_split(res))
    elif a.mode == "cv":
        folds = S.kfold_splits(df, a.k, seed=a.seed)
        for i, f in enumerate(folds):
            save_manifest(f.drop(columns="fold"), out / f"manifest_fold{i}.csv")
        _jprint([S.audit_split(f)["totals"] for f in folds])
    elif a.mode == "source":
        res = S.source_held_out_split(df, a.test_source, seed=a.seed)
        save_manifest(res, out / f"manifest_holdout_{a.test_source}.csv")
        _jprint(S.check_source_compatibility(df, a.test_source))
    elif a.mode == "import":
        res = S.import_split_manifest(df, a.split_csv)
        save_manifest(res, out / "manifest_imported.csv")
        _jprint(S.audit_split(res))


# ---------------------------------------------------------------- training / evaluation
def _model_builder(name: str, pretrained: bool):
    from neurofusion.models.factory import build_model

    return lambda: build_model(name, pretrained=pretrained)


def cmd_train(a):
    from neurofusion.data.manifest import load_manifest
    from neurofusion.data.transforms import AugmentationConfig
    from neurofusion.training.experiment import run_experiment
    from neurofusion.training.trainer import TrainConfig

    cfg = _load_cfg(a.config)
    tc = TrainConfig(**{**cfg.get("train", {}), "seed": a.seed, **({"batch_size": a.batch_size} if a.batch_size else {})})
    ac = AugmentationConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("augmentation", {}).items()})
    rec = run_experiment(load_manifest(a.manifest), a.root, a.out, _model_builder(a.model, not a.no_pretrained), a.model, tc, ac,
                         device=a.device, max_epochs=a.max_epochs, evaluate_test=not a.skip_test)
    _jprint({k: rec[k] for k in ("train_summary", "status") if k in rec} | {"test_accuracy": rec.get("test_report", {}).get("accuracy")})


def cmd_seeds(a):
    from neurofusion.data.manifest import load_manifest
    from neurofusion.training.experiment import run_repeated_seeds

    df = run_repeated_seeds(load_manifest(a.manifest), a.root, a.out, _model_builder(a.model, not a.no_pretrained), a.model,
                            [int(s) for s in a.seeds.split(",")], device=a.device, max_epochs=a.max_epochs)
    print(df.to_string(index=False))


def cmd_cv(a):
    from neurofusion.data.manifest import load_manifest
    from neurofusion.training.experiment import run_cross_validation

    folds = [load_manifest(m) for m in a.manifests]
    df = run_cross_validation(folds, a.root, a.out, _model_builder(a.model, not a.no_pretrained), a.model, device=a.device, max_epochs=a.max_epochs)
    print(df.to_string(index=False))


def cmd_predict(a):
    """Evaluate a checkpoint on a manifest split and save sample-level predictions."""
    import torch

    from neurofusion.data.dataset import ManifestDataset, make_loader
    from neurofusion.data.manifest import load_manifest
    from neurofusion.data.transforms import build_eval_transform
    from neurofusion.evaluation.predict import predict_loader, save_predictions
    from neurofusion.training.checkpoint import load_checkpoint

    df = load_manifest(a.manifest)
    df = df[df["split"] == a.split]
    ck = load_checkpoint(a.checkpoint)
    model = _model_builder(a.model, False)()
    model.load_state_dict(ck["model_state"])
    loader = make_loader(ManifestDataset(df, a.root, build_eval_transform()), a.batch_size, False)
    pred = predict_loader(model, loader, a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    save_predictions(pred, a.out, {"checkpoint": a.checkpoint, "config_hash": ck.get("config_hash"), "split": a.split})
    print(f"{len(pred)} predictions -> {a.out}")


def cmd_evaluate(a):
    from neurofusion.evaluation.calibration import calibration_report
    from neurofusion.evaluation.metrics import classification_report
    from neurofusion.evaluation.predict import load_predictions

    df = load_predictions(a.predictions)
    _jprint({"classification": classification_report(df), "calibration": calibration_report(df)})


def cmd_compare(a):
    from neurofusion.evaluation.bootstrap import paired_bootstrap_difference
    from neurofusion.evaluation.metrics import macro_auc
    from neurofusion.evaluation.paired import compare_to_reference
    from neurofusion.evaluation.predict import load_predictions

    ref = load_predictions(a.reference)
    others = {Path(p).stem: load_predictions(p) for p in a.others}
    table = compare_to_reference(ref, others, family_name=a.family)
    print(table.to_string(index=False))
    for name, df in others.items():
        print(name, "macro-AUC paired bootstrap:", json.dumps(paired_bootstrap_difference(ref, df, macro_auc, n_boot=a.n_boot, seed=a.seed)))


def cmd_gradcam(a):
    import numpy as np
    import torch
    from PIL import Image

    from neurofusion.data.transforms import build_eval_transform, denormalize
    from neurofusion.explainability.gradcam import CAMExtractor, export_attribution, target_layer
    from neurofusion.training.checkpoint import load_checkpoint

    model = _model_builder(a.model, False)()
    model.load_state_dict(load_checkpoint(a.checkpoint)["model_state"])
    layer, reshape, desc = target_layer(model, a.branch)
    x = build_eval_transform()(Image.open(a.image)).unsqueeze(0)
    with CAMExtractor(model, layer, a.method, reshape) as cam_fn:
        cam, logits, idx = cam_fn(x, a.target_class)
    probs = torch.softmax(logits, 1)[0]
    img = denormalize(x[0]).permute(1, 2, 0).numpy()
    meta = export_attribution(a.out, Path(a.image).stem, img, cam[0].numpy(), {
        "method": a.method, "branch": a.branch, "layer": desc, "checkpoint": a.checkpoint, "target_class_index": int(idx[0]),
        "probabilities": probs.tolist(), "preprocessing": "eval transform (RGB, bilinear 224, ImageNet norm)"})
    _jprint(meta)
    np.save(Path(a.out) / f"{Path(a.image).stem}.probs.npy", probs.numpy())


def cmd_robustness(a):
    import torch

    from neurofusion.data.manifest import load_manifest
    from neurofusion.evaluation.robustness import evaluate_robustness
    from neurofusion.training.checkpoint import load_checkpoint

    df = load_manifest(a.manifest)
    df = df[df["split"] == "test"]
    model = _model_builder(a.model, False)()
    model.load_state_dict(load_checkpoint(a.checkpoint)["model_state"])
    res = evaluate_robustness(model, df, a.root, device=a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    print(res.to_string(index=False))


def cmd_efficiency(a):
    from neurofusion.evaluation.efficiency import efficiency_report

    model = _model_builder(a.model, False)()
    rep = efficiency_report(model, iters=a.iters, device=a.device)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
    _jprint(rep)


def cmd_audit_manuscript(a):
    from neurofusion.reporting.numeric_audit import render_markdown, run_audit

    md = render_markdown(run_audit())
    Path(a.out).write_text(md, encoding="utf-8")
    print(md.split("\n\n")[3] if len(md.split("\n\n")) > 3 else "")
    print(f"written {a.out}")


def cmd_audit_parameters(a):
    from neurofusion.reporting.parameter_audit import parameter_audit

    rep = parameter_audit()
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    _jprint(rep)


def cmd_report_tables(a):
    from neurofusion.reporting.tables import per_class_table, results_table, save_table

    files = dict(kv.split("=", 1) for kv in a.predictions)
    tbl = results_table(files, n_boot=a.n_boot)
    save_table(tbl, a.out, "results_overall")
    first = next(iter(files.values()))
    save_table(per_class_table(first), a.out, "results_per_class")
    print(tbl.to_string(index=False))


def cmd_figures(a):
    from neurofusion.reporting.tables import plot_confusion

    print(plot_confusion(a.predictions, Path(a.out) / "confusion_matrix.png"))


# ---------------------------------------------------------------- RAG
def _embedder(a):
    from neurofusion.rag.embedding import HashingEmbedder, SentenceTransformerEmbedder

    if a.embedder == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(a.embedder_model, a.embedder_revision)


def _tokenizer(a):
    from neurofusion.rag.chunking import HFTokenizerAdapter, WhitespaceTokenizer

    if a.embedder == "hashing":
        return WhitespaceTokenizer()
    from transformers import AutoTokenizer

    return HFTokenizerAdapter(AutoTokenizer.from_pretrained(a.embedder_model, revision=a.embedder_revision))


def cmd_rag_demo_collection(a):
    from neurofusion.rag.demo_collection import write_demo_collection

    print(write_demo_collection(a.out), "(SYNTHETIC demonstration passages; not sourced references)")


def cmd_rag_build(a):
    from neurofusion.rag.index import NumpyStore, QdrantStore
    from neurofusion.rag.retrieval import build_index
    from neurofusion.rag.sources import indexable, load_source_manifest, verify_documents

    docs = load_source_manifest(a.sources)
    base = Path(a.sources).parent
    problems = verify_documents(docs, base)
    if problems:
        sys.exit("source verification failed:\n" + "\n".join(problems))
    use = indexable(docs, allow_demo=a.allow_demo)
    emb = _embedder(a)
    store = QdrantStore(emb.dim, a.qdrant, recreate=True) if a.store == "qdrant" else NumpyStore(emb.dim)
    rep = build_index(use, base, _tokenizer(a), emb, store, a.max_tokens, a.overlap)
    rep["excluded_unreviewed"] = len(docs) - len(use)
    if a.store == "numpy":
        store.save(a.out)
    _jprint(rep)


def _load_store(a, emb):
    from neurofusion.rag.index import NumpyStore, QdrantStore

    if a.store == "numpy":
        return NumpyStore.load(a.index)
    return QdrantStore(emb.dim, a.qdrant, recreate=False)


def cmd_rag_generate(a):
    from neurofusion.rag.generation import ExtractiveGenerator, GenerationConfig, QwenGenerator
    from neurofusion.rag.pipeline import NeuroRAGPipeline
    from neurofusion.rag.retrieval import Retriever
    from neurofusion.rag.safeguards import SafeguardPolicy

    emb = _embedder(a)
    retr = Retriever(_load_store(a, emb), emb)
    gen = ExtractiveGenerator(a.category) if a.generator == "extractive" else QwenGenerator(GenerationConfig(model_id=a.llm, revision=a.llm_revision))
    _jprint(NeuroRAGPipeline(retr, gen, SafeguardPolicy(confidence_threshold=a.confidence_threshold)).run(a.category, a.confidence))


def cmd_rag_eval(a):
    from neurofusion.rag.evaluation import retrieval_metrics

    data = json.loads(Path(a.run).read_text())
    _jprint(retrieval_metrics(data["runs"], {q: set(v) for q, v in data["qrels"].items()}))


def cmd_claims(a):
    from neurofusion.rag.annotation import import_annotations
    from neurofusion.rag.evaluation import claim_summary

    df = import_annotations(a.annotations)
    _jprint(claim_summary(df["label"].tolist()))


# ---------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="neurofusion", description=__doc__)
    sp = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_):
        s = sp.add_parser(name, help=help_)
        s.set_defaults(fn=fn)
        return s

    add("env", cmd_env, "print environment and hardware information")
    s = add("registry", cmd_registry, "show the dataset source registry")
    s.add_argument("--source")
    s = add("download-figshare", cmd_download_figshare, "download + md5-verify the figshare release")
    s.add_argument("--dest", required=True)
    s.add_argument("--no-extract", action="store_true")
    s = add("import-figshare", cmd_import_figshare, "convert figshare .mat files to PNG + manifest (source patient ids kept)")
    s.add_argument("--mat-dir", required=True)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--manifest", required=True)
    s = add("import-folder", cmd_import_folder, "build a manifest from a local class-folder image tree")
    s.add_argument("--root", required=True)
    s.add_argument("--source-id", required=True)
    s.add_argument("--release", default="unspecified")
    s.add_argument("--manifest", required=True)
    s = add("audit-manifest", cmd_audit_manifest, "audit a manifest (duplicates, splits, groups)")
    s.add_argument("--manifest", required=True)
    s.add_argument("--expected", action="store_true", help="also report differences from manuscript Table 1")
    s = add("dedup", cmd_dedup, "export exact duplicates and perceptual candidate pairs for manual review")
    s.add_argument("--manifest", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--method", choices=["dhash", "phash"], default="dhash")
    s.add_argument("--max-hamming", type=int, default=6)
    s = add("split", cmd_split, "create or import splits")
    s.add_argument("--manifest", required=True)
    s.add_argument("--mode", choices=["fixed", "cv", "source", "import"], required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--val-fraction", type=float, default=0.1)
    s.add_argument("--test-fraction", type=float, default=0.2)
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--test-source")
    s.add_argument("--split-csv")
    s.add_argument("--filename-pattern")

    def common_train(s):
        s.add_argument("--manifest", required=True)
        s.add_argument("--root", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--model", default="proposed")
        s.add_argument("--device")
        s.add_argument("--max-epochs", type=int, help="smoke-test limit; flags the run as truncated")
        s.add_argument("--no-pretrained", action="store_true")

    s = add("train", cmd_train, "train a model (new experiment)")
    common_train(s)
    s.add_argument("--config")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--batch-size", type=int)
    s.add_argument("--skip-test", action="store_true")
    s = add("seeds", cmd_seeds, "repeated-seed runs on one split")
    common_train(s)
    s.add_argument("--seeds", default="0,1,2")
    s = add("cv", cmd_cv, "cross-validation over fold manifests")
    s.add_argument("--manifests", nargs="+", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="proposed")
    s.add_argument("--device")
    s.add_argument("--max-epochs", type=int)
    s.add_argument("--no-pretrained", action="store_true")
    s = add("predict", cmd_predict, "run a checkpoint on a split and save sample-level predictions")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="proposed")
    s.add_argument("--split", default="test")
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--device")
    s = add("evaluate", cmd_evaluate, "metrics + calibration from a predictions file")
    s.add_argument("--predictions", required=True)
    s = add("compare", cmd_compare, "paired McNemar (exact + chi2) and paired bootstrap macro-AUC vs a reference")
    s.add_argument("--reference", required=True)
    s.add_argument("--others", nargs="+", required=True)
    s.add_argument("--family", default="vs-reference")
    s.add_argument("--n-boot", type=int, default=2000)
    s.add_argument("--seed", type=int, default=0)
    s = add("gradcam", cmd_gradcam, "Grad-CAM / Grad-CAM++ from a trained checkpoint")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--image", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="proposed")
    s.add_argument("--branch", choices=["a", "b"], default="a")
    s.add_argument("--method", choices=["gradcam", "gradcampp"], default="gradcampp")
    s.add_argument("--target-class", type=int)
    s = add("robustness", cmd_robustness, "accuracy under image perturbations (test split)")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--manifest", required=True)
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--model", default="proposed")
    s.add_argument("--device")
    s = add("efficiency", cmd_efficiency, "params, FLOPs, latency on this hardware")
    s.add_argument("--model", default="proposed")
    s.add_argument("--iters", type=int, default=50)
    s.add_argument("--device")
    s.add_argument("--out")
    s = add("audit-manuscript", cmd_audit_manuscript, "numerical audit of the manuscript's reported numbers")
    s.add_argument("--out", default="docs/numerical_audit.md")
    s = add("audit-parameters", cmd_audit_parameters, "instantiate models and compare parameter counts with the manuscript")
    s.add_argument("--out")
    s = add("report-tables", cmd_report_tables, "results tables from saved predictions (name=path ...)")
    s.add_argument("--predictions", nargs="+", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--n-boot", type=int, default=1000)
    s = add("figures", cmd_figures, "confusion-matrix figure from saved predictions")
    s.add_argument("--predictions", required=True)
    s.add_argument("--out", required=True)

    def rag_common(s):
        s.add_argument("--embedder", choices=["hashing", "sentence-transformer"], default="hashing")
        s.add_argument("--embedder-model", default="BAAI/bge-base-en-v1.5")
        s.add_argument("--embedder-revision")
        s.add_argument("--store", choices=["numpy", "qdrant"], default="numpy")
        s.add_argument("--qdrant", default=":memory:")

    s = add("rag-demo-collection", cmd_rag_demo_collection, "write the SYNTHETIC demo collection")
    s.add_argument("--out", required=True)
    s = add("rag-build", cmd_rag_build, "build the retrieval index from a source manifest")
    rag_common(s)
    s.add_argument("--sources", required=True)
    s.add_argument("--out", default="rag_index")
    s.add_argument("--max-tokens", type=int, default=512)
    s.add_argument("--overlap", type=int, default=64)
    s.add_argument("--allow-demo", action="store_true")
    s = add("rag-generate", cmd_rag_generate, "generate a contextual summary for a predicted category")
    rag_common(s)
    s.add_argument("--index", default="rag_index")
    s.add_argument("--category", choices=["glioma", "meningioma", "notumor", "pituitary"], required=True)
    s.add_argument("--confidence", type=float)
    s.add_argument("--confidence-threshold", type=float)
    s.add_argument("--generator", choices=["extractive", "qwen"], default="extractive")
    s.add_argument("--llm", default="Qwen/Qwen2.5-7B-Instruct")
    s.add_argument("--llm-revision")
    s = add("rag-eval", cmd_rag_eval, "retrieval metrics from a run/qrels JSON")
    s.add_argument("--run", required=True)
    s = add("claims", cmd_claims, "claim-level summary from an imported annotation sheet")
    s.add_argument("--annotations", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pd.set_option("display.width", 200)
    args.fn(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
