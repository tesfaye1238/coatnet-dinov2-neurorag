"""Local research demonstration (Streamlit). Run:  streamlit run app/streamlit_app.py

GitHub hosting does not run this app; it is a local tool. See src/neurofusion/app_logic.py for the rules it follows.
"""
from __future__ import annotations

import os

import streamlit as st

from neurofusion.app_logic import (
    EXPERIMENTAL_BANNER,
    PRIVACY_NOTE,
    UploadError,
    WeightsMissing,
    load_checkpoint_model,
    predict_image,
    validate_upload,
)

st.set_page_config(page_title="NeuroFusion research demo", layout="wide")
st.title("NeuroFusion: brain MRI category classification (experimental)")
st.warning(EXPERIMENTAL_BANNER)
st.caption(PRIVACY_NOTE)

with st.sidebar:
    st.header("Model")
    ckpt = st.text_input("Checkpoint path (required)", value=os.environ.get("NEUROFUSION_CHECKPOINT", ""))
    model_name = st.selectbox("Architecture", ["proposed", "concat_no_attention", "concat_se_r16", "sum_se_r8", "coatnet_only", "dinov2_only"])
    show_cam = st.checkbox("Show attribution (Grad-CAM++)", value=False)
    cam_branch = st.radio("Attribution branch", ["a", "b"], format_func=lambda b: "CoAtNet (a)" if b == "a" else "DINOv2 (b)")
    show_ctx = st.checkbox("Show retrieved context", value=False)
    index_dir = st.text_input("RAG index directory", value=os.environ.get("NEUROFUSION_RAG_INDEX", "rag_index"), disabled=not show_ctx)
    st.caption("Context uses a local retrieval index and a deterministic extractive summary. No external service is called.")

upload = st.file_uploader("MRI image (PNG/JPG/BMP/TIFF)", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])

if upload is not None:
    try:
        image = validate_upload(upload.getvalue(), upload.name)
    except UploadError as exc:
        st.error(f"Invalid upload: {exc}")
        st.stop()
    try:
        lm = load_checkpoint_model(ckpt, model_name)
    except WeightsMissing as exc:
        st.error(str(exc))
        st.image(image, caption="uploaded image (no prediction without weights)", width=320)
        st.stop()
    res = predict_image(lm, image)
    c1, c2 = st.columns(2)
    c1.image(image, caption="uploaded image", width=320)
    c2.subheader(f"Predicted category: {res['predicted_display']}")
    c2.write(f"Model score: {res['model_score']:.3f} ({res['label']})")
    c2.bar_chart(res["probabilities"])
    if show_cam:
        import numpy as np
        import torch

        from neurofusion.data.transforms import build_eval_transform, denormalize
        from neurofusion.explainability.gradcam import CAMExtractor, normalise_cam, target_layer

        layer, reshape, desc = target_layer(lm.model, cam_branch)
        x = build_eval_transform()(image).unsqueeze(0)
        with CAMExtractor(lm.model, layer, "gradcampp", reshape) as f:
            cam, _, _ = f(x)
        import matplotlib.cm as cm

        base = denormalize(x[0]).permute(1, 2, 0).numpy()
        heat = cm.jet(normalise_cam(cam)[0].numpy())[..., :3]
        st.image(np.clip(0.6 * base + 0.4 * heat, 0, 1), caption=f"attribution via {desc}. Not a segmentation mask.", width=320)
        del torch
    if show_ctx:
        try:
            from neurofusion.rag.embedding import HashingEmbedder
            from neurofusion.rag.generation import ExtractiveGenerator
            from neurofusion.rag.index import NumpyStore
            from neurofusion.rag.pipeline import NeuroRAGPipeline
            from neurofusion.rag.retrieval import Retriever

            emb = HashingEmbedder()
            rec = NeuroRAGPipeline(Retriever(NumpyStore.load(index_dir), emb), ExtractiveGenerator(res["predicted_category"])).run(
                res["predicted_category"], res["model_score"])
            st.subheader("Contextual summary (experimental)")
            out = rec["output"]
            st.write(out.get("general_background", ""))
            for s in out.get("contextual_summary", []):
                st.write(f"- {s['statement']}  `{', '.join(s['passage_ids'])}`")
            st.caption(out.get("evidence_limitations", ""))
            st.caption("Citations were checked for existence only; support and clinical correctness are not verified.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Context unavailable: {exc}")
