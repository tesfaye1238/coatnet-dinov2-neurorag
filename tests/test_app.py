import io
import os
import pathlib

import pytest
import torch
from PIL import Image

from conftest import TinyBranch, TinyViT
from neurofusion import app_logic as A
from neurofusion.training.checkpoint import save_checkpoint


def png_bytes(size=(64, 64), mode="L", fmt="PNG"):
    b = io.BytesIO()
    Image.new(mode, size, 100).save(b, format=fmt)
    return b.getvalue()


def test_valid_upload_roundtrip_in_memory(tmp_path):
    before = set(os.listdir(tmp_path))
    im = A.validate_upload(png_bytes(), "x.png")
    assert im.mode == "RGB" and im.size == (64, 64)
    assert set(os.listdir(tmp_path)) == before  # nothing persisted


@pytest.mark.parametrize("data,name,msg", [
    (b"", "a.png", "empty"),
    (b"notanimage", "a.png", "not a readable image"),
    (png_bytes(), "a.exe", "unsupported file type"),
    (png_bytes((8, 8)), "a.png", "between"),
    ("TOO_BIG", "a.png", "larger than"),
])
def test_invalid_uploads_rejected(data, name, msg):
    if data == "TOO_BIG":
        data = b"A" * (A.MAX_UPLOAD_BYTES + 1)
    with pytest.raises(A.UploadError, match=msg):
        A.validate_upload(data, name)


def test_truncated_png_rejected():
    with pytest.raises(A.UploadError):
        A.validate_upload(png_bytes((64, 64))[:40], "t.png")


def test_missing_weights_never_fake_a_prediction(tmp_path):
    with pytest.raises(A.WeightsMissing, match="no prediction will be invented"):
        A.load_checkpoint_model(tmp_path / "nope.pt")
    with pytest.raises(A.WeightsMissing):
        A.load_checkpoint_model("")


def test_class_mapping_mismatch_reported_as_missing_weights(tmp_path):
    model = torch.nn.Linear(2, 2)
    ck = tmp_path / "c.pt"
    save_checkpoint(ck, model, epoch=0, class_names=("a", "b", "c", "d"))
    with pytest.raises(A.WeightsMissing, match="class mapping"):
        A.load_checkpoint_model(ck)


def test_prediction_from_explicit_checkpoint(tmp_path):
    from neurofusion.models.classifier import FusionClassifier, ModelConfig

    mk = lambda: FusionClassifier(ModelConfig(pretrained=False, hidden1=16, hidden2=8), TinyBranch(24), TinyViT(16), 24, 16)  # noqa: E731
    m = mk()
    ck = tmp_path / "c.pt"
    save_checkpoint(ck, m, epoch=0, config_hash="h1")
    lm = A.load_checkpoint_model(ck, builder=mk)
    res = A.predict_image(lm, A.validate_upload(png_bytes((64, 64), "RGB"), "x.png"))
    assert abs(sum(res["probabilities"].values()) - 1) < 1e-5 and res["config_hash"] == "h1"
    assert res["predicted_category"] in res["probabilities"] and "not diagnostic certainty" in res["label"]


def test_banners_state_limits():
    assert "not a diagnosis" in A.EXPERIMENTAL_BANNER and "does not exclude" in A.EXPERIMENTAL_BANNER
    assert "not saved" in A.PRIVACY_NOTE


def test_streamlit_app_without_weights_shows_error_not_prediction():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(pathlib.Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert any("EXPERIMENTAL" in w.value for w in at.warning)
    assert not at.error  # nothing uploaded yet -> no error, no prediction
