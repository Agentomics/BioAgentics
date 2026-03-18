"""Tests for DR screening model export module."""

import numpy as np
import torch

from bioagentics.diagnostics.retinal_dr_screening.export import (
    export_onnx,
)
from bioagentics.diagnostics.retinal_dr_screening.training import create_model


def _make_checkpoint(tmp_path, model_name="mobilenetv3_small_100"):
    model = create_model(model_name, num_classes=5, pretrained=False)
    ckpt_path = tmp_path / "model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {"model_name": model_name, "num_classes": 5},
    }, ckpt_path)
    return ckpt_path


def test_export_onnx(tmp_path):
    ckpt_path = _make_checkpoint(tmp_path)
    onnx_path = tmp_path / "model.onnx"

    result = export_onnx(
        ckpt_path,
        output_path=onnx_path,
        model_name="mobilenetv3_small_100",
        image_size=32,
    )

    assert result.exists()
    assert result.stat().st_size > 0


def test_export_onnx_validates(tmp_path):
    """Exported ONNX should pass onnx.checker validation."""
    import onnx

    ckpt_path = _make_checkpoint(tmp_path)
    onnx_path = export_onnx(ckpt_path, model_name="mobilenetv3_small_100", image_size=32)

    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)


def test_export_onnx_inference(tmp_path):
    """Exported ONNX should produce valid inference results."""
    import onnxruntime as ort

    ckpt_path = _make_checkpoint(tmp_path)
    onnx_path = export_onnx(ckpt_path, model_name="mobilenetv3_small_100", image_size=32)

    session = ort.InferenceSession(str(onnx_path))
    dummy = np.random.randn(1, 3, 32, 32).astype(np.float32)
    input_name = session.get_inputs()[0].name

    outputs = session.run(None, {input_name: dummy})
    assert outputs[0].shape == (1, 5)
