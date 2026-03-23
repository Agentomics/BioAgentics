"""Convert SpectrogramCNN from PyTorch to ONNX and TFLite formats.

Conversion pipeline: PyTorch → ONNX → TFLite
ONNX export uses torch.onnx; TFLite conversion requires optional
tensorflow/onnx-tf dependencies.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch

from bioagentics.voice_pd.config import MODELS_DIR, N_MELS
from bioagentics.voice_pd.deep.cnn_model import SpectrogramCNN, build_model

log = logging.getLogger(__name__)

# Fixed input dimensions for the voice protocol (5-sec clip → spectrogram)
INPUT_N_MELS = N_MELS
INPUT_TIME_FRAMES = 157  # ceil(5.0 * 16000 / 512)
INPUT_SHAPE = (1, 3, INPUT_N_MELS, INPUT_TIME_FRAMES)


def export_onnx(
    model: SpectrogramCNN | None = None,
    state_dict_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Export SpectrogramCNN to ONNX format.

    Provide either a live model or a path to saved state_dict weights.

    Args:
        model: A SpectrogramCNN instance (already loaded).
        state_dict_path: Path to .pt state_dict file. Ignored if model given.
        output_path: Where to save the .onnx file.

    Returns:
        Path to the saved ONNX file.
    """
    if model is None:
        model = build_model(pretrained=False)
        if state_dict_path is None:
            state_dict_path = MODELS_DIR / "deep_cnn_model.pt"
        state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)

    if output_path is None:
        output_path = MODELS_DIR / "voice_pd_model.onnx"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(*INPUT_SHAPE)

    torch.onnx.export(
        model,
        (dummy_input,),
        str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["spectrogram"],
        output_names=["logit"],
        dynamic_axes={
            "spectrogram": {0: "batch_size"},
            "logit": {0: "batch_size"},
        },
    )
    log.info("ONNX model exported to %s (%.1f KB)", output_path, output_path.stat().st_size / 1024)
    return output_path


def convert_onnx_to_tflite(
    onnx_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    """Convert ONNX model to TFLite format.

    Requires: pip install onnx-tf tensorflow  (heavy optional deps).

    Args:
        onnx_path: Path to .onnx model file.
        output_path: Where to save the .tflite file.

    Returns:
        Path to the saved TFLite file.
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
    except ImportError as e:
        raise ImportError(
            "TFLite conversion requires 'onnx' and 'onnx-tf' packages. "
            "Install with: uv add --optional research onnx onnx-tf tensorflow"
        ) from e

    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.with_suffix(".tflite")
    output_path = Path(output_path)

    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)

    # Save as SavedModel then convert to TFLite
    import tempfile

    import tensorflow as tf

    with tempfile.TemporaryDirectory() as tmpdir:
        saved_model_dir = Path(tmpdir) / "saved_model"
        tf_rep.export_graph(str(saved_model_dir))

        converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    log.info("TFLite model saved to %s (%.1f KB)", output_path, len(tflite_model) / 1024)
    return output_path


def run_tflite_inference(
    tflite_path: str | Path,
    spectrogram: np.ndarray,
) -> float:
    """Run inference with a TFLite model.

    Args:
        tflite_path: Path to .tflite model file.
        spectrogram: Input array of shape (3, n_mels, time_frames).

    Returns:
        Predicted probability of PD (sigmoid of logit).
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TFLite inference requires 'tensorflow' or 'tflite-runtime'. "
            "Install with: uv add --optional research tensorflow"
        ) from e

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = spectrogram[np.newaxis].astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    logit = interpreter.get_tensor(output_details[0]["index"])[0, 0]
    prob = 1.0 / (1.0 + np.exp(-logit))
    return float(prob)


@torch.no_grad()
def run_pytorch_inference(
    model: SpectrogramCNN,
    spectrogram: np.ndarray,
) -> float:
    """Run inference with the PyTorch model (for benchmarking without TFLite deps).

    Args:
        model: SpectrogramCNN instance.
        spectrogram: Input array of shape (3, n_mels, time_frames).

    Returns:
        Predicted probability of PD.
    """
    model.eval()
    x = torch.from_numpy(spectrogram[np.newaxis].astype(np.float32))
    logit = model(x)
    prob = torch.sigmoid(logit).item()
    return prob


def benchmark_inference(
    model: SpectrogramCNN | None = None,
    state_dict_path: str | Path | None = None,
    n_runs: int = 20,
    warmup: int = 3,
) -> dict:
    """Benchmark PyTorch CPU inference time.

    Measures end-to-end time from spectrogram input to probability output
    on CPU, simulating mobile inference conditions.

    Returns:
        Dict with mean_sec, std_sec, median_sec, min_sec, max_sec,
        meets_target (< MOBILE_INFERENCE_SEC).
    """
    from bioagentics.voice_pd.config import MOBILE_INFERENCE_SEC

    if model is None:
        model = build_model(pretrained=False)
        if state_dict_path is not None:
            state_dict = torch.load(state_dict_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
    model.eval()

    dummy = np.random.default_rng(42).random(INPUT_SHAPE[1:]).astype(np.float32)

    # Warmup
    for _ in range(warmup):
        run_pytorch_inference(model, dummy)

    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_pytorch_inference(model, dummy)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_arr = np.array(times)
    results = {
        "mean_sec": float(times_arr.mean()),
        "std_sec": float(times_arr.std()),
        "median_sec": float(np.median(times_arr)),
        "min_sec": float(times_arr.min()),
        "max_sec": float(times_arr.max()),
        "n_runs": n_runs,
        "target_sec": MOBILE_INFERENCE_SEC,
        "meets_target": float(np.median(times_arr)) < MOBILE_INFERENCE_SEC,
        "input_shape": list(INPUT_SHAPE),
    }
    log.info(
        "Inference benchmark: median=%.3fs, mean=%.3fs (target <%.1fs, %s)",
        results["median_sec"], results["mean_sec"], MOBILE_INFERENCE_SEC,
        "PASS" if results["meets_target"] else "FAIL",
    )
    return results
