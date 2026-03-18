"""Model export to ONNX and TFLite with mobile latency benchmarking.

Usage:
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.export \\
        --model-path output/.../best_model.pt \\
        --format onnx tflite
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch

from bioagentics.diagnostics.retinal_dr_screening.config import (
    MOBILE_IMAGE_SIZE,
    MODEL_DIR,
    NUM_CLASSES,
)
from bioagentics.diagnostics.retinal_dr_screening.training import create_model

logger = logging.getLogger(__name__)


def export_onnx(
    model_path: Path,
    output_path: Path | None = None,
    model_name: str = "mobilenetv3_small_100",
    image_size: int = MOBILE_IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    dynamic_batch: bool = True,
) -> Path:
    """Export PyTorch model to ONNX format.

    Args:
        model_path: Path to PyTorch checkpoint.
        output_path: Path for ONNX file (auto-generated if None).
        model_name: timm model name.
        image_size: Input image size.
        num_classes: Number of output classes.
        dynamic_batch: Enable dynamic batch size.

    Returns:
        Path to exported ONNX file.
    """
    import onnx

    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if output_path is None:
        output_path = model_path.parent / f"{model_name}.onnx"

    dummy = torch.randn(1, 3, image_size, image_size)
    dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None

    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
    )

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("ONNX exported: %s (%.1f MB)", output_path, size_mb)
    return output_path


def export_tflite(
    model_path: Path,
    output_path: Path | None = None,
    model_name: str = "mobilenetv3_small_100",
    image_size: int = MOBILE_IMAGE_SIZE,
    num_classes: int = NUM_CLASSES,
    quantize: bool = False,
) -> Path:
    """Export PyTorch model to TFLite via ONNX intermediate.

    Note: Requires onnx2tf or tf2onnx. Falls back to ONNX-only if unavailable.
    """
    # First export to ONNX
    onnx_path = model_path.parent / f"{model_name}_temp.onnx"
    export_onnx(model_path, onnx_path, model_name, image_size, num_classes)

    if output_path is None:
        output_path = model_path.parent / f"{model_name}.tflite"

    try:
        import onnx2tf  # noqa: F401

        onnx2tf.convert(
            input_onnx_file_path=str(onnx_path),
            output_folder_path=str(output_path.parent),
            output_file_name=output_path.name,
        )
        logger.info("TFLite exported: %s", output_path)
    except ImportError:
        logger.warning(
            "onnx2tf not installed — TFLite conversion skipped.\n"
            "  Install: pip install onnx2tf\n"
            "  ONNX file available at: %s",
            onnx_path,
        )
        # Clean up temp ONNX if we can't convert
        if onnx_path.exists() and "_temp" in onnx_path.name:
            onnx_path.unlink()
        return output_path

    # Clean up temp ONNX
    if onnx_path.exists() and "_temp" in onnx_path.name:
        onnx_path.unlink()

    return output_path


def benchmark_onnx(
    onnx_path: Path,
    image_size: int = MOBILE_IMAGE_SIZE,
    n_warmup: int = 5,
    n_runs: int = 50,
) -> dict:
    """Benchmark ONNX Runtime inference latency.

    Returns dict with mean/std latency in ms and model size.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path))
    dummy = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(n_warmup):
        session.run(None, {input_name: dummy})

    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)

    return {
        "format": "onnx",
        "file_size_mb": round(size_mb, 2),
        "latency_mean_ms": round(float(np.mean(timings)), 1),
        "latency_std_ms": round(float(np.std(timings)), 1),
        "latency_p95_ms": round(float(np.percentile(timings, 95)), 1),
        "n_runs": n_runs,
    }


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Export DR model to mobile formats")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-name", default="mobilenetv3_small_100")
    parser.add_argument("--image-size", type=int, default=MOBILE_IMAGE_SIZE)
    parser.add_argument("--format", nargs="+", default=["onnx"], choices=["onnx", "tflite"])
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if "onnx" in args.format:
        onnx_path = export_onnx(args.model_path, model_name=args.model_name,
                                image_size=args.image_size)
        if args.benchmark:
            results = benchmark_onnx(onnx_path, args.image_size)
            import json
            print(json.dumps(results, indent=2))

    if "tflite" in args.format:
        export_tflite(args.model_path, model_name=args.model_name,
                      image_size=args.image_size)


if __name__ == "__main__":
    main()
