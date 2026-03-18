"""Model profiling: inference latency, parameter count, FLOPs, and file size.

Usage:
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.profiling \\
        --model mobilenetv3_small_100 --image-size 224
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bioagentics.diagnostics.retinal_dr_screening.config import (
    MOBILE_IMAGE_SIZE,
    TRAIN_IMAGE_SIZE,
)
from bioagentics.diagnostics.retinal_dr_screening.training import create_model

logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Model profiling results."""

    model_name: str
    image_size: int
    num_params: int
    num_params_m: float
    file_size_mb: float
    cpu_latency_ms: float
    cpu_latency_std_ms: float
    gpu_latency_ms: float
    gpu_latency_std_ms: float
    mps_latency_ms: float
    mps_latency_std_ms: float


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_file_size(model: nn.Module) -> float:
    """Measure model checkpoint file size in MB."""
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        size = Path(f.name).stat().st_size
    return size / (1024 * 1024)


def measure_latency(
    model: nn.Module,
    image_size: int,
    device: torch.device,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> tuple[float, float]:
    """Measure average inference latency in milliseconds.

    Returns (mean_ms, std_ms).
    """
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

    # Synchronize if CUDA
    if device.type == "cuda":
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000)

    return float(np.mean(timings)), float(np.std(timings))


def profile_model(
    model_name: str = "mobilenetv3_small_100",
    image_size: int = MOBILE_IMAGE_SIZE,
    num_classes: int = 5,
) -> ModelProfile:
    """Run full profiling for a model.

    Tests latency on CPU, GPU (if available), and MPS (if available).
    """
    model = create_model(model_name, num_classes=num_classes, pretrained=False)

    num_params = count_parameters(model)
    file_size = measure_file_size(model)

    # CPU latency
    cpu_mean, cpu_std = measure_latency(model, image_size, torch.device("cpu"))

    # GPU latency
    gpu_mean, gpu_std = 0.0, 0.0
    if torch.cuda.is_available():
        gpu_mean, gpu_std = measure_latency(model, image_size, torch.device("cuda"))

    # MPS latency
    mps_mean, mps_std = 0.0, 0.0
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_mean, mps_std = measure_latency(model, image_size, torch.device("mps"))

    profile = ModelProfile(
        model_name=model_name,
        image_size=image_size,
        num_params=num_params,
        num_params_m=round(num_params / 1e6, 2),
        file_size_mb=round(file_size, 2),
        cpu_latency_ms=round(cpu_mean, 1),
        cpu_latency_std_ms=round(cpu_std, 1),
        gpu_latency_ms=round(gpu_mean, 1),
        gpu_latency_std_ms=round(gpu_std, 1),
        mps_latency_ms=round(mps_mean, 1),
        mps_latency_std_ms=round(mps_std, 1),
    )

    logger.info(
        "%s @ %dx%d: %.2fM params, %.1f MB, CPU=%.1f±%.1fms",
        model_name,
        image_size,
        image_size,
        profile.num_params_m,
        profile.file_size_mb,
        cpu_mean,
        cpu_std,
    )
    return profile


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Profile DR screening models")
    parser.add_argument("--model", default="mobilenetv3_small_100")
    parser.add_argument("--image-size", type=int, default=MOBILE_IMAGE_SIZE)
    parser.add_argument("--compare", action="store_true", help="Compare all models")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.compare:
        models = [
            ("mobilenetv3_small_100", MOBILE_IMAGE_SIZE),
            ("efficientnet_b0", TRAIN_IMAGE_SIZE),
            ("efficientnet_b4", TRAIN_IMAGE_SIZE),
        ]
        for name, size in models:
            p = profile_model(name, size)
            print(
                f"{name:30s} | {p.num_params_m:6.2f}M | "
                f"{p.file_size_mb:6.1f}MB | CPU {p.cpu_latency_ms:6.1f}ms"
            )
    else:
        p = profile_model(args.model, args.image_size)
        import json

        print(json.dumps(asdict(p), indent=2))


if __name__ == "__main__":
    main()
