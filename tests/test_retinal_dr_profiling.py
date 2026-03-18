"""Tests for DR screening model profiling."""

import torch

from bioagentics.diagnostics.retinal_dr_screening.profiling import (
    count_parameters,
    measure_file_size,
    measure_latency,
    profile_model,
)
from bioagentics.diagnostics.retinal_dr_screening.training import create_model


def test_count_parameters():
    model = create_model("efficientnet_b0", num_classes=5, pretrained=False)
    n = count_parameters(model)
    assert n > 1_000_000  # EfficientNet-B0 has ~5.3M params
    assert n < 10_000_000


def test_count_parameters_mobilenet():
    model = create_model("mobilenetv3_small_100", num_classes=5, pretrained=False)
    n = count_parameters(model)
    assert n > 500_000  # MobileNetV3-Small has ~2.5M params
    assert n < 5_000_000


def test_measure_file_size():
    model = create_model("mobilenetv3_small_100", num_classes=5, pretrained=False)
    size_mb = measure_file_size(model)
    assert size_mb > 1  # should be at least a few MB
    assert size_mb < 50


def test_measure_latency_cpu():
    model = create_model("mobilenetv3_small_100", num_classes=5, pretrained=False)
    mean_ms, std_ms = measure_latency(
        model, image_size=32, device=torch.device("cpu"), n_warmup=1, n_runs=3,
    )
    assert mean_ms > 0
    assert std_ms >= 0


def test_profile_model():
    profile = profile_model("mobilenetv3_small_100", image_size=32, num_classes=5)
    assert profile.model_name == "mobilenetv3_small_100"
    assert profile.num_params > 0
    assert profile.num_params_m > 0
    assert profile.file_size_mb > 0
    assert profile.cpu_latency_ms > 0
