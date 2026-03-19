"""Tests for calibration analysis module."""

import json

import h5py
import numpy as np
import pytest

from bioagentics.models.pathology_msi.calibration import (
    CalibrationMetrics,
    TemperatureScaling,
    calibrate_model,
    collect_predictions,
    expected_calibration_error,
    fit_temperature_scaling,
)
from bioagentics.models.pathology_msi.mil_models import create_mil_model


@pytest.fixture
def calibration_data(tmp_path):
    """Create test data for calibration tests."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    slide_ids = [f"slide_{i:03d}" for i in range(20)]
    for sid in slide_ids:
        n_patches = np.random.randint(10, 25)
        features = np.random.randn(n_patches, 256).astype(np.float32)
        with h5py.File(features_dir / f"{sid}.h5", "w") as f:
            f.create_dataset("features", data=features)

    labels = {sid: (1 if i % 4 == 0 else 0) for i, sid in enumerate(slide_ids)}
    return features_dir, slide_ids, labels


class TestExpectedCalibrationError:
    def test_perfect_calibration(self):
        # Perfectly calibrated: probs match true frequency
        probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        labels = np.array([0, 0, 0, 1, 1])
        metrics = expected_calibration_error(probs, labels, n_bins=5)
        assert isinstance(metrics, CalibrationMetrics)
        assert metrics.ece >= 0.0
        assert metrics.n_bins == 5

    def test_worst_calibration(self):
        # All predictions are 1.0 but labels are 0
        probs = np.ones(10)
        labels = np.zeros(10, dtype=int)
        metrics = expected_calibration_error(probs, labels, n_bins=5)
        assert metrics.ece == pytest.approx(1.0, abs=0.01)

    def test_bin_counts_sum_to_total(self):
        np.random.seed(42)
        probs = np.random.rand(100)
        labels = (probs > 0.5).astype(int)
        metrics = expected_calibration_error(probs, labels, n_bins=10)
        assert sum(metrics.bin_counts) == 100

    def test_ece_between_zero_and_one(self):
        np.random.seed(42)
        probs = np.random.rand(50)
        labels = np.random.randint(0, 2, 50)
        metrics = expected_calibration_error(probs, labels)
        assert 0.0 <= metrics.ece <= 1.0
        assert 0.0 <= metrics.mce <= 1.0

    def test_empty_bins_handled(self):
        # All predictions in one bin
        probs = np.full(10, 0.95)
        labels = np.ones(10, dtype=int)
        metrics = expected_calibration_error(probs, labels, n_bins=10)
        assert metrics.ece >= 0.0


class TestTemperatureScaling:
    def test_identity_at_temp_one(self):
        import torch

        ts = TemperatureScaling()
        logits = torch.randn(5, 2)
        scaled = ts(logits)
        assert torch.allclose(logits, scaled, atol=1e-5)

    def test_scaling_reduces_confidence(self):
        import torch

        ts = TemperatureScaling()
        ts.temperature.data = torch.tensor([2.0])
        logits = torch.tensor([[3.0, 1.0]])
        scaled = ts(logits)
        assert scaled[0, 0].item() == pytest.approx(1.5, abs=1e-5)

    def test_fit_returns_positive_temp(self):
        np.random.seed(42)
        logits = np.random.randn(50, 2).astype(np.float32)
        labels = np.random.randint(0, 2, 50)
        temp = fit_temperature_scaling(logits, labels)
        assert temp > 0


class TestCollectPredictions:
    def test_returns_correct_shapes(self, calibration_data):
        features_dir, slide_ids, labels = calibration_data
        model = create_mil_model("abmil", input_dim=256, n_classes=2)

        logits, probs, true_labels = collect_predictions(
            model, slide_ids[:5], labels, features_dir,
        )
        assert logits.shape == (5, 2)
        assert probs.shape == (5,)
        assert true_labels.shape == (5,)
        assert np.all((probs >= 0) & (probs <= 1))


class TestCalibrateModel:
    def test_full_pipeline(self, calibration_data, tmp_path):
        features_dir, slide_ids, labels = calibration_data
        model = create_mil_model("abmil", input_dim=256, n_classes=2)
        output_dir = tmp_path / "cal_output"

        results = calibrate_model(
            model,
            val_ids=slide_ids[:10],
            test_ids=slide_ids[10:],
            labels=labels,
            features_dir=features_dir,
            output_dir=output_dir,
        )

        assert "pre_calibration" in results
        assert "post_calibration" in results
        assert "ece_reduction" in results
        assert results["post_calibration"]["temperature"] > 0

        assert (output_dir / "calibration_results.json").exists()
        with open(output_dir / "calibration_results.json") as f:
            saved = json.load(f)
        assert saved["pre_calibration"]["ece"] >= 0
