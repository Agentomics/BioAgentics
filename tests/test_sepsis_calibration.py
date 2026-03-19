"""Tests for Phase 3 calibration module."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.sepsis.calibration.calibration import (
    calibrate_cv,
    calibration_curve_data,
    compute_ece,
    evaluate_calibration,
)


@pytest.fixture
def calibration_data():
    """Synthetic predictions with known miscalibration."""
    rng = np.random.default_rng(42)
    n = 300
    # Generate well-separated classes
    y_true = np.concatenate([np.zeros(200), np.ones(100)]).astype(int)
    # Overconfident predictions (miscalibrated)
    y_prob = np.clip(
        np.where(y_true == 1, rng.beta(5, 2, n), rng.beta(2, 5, n)),
        0.01,
        0.99,
    )
    return y_true, y_prob


def test_compute_ece_perfect():
    """ECE should be 0 for perfectly calibrated predictions."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    # Perfectly calibrated: p=0.0 for negatives, p=1.0 for positives
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    ece = compute_ece(y_true, y_prob, n_bins=10)
    assert ece == pytest.approx(0.0, abs=1e-10)


def test_compute_ece_range(calibration_data):
    """ECE should be between 0 and 1."""
    y_true, y_prob = calibration_data
    ece = compute_ece(y_true, y_prob)
    assert 0.0 <= ece <= 1.0


def test_compute_ece_bins():
    """ECE with more bins gives finer resolution."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_prob = rng.random(500)
    ece_10 = compute_ece(y_true, y_prob, n_bins=10)
    ece_20 = compute_ece(y_true, y_prob, n_bins=20)
    # Both should be valid
    assert 0.0 <= ece_10 <= 1.0
    assert 0.0 <= ece_20 <= 1.0


def test_calibration_curve_data_structure(calibration_data):
    """Calibration curve data has expected fields."""
    y_true, y_prob = calibration_data
    result = calibration_curve_data(y_true, y_prob, n_bins=10)
    assert "fraction_of_positives" in result
    assert "mean_predicted_value" in result
    assert "n_bins_used" in result
    assert len(result["fraction_of_positives"]) == len(result["mean_predicted_value"])
    # All values in [0, 1]
    assert all(0 <= v <= 1 for v in result["fraction_of_positives"])
    assert all(0 <= v <= 1 for v in result["mean_predicted_value"])


def test_calibrate_cv_sigmoid(calibration_data):
    """Platt scaling produces valid probabilities."""
    y_true, y_prob = calibration_data
    calibrated = calibrate_cv(y_true, y_prob, method="sigmoid", n_folds=3)
    assert calibrated.shape == y_prob.shape
    assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)


def test_calibrate_cv_isotonic(calibration_data):
    """Isotonic calibration produces valid probabilities."""
    y_true, y_prob = calibration_data
    calibrated = calibrate_cv(y_true, y_prob, method="isotonic", n_folds=3)
    assert calibrated.shape == y_prob.shape
    assert np.all(calibrated >= 0.0) and np.all(calibrated <= 1.0)


def test_calibrate_cv_invalid_method(calibration_data):
    """Invalid method raises ValueError."""
    y_true, y_prob = calibration_data
    with pytest.raises(ValueError, match="Unknown calibration method"):
        calibrate_cv(y_true, y_prob, method="invalid")


def test_evaluate_calibration_structure(calibration_data):
    """Full evaluation returns expected structure."""
    y_true, y_prob = calibration_data
    result = evaluate_calibration(y_true, y_prob, n_bins=10, n_folds=3)
    assert "uncalibrated" in result
    assert "platt_scaling" in result
    assert "isotonic" in result
    assert "best_method" in result
    assert "best_ece" in result
    assert "meets_target" in result
    # ECEs should be non-negative
    for method in ["uncalibrated", "platt_scaling", "isotonic"]:
        assert result[method]["ece"] >= 0.0


def test_evaluate_calibration_improvement(calibration_data):
    """At least one calibration method should not worsen ECE significantly."""
    y_true, y_prob = calibration_data
    result = evaluate_calibration(y_true, y_prob, n_bins=10, n_folds=3)
    # Best ECE should be <= uncalibrated ECE (within tolerance)
    assert result["best_ece"] <= result["uncalibrated"]["ece"] + 0.05
