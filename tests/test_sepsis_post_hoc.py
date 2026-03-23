"""Tests for Phase 3 post-hoc calibration with conformal comparison."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.sepsis.calibration.post_hoc import (
    compute_mce,
    evaluate_post_hoc,
)


@pytest.fixture
def calibration_data():
    """Synthetic predictions with known miscalibration."""
    rng = np.random.default_rng(42)
    n = 400
    y_true = np.concatenate([np.zeros(280), np.ones(120)]).astype(int)
    y_prob = np.clip(
        np.where(y_true == 1, rng.beta(5, 2, n), rng.beta(2, 5, n)),
        0.01,
        0.99,
    )
    return y_true, y_prob


def test_compute_mce_perfect():
    """MCE should be 0 for perfectly calibrated predictions."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    mce = compute_mce(y_true, y_prob, n_bins=10)
    assert mce == pytest.approx(0.0, abs=1e-10)


def test_compute_mce_range(calibration_data):
    """MCE should be in [0, 1]."""
    y_true, y_prob = calibration_data
    mce = compute_mce(y_true, y_prob)
    assert 0.0 <= mce <= 1.0


def test_mce_geq_ece(calibration_data):
    """MCE should always be >= ECE (max >= weighted avg)."""
    from bioagentics.diagnostics.sepsis.calibration.calibration import compute_ece

    y_true, y_prob = calibration_data
    ece = compute_ece(y_true, y_prob)
    mce = compute_mce(y_true, y_prob)
    assert mce >= ece - 1e-10


def test_evaluate_post_hoc_structure(calibration_data):
    """Result dict should have all expected fields."""
    y_true, y_prob = calibration_data
    result = evaluate_post_hoc(y_true, y_prob, n_bins=10, n_folds=3)

    assert "uncalibrated" in result
    assert "platt_scaling" in result
    assert "isotonic" in result
    assert "conformal" in result
    assert "best_method" in result
    assert "best_ece" in result
    assert "best_mce" in result
    assert "meets_ece_target" in result

    # Each method should have ECE and MCE
    for method in ("uncalibrated", "platt_scaling", "isotonic"):
        assert "ece" in result[method]
        assert "mce" in result[method]
        assert "calibration_curve" in result[method]


def test_evaluate_post_hoc_conformal_comparison(calibration_data):
    """Conformal comparison should include coverage and alarm info."""
    y_true, y_prob = calibration_data
    result = evaluate_post_hoc(y_true, y_prob, n_folds=3)

    assert "coverage" in result["conformal"]
    assert "avg_set_size" in result["conformal"]
    assert "false_alarm_reduction" in result["conformal"]
    assert result["conformal"]["coverage"] >= 0.0
    assert result["conformal"]["coverage"] <= 1.0


def test_evaluate_post_hoc_mce_values(calibration_data):
    """MCE values should be non-negative for all methods."""
    y_true, y_prob = calibration_data
    result = evaluate_post_hoc(y_true, y_prob, n_folds=3)

    for method in ("uncalibrated", "platt_scaling", "isotonic"):
        assert result[method]["mce"] >= 0.0
        assert result[method]["mce"] <= 1.0


def test_evaluate_post_hoc_best_method(calibration_data):
    """Best method should have the lowest ECE."""
    y_true, y_prob = calibration_data
    result = evaluate_post_hoc(y_true, y_prob, n_folds=3)

    eces = {
        "uncalibrated": result["uncalibrated"]["ece"],
        "platt_scaling": result["platt_scaling"]["ece"],
        "isotonic": result["isotonic"]["ece"],
    }
    actual_best = min(eces, key=eces.get)
    assert result["best_method"] == actual_best
    assert result["best_ece"] == pytest.approx(eces[actual_best])
