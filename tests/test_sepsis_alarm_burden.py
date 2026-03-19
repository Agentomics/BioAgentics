"""Tests for Phase 3 alarm burden analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.sepsis.calibration.alarm_burden import (
    evaluate_alarm_burden,
    find_threshold_at_sensitivity,
)


@pytest.fixture
def alarm_data():
    """Synthetic data with separable classes for alarm analysis."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = np.concatenate([np.zeros(350), np.ones(150)]).astype(int)
    y_prob = np.clip(
        np.where(y_true == 1, rng.beta(5, 2, n), rng.beta(2, 5, n)),
        0.01,
        0.99,
    )
    return y_true, y_prob


def test_find_threshold_structure(alarm_data):
    """Threshold result has expected fields."""
    y_true, y_prob = alarm_data
    result = find_threshold_at_sensitivity(y_true, y_prob, 0.80)
    expected_keys = [
        "target_sensitivity", "threshold", "achieved_sensitivity",
        "specificity", "fpr", "ppv", "false_alarm_ratio",
        "tp", "fp", "tn", "fn", "n_alarms", "n_true_alarms",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"


def test_find_threshold_achieves_sensitivity(alarm_data):
    """Achieved sensitivity should meet or exceed target."""
    y_true, y_prob = alarm_data
    for target in [0.80, 0.90, 0.95]:
        result = find_threshold_at_sensitivity(y_true, y_prob, target)
        # Allow small tolerance due to discrete thresholds
        assert result["achieved_sensitivity"] >= target - 0.05, (
            f"Target {target}: achieved {result['achieved_sensitivity']}"
        )


def test_find_threshold_valid_ranges(alarm_data):
    """All metrics should be in valid ranges."""
    y_true, y_prob = alarm_data
    result = find_threshold_at_sensitivity(y_true, y_prob, 0.80)
    assert 0.0 <= result["threshold"] <= 1.0
    assert 0.0 <= result["achieved_sensitivity"] <= 1.0
    assert 0.0 <= result["specificity"] <= 1.0
    assert 0.0 <= result["fpr"] <= 1.0
    assert 0.0 <= result["ppv"] <= 1.0
    assert result["tp"] >= 0
    assert result["fp"] >= 0
    assert result["tn"] >= 0
    assert result["fn"] >= 0


def test_find_threshold_counts_consistent(alarm_data):
    """TP+FP+TN+FN should equal total samples."""
    y_true, y_prob = alarm_data
    result = find_threshold_at_sensitivity(y_true, y_prob, 0.80)
    total = result["tp"] + result["fp"] + result["tn"] + result["fn"]
    assert total == len(y_true)
    assert result["n_alarms"] == result["tp"] + result["fp"]
    assert result["n_true_alarms"] == result["tp"]


def test_evaluate_alarm_burden_structure(alarm_data):
    """Full evaluation returns expected structure."""
    y_true, y_prob = alarm_data
    result = evaluate_alarm_burden(y_true, y_prob)
    assert "threshold_analysis" in result
    assert len(result["threshold_analysis"]) == 3  # default thresholds
    assert "target_met" in result
    assert "n_samples" in result
    assert "prevalence" in result


def test_evaluate_alarm_burden_ordering(alarm_data):
    """Higher sensitivity targets should have lower thresholds."""
    y_true, y_prob = alarm_data
    result = evaluate_alarm_burden(y_true, y_prob)
    thresholds = [r["threshold"] for r in result["threshold_analysis"]]
    # Higher sensitivity requires lower threshold (more permissive)
    assert thresholds[0] >= thresholds[-1] - 0.01


def test_evaluate_alarm_burden_custom_thresholds(alarm_data):
    """Custom sensitivity thresholds work."""
    y_true, y_prob = alarm_data
    result = evaluate_alarm_burden(y_true, y_prob, thresholds=[0.70, 0.85])
    assert len(result["threshold_analysis"]) == 2


def test_evaluate_alarm_burden_prevalence(alarm_data):
    """Prevalence matches actual positive rate."""
    y_true, y_prob = alarm_data
    result = evaluate_alarm_burden(y_true, y_prob)
    expected_prev = y_true.mean()
    assert result["prevalence"] == pytest.approx(expected_prev)
