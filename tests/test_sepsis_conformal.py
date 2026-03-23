"""Tests for Phase 3 conformal prediction module."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.sepsis.calibration.conformal import (
    compute_nonconformity_scores,
    conformal_prediction_sets,
    evaluate_conformal,
    evaluate_mondrian,
    mondrian_conformal_calibrate,
    split_conformal_calibrate,
)


@pytest.fixture
def conformal_data():
    """Synthetic predictions with moderate discrimination."""
    rng = np.random.default_rng(42)
    n = 500
    y_true = np.concatenate([np.zeros(350), np.ones(150)]).astype(int)
    # Reasonably discriminative model
    y_prob = np.clip(
        np.where(y_true == 1, rng.beta(5, 2, n), rng.beta(2, 5, n)),
        0.01,
        0.99,
    )
    return y_true, y_prob


@pytest.fixture
def grouped_data(conformal_data):
    """Conformal data with group labels."""
    y_true, y_prob = conformal_data
    rng = np.random.default_rng(123)
    groups = rng.choice(["A", "B", "C"], size=len(y_true))
    return y_true, y_prob, groups


def test_nonconformity_scores_range(conformal_data):
    """Scores should be in [0, 1]."""
    y_true, y_prob = conformal_data
    scores = compute_nonconformity_scores(y_true, y_prob)
    assert scores.shape == y_true.shape
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_nonconformity_perfect_predictions():
    """Perfect predictions should have score 0."""
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    scores = compute_nonconformity_scores(y_true, y_prob)
    np.testing.assert_array_almost_equal(scores, [0.0, 0.0, 0.0, 0.0])


def test_nonconformity_worst_predictions():
    """Completely wrong predictions should have score 1."""
    y_true = np.array([0, 1])
    y_prob = np.array([1.0, 0.0])
    scores = compute_nonconformity_scores(y_true, y_prob)
    np.testing.assert_array_almost_equal(scores, [1.0, 1.0])


def test_split_conformal_quantile():
    """Quantile threshold should be within score range."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    q = split_conformal_calibrate(scores, alpha=0.10)
    assert 0.0 <= q <= 1.0
    # Should cover at least (1-alpha) fraction of scores
    assert (scores <= q).mean() >= 0.90 - 0.01


def test_split_conformal_alpha_monotonic():
    """Lower alpha (higher coverage) -> higher threshold."""
    rng = np.random.default_rng(42)
    scores = rng.random(100)
    q_90 = split_conformal_calibrate(scores, alpha=0.10)
    q_95 = split_conformal_calibrate(scores, alpha=0.05)
    assert q_95 >= q_90


def test_prediction_sets_structure(conformal_data):
    """Prediction sets should have correct shape and values."""
    y_true, y_prob = conformal_data
    scores = compute_nonconformity_scores(y_true, y_prob)
    q = split_conformal_calibrate(scores, alpha=0.10)
    psets = conformal_prediction_sets(y_prob, q)

    assert psets["include_0"].shape == y_prob.shape
    assert psets["include_1"].shape == y_prob.shape
    assert psets["set_size"].shape == y_prob.shape
    assert np.all((psets["set_size"] >= 0) & (psets["set_size"] <= 2))


def test_prediction_sets_at_least_one():
    """With reasonable alpha, most samples should have non-empty sets."""
    rng = np.random.default_rng(42)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.clip(rng.random(n), 0.01, 0.99)
    scores = compute_nonconformity_scores(y_true, y_prob)
    q = split_conformal_calibrate(scores, alpha=0.10)
    psets = conformal_prediction_sets(y_prob, q)
    # Most sets should be non-empty
    assert (psets["set_size"] >= 1).mean() > 0.5


def test_evaluate_conformal_coverage(conformal_data):
    """Coverage should approximate target (1 - alpha)."""
    y_true, y_prob = conformal_data
    result = evaluate_conformal(y_true, y_prob, alpha=0.10, n_folds=3)
    # Coverage should be near 90%, within tolerance
    assert result["mean_coverage"] >= 0.80
    assert result["mean_coverage"] <= 1.0
    assert result["target_coverage"] == 0.90


def test_evaluate_conformal_structure(conformal_data):
    """Result dict should have all expected fields."""
    y_true, y_prob = conformal_data
    result = evaluate_conformal(y_true, y_prob, alpha=0.10, n_folds=3)
    expected_keys = {
        "alpha",
        "target_coverage",
        "mean_coverage",
        "coverage_per_fold",
        "mean_set_size",
        "set_sizes_per_fold",
        "standard_false_alarms",
        "standard_true_alarms",
        "conformal_false_alarms",
        "conformal_true_alarms",
        "false_alarm_reduction",
        "meets_50pct_reduction_target",
        "coverage_valid",
    }
    assert expected_keys.issubset(result.keys())


def test_evaluate_conformal_alarm_counts(conformal_data):
    """Alarm counts should be non-negative integers."""
    y_true, y_prob = conformal_data
    result = evaluate_conformal(y_true, y_prob, alpha=0.10, n_folds=3)
    assert result["standard_false_alarms"] >= 0
    assert result["standard_true_alarms"] >= 0
    assert result["conformal_false_alarms"] >= 0
    assert result["conformal_true_alarms"] >= 0


def test_mondrian_calibrate_per_group(grouped_data):
    """Mondrian should produce a threshold per group."""
    y_true, y_prob, groups = grouped_data
    scores = compute_nonconformity_scores(y_true, y_prob)
    thresholds = mondrian_conformal_calibrate(scores, groups, alpha=0.10)
    unique = np.unique(groups)
    assert len(thresholds) == len(unique)
    for g in unique:
        assert str(g) in thresholds
        assert 0.0 <= thresholds[str(g)] <= 1.0


def test_evaluate_mondrian_coverage(grouped_data):
    """Mondrian should give fair coverage across groups."""
    y_true, y_prob, groups = grouped_data
    result = evaluate_mondrian(y_true, y_prob, groups, alpha=0.10, n_folds=3)
    assert result["overall_mean_coverage"] >= 0.75
    assert "per_group" in result
    for _, metrics in result["per_group"].items():
        assert "mean_coverage" in metrics
        assert "mean_set_size" in metrics


def test_evaluate_mondrian_structure(grouped_data):
    """Mondrian result dict should have expected fields."""
    y_true, y_prob, groups = grouped_data
    result = evaluate_mondrian(y_true, y_prob, groups, alpha=0.10, n_folds=3)
    assert "alpha" in result
    assert "target_coverage" in result
    assert "max_coverage_disparity" in result
    assert "fair_coverage" in result
    assert result["max_coverage_disparity"] >= 0.0


def test_conformal_different_alphas(conformal_data):
    """Tighter alpha should give higher coverage but larger sets."""
    y_true, y_prob = conformal_data
    r_90 = evaluate_conformal(y_true, y_prob, alpha=0.10, n_folds=3)
    r_95 = evaluate_conformal(y_true, y_prob, alpha=0.05, n_folds=3)
    assert r_95["mean_coverage"] >= r_90["mean_coverage"] - 0.05
    assert r_95["mean_set_size"] >= r_90["mean_set_size"] - 0.1
