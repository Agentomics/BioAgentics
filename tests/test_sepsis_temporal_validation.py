"""Tests for Phase 4 temporal validation module."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.sepsis.validation.temporal_validation import (
    _compute_metrics,
    _train_lr,
    evaluate_temporal_holdout,
    temporal_split,
)


@pytest.fixture
def synth_data():
    """Synthetic sepsis-like dataset with admission times.

    Ensures both classes are present throughout the time range
    so temporal splits always have both classes.
    """
    rng = np.random.default_rng(42)
    n = 200
    n_features = 26

    # Interleave positive/negative throughout time to ensure both
    # classes appear in train and test after temporal split
    y = np.zeros(n, dtype=int)
    # Spread 60 positives evenly across the 200 samples
    pos_idx = np.linspace(0, n - 1, 60, dtype=int)
    y[pos_idx] = 1

    X = rng.standard_normal((n, n_features))
    X[y == 1, 0] += 1.5
    X[y == 1, 3] += 1.0

    # Admission times: sequential integers (already ordered by time)
    admit_times = np.arange(n, dtype=float)

    return X, y, admit_times


def test_temporal_split_sizes(synth_data):
    """Split produces correct train/test sizes."""
    X, y, admit_times = synth_data
    result = temporal_split(X, y, admit_times, holdout_frac=0.2)
    assert len(result["y_train"]) == 160
    assert len(result["y_test"]) == 40
    assert result["X_train"].shape[0] == 160
    assert result["X_test"].shape[0] == 40


def test_temporal_split_ordering(synth_data):
    """Train set should contain earlier admissions than test set."""
    X, y, admit_times = synth_data
    result = temporal_split(X, y, admit_times, holdout_frac=0.2)
    train_max_time = admit_times[result["train_idx"]].max()
    test_min_time = admit_times[result["test_idx"]].min()
    assert train_max_time <= test_min_time


def test_temporal_split_no_overlap(synth_data):
    """Train and test indices should not overlap."""
    X, y, admit_times = synth_data
    result = temporal_split(X, y, admit_times, holdout_frac=0.2)
    assert len(set(result["train_idx"]) & set(result["test_idx"])) == 0


def test_temporal_split_custom_frac(synth_data):
    """Custom holdout fraction works correctly."""
    X, y, admit_times = synth_data
    result = temporal_split(X, y, admit_times, holdout_frac=0.3)
    assert len(result["y_train"]) == 140
    assert len(result["y_test"]) == 60


def test_compute_metrics_structure():
    """Metrics dict has expected keys and valid ranges."""
    rng = np.random.default_rng(42)
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_prob = rng.random(10)
    result = _compute_metrics(y_true, y_prob)
    for key in ["auroc", "auprc", "sensitivity", "specificity", "ppv", "npv"]:
        assert key in result
        assert 0.0 <= result[key] <= 1.0
    assert result["n_samples"] == 10
    assert result["n_positive"] == 5


def test_compute_metrics_perfect():
    """Perfect predictions yield AUROC=1."""
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    result = _compute_metrics(y_true, y_prob)
    assert result["auroc"] == 1.0


def test_train_lr(synth_data):
    """LR training returns valid probability array."""
    X, y, _ = synth_data
    probs = _train_lr(X[:160], y[:160], X[160:])
    assert probs.shape == (40,)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_evaluate_temporal_holdout_structure(synth_data):
    """Temporal holdout evaluation returns expected structure."""
    X, y, admit_times = synth_data
    split = temporal_split(X, y, admit_times, holdout_frac=0.2)
    result = evaluate_temporal_holdout(
        split["X_train"], split["y_train"],
        split["X_test"], split["y_test"],
    )
    for model in ["logistic_regression", "xgboost", "lightgbm", "ensemble_avg"]:
        assert model in result
        assert "auroc" in result[model]
        assert 0.0 <= result[model]["auroc"] <= 1.0
    assert "best_auroc" in result
    assert "meets_target_085" in result


def test_evaluate_temporal_holdout_reasonable_auroc(synth_data):
    """With signal in data, models should achieve above-chance AUROC."""
    X, y, admit_times = synth_data
    split = temporal_split(X, y, admit_times, holdout_frac=0.2)
    result = evaluate_temporal_holdout(
        split["X_train"], split["y_train"],
        split["X_test"], split["y_test"],
    )
    # With planted signal, best model should beat random
    assert result["best_auroc"] > 0.5


def test_evaluate_temporal_holdout_calibration(synth_data):
    """Temporal holdout includes ECE calibration metrics."""
    X, y, admit_times = synth_data
    split = temporal_split(X, y, admit_times, holdout_frac=0.2)
    result = evaluate_temporal_holdout(
        split["X_train"], split["y_train"],
        split["X_test"], split["y_test"],
    )
    assert "calibration" in result
    for model in ["logistic_regression", "xgboost", "lightgbm", "ensemble_avg"]:
        assert model in result["calibration"]
        assert "ece" in result["calibration"][model]
        assert 0.0 <= result["calibration"][model]["ece"] <= 1.0
        assert "meets_ece_target" in result["calibration"][model]
