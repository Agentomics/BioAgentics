"""Tests for Phase 4 external validation module."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.sepsis.validation.external_validation import (
    _compute_metrics,
    align_features,
    evaluate_external,
    map_eicu_features,
    per_center_analysis,
    train_on_source,
)


@pytest.fixture
def source_data():
    """Synthetic MIMIC-IV-like training data."""
    rng = np.random.default_rng(42)
    n = 200
    n_features = 26
    X = rng.standard_normal((n, n_features))
    y = np.concatenate([np.zeros(140), np.ones(60)]).astype(int)
    X[y == 1, 0] += 1.5
    X[y == 1, 3] += 1.0
    return X, y


@pytest.fixture
def external_data():
    """Synthetic eICU-like external data with both classes per center."""
    rng = np.random.default_rng(123)
    n = 120
    n_features = 26
    # Ensure each center has both classes
    # Center A: 40 samples (28 neg, 12 pos)
    # Center B: 40 samples (28 neg, 12 pos)
    # Center C: 40 samples (28 neg, 12 pos)
    y = np.array(
        [0] * 28 + [1] * 12
        + [0] * 28 + [1] * 12
        + [0] * 28 + [1] * 12
    )
    X = rng.standard_normal((n, n_features))
    X[y == 1, 0] += 1.2
    X[y == 1, 3] += 0.8
    center_ids = np.array(["A"] * 40 + ["B"] * 40 + ["C"] * 40)
    return X, y, center_ids


def test_map_eicu_features():
    """Feature mapping identifies correct mappings and missing features."""
    eicu_cols = ["heartrate", "systemicsystolic", "WBC x 1000", "unknown_col"]
    target = ["heart_rate", "sbp", "wbc", "lactate"]
    result = map_eicu_features(eicu_cols, target)
    assert result["heart_rate"] == "heartrate"
    assert result["sbp"] == "systemicsystolic"
    assert result["wbc"] == "WBC x 1000"
    assert result["lactate"] is None  # not in eicu_cols


def test_align_features_correct_order():
    """Feature alignment reorders correctly and fills missing with NaN."""
    X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    external_names = ["feat_b", "feat_a", "feat_c"]
    target_names = ["feat_a", "feat_b", "feat_c", "feat_d"]
    result = align_features(X, external_names, target_names)
    assert result.shape == (2, 4)
    np.testing.assert_array_equal(result[:, 0], [2.0, 5.0])  # feat_a
    np.testing.assert_array_equal(result[:, 1], [1.0, 4.0])  # feat_b
    np.testing.assert_array_equal(result[:, 2], [3.0, 6.0])  # feat_c
    assert np.all(np.isnan(result[:, 3]))  # feat_d missing


def test_align_features_all_missing():
    """All missing features produces all-NaN matrix."""
    X = np.array([[1.0, 2.0]])
    result = align_features(X, ["a", "b"], ["c", "d"])
    assert result.shape == (1, 2)
    assert np.all(np.isnan(result))


def test_train_on_source(source_data):
    """Training returns fitted model objects."""
    X, y = source_data
    trained = train_on_source(X, y)
    assert "lr" in trained
    assert "xgb" in trained
    assert "lgbm" in trained
    assert "imputer_lr" in trained
    assert "scaler_lr" in trained
    assert "imputer_gbm" in trained


def test_evaluate_external_structure(source_data, external_data):
    """External evaluation returns expected structure."""
    X_train, y_train = source_data
    X_ext, y_ext, _ = external_data
    trained = train_on_source(X_train, y_train)
    result = evaluate_external(trained, X_ext, y_ext)
    for model in ["logistic_regression", "xgboost", "lightgbm", "ensemble_avg"]:
        assert model in result
        assert "auroc" in result[model]
    assert "best_auroc" in result
    assert "meets_target_080" in result


def test_evaluate_external_reasonable_auroc(source_data, external_data):
    """Models trained on source should generalize above chance."""
    X_train, y_train = source_data
    X_ext, y_ext, _ = external_data
    trained = train_on_source(X_train, y_train)
    result = evaluate_external(trained, X_ext, y_ext)
    assert result["best_auroc"] > 0.5


def test_per_center_analysis(source_data, external_data):
    """Per-center analysis returns valid structure."""
    X_train, y_train = source_data
    X_ext, y_ext, center_ids = external_data
    trained = train_on_source(X_train, y_train)
    result = per_center_analysis(
        trained, X_ext, y_ext, center_ids,
        min_samples=10, min_positive=3,
    )
    assert "n_centers" in result
    assert result["n_centers"] >= 1
    assert "auroc_mean" in result
    assert 0.0 <= result["auroc_mean"] <= 1.0


def test_per_center_analysis_insufficient_data(source_data):
    """Centers with too few samples are excluded."""
    X_train, y_train = source_data
    X_ext = np.random.default_rng(42).standard_normal((20, 26))
    y_ext = np.array([0] * 15 + [1] * 5)
    # Only one center but with high threshold
    center_ids = np.array(["A"] * 20)
    trained = train_on_source(X_train, y_train)
    result = per_center_analysis(
        trained, X_ext, y_ext, center_ids,
        min_samples=50, min_positive=10,
    )
    assert result["n_centers"] == 0


def test_compute_metrics_valid_range():
    """Metrics are in valid ranges."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.2, 0.3, 0.8, 0.7, 0.4, 0.6])
    result = _compute_metrics(y_true, y_prob)
    assert 0 <= result["auroc"] <= 1
    assert result["n_samples"] == 6
    assert result["n_positive"] == 3


def test_evaluate_external_calibration(source_data, external_data):
    """External evaluation includes ECE calibration metrics."""
    X_train, y_train = source_data
    X_ext, y_ext, _ = external_data
    trained = train_on_source(X_train, y_train)
    result = evaluate_external(trained, X_ext, y_ext)
    assert "calibration" in result
    for model in ["logistic_regression", "xgboost", "lightgbm", "ensemble_avg"]:
        assert model in result["calibration"]
        assert "ece" in result["calibration"][model]
        assert 0.0 <= result["calibration"][model]["ece"] <= 1.0
