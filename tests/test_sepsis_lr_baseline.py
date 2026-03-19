"""Tests for logistic regression baseline model."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.sepsis.models.lr_baseline import (
    _build_pipeline,
    _select_best_C,
    evaluate_lr_nested_cv,
)


@pytest.fixture
def synthetic_dataset():
    """Generate a synthetic binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 200
    n_features = 10
    X = rng.standard_normal((n, n_features))
    # Make a separable signal in first two features
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int64)
    # Inject some NaNs
    mask = rng.random((n, n_features)) < 0.05
    X[mask] = np.nan
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


def test_build_pipeline():
    """Pipeline has imputer, scaler, and LR with L1 penalty."""
    pipe = _build_pipeline(C=1.0)
    assert "imputer" in pipe.named_steps
    assert "scaler" in pipe.named_steps
    assert "lr" in pipe.named_steps
    assert pipe.named_steps["lr"].l1_ratio == 1.0


def test_build_pipeline_can_fit(synthetic_dataset):
    """Pipeline fits and predicts on data with NaNs."""
    X, y, _ = synthetic_dataset
    pipe = _build_pipeline(C=1.0)
    pipe.fit(X, y)
    probs = pipe.predict_proba(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0) and np.all(probs <= 1)


def test_select_best_C(synthetic_dataset):
    """Inner CV selects a C value from the grid."""
    X, y, _ = synthetic_dataset
    from bioagentics.diagnostics.sepsis.models.lr_baseline import C_GRID

    best_C = _select_best_C(X, y, n_inner_folds=3)
    assert best_C in C_GRID


def test_evaluate_lr_nested_cv_returns_metrics(synthetic_dataset):
    """Nested CV returns all expected metric keys."""
    X, y, feature_names = synthetic_dataset
    result = evaluate_lr_nested_cv(
        X, y, feature_names=feature_names, n_outer_folds=3, n_inner_folds=2
    )

    assert "auroc_mean" in result
    assert "auprc_mean" in result
    assert "sensitivity_mean" in result
    assert "specificity_mean" in result
    assert "fold_results" in result
    assert "best_Cs" in result
    assert "coef_importance" in result

    # Metrics should be in valid ranges
    assert 0.0 <= result["auroc_mean"] <= 1.0
    assert 0.0 <= result["auprc_mean"] <= 1.0
    assert 0.0 <= result["sensitivity_mean"] <= 1.0
    assert 0.0 <= result["specificity_mean"] <= 1.0

    # Should have one result per fold
    assert len(result["fold_results"]) == 3
    assert len(result["best_Cs"]) == 3


def test_evaluate_lr_nested_cv_reasonable_auroc(synthetic_dataset):
    """On a separable dataset, AUROC should be well above chance."""
    X, y, _ = synthetic_dataset
    result = evaluate_lr_nested_cv(X, y, n_outer_folds=3, n_inner_folds=2)
    # Signal is strong in first two features, AUROC should be > 0.7
    assert result["auroc_mean"] > 0.7


def test_evaluate_lr_nested_cv_no_feature_names(synthetic_dataset):
    """Without feature_names, coef_importance is not included."""
    X, y, _ = synthetic_dataset
    result = evaluate_lr_nested_cv(
        X, y, feature_names=None, n_outer_folds=3, n_inner_folds=2
    )
    assert "coef_importance" not in result


def test_coef_importance_sorted(synthetic_dataset):
    """Feature importance is sorted by absolute coefficient."""
    X, y, feature_names = synthetic_dataset
    result = evaluate_lr_nested_cv(
        X, y, feature_names=feature_names, n_outer_folds=3, n_inner_folds=2
    )
    importances = result["coef_importance"]
    abs_coefs = [abs(item["coef"]) for item in importances]
    assert abs_coefs == sorted(abs_coefs, reverse=True)
