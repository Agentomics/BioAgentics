"""Tests for ensemble stacking framework."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.sepsis.models.ensemble import (
    _get_oof_predictions,
    _get_test_predictions,
    evaluate_ensemble,
    train_stacking_ensemble,
)


@pytest.fixture
def synthetic_dataset():
    """Synthetic binary classification dataset with NaNs."""
    rng = np.random.default_rng(42)
    n = 150
    n_features = 8
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int64)
    mask = rng.random((n, n_features)) < 0.03
    X[mask] = np.nan
    return X, y


def test_get_oof_predictions_shape(synthetic_dataset):
    """OOF predictions have correct shape (n_samples, 5)."""
    X, y = synthetic_dataset
    oof = _get_oof_predictions(X, y, n_folds=3)
    assert oof.shape == (len(X), 5)
    # All predictions should be in [0, 1]
    assert np.all(oof >= 0.0) and np.all(oof <= 1.0)


def test_get_test_predictions_shape(synthetic_dataset):
    """Test predictions have correct shape."""
    X, y = synthetic_dataset
    X_train, X_test = X[:100], X[100:]
    y_train = y[:100]
    preds = _get_test_predictions(X_train, y_train, X_test)
    assert preds.shape == (50, 5)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_train_stacking_ensemble():
    """Meta-learner trains on stacked predictions."""
    rng = np.random.default_rng(42)
    oof = rng.random((100, 5))
    y = (rng.random(100) > 0.5).astype(np.int64)
    meta = train_stacking_ensemble(oof, y)
    probs = meta.predict_proba(oof)
    assert probs.shape == (100, 2)


def test_evaluate_ensemble(synthetic_dataset):
    """Full ensemble evaluation returns valid metrics."""
    X, y = synthetic_dataset
    result = evaluate_ensemble(X, y, n_folds=2)

    assert result["model"] == "stacking_ensemble"
    assert 0.0 <= result["auroc_mean"] <= 1.0
    assert 0.0 <= result["auprc_mean"] <= 1.0
    assert len(result["fold_results"]) == 2
    assert "individual_aurocs" in result
    assert "lr" in result["individual_aurocs"]
    assert "xgboost" in result["individual_aurocs"]
