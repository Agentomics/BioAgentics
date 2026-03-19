"""Tests for gradient boosting models (XGBoost + LightGBM)."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.sepsis.models.gbm import (
    _impute,
    _inner_cv_select,
    _make_lgbm_model,
    _make_xgb_model,
    evaluate_gbm_nested_cv,
    PARAM_GRID,
)


@pytest.fixture
def synthetic_dataset():
    """Synthetic binary classification dataset with NaNs."""
    rng = np.random.default_rng(42)
    n = 200
    n_features = 10
    X = rng.standard_normal((n, n_features))
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int64)
    mask = rng.random((n, n_features)) < 0.05
    X[mask] = np.nan
    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


def test_make_xgb_model():
    """XGBoost model is created with correct params."""
    model = _make_xgb_model(PARAM_GRID[0])
    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")


def test_make_lgbm_model():
    """LightGBM model is created with correct params."""
    model = _make_lgbm_model(PARAM_GRID[0])
    assert hasattr(model, "fit")
    assert hasattr(model, "predict_proba")


def test_impute_removes_nans(synthetic_dataset):
    """Imputation removes all NaN values."""
    X, _, _ = synthetic_dataset
    X_train, X_test = X[:150], X[150:]
    X_train_imp, X_test_imp = _impute(X_train, X_test)
    assert not np.any(np.isnan(X_train_imp))
    assert not np.any(np.isnan(X_test_imp))


def test_inner_cv_select_xgb(synthetic_dataset):
    """Inner CV selects valid params for XGBoost."""
    X, y, _ = synthetic_dataset
    best_params = _inner_cv_select(X, y, _make_xgb_model, n_inner_folds=2)
    assert best_params in PARAM_GRID


def test_inner_cv_select_lgbm(synthetic_dataset):
    """Inner CV selects valid params for LightGBM."""
    X, y, _ = synthetic_dataset
    best_params = _inner_cv_select(X, y, _make_lgbm_model, n_inner_folds=2)
    assert best_params in PARAM_GRID


def test_evaluate_xgboost_nested_cv(synthetic_dataset):
    """XGBoost nested CV returns valid metrics."""
    X, y, feature_names = synthetic_dataset
    result = evaluate_gbm_nested_cv(
        X, y, model_name="xgboost", feature_names=feature_names,
        n_outer_folds=3, n_inner_folds=2,
    )
    assert 0.0 <= result["auroc_mean"] <= 1.0
    assert 0.0 <= result["auprc_mean"] <= 1.0
    assert result["model"] == "xgboost"
    assert len(result["fold_results"]) == 3
    assert "feature_importance" in result


def test_evaluate_lightgbm_nested_cv(synthetic_dataset):
    """LightGBM nested CV returns valid metrics."""
    X, y, feature_names = synthetic_dataset
    result = evaluate_gbm_nested_cv(
        X, y, model_name="lightgbm", feature_names=feature_names,
        n_outer_folds=3, n_inner_folds=2,
    )
    assert 0.0 <= result["auroc_mean"] <= 1.0
    assert result["model"] == "lightgbm"
    assert len(result["fold_results"]) == 3


def test_gbm_reasonable_auroc(synthetic_dataset):
    """On separable data, GBM AUROC should be well above chance."""
    X, y, _ = synthetic_dataset
    result = evaluate_gbm_nested_cv(
        X, y, model_name="xgboost", n_outer_folds=3, n_inner_folds=2,
    )
    assert result["auroc_mean"] > 0.7


def test_feature_importance_sorted(synthetic_dataset):
    """Feature importance is sorted descending."""
    X, y, feature_names = synthetic_dataset
    result = evaluate_gbm_nested_cv(
        X, y, model_name="xgboost", feature_names=feature_names,
        n_outer_folds=3, n_inner_folds=2,
    )
    importances = result["feature_importance"]
    values = [item["importance"] for item in importances]
    assert values == sorted(values, reverse=True)
