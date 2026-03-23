"""Tests for Phase 4 SHAP explainability module."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.sepsis.validation.explainability import (
    compare_centers_ranking,
    compute_shap_tree,
    feature_importance_stability,
    generate_plot_data,
    generate_waterfall_data,
    run_explainability,
)


@pytest.fixture
def synth_data():
    """Synthetic dataset with known feature importance."""
    rng = np.random.default_rng(42)
    n = 200
    n_features = 10

    X = rng.standard_normal((n, n_features))
    y = np.concatenate([np.zeros(130), np.ones(70)]).astype(int)
    # Features 0 and 1 are predictive
    X[y == 1, 0] += 2.0
    X[y == 1, 1] += 1.5
    # Sprinkle NaNs (10%)
    nan_mask = rng.random((n, n_features)) < 0.1
    X[nan_mask] = np.nan

    feature_names = [f"feat_{i}" for i in range(n_features)]
    return X, y, feature_names


@pytest.fixture
def fitted_xgb(synth_data):
    """Fitted XGBoost model on synthetic data."""
    import xgboost as xgb
    from sklearn.impute import SimpleImputer

    X, y, _ = synth_data
    imp = SimpleImputer(strategy="median")
    X_clean = imp.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, n_jobs=1, verbosity=0,
    )
    model.fit(X_clean, y)
    return model, X_clean


def test_compute_shap_tree_structure(fitted_xgb, synth_data):
    """SHAP computation returns expected structure."""
    model, X_clean = fitted_xgb
    _, _, feature_names = synth_data
    result = compute_shap_tree(model, X_clean, feature_names, max_samples=100)
    assert "shap_values" in result
    assert "mean_abs_shap" in result
    assert "feature_ranking" in result
    assert len(result["feature_ranking"]) == len(feature_names)
    assert result["n_samples_explained"] <= 100


def test_compute_shap_tree_values_shape(fitted_xgb, synth_data):
    """SHAP values have correct shape."""
    model, X_clean = fitted_xgb
    _, _, feature_names = synth_data
    result = compute_shap_tree(model, X_clean, feature_names, max_samples=50)
    assert result["shap_values"].shape == (50, len(feature_names))


def test_compute_shap_top_features(fitted_xgb, synth_data):
    """Most important features should be the planted signal features."""
    model, X_clean = fitted_xgb
    _, _, feature_names = synth_data
    result = compute_shap_tree(model, X_clean, feature_names, max_samples=200)
    top_2 = {r["feature"] for r in result["feature_ranking"][:2]}
    # feat_0 and feat_1 have planted signal
    assert "feat_0" in top_2 or "feat_1" in top_2


def test_compute_shap_no_feature_names(fitted_xgb):
    """Works without explicit feature names."""
    model, X_clean = fitted_xgb
    result = compute_shap_tree(model, X_clean, feature_names=None)
    assert result["feature_ranking"][0]["feature"].startswith("feature_")


def test_feature_importance_stability_structure(synth_data):
    """Stability analysis returns expected fields."""
    X, y, feature_names = synth_data
    result = feature_importance_stability(
        X, y, feature_names, n_folds=3, max_samples_per_fold=50,
    )
    assert "n_folds" in result
    assert result["n_folds"] == 3
    assert "rank_correlations" in result
    assert "mean_rank_correlation" in result
    assert "stable_top_features" in result
    assert "overall_ranking" in result
    assert len(result["overall_ranking"]) == len(feature_names)


def test_feature_importance_stability_correlations(synth_data):
    """Rank correlations should be positive for consistent data."""
    X, y, feature_names = synth_data
    result = feature_importance_stability(
        X, y, feature_names, n_folds=3, max_samples_per_fold=50,
    )
    # With planted signal, folds should agree on important features
    assert result["mean_rank_correlation"] > 0.0
    # Should have 3 pairwise correlations for 3 folds
    assert len(result["rank_correlations"]) == 3


def test_feature_importance_stability_top_features(synth_data):
    """Planted signal features should appear in stable top set."""
    X, y, feature_names = synth_data
    result = feature_importance_stability(
        X, y, feature_names, n_folds=3, max_samples_per_fold=100,
    )
    # At least one planted feature should be consistently top
    top_feats = {r["feature"] for r in result["overall_ranking"][:3]}
    assert "feat_0" in top_feats or "feat_1" in top_feats


def test_run_explainability(synth_data, tmp_path):
    """Full explainability pipeline produces output file."""
    X, y, feature_names = synth_data
    result = run_explainability(
        X, y, feature_names, results_dir=tmp_path, label="test",
    )
    assert "feature_ranking" in result
    assert "stability" in result
    assert (tmp_path / "explainability_test.json").exists()


def test_compare_centers_ranking(fitted_xgb, synth_data):
    """Cross-center comparison produces valid correlations."""
    model, X_clean = fitted_xgb
    _, _, feature_names = synth_data

    result_a = compute_shap_tree(model, X_clean[:100], feature_names, max_samples=50)
    result_b = compute_shap_tree(model, X_clean[100:], feature_names, max_samples=50)

    comparison = compare_centers_ranking({
        "mimic": result_a,
        "eicu": result_b,
    })
    assert comparison["n_centers"] == 2
    assert len(comparison["pairwise_correlations"]) == 1
    assert -1.0 <= comparison["mean_correlation"] <= 1.0
    assert isinstance(comparison["shared_top_features"], list)


def test_generate_plot_data(fitted_xgb, synth_data):
    """Plot data generation returns bar and beeswarm data."""
    model, X_clean = fitted_xgb
    _, _, feature_names = synth_data
    result = compute_shap_tree(model, X_clean, feature_names, max_samples=50)

    plot = generate_plot_data(
        result["shap_values"], X_clean[:result["n_samples_explained"]],
        feature_names, top_k=5,
    )
    assert "bar" in plot
    assert "beeswarm" in plot
    assert len(plot["bar"]) == 5
    assert len(plot["beeswarm"]) == 5
    assert "feature" in plot["bar"][0]
    assert "mean_abs_shap" in plot["bar"][0]


def test_generate_waterfall_data(fitted_xgb, synth_data):
    """Waterfall data for a single prediction."""
    model, X_clean = fitted_xgb
    _, _, feature_names = synth_data
    result = compute_shap_tree(model, X_clean, feature_names, max_samples=50)

    wf = generate_waterfall_data(
        result["shap_values"],
        X_clean[:result["n_samples_explained"]],
        feature_names,
        sample_idx=0,
        top_k=5,
    )
    assert wf["sample_idx"] == 0
    assert "base_value" in wf
    assert "final_value" in wf
    assert len(wf["contributions"]) == 5
    assert "feature" in wf["contributions"][0]
    assert "shap_value" in wf["contributions"][0]
