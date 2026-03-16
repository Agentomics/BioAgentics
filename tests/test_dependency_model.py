"""Tests for elastic-net dependency model training (small synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.dependency_model import (
    ModelResults,
    load_models,
    save_results,
    train_all_models,
)


@pytest.fixture
def synthetic_data():
    """Create a small synthetic dataset with one predictable and one noisy gene."""
    rng = np.random.default_rng(42)
    n_samples = 30
    n_features = 20

    X = pd.DataFrame(
        rng.normal(0, 1, (n_samples, n_features)),
        columns=[f"FEAT{i}" for i in range(n_features)],
        index=[f"ACH-{i:06d}" for i in range(n_samples)],
    )

    # Predictable gene: strong linear relationship with first 3 features
    y_pred = 0.5 * X["FEAT0"] - 0.3 * X["FEAT1"] + 0.8 * X["FEAT2"] + rng.normal(0, 0.2, n_samples)
    # Pure noise gene
    y_noise = rng.normal(0, 1, n_samples)

    Y = pd.DataFrame(
        {"PREDICTABLE": y_pred, "NOISE": y_noise},
        index=X.index,
    )
    return X, Y


def test_train_all_models_basic(synthetic_data):
    """Test basic model training on synthetic data."""
    X, Y = synthetic_data
    results = train_all_models(X, Y, n_folds=3, min_r=0.3, random_state=42)

    assert isinstance(results, ModelResults)
    assert results.n_total == 2
    assert len(results.metrics) == 2
    assert "cv_r" in results.metrics.columns
    assert "rmse" in results.metrics.columns


def test_predictable_gene_detected(synthetic_data):
    """Test that a strongly predictable gene passes the CV r threshold."""
    X, Y = synthetic_data
    results = train_all_models(X, Y, n_folds=3, min_r=0.3, random_state=42)

    # The predictable gene should have high CV r
    assert results.metrics.loc["PREDICTABLE", "cv_r"] > 0.3
    assert "PREDICTABLE" in results.predictable_genes
    assert "PREDICTABLE" in results.models


def test_noise_gene_filtered(synthetic_data):
    """Test that a noise gene is filtered out."""
    X, Y = synthetic_data
    results = train_all_models(X, Y, n_folds=3, min_r=0.3, random_state=42)

    # Pure noise should have low CV r
    assert results.metrics.loc["NOISE", "cv_r"] < 0.5


def test_save_and_load(synthetic_data, tmp_path):
    """Test saving and loading model results."""
    X, Y = synthetic_data
    results = train_all_models(X, Y, n_folds=3, min_r=0.3, random_state=42)

    results_dir = tmp_path / "results"
    save_results(results, results_dir)

    # Check files exist
    assert (results_dir / "gene_metrics.csv").exists()
    assert (results_dir / "predictable_genes.txt").exists()

    # Load models and verify
    loaded = load_models(results_dir)
    assert set(loaded.keys()) == set(results.models.keys())

    # Verify loaded model produces same predictions
    for gene in loaded:
        pred_orig = results.models[gene].predict(X.values)
        pred_loaded = loaded[gene].predict(X.values)
        np.testing.assert_array_almost_equal(pred_orig, pred_loaded)


def test_nan_handling():
    """Test that genes with NaN dependency scores are handled gracefully."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, (20, 5)), columns=[f"F{i}" for i in range(5)])
    Y = pd.DataFrame({"ALL_NAN": np.full(20, np.nan), "SOME_NAN": rng.normal(0, 1, 20)})
    Y.loc[:5, "SOME_NAN"] = np.nan

    results = train_all_models(X, Y, n_folds=3, min_r=0.3, random_state=42)
    # ALL_NAN gene should not crash, should have cv_r = 0
    assert results.metrics.loc["ALL_NAN", "cv_r"] == 0.0


def test_zero_variance_gene():
    """Test that genes with zero variance in dependency scores are skipped."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.normal(0, 1, (20, 5)), columns=[f"F{i}" for i in range(5)])
    Y = pd.DataFrame({"CONSTANT": np.zeros(20), "NORMAL": rng.normal(0, 1, 20)})

    results = train_all_models(X, Y, n_folds=3, min_r=0.3, random_state=42)
    assert results.metrics.loc["CONSTANT", "cv_r"] == 0.0
