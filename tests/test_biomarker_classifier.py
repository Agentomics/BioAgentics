"""Tests for bioagentics.models.classifier (transcriptomic biomarker panel)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

from bioagentics.models.classifier import (
    select_features_rfecv,
    train_nested_cv,
    run_classifier_pipeline,
    _compute_metrics,
)


def _make_classifier_adata(
    n_genes: int = 50, n_per_group: int = 30, n_signal_genes: int = 5, seed: int = 42,
) -> ad.AnnData:
    """Create synthetic AnnData with an injected classification signal."""
    rng = np.random.default_rng(seed)
    n_total = n_per_group * 2

    X = rng.normal(50, 10, (n_total, n_genes)).clip(min=0).astype(np.float32)
    X[:n_per_group, :n_signal_genes] += 30

    obs = pd.DataFrame({
        "condition": ["case"] * n_per_group + ["control"] * n_per_group,
        "sex": (["M", "F"] * n_total)[:n_total],
    }, index=[f"S{i}" for i in range(n_total)])

    var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestSelectFeaturesRFECV:
    def test_selects_features(self):
        rng = np.random.default_rng(42)
        n = 60
        X = rng.normal(0, 1, (n, 30))
        X[:30, :5] += 3
        y = np.array([1] * 30 + [0] * 30)

        features = [f"G{i}" for i in range(30)]
        selected, rfecv = select_features_rfecv(X, y, features, min_features=5, cv_folds=3)

        assert len(selected) >= 5
        signal_feats = {f"G{i}" for i in range(5)}
        assert len(signal_feats & set(selected)) >= 3


class TestComputeMetrics:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.95])
        metrics = _compute_metrics(y_true, y_prob)
        assert metrics["auc"] > 0.95
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0

    def test_random_predictions(self):
        rng = np.random.default_rng(42)
        y_true = np.array([0] * 50 + [1] * 50)
        y_prob = rng.random(100)
        metrics = _compute_metrics(y_true, y_prob)
        assert 0.3 < metrics["auc"] < 0.7


class TestTrainNestedCV:
    def test_returns_three_classifiers(self):
        adata = _make_classifier_adata()
        X = np.array(adata.X, dtype=float)
        y = np.array([1] * 30 + [0] * 30)
        features = list(adata.var_names)

        results = train_nested_cv(X, y, features, mode="combined", outer_folds=3, inner_folds=3)

        assert len(results) == 3
        names = {r.name for r in results}
        assert "RandomForest" in names
        assert "XGBoost" in names
        assert "LogisticRegression" in names

    def test_classifiers_learn_signal(self):
        adata = _make_classifier_adata(n_per_group=40, n_signal_genes=10)
        X = np.array(adata.X, dtype=float)
        y = np.array([1] * 40 + [0] * 40)
        features = list(adata.var_names)

        results = train_nested_cv(X, y, features, mode="combined", outer_folds=3, inner_folds=3)

        best_auc = max(r.auc for r in results)
        assert best_auc > 0.7, f"Best AUC only {best_auc:.3f}"


class TestRunClassifierPipeline:
    def test_pipeline_returns_all_modes(self, tmp_path):
        adata = _make_classifier_adata(n_per_group=30)
        results = run_classifier_pipeline(
            adata,
            condition_key="condition",
            sex_key="sex",
            dest_dir=tmp_path,
        )

        assert "combined" in results
        assert "male" in results
        assert "female" in results
        assert len(results["combined"]) == 3

    def test_saves_outputs(self, tmp_path):
        adata = _make_classifier_adata(n_per_group=30)
        run_classifier_pipeline(
            adata,
            condition_key="condition",
            sex_key="sex",
            dest_dir=tmp_path,
        )

        assert (tmp_path / "classifier_metrics.csv").exists()
        assert (tmp_path / "selected_features.csv").exists()
        assert (tmp_path / "models").is_dir()
