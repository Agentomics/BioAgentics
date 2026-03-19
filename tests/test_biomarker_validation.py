"""Tests for bioagentics.models.biomarker_validation."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import anndata as ad
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from bioagentics.models.biomarker_validation import (
    ValidationResult,
    validate_on_holdout,
    compare_with_cunningham_genes,
    plot_roc_curves,
    plot_confusion_matrix,
    validation_pipeline,
    _compute_metrics,
)


def _make_model_data(features: list[str], seed: int = 42) -> dict:
    """Create a simple trained model dict for testing."""
    rng = np.random.default_rng(seed)
    n_train = 40
    n_feat = len(features)

    X_train = rng.normal(0, 1, (n_train, n_feat))
    X_train[:20, :5] += 3  # inject signal in first 20 = "case"

    le = LabelEncoder()
    le.fit(["case", "control"])
    # LabelEncoder maps alphabetically: case=0, control=1
    # First 20 samples are "case" (0), last 20 are "control" (1)
    y_train = le.transform(["case"] * 20 + ["control"] * 20)

    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X_train, y_train)

    return {"model": clf, "features": features, "label_encoder": le}


def _make_validation_adata(
    features: list[str], n_per_group: int = 15, seed: int = 99,
) -> ad.AnnData:
    """Create synthetic validation AnnData with separable groups."""
    rng = np.random.default_rng(seed)
    n = n_per_group * 2
    n_feat = len(features)

    X = rng.normal(0, 1, (n, n_feat)).astype(np.float32)
    X[:n_per_group, :5] += 3

    obs = pd.DataFrame({
        "condition": ["case"] * n_per_group + ["control"] * n_per_group,
    }, index=[f"V{i}" for i in range(n)])
    var = pd.DataFrame(index=features)
    return ad.AnnData(X=X, obs=obs, var=var)


class TestComputeMetrics:
    def test_perfect(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.0, 0.2, 0.9, 0.95, 0.8])
        m = _compute_metrics(y_true, y_prob)
        assert m["auc"] > 0.95
        assert m["sensitivity"] == 1.0
        assert m["specificity"] == 1.0

    def test_single_class(self):
        y_true = np.array([0, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.3])
        m = _compute_metrics(y_true, y_prob)
        assert m["auc"] == 0.0


class TestValidateOnHoldout:
    def test_validates_successfully(self):
        features = [f"GENE{i}" for i in range(30)]
        model_data = _make_model_data(features)
        val_adata = _make_validation_adata(features)

        result = validate_on_holdout(
            model_data, val_adata,
            condition_key="condition",
            classifier_name="RF",
            mode="combined",
        )

        assert isinstance(result, ValidationResult)
        assert result.auc > 0.5
        assert result.n_samples == 30
        assert result.confusion.shape == (2, 2)

    def test_missing_features_filled_with_zeros(self):
        features = [f"GENE{i}" for i in range(30)]
        model_data = _make_model_data(features)
        # Validation data has only first 20 features
        partial_features = features[:20] + [f"EXTRA{i}" for i in range(10)]
        val_adata = _make_validation_adata(partial_features)

        result = validate_on_holdout(
            model_data, val_adata,
            condition_key="condition",
        )

        assert result.n_features_used == 20

    def test_too_few_features_raises(self):
        features = [f"GENE{i}" for i in range(30)]
        model_data = _make_model_data(features)
        # Only 5 overlapping features (< 50%)
        bad_features = features[:5] + [f"X{i}" for i in range(25)]
        val_adata = _make_validation_adata(bad_features)

        try:
            validate_on_holdout(model_data, val_adata)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "features found" in str(e)


class TestCompareWithCunninghamGenes:
    def test_overlap_detection(self):
        selected = ["DRD1", "DRD2", "NOVEL1", "NOVEL2", "NOVEL3"]
        df = compare_with_cunningham_genes(selected)

        both = df[df["category"] == "both"]
        assert len(both) == 2  # DRD1 and DRD2

        novel = df[df["category"] == "classifier_only"]
        assert len(novel) == 3

    def test_no_overlap(self):
        df = compare_with_cunningham_genes(["NOVELX", "NOVELY"])
        both = df[df["category"] == "both"]
        assert len(both) == 0

    def test_all_columns_present(self):
        df = compare_with_cunningham_genes(["DRD1"])
        assert "gene" in df.columns
        assert "in_classifier" in df.columns
        assert "in_cunningham" in df.columns
        assert "category" in df.columns


class TestPlots:
    def test_roc_curves_saved(self, tmp_path):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.3, 0.2, 0.8, 0.9, 0.7])
        vr = ValidationResult(
            classifier_name="RF", mode="combined",
            auc=0.9, sensitivity=1.0, specificity=1.0,
            n_samples=6, n_features_used=10,
            y_true=y_true, y_prob=y_prob,
            confusion=np.array([[3, 0], [0, 3]]),
        )
        save_path = tmp_path / "roc.png"
        plot_roc_curves([vr], save_path=save_path)
        assert save_path.exists()

    def test_confusion_matrix_saved(self, tmp_path):
        vr = ValidationResult(
            classifier_name="RF", mode="combined",
            auc=0.9, sensitivity=1.0, specificity=1.0,
            n_samples=6, n_features_used=10,
            y_true=np.array([0, 0, 0, 1, 1, 1]),
            y_prob=np.array([0.1, 0.3, 0.2, 0.8, 0.9, 0.7]),
            confusion=np.array([[3, 0], [0, 3]]),
        )
        save_path = tmp_path / "cm.png"
        plot_confusion_matrix(vr, save_path=save_path)
        assert save_path.exists()

    def test_empty_results_no_crash(self, tmp_path):
        plot_roc_curves([], save_path=tmp_path / "empty.png")
        assert not (tmp_path / "empty.png").exists()


class TestValidationPipeline:
    def test_full_pipeline(self, tmp_path):
        features = [f"GENE{i}" for i in range(30)]
        model_data = _make_model_data(features)

        # Save model to disk
        model_path = tmp_path / "RandomForest_combined.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        val_adata = _make_validation_adata(features)

        results = validation_pipeline(
            model_paths=[model_path],
            validation_adata=val_adata,
            condition_key="condition",
            dest_dir=tmp_path,
        )

        assert "results" in results
        assert "metrics" in results
        assert "cunningham_comparison" in results
        assert (tmp_path / "validation_metrics.csv").exists()
        assert (tmp_path / "validation_roc.png").exists()
        assert (tmp_path / "cunningham_gene_comparison.csv").exists()

    def test_metrics_have_expected_columns(self, tmp_path):
        features = [f"GENE{i}" for i in range(20)]
        model_data = _make_model_data(features)

        model_path = tmp_path / "XGBoost_combined.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model_data, f)

        val_adata = _make_validation_adata(features)
        results = validation_pipeline(
            model_paths=[model_path],
            validation_adata=val_adata,
            dest_dir=tmp_path,
        )

        metrics = results["metrics"]
        assert "auc" in metrics.columns
        assert "sensitivity" in metrics.columns
        assert "specificity" in metrics.columns
