"""Tests for TCGA dependency prediction (mock models, synthetic data)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import ElasticNet

from bioagentics.models.tcga_prediction import predict_tcga_dependencies, save_predictions


@pytest.fixture
def mock_models_and_data():
    """Create mock trained models and TCGA expression data."""
    rng = np.random.default_rng(42)
    feature_genes = [f"FEAT{i}" for i in range(20)]
    n_patients = 50

    # Create simple linear models
    models = {}
    for gene_name in ["TARGET_A", "TARGET_B"]:
        model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)
        X_train = rng.normal(0, 1, (30, 20))
        y_train = X_train[:, 0] * 0.5 + rng.normal(0, 0.1, 30)
        model.fit(X_train, y_train)
        models[gene_name] = model

    # TCGA expression: all features present
    tcga_expr = pd.DataFrame(
        rng.normal(5, 2, (n_patients, 20)),
        columns=feature_genes,
        index=[f"TCGA-XX-{i:04d}" for i in range(n_patients)],
    )

    return models, feature_genes, tcga_expr


def test_predict_basic(mock_models_and_data):
    """Test basic prediction produces correct shape."""
    models, feature_genes, tcga_expr = mock_models_and_data
    result = predict_tcga_dependencies(models, feature_genes, tcga_expr)

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (50, 2)
    assert set(result.columns) == {"TARGET_A", "TARGET_B"}
    assert list(result.index) == list(tcga_expr.index)


def test_missing_features_within_threshold(mock_models_and_data):
    """Test that <20% missing features works with warning."""
    models, feature_genes, tcga_expr = mock_models_and_data
    # Drop 2/20 = 10% features
    tcga_expr_partial = tcga_expr.drop(columns=["FEAT0", "FEAT1"])
    result = predict_tcga_dependencies(models, feature_genes, tcga_expr_partial)
    assert result.shape == (50, 2)


def test_missing_features_exceeds_threshold(mock_models_and_data):
    """Test that >20% missing features raises error."""
    models, feature_genes, tcga_expr = mock_models_and_data
    # Drop 5/20 = 25% features
    tcga_expr_partial = tcga_expr.drop(columns=["FEAT0", "FEAT1", "FEAT2", "FEAT3", "FEAT4"])
    with pytest.raises(ValueError, match="missing"):
        predict_tcga_dependencies(models, feature_genes, tcga_expr_partial)


def test_extra_tcga_genes_ignored(mock_models_and_data):
    """Test that extra genes in TCGA data don't affect predictions."""
    models, feature_genes, tcga_expr = mock_models_and_data
    tcga_expr["EXTRA_GENE"] = 42.0
    result = predict_tcga_dependencies(models, feature_genes, tcga_expr)
    assert result.shape == (50, 2)
    assert "EXTRA_GENE" not in result.columns


def test_save_predictions(mock_models_and_data, tmp_path):
    """Test saving predictions to disk."""
    models, feature_genes, tcga_expr = mock_models_and_data
    result = predict_tcga_dependencies(models, feature_genes, tcga_expr)
    out_path = save_predictions(result, tmp_path)
    assert out_path.exists()
    loaded = pd.read_csv(out_path, index_col=0)
    assert loaded.shape == result.shape


def test_save_with_metadata(mock_models_and_data, tmp_path):
    """Test saving predictions with patient metadata."""
    models, feature_genes, tcga_expr = mock_models_and_data
    result = predict_tcga_dependencies(models, feature_genes, tcga_expr)
    meta = pd.DataFrame({
        "patient_id": tcga_expr.index,
        "subtype": ["KP"] * 25 + ["KL"] * 25,
    })
    save_predictions(result, tmp_path, patient_meta=meta)
    assert (tmp_path / "tcga_patient_metadata.csv").exists()
