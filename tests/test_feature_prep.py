"""Tests for feature matrix preparation (synthetic data, no real DepMap files needed)."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.feature_prep import (
    FeatureResult,
    prepare_features,
)


def _make_nsclc_meta(line_ids: list[str]) -> pd.DataFrame:
    """Create mock NSCLC cell line metadata."""
    subtypes = ["KP", "KL", "KPL", "KOnly", "KRAS-WT"]
    rows = []
    for i, _lid in enumerate(line_ids):
        rows.append({
            "CellLineName": f"Line{i}",
            "StrippedCellLineName": f"LINE{i}",
            "OncotreeSubtype": "Lung Adenocarcinoma",
            "molecular_subtype": subtypes[i % len(subtypes)],
            "KRAS_mutated": subtypes[i % len(subtypes)] != "KRAS-WT",
            "TP53_mutated": subtypes[i % len(subtypes)] in ("KP", "KPL"),
            "STK11_mutated": subtypes[i % len(subtypes)] in ("KL", "KPL"),
            "KRAS_allele": "G12C" if subtypes[i % len(subtypes)] != "KRAS-WT" else "WT",
        })
    return pd.DataFrame(rows, index=line_ids)


def _make_expr_matrix(line_ids: list[str], n_genes: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create mock expression matrix with varying gene variance."""
    rng = np.random.default_rng(seed)
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    # Make some genes high-variance, others low
    scales = np.linspace(0.1, 5.0, n_genes)
    data = rng.normal(5.0, scales, size=(len(line_ids), n_genes))
    return pd.DataFrame(data, index=line_ids, columns=gene_names)


def _make_crispr_matrix(line_ids: list[str], n_genes: int = 80, seed: int = 99) -> pd.DataFrame:
    """Create mock CRISPR gene effect matrix."""
    rng = np.random.default_rng(seed)
    gene_names = [f"TARGET{i}" for i in range(n_genes)]
    data = rng.normal(-0.3, 0.5, size=(len(line_ids), n_genes))
    return pd.DataFrame(data, index=line_ids, columns=gene_names)


@pytest.fixture
def mock_data():
    """Set up mock data for feature prep tests."""
    # 20 cell lines, 15 overlap all three datasets
    all_ids = [f"ACH-{i:06d}" for i in range(20)]
    meta_ids = all_ids[:18]   # 18 lines in NSCLC meta
    expr_ids = all_ids[2:20]  # 18 lines in expression (overlap: indices 2-17 = 16 lines)
    crispr_ids = all_ids[1:19]  # 18 lines in CRISPR (overlap with above: indices 2-17 = 16 lines)
    # Triple intersection = indices 2-17 = 16 lines

    meta = _make_nsclc_meta(meta_ids)
    expr = _make_expr_matrix(expr_ids, n_genes=100)
    crispr = _make_crispr_matrix(crispr_ids, n_genes=80)

    return meta, expr, crispr


def test_prepare_features_basic(mock_data, tmp_path):
    """Test basic feature preparation with synthetic data."""
    meta, expr, crispr = mock_data

    with (
        patch("bioagentics.models.feature_prep.annotate_nsclc_lines", return_value=meta),
        patch("bioagentics.models.feature_prep.load_depmap_matrix", side_effect=[expr, crispr]),
    ):
        result = prepare_features(depmap_dir=tmp_path, n_features=50)

    assert isinstance(result, FeatureResult)
    # 16 lines in triple intersection
    assert result.n_lines == 16
    assert result.X.shape[0] == 16
    assert result.Y.shape[0] == 16
    # Requested 50 features, 100 available → 50 selected
    assert result.n_features == 50
    assert result.X.shape[1] == 50
    # All 80 CRISPR targets
    assert result.n_targets == 80
    assert len(result.feature_genes) == 50
    assert len(result.target_genes) == 80


def test_feature_selection_prefers_high_variance(mock_data, tmp_path):
    """Test that feature selection favors higher-variance genes."""
    meta, expr, crispr = mock_data

    with (
        patch("bioagentics.models.feature_prep.annotate_nsclc_lines", return_value=meta),
        patch("bioagentics.models.feature_prep.load_depmap_matrix", side_effect=[expr, crispr]),
    ):
        result = prepare_features(depmap_dir=tmp_path, n_features=30)

    # With scales = linspace(0.1, 5.0, 100), genes with higher indices
    # have higher expected variance. The top 30 should skew toward high indices.
    selected_indices = [int(g.replace("GENE", "")) for g in result.feature_genes]
    mean_index = np.mean(selected_indices)
    # Mean index of top-30 should be well above the overall mean (49.5)
    assert mean_index > 60


def test_kpl_grouping_kl(mock_data, tmp_path):
    """Test that KPL lines are grouped with KL when grouping='kl'."""
    meta, expr, crispr = mock_data

    with (
        patch("bioagentics.models.feature_prep.annotate_nsclc_lines", return_value=meta),
        patch("bioagentics.models.feature_prep.load_depmap_matrix", side_effect=[expr, crispr]),
    ):
        result = prepare_features(depmap_dir=tmp_path, grouping="kl")

    subtypes = result.cell_line_meta["molecular_subtype"].unique()
    assert "KPL" not in subtypes
    assert "KL" in subtypes


def test_kpl_grouping_separate(mock_data, tmp_path):
    """Test that KPL lines remain distinct when grouping='separate'."""
    meta, expr, crispr = mock_data

    with (
        patch("bioagentics.models.feature_prep.annotate_nsclc_lines", return_value=meta),
        patch("bioagentics.models.feature_prep.load_depmap_matrix", side_effect=[expr, crispr]),
    ):
        result = prepare_features(depmap_dir=tmp_path, grouping="separate")

    subtypes = result.cell_line_meta["molecular_subtype"].unique()
    assert "KPL" in subtypes


def test_invalid_grouping(tmp_path):
    """Test that invalid grouping raises ValueError."""
    with pytest.raises(ValueError, match="Unknown grouping"):
        prepare_features(depmap_dir=tmp_path, grouping="invalid")


def test_n_features_capped(mock_data, tmp_path):
    """Test that n_features is capped at available genes."""
    meta, expr, crispr = mock_data

    with (
        patch("bioagentics.models.feature_prep.annotate_nsclc_lines", return_value=meta),
        patch("bioagentics.models.feature_prep.load_depmap_matrix", side_effect=[expr, crispr]),
    ):
        result = prepare_features(depmap_dir=tmp_path, n_features=9999)

    # Only 100 expression genes available
    assert result.n_features == 100


def test_xy_index_alignment(mock_data, tmp_path):
    """Test that X and Y have identical row indices."""
    meta, expr, crispr = mock_data

    with (
        patch("bioagentics.models.feature_prep.annotate_nsclc_lines", return_value=meta),
        patch("bioagentics.models.feature_prep.load_depmap_matrix", side_effect=[expr, crispr]),
    ):
        result = prepare_features(depmap_dir=tmp_path)

    pd.testing.assert_index_equal(result.X.index, result.Y.index)
    pd.testing.assert_index_equal(result.X.index, result.cell_line_meta.index)
