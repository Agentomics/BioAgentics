"""Tests for striatal cell-type deconvolution module."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.analysis.tourettes.striatal_celltypes import (
    STRIATAL_CELL_TYPES,
    annotate_with_hmba,
    build_marker_expression_matrix,
    compute_gene_celltype_specificity,
    compute_tau_specificity,
    compute_celltype_enrichment,
)
from bioagentics.data.tourettes.gene_sets import (
    get_celltype_markers,
    list_celltype_markers,
)


def _make_expression_df(
    n_donors: int = 2,
    include_markers: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic AHBA expression data for testing.

    Generates expression records across striatal + other regions for
    TS risk genes and (optionally) cell-type marker genes.
    """
    rng = np.random.default_rng(seed)
    regions = ["caudate", "putamen", "GPe", "thalamus", "other"]
    donors = list(range(1, n_donors + 1))

    # TS risk genes
    genes = ["FLT3", "MEIS1", "PTPRD", "SEMA6D", "HDC", "SLITRK1", "LHX6"]

    # Add cell-type markers
    if include_markers:
        for ct in list_celltype_markers():
            markers = get_celltype_markers(ct)
            genes.extend(markers.keys())
    genes = sorted(set(genes))

    records = []
    for gene in genes:
        for donor in donors:
            for region in regions:
                records.append({
                    "gene_symbol": gene,
                    "cstc_region": region,
                    "donor_id": donor,
                    "mean_zscore": float(rng.normal(0, 1)),
                    "n_samples": 5,
                    "n_probes": 2,
                })
    return pd.DataFrame(records)


class TestComputeTauSpecificity:
    def test_ubiquitous_gene(self):
        """Equal expression → tau = 0."""
        vec = np.array([1.0, 1.0, 1.0, 1.0])
        tau = compute_tau_specificity(vec)
        assert tau == pytest.approx(0.0, abs=1e-10)

    def test_specific_gene(self):
        """Expression in one cell type → tau = 1."""
        vec = np.array([10.0, 0.0, 0.0, 0.0])
        tau = compute_tau_specificity(vec)
        assert tau == pytest.approx(1.0, abs=1e-10)

    def test_partial_specificity(self):
        """Mixed expression → 0 < tau < 1."""
        vec = np.array([10.0, 5.0, 0.0, 0.0])
        tau = compute_tau_specificity(vec)
        assert 0 < tau < 1

    def test_nan_handling(self):
        """NaN values should be excluded."""
        vec = np.array([10.0, np.nan, 0.0, 0.0])
        tau = compute_tau_specificity(vec)
        assert 0 < tau <= 1

    def test_single_value(self):
        """Single value → NaN."""
        vec = np.array([5.0])
        tau = compute_tau_specificity(vec)
        assert np.isnan(tau)

    def test_all_nan(self):
        """All NaN → NaN."""
        vec = np.array([np.nan, np.nan])
        tau = compute_tau_specificity(vec)
        assert np.isnan(tau)

    def test_negative_values(self):
        """Tau handles negative values by shifting to non-negative."""
        vec = np.array([-1.0, 0.0, 5.0, 2.0])
        tau = compute_tau_specificity(vec)
        assert 0 <= tau <= 1


class TestBuildMarkerExpressionMatrix:
    def test_basic_output(self):
        expr_df = _make_expression_df()
        matrix = build_marker_expression_matrix(expr_df)
        assert not matrix.empty
        assert matrix.index.name == "cell_type"
        assert len(matrix) == len(STRIATAL_CELL_TYPES)

    def test_only_striatal_data_used(self):
        """Verify only caudate/putamen data contributes."""
        expr_df = _make_expression_df()
        # Remove striatal data
        non_striatal = expr_df[~expr_df["cstc_region"].isin(["caudate", "putamen"])]
        matrix = build_marker_expression_matrix(non_striatal)
        assert matrix.empty

    def test_cell_types_present(self):
        expr_df = _make_expression_df()
        matrix = build_marker_expression_matrix(expr_df)
        for ct in STRIATAL_CELL_TYPES:
            assert ct in matrix.index


class TestComputeGeneCelltypeSpecificity:
    def test_returns_dataframe(self):
        expr_df = _make_expression_df()
        genes = ["FLT3", "MEIS1", "PTPRD"]
        result = compute_gene_celltype_specificity(expr_df, genes)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_has_tau_column(self):
        expr_df = _make_expression_df()
        genes = ["FLT3", "MEIS1"]
        result = compute_gene_celltype_specificity(expr_df, genes)
        assert "tau" in result.columns

    def test_has_top_cell_type(self):
        expr_df = _make_expression_df()
        genes = ["FLT3"]
        result = compute_gene_celltype_specificity(expr_df, genes)
        assert "top_cell_type" in result.columns
        assert result["top_cell_type"].iloc[0] in STRIATAL_CELL_TYPES or result["top_cell_type"].iloc[0] == "none"

    def test_correlation_values_bounded(self):
        """Correlations should be in [-1, 1]."""
        expr_df = _make_expression_df()
        genes = ["FLT3", "MEIS1", "PTPRD"]
        result = compute_gene_celltype_specificity(expr_df, genes)
        ct_cols = [c for c in result.columns if c in STRIATAL_CELL_TYPES]
        for col in ct_cols:
            vals = result[col].dropna()
            assert (vals >= -1.0 - 1e-10).all()
            assert (vals <= 1.0 + 1e-10).all()

    def test_missing_gene_excluded(self):
        expr_df = _make_expression_df()
        genes = ["FLT3", "NONEXISTENT_GENE"]
        result = compute_gene_celltype_specificity(expr_df, genes)
        assert "NONEXISTENT_GENE" not in result.index

    def test_empty_input(self):
        expr_df = _make_expression_df()
        result = compute_gene_celltype_specificity(expr_df, [])
        assert result.empty


class TestCelltypeEnrichment:
    def test_returns_dataframe(self):
        expr_df = _make_expression_df()
        genes = ["FLT3", "MEIS1", "PTPRD", "SEMA6D", "HDC"]
        specificity = compute_gene_celltype_specificity(expr_df, genes)
        enrichment = compute_celltype_enrichment(specificity, n_permutations=100)
        assert isinstance(enrichment, pd.DataFrame)
        assert not enrichment.empty

    def test_expected_columns(self):
        expr_df = _make_expression_df()
        genes = ["FLT3", "MEIS1", "PTPRD", "SEMA6D", "HDC"]
        specificity = compute_gene_celltype_specificity(expr_df, genes)
        enrichment = compute_celltype_enrichment(specificity, n_permutations=100)
        expected_cols = {"mean_corr", "std_corr", "z_score", "p_value_bootstrap",
                         "n_genes", "significant"}
        assert expected_cols.issubset(set(enrichment.columns))

    def test_p_values_bounded(self):
        expr_df = _make_expression_df()
        genes = ["FLT3", "MEIS1", "PTPRD"]
        specificity = compute_gene_celltype_specificity(expr_df, genes)
        enrichment = compute_celltype_enrichment(specificity, n_permutations=100)
        assert (enrichment["p_value_bootstrap"] >= 0).all()
        assert (enrichment["p_value_bootstrap"] <= 1).all()

    def test_reproducible_with_seed(self):
        expr_df = _make_expression_df()
        genes = ["FLT3", "MEIS1", "PTPRD"]
        specificity = compute_gene_celltype_specificity(expr_df, genes)
        e1 = compute_celltype_enrichment(specificity, n_permutations=100, seed=99)
        e2 = compute_celltype_enrichment(specificity, n_permutations=100, seed=99)
        pd.testing.assert_frame_equal(e1, e2)


class TestAnnotateWithHmba:
    def test_returns_annotations(self):
        expr_df = _make_expression_df()
        genes = ["FLT3", "MEIS1"]
        specificity = compute_gene_celltype_specificity(expr_df, genes)
        annot = annotate_with_hmba(specificity)
        assert not annot.empty
        assert len(annot) == len(STRIATAL_CELL_TYPES)

    def test_has_hmba_columns(self):
        expr_df = _make_expression_df()
        genes = ["FLT3"]
        specificity = compute_gene_celltype_specificity(expr_df, genes)
        annot = annotate_with_hmba(specificity)
        assert "hmba_label" in annot.columns
        assert "cell_class" in annot.columns
        assert "region" in annot.columns

    def test_known_types_mapped(self):
        """Known HMBA cell types should not be 'unmapped'."""
        expr_df = _make_expression_df()
        genes = ["FLT3"]
        specificity = compute_gene_celltype_specificity(expr_df, genes)
        annot = annotate_with_hmba(specificity)
        # At least some should map (d1_msn maps to D1_MSN_matrix etc.)
        mapped = annot[annot["hmba_label"] != "unmapped"]
        assert len(mapped) > 0


class TestStriatalCellTypeConstants:
    def test_cell_types_match_markers(self):
        """All STRIATAL_CELL_TYPES should have corresponding marker panels."""
        for ct in STRIATAL_CELL_TYPES:
            markers = get_celltype_markers(ct)
            assert len(markers) >= 3, f"{ct} has fewer than 3 markers"

    def test_no_duplicate_cell_types(self):
        assert len(STRIATAL_CELL_TYPES) == len(set(STRIATAL_CELL_TYPES))
