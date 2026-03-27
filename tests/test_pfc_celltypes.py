"""Tests for PFC cell-type expression profiling module."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.analysis.tourettes.pfc_celltypes import (
    BROAD_CATEGORIES,
    IEG_GENES,
    PFC_CELL_TYPES,
    compute_celltype_mean_expression,
    compute_ieg_overlap,
    compute_specificity_scores,
    compute_tau_specificity,
    compute_td_vs_ctl_enrichment,
)


def _make_expr_df(n_genes: int = 10, n_cells: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create synthetic genes x cells expression DataFrame."""
    rng = np.random.default_rng(seed)
    genes = [f"GENE_{i}" for i in range(n_genes)]
    cells = [f"CELL_{i}" for i in range(n_cells)]
    # Sparse counts (mostly zeros with some non-zero values)
    data = rng.poisson(lam=0.5, size=(n_genes, n_cells)).astype(np.float32)
    return pd.DataFrame(data, index=genes, columns=cells)


def _make_metadata(n_cells: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create synthetic cell metadata with PFC cell-type labels."""
    rng = np.random.default_rng(seed)
    cells = [f"CELL_{i}" for i in range(n_cells)]
    # Assign cell types proportionally
    ct_pool = PFC_CELL_TYPES * (n_cells // len(PFC_CELL_TYPES) + 1)
    ct_labels = ct_pool[:n_cells]
    rng.shuffle(ct_labels)

    groups = rng.choice(["TD", "CTL"], size=n_cells, p=[0.6, 0.4])
    ieg_scores = rng.normal(0, 1, size=n_cells)

    return pd.DataFrame(
        {
            "predicted.subclass_label": ct_labels,
            "group": groups,
            "IEG_Module": ieg_scores,
        },
        index=cells,
    )


class TestComputeTauSpecificity:
    def test_ubiquitous(self):
        vec = np.array([1.0, 1.0, 1.0, 1.0])
        assert compute_tau_specificity(vec) == pytest.approx(0.0, abs=1e-10)

    def test_specific(self):
        vec = np.array([10.0, 0.0, 0.0, 0.0])
        assert compute_tau_specificity(vec) == pytest.approx(1.0, abs=1e-10)

    def test_partial(self):
        vec = np.array([10.0, 5.0, 0.0, 0.0])
        tau = compute_tau_specificity(vec)
        assert 0 < tau < 1

    def test_nan_excluded(self):
        vec = np.array([10.0, np.nan, 0.0, 0.0])
        tau = compute_tau_specificity(vec)
        assert 0 < tau <= 1

    def test_single_value_nan(self):
        assert np.isnan(compute_tau_specificity(np.array([5.0])))

    def test_all_zeros(self):
        assert compute_tau_specificity(np.array([0.0, 0.0, 0.0])) == 0.0


class TestComputecelltypeMeanExpression:
    def test_returns_dataframe(self):
        expr = _make_expr_df()
        meta = _make_metadata()
        result = compute_celltype_mean_expression(expr, meta)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty

    def test_genes_are_rows(self):
        expr = _make_expr_df(n_genes=5)
        meta = _make_metadata()
        result = compute_celltype_mean_expression(expr, meta)
        assert len(result) == 5

    def test_cell_types_are_columns(self):
        expr = _make_expr_df()
        meta = _make_metadata()
        result = compute_celltype_mean_expression(expr, meta)
        for ct in PFC_CELL_TYPES:
            assert ct in result.columns

    def test_non_negative_means(self):
        expr = _make_expr_df()
        meta = _make_metadata()
        result = compute_celltype_mean_expression(expr, meta)
        ct_cols = [c for c in PFC_CELL_TYPES if c in result.columns]
        assert (result[ct_cols] >= 0).all().all()

    def test_no_common_cells_empty(self):
        expr = _make_expr_df(n_cells=10)
        # Metadata with different cell IDs
        meta = pd.DataFrame(
            {"predicted.subclass_label": ["IT"] * 5, "group": ["TD"] * 5},
            index=[f"OTHER_{i}" for i in range(5)],
        )
        result = compute_celltype_mean_expression(expr, meta)
        assert result.empty


class TestComputeSpecificityScores:
    def test_adds_tau_column(self):
        expr = _make_expr_df()
        meta = _make_metadata()
        mean_expr = compute_celltype_mean_expression(expr, meta)
        result = compute_specificity_scores(mean_expr)
        assert "tau" in result.columns

    def test_adds_top_cell_type(self):
        expr = _make_expr_df()
        meta = _make_metadata()
        mean_expr = compute_celltype_mean_expression(expr, meta)
        result = compute_specificity_scores(mean_expr)
        assert "top_cell_type" in result.columns
        for ct in result["top_cell_type"]:
            assert ct in PFC_CELL_TYPES or ct == "none"

    def test_adds_broad_category(self):
        expr = _make_expr_df()
        meta = _make_metadata()
        mean_expr = compute_celltype_mean_expression(expr, meta)
        result = compute_specificity_scores(mean_expr)
        assert "broad_category" in result.columns
        valid_cats = set(BROAD_CATEGORIES.values()) | {"unknown"}
        for cat in result["broad_category"]:
            assert cat in valid_cats

    def test_tau_bounded(self):
        expr = _make_expr_df()
        meta = _make_metadata()
        mean_expr = compute_celltype_mean_expression(expr, meta)
        result = compute_specificity_scores(mean_expr)
        valid_taus = result["tau"].dropna()
        assert (valid_taus >= 0).all()
        assert (valid_taus <= 1).all()


class TestTdVsCtlEnrichment:
    def test_returns_dataframe(self):
        expr = _make_expr_df(n_genes=5, n_cells=500)
        meta = _make_metadata(n_cells=500)
        genes = expr.index.tolist()
        result = compute_td_vs_ctl_enrichment(expr, meta, genes)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self):
        expr = _make_expr_df(n_genes=3, n_cells=500)
        meta = _make_metadata(n_cells=500)
        genes = expr.index.tolist()
        result = compute_td_vs_ctl_enrichment(expr, meta, genes)
        if not result.empty:
            expected = {"gene_symbol", "cell_type", "mean_td", "mean_ctl",
                        "log2fc", "p_value", "significant"}
            assert expected.issubset(set(result.columns))

    def test_p_values_bounded(self):
        expr = _make_expr_df(n_genes=3, n_cells=500)
        meta = _make_metadata(n_cells=500)
        genes = expr.index.tolist()
        result = compute_td_vs_ctl_enrichment(expr, meta, genes)
        if not result.empty:
            assert (result["p_value"] >= 0).all()
            assert (result["p_value"] <= 1).all()


class TestIegOverlap:
    def test_with_ieg_module(self):
        expr = _make_expr_df(n_genes=5, n_cells=500)
        meta = _make_metadata(n_cells=500)
        spec = compute_specificity_scores(
            compute_celltype_mean_expression(expr, meta)
        )
        result = compute_ieg_overlap(spec, meta, expr)
        assert result["available"] is True
        assert "per_cell_type" in result

    def test_without_ieg_module(self):
        expr = _make_expr_df(n_genes=5, n_cells=100)
        meta = _make_metadata(n_cells=100)
        meta = meta.drop(columns=["IEG_Module"])
        spec = compute_specificity_scores(
            compute_celltype_mean_expression(expr, meta)
        )
        result = compute_ieg_overlap(spec, meta, expr)
        assert result["available"] is False

    def test_overlap_detection(self):
        # Create expression with an IEG gene as a TS risk gene
        expr = _make_expr_df(n_genes=3, n_cells=100)
        expr.index = pd.Index(["FOS", "GENE_1", "GENE_2"])
        meta = _make_metadata(n_cells=100)
        spec = compute_specificity_scores(
            compute_celltype_mean_expression(expr, meta)
        )
        result = compute_ieg_overlap(spec, meta, expr)
        assert "FOS" in result.get("ts_ieg_overlap_genes", [])


class TestPFCCellTypeConstants:
    def test_all_cell_types_have_broad_category(self):
        for ct in PFC_CELL_TYPES:
            assert ct in BROAD_CATEGORIES

    def test_no_duplicate_cell_types(self):
        assert len(PFC_CELL_TYPES) == len(set(PFC_CELL_TYPES))

    def test_ieg_genes_not_empty(self):
        assert len(IEG_GENES) > 0

    def test_expected_broad_categories(self):
        cats = set(BROAD_CATEGORIES.values())
        assert "Excitatory" in cats
        assert "Inhibitory" in cats
        assert "Non-neuronal" in cats
