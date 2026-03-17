"""Tests for IL-23/Th17 pathway scoring module."""

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from bioagentics.scrna.pathway_scoring import (
    IL23_PATHWAY_MODULES,
    PathwayActivity,
    activity_to_dataframe,
    compute_celltype_activity,
    rank_cell_types_by_pathway,
    score_il23_pathway,
    score_pathway_modules,
)


def make_pathway_adata(n_cells: int = 200, seed: int = 42) -> ad.AnnData:
    """Create synthetic AnnData with IL-23 pathway genes."""
    rng = np.random.default_rng(seed)

    # Include key pathway genes
    pathway_genes = ["IL23R", "IL12RB1", "RORC", "IL17A", "IL17F", "STAT3", "JAK2", "IL23A", "FCGR1A"]
    filler = [f"GENE{i}" for i in range(491)]
    gene_names = pathway_genes + filler
    n_genes = len(gene_names)

    data = rng.negative_binomial(2, 0.5, size=(n_cells, n_genes)).astype(np.float32)

    # Make first 60 cells express IL23R/RORC/IL17A (Th17-like)
    for i, g in enumerate(gene_names):
        if g in ["IL23R", "RORC", "IL17A", "IL17F"]:
            data[:60, i] = rng.negative_binomial(15, 0.3, size=60)

    # Make cells 60-120 express FCGR1A/IL23A (Mac-like)
    for i, g in enumerate(gene_names):
        if g in ["FCGR1A", "IL23A"]:
            data[60:120, i] = rng.negative_binomial(15, 0.3, size=60)

    X = sp.csr_matrix(data)
    cell_types = ["Th17"] * 60 + ["Macrophage"] * 60 + ["Other"] * (n_cells - 120)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"cell_type": cell_types}, index=[f"CELL_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=gene_names),
    )

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


class TestScorePathwayModules:
    def test_adds_score_columns(self):
        adata = make_pathway_adata()
        modules = {"test_module": ["IL23R", "RORC"]}
        adata = score_pathway_modules(adata, modules=modules)
        assert "il23_test_module" in adata.obs.columns

    def test_missing_genes_zero_score(self):
        adata = make_pathway_adata()
        modules = {"empty": ["NONEXISTENT"]}
        adata = score_pathway_modules(adata, modules=modules)
        assert (adata.obs["il23_empty"] == 0.0).all()


class TestComputeCelltypeActivity:
    def test_returns_activity_records(self):
        adata = make_pathway_adata()
        adata = score_pathway_modules(adata, modules={"test": ["IL23R", "RORC"]})
        activities = compute_celltype_activity(adata)
        assert len(activities) > 0
        assert all(isinstance(a, PathwayActivity) for a in activities)

    def test_activity_per_celltype(self):
        adata = make_pathway_adata()
        adata = score_pathway_modules(adata, modules={"test": ["IL23R", "RORC"]})
        activities = compute_celltype_activity(adata)
        cell_types_in_results = {a.cell_type for a in activities}
        assert "Th17" in cell_types_in_results
        assert "Macrophage" in cell_types_in_results


class TestActivityToDataframe:
    def test_converts_to_df(self):
        activities = [
            PathwayActivity("mod1", "Th17", 0.5, 0.4, 80.0, 60),
            PathwayActivity("mod1", "Mac", 0.2, 0.1, 40.0, 60),
        ]
        df = activity_to_dataframe(activities)
        assert len(df) == 2
        assert "module" in df.columns
        assert "cell_type" in df.columns

    def test_empty_list(self):
        df = activity_to_dataframe([])
        assert df.empty


class TestRankCellTypes:
    def test_ranking_order(self):
        activities = [
            PathwayActivity("il23_th17_combined", "Th17", 0.8, 0.7, 90.0, 60),
            PathwayActivity("il23_th17_combined", "Mac", 0.3, 0.2, 50.0, 60),
            PathwayActivity("il23_th17_combined", "Other", 0.1, 0.05, 20.0, 80),
        ]
        ranking = rank_cell_types_by_pathway(activities)
        assert ranking.iloc[0]["cell_type"] == "Th17"
        assert ranking.iloc[-1]["cell_type"] == "Other"


class TestScoreIL23Pathway:
    def test_full_pipeline(self):
        adata = make_pathway_adata()
        result, activities = score_il23_pathway(adata)
        assert len(activities) > 0

        # Check that IL-23 module scores are in obs
        score_cols = [c for c in result.obs.columns if c.startswith("il23_")]
        assert len(score_cols) > 0
