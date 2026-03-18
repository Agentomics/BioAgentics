"""Tests for scrna regulon analysis module."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from bioagentics.scrna.regulon import (
    KNOWN_IL23_TFS,
    TFActivity,
    compute_celltype_tf_activity,
    identify_novel_regulons,
    infer_tf_activity,
    rank_tfs_per_celltype,
    run_regulon_analysis,
)


def _make_test_adata(n_cells: int = 200, n_genes: int = 500) -> ad.AnnData:
    """Create a synthetic AnnData with cell types and gene expression."""
    rng = np.random.default_rng(42)

    # Create count matrix
    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)

    # Gene names — include some real TF targets
    real_genes = [
        "IL17A", "IL17F", "RORC", "CCR6", "IL23R", "STAT3", "JAK2",
        "IL22", "AHR", "BATF", "IRF4", "HIF1A", "RELA", "NFKB1",
        "CD14", "FCGR1A", "IL23A", "IL1B", "TNF", "CD68",
    ]
    fake_genes = [f"GENE_{i}" for i in range(n_genes - len(real_genes))]
    gene_names = real_genes + fake_genes

    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=gene_names[:n_genes]),
    )

    # Assign cell types
    cell_types = ["Th17"] * 60 + ["ILC3"] * 40 + ["Inflammatory_Mac"] * 50 + ["B_cell"] * 50
    adata.obs["cell_type"] = cell_types[:n_cells]

    # Log-normalize
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


def _make_mock_net() -> pd.DataFrame:
    """Create a small mock TF-target network for testing."""
    records = []
    # RORC targets (known IL-23 TF)
    for target in ["IL17A", "IL17F", "IL22", "CCR6", "IL23R", "AHR"]:
        records.append({"source": "RORC", "target": target, "weight": 1.0})
    # STAT3 targets (known)
    for target in ["JAK2", "IL17A", "STAT3", "BATF", "IRF4", "HIF1A"]:
        records.append({"source": "STAT3", "target": target, "weight": 1.0})
    # BATF targets (novel candidate)
    for target in ["IL17A", "IL17F", "RORC", "IL22", "CCR6", "IRF4"]:
        records.append({"source": "BATF", "target": target, "weight": 1.0})
    # NFKB1 targets (novel candidate)
    for target in ["TNF", "IL1B", "RELA", "NFKB1", "IL23A", "CD14"]:
        records.append({"source": "NFKB1", "target": target, "weight": 1.0})
    return pd.DataFrame(records)


class TestInferTFActivity:
    def test_adds_obsm_keys(self):
        adata = _make_test_adata()
        net = _make_mock_net()
        adata = infer_tf_activity(adata, net, method="ulm", min_targets=3)
        assert "score_ulm" in adata.obsm
        assert "padj_ulm" in adata.obsm

    def test_activity_shape(self):
        adata = _make_test_adata()
        net = _make_mock_net()
        adata = infer_tf_activity(adata, net, method="ulm", min_targets=3)
        act = adata.obsm["score_ulm"]
        assert act.shape[0] == adata.n_obs
        assert act.shape[1] > 0  # At least one TF


class TestComputeCelltypeTFActivity:
    def test_returns_activities(self):
        adata = _make_test_adata()
        net = _make_mock_net()
        adata = infer_tf_activity(adata, net, method="ulm", min_targets=3)
        activities = compute_celltype_tf_activity(adata, net, il23_types_only=True)
        assert len(activities) > 0
        assert all(isinstance(a, TFActivity) for a in activities)

    def test_il23_types_filter(self):
        adata = _make_test_adata()
        net = _make_mock_net()
        adata = infer_tf_activity(adata, net, method="ulm", min_targets=3)
        activities = compute_celltype_tf_activity(adata, net, il23_types_only=True)
        cell_types = {a.cell_type for a in activities}
        # B_cell should be excluded when il23_types_only=True
        assert "B_cell" not in cell_types

    def test_known_flag(self):
        adata = _make_test_adata()
        net = _make_mock_net()
        adata = infer_tf_activity(adata, net, method="ulm", min_targets=3)
        activities = compute_celltype_tf_activity(adata, net, il23_types_only=True)
        rorc_acts = [a for a in activities if a.tf_name == "RORC"]
        if rorc_acts:
            assert all(a.is_known_il23 for a in rorc_acts)


class TestRankTFsPerCelltype:
    def test_returns_rankings(self):
        activities = [
            TFActivity("RORC", "Th17", 2.0, 1.5, 80.0, 60, 22, True),
            TFActivity("STAT3", "Th17", 1.5, 1.0, 70.0, 60, 437, True),
            TFActivity("BATF", "Th17", 1.8, 1.2, 75.0, 60, 16, False),
        ]
        rankings = rank_tfs_per_celltype(activities)
        assert "Th17" in rankings
        assert len(rankings["Th17"]) == 3

    def test_ranking_order(self):
        activities = [
            TFActivity("RORC", "Th17", 2.0, 1.5, 80.0, 60, 22, True),
            TFActivity("BATF", "Th17", 3.0, 2.0, 90.0, 60, 16, False),
        ]
        rankings = rank_tfs_per_celltype(activities)
        assert rankings["Th17"].iloc[0]["tf"] == "BATF"  # Higher mean_activity


class TestIdentifyNovelRegulons:
    def test_excludes_known_tfs(self):
        rankings = {
            "Th17": pd.DataFrame({
                "tf": ["RORC", "BATF", "STAT3", "NFKB1"],
                "mean_activity": [2.0, 1.8, 1.5, 1.2],
                "is_known_il23": [True, False, True, False],
            })
        }
        novel = identify_novel_regulons(rankings)
        assert "BATF" in novel
        assert "NFKB1" in novel
        assert "RORC" not in novel
        assert "STAT3" not in novel


class TestRunRegulonAnalysis:
    def test_full_pipeline(self):
        adata = _make_test_adata()
        adata_result, result = run_regulon_analysis(
            adata, method="ulm", min_targets=3, il23_types_only=True,
        )
        assert result.n_cells_analyzed > 0
        assert result.n_tfs_tested > 0
        assert result.n_cell_types > 0

    def test_summary(self):
        adata = _make_test_adata()
        _, result = run_regulon_analysis(
            adata, method="ulm", min_targets=3, il23_types_only=True,
        )
        summary = result.summary()
        assert "Regulon Analysis" in summary
        assert "TFs tested" in summary

    def test_subsampling(self):
        adata = _make_test_adata(n_cells=200)
        _, result = run_regulon_analysis(
            adata, method="ulm", min_targets=3,
            max_cells_per_type=20, il23_types_only=True,
        )
        assert result.n_cells_analyzed <= 80  # 4 types * 20 max
