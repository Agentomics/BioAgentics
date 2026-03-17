"""Tests for scRNA-seq integration module."""

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from bioagentics.scrna.integration import (
    IntegrationStats,
    compute_lisi,
    integrate_datasets,
    normalize_and_hvg,
    run_harmony,
    run_pca,
)


def make_two_batch_adata(n_cells: int = 300, n_genes: int = 200, seed: int = 42) -> ad.AnnData:
    """Create synthetic AnnData with two batches."""
    rng = np.random.default_rng(seed)

    # Two batches with slightly different means (simulating batch effect)
    n_half = n_cells // 2
    data1 = rng.negative_binomial(5, 0.3, size=(n_half, n_genes)).astype(np.float32)
    data2 = rng.negative_binomial(6, 0.3, size=(n_cells - n_half, n_genes)).astype(np.float32)

    data = np.vstack([data1, data2])
    X = sp.csr_matrix(data)

    gene_names = [f"GENE{i}" for i in range(n_genes)]
    cell_names = [f"CELL_{i}" for i in range(n_cells)]
    batch_labels = ["batch_A"] * n_half + ["batch_B"] * (n_cells - n_half)

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame({"dataset": batch_labels}, index=cell_names),
        var=pd.DataFrame(index=gene_names),
    )
    return adata


class TestComputeLISI:
    def test_perfect_mixing(self):
        """Perfectly interleaved labels should give LISI close to n_labels."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        labels = np.array(["A", "B"] * 50)
        lisi = compute_lisi(X, labels)
        # Perfect mixing should give LISI close to 2
        assert np.median(lisi) > 1.3

    def test_no_mixing(self):
        """Completely separated clusters with enough cells should give low LISI."""
        rng = np.random.default_rng(42)
        # Need enough cells so k << N for meaningful LISI
        X = np.vstack([
            rng.standard_normal((200, 10)),
            rng.standard_normal((200, 10)) + 1000,
        ])
        labels = np.array(["A"] * 200 + ["B"] * 200)
        lisi = compute_lisi(X, labels, perplexity=10.0)
        assert np.median(lisi) < 1.5


class TestNormalizeAndHVG:
    def test_stores_raw_counts(self):
        adata = make_two_batch_adata(n_cells=100, n_genes=50)
        adata = normalize_and_hvg(adata, n_top_genes=20)
        assert "counts" in adata.layers

    def test_selects_hvgs(self):
        adata = make_two_batch_adata(n_cells=100, n_genes=50)
        adata = normalize_and_hvg(adata, n_top_genes=20)
        assert "highly_variable" in adata.var.columns
        assert adata.var["highly_variable"].sum() > 0


class TestRunPCA:
    def test_pca_embedding(self):
        adata = make_two_batch_adata(n_cells=100, n_genes=50)
        adata = normalize_and_hvg(adata, n_top_genes=30)
        adata = run_pca(adata, n_comps=10)
        assert "X_pca" in adata.obsm
        assert adata.obsm["X_pca"].shape == (100, 10)


class TestRunHarmony:
    def test_harmony_correction(self):
        adata = make_two_batch_adata(n_cells=100, n_genes=50)
        adata = normalize_and_hvg(adata, n_top_genes=30)
        adata = run_pca(adata, n_comps=10)
        adata = run_harmony(adata, batch_key="dataset")
        assert "X_pca_harmony" in adata.obsm
        assert adata.obsm["X_pca_harmony"].shape == adata.obsm["X_pca"].shape


class TestIntegrateDatasets:
    def test_full_pipeline(self):
        adata = make_two_batch_adata(n_cells=200, n_genes=100)
        result, stats = integrate_datasets(adata, n_top_genes=50, n_pcs=10)

        assert isinstance(stats, IntegrationStats)
        assert stats.n_datasets == 2
        assert stats.n_cells_total == 200
        assert "X_pca_harmony" in result.obsm
        assert "X_umap" in result.obsm
        assert "leiden" in result.obs.columns
        assert "batch_lisi" in result.obs.columns

    def test_from_list(self):
        """Test integration from a list of AnnData objects."""
        rng = np.random.default_rng(42)
        n_genes = 80

        adata1 = ad.AnnData(
            X=sp.csr_matrix(rng.negative_binomial(5, 0.3, size=(100, n_genes)).astype(np.float32)),
            obs=pd.DataFrame({"dataset": "ds1"}, index=[f"A_{i}" for i in range(100)]),
            var=pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)]),
        )
        adata2 = ad.AnnData(
            X=sp.csr_matrix(rng.negative_binomial(6, 0.3, size=(100, n_genes)).astype(np.float32)),
            obs=pd.DataFrame({"dataset": "ds2"}, index=[f"B_{i}" for i in range(100)]),
            var=pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)]),
        )

        result, stats = integrate_datasets([adata1, adata2], n_top_genes=40, n_pcs=10)
        assert stats.n_datasets == 2
        assert stats.n_cells_total == 200

    def test_stats_summary(self):
        adata = make_two_batch_adata(n_cells=100, n_genes=60)
        _, stats = integrate_datasets(adata, n_top_genes=30, n_pcs=10)
        summary = stats.summary()
        assert "Integration Summary" in summary
        assert "batch_A" in summary
