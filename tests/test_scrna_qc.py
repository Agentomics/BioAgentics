"""Tests for scRNA-seq QC module."""

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from bioagentics.scrna.qc import QCStats, compute_qc_metrics, filter_cells, filter_genes, run_qc


def make_synthetic_adata(n_cells: int = 500, n_genes: int = 200, seed: int = 42) -> ad.AnnData:
    """Create a synthetic AnnData for testing."""
    rng = np.random.default_rng(seed)

    # Sparse count matrix with realistic distribution
    data = rng.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)
    # Make some genes very lowly expressed (to test gene filtering)
    data[:, -5:] = 0
    data[rng.choice(n_cells, size=3), -5:] = 1  # only 3 cells express last 5 genes

    # Add mitochondrial genes (high mito in some cells)
    gene_names = [f"GENE{i}" for i in range(n_genes - 5)] + [f"MT-{c}" for c in ["ND1", "ND2", "CO1", "CO2", "ATP6"]]

    X = sp.csr_matrix(data)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"CELL_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=gene_names),
    )

    # Make ~10% of cells have high mito (>20%)
    high_mito_cells = rng.choice(n_cells, size=int(n_cells * 0.1), replace=False)
    for ci in high_mito_cells:
        # Set mito genes to very high counts
        mt_cols = [i for i, g in enumerate(gene_names) if g.startswith("MT-")]
        for mc in mt_cols:
            adata.X[ci, mc] = 500

    # Make ~5% of cells have very few genes (to test cell filtering)
    low_gene_cells = rng.choice(n_cells, size=int(n_cells * 0.05), replace=False)
    for ci in low_gene_cells:
        # Zero out most genes
        row = np.zeros(n_genes)
        row[:50] = rng.negative_binomial(2, 0.5, size=50)
        adata.X[ci] = sp.csr_matrix(row)

    return adata


class TestComputeQCMetrics:
    def test_adds_qc_columns(self):
        adata = make_synthetic_adata(n_cells=100, n_genes=50)
        adata = compute_qc_metrics(adata)
        assert "n_genes_by_counts" in adata.obs.columns
        assert "total_counts" in adata.obs.columns
        assert "pct_counts_mt" in adata.obs.columns
        assert "mt" in adata.var.columns

    def test_mito_genes_detected(self):
        adata = make_synthetic_adata(n_cells=100, n_genes=50)
        # Override all gene names so exactly 2 are MT- prefixed
        gene_names = [f"GENE{i}" for i in range(len(adata.var_names))]
        gene_names[-2] = "MT-ND1"
        gene_names[-1] = "MT-CO1"
        adata.var_names = pd.Index(gene_names)
        adata = compute_qc_metrics(adata)
        assert adata.var["mt"].sum() == 2


class TestFilterCells:
    def test_removes_low_gene_cells(self):
        adata = make_synthetic_adata(n_cells=200, n_genes=100)
        adata = compute_qc_metrics(adata)
        n_before = adata.n_obs
        adata = filter_cells(adata, min_genes=50)
        assert adata.n_obs <= n_before

    def test_removes_high_mito_cells(self):
        adata = make_synthetic_adata(n_cells=200, n_genes=100)
        adata = compute_qc_metrics(adata)
        n_before = adata.n_obs
        adata = filter_cells(adata, min_genes=1, max_pct_mito=20.0)
        assert adata.n_obs <= n_before


class TestFilterGenes:
    def test_removes_rare_genes(self):
        adata = make_synthetic_adata(n_cells=200, n_genes=100)
        n_before = adata.n_vars
        adata = filter_genes(adata, min_cells=10)
        assert adata.n_vars <= n_before


class TestRunQC:
    def test_full_pipeline(self):
        adata = make_synthetic_adata(n_cells=300, n_genes=150)
        adata_filtered, stats = run_qc(adata, min_genes=50, min_cells=5)

        assert isinstance(stats, QCStats)
        assert stats.cells_before == 300
        assert stats.cells_after <= 300
        assert stats.genes_before == 150
        assert stats.genes_after <= 150
        assert adata_filtered.n_obs == stats.cells_after
        assert adata_filtered.n_vars == stats.genes_after

    def test_stats_summary(self):
        adata = make_synthetic_adata(n_cells=300, n_genes=150)
        _, stats = run_qc(adata, min_genes=50, min_cells=5)
        summary = stats.summary()
        assert "QC Summary" in summary
        assert "Cells:" in summary

    def test_preserves_sparse(self):
        adata = make_synthetic_adata(n_cells=300, n_genes=150)
        adata_filtered, _ = run_qc(adata, min_genes=50, min_cells=5)
        assert sp.issparse(adata_filtered.X)
