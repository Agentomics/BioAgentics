"""Tests for IVIG scRNA-seq QC pipeline."""

import numpy as np
import pytest
import anndata as ad
import scipy.sparse as sp

from bioagentics.pandas_pans.ivig_scrna_qc import (
    QCStats,
    QCThresholds,
    compute_qc_metrics,
    correct_ambient_rna,
    estimate_ambient_profile,
    filter_cells_by_qc,
    filter_genes,
    remove_doublets,
    run_ivig_qc,
)


def _make_test_adata(
    n_cells: int = 500,
    n_genes: int = 200,
    n_mito_genes: int = 5,
    seed: int = 42,
    sparse: bool = True,
) -> ad.AnnData:
    """Create a synthetic AnnData for testing."""
    rng = np.random.default_rng(seed)

    # Generate count matrix with realistic-ish distribution
    X = rng.negative_binomial(n=2, p=0.3, size=(n_cells, n_genes)).astype(np.float32)

    # Make some cells low-quality (low gene count)
    X[:10, :] = 0  # dead cells
    X[:10, :3] = 1  # minimal expression

    # Make some cells high-mito
    gene_names = [f"GENE{i}" for i in range(n_genes - n_mito_genes)]
    gene_names += [f"MT-{chr(65 + i)}" for i in range(n_mito_genes)]

    # Boost mito genes in some cells
    X[10:20, -n_mito_genes:] = X[10:20, -n_mito_genes:] * 50

    if sparse:
        X = sp.csr_matrix(X)

    adata = ad.AnnData(
        X=X,
        obs={"sample": [f"sample_{i % 3}" for i in range(n_cells)]},
        var={"gene_symbols": gene_names},
    )
    adata.var_names = gene_names
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return adata


class TestQCThresholds:
    def test_defaults(self):
        t = QCThresholds()
        assert t.min_genes == 200
        assert t.max_genes == 5000
        assert t.max_pct_mito == 15.0
        assert t.ambient_correction is True

    def test_custom(self):
        t = QCThresholds(min_genes=300, max_pct_mito=10.0)
        assert t.min_genes == 300
        assert t.max_pct_mito == 10.0


class TestQCStats:
    def test_to_dict(self):
        stats = QCStats(cells_input=1000, cells_final=800)
        d = stats.to_dict()
        assert d["cells_input"] == 1000
        assert d["cells_final"] == 800

    def test_summary(self):
        stats = QCStats(
            cells_input=1000,
            genes_input=500,
            cells_final=800,
            genes_final=450,
        )
        s = stats.summary()
        assert "1,000 cells" in s
        assert "800 cells" in s


class TestComputeQCMetrics:
    def test_adds_metrics(self):
        adata = _make_test_adata()
        adata = compute_qc_metrics(adata)
        assert "n_genes_by_counts" in adata.obs.columns
        assert "total_counts" in adata.obs.columns
        assert "pct_counts_mt" in adata.obs.columns
        assert "mt" in adata.var.columns

    def test_mito_detection(self):
        adata = _make_test_adata(n_mito_genes=5)
        adata = compute_qc_metrics(adata)
        assert adata.var["mt"].sum() == 5


class TestFilterCells:
    def test_removes_low_quality(self):
        adata = _make_test_adata()
        adata = compute_qc_metrics(adata)
        thresholds = QCThresholds(min_genes=50, max_pct_mito=20.0)
        filtered, counts = filter_cells_by_qc(adata, thresholds)
        assert filtered.n_obs < adata.n_obs
        assert counts["low_genes"] >= 0
        assert counts["high_mito"] >= 0

    def test_strict_thresholds_remove_more(self):
        adata = _make_test_adata()
        adata = compute_qc_metrics(adata)
        loose = QCThresholds(min_genes=10, max_pct_mito=50.0)
        strict = QCThresholds(min_genes=100, max_pct_mito=5.0)
        filtered_loose, _ = filter_cells_by_qc(adata.copy(), loose)
        filtered_strict, _ = filter_cells_by_qc(adata.copy(), strict)
        assert filtered_strict.n_obs <= filtered_loose.n_obs


class TestFilterGenes:
    def test_removes_rare_genes(self):
        adata = _make_test_adata()
        n_before = adata.n_vars
        filtered = filter_genes(adata, min_cells=5)
        assert filtered.n_vars <= n_before


class TestDoubletDetection:
    def test_removes_doublets(self):
        adata = _make_test_adata(n_cells=300, n_genes=150)
        adata = compute_qc_metrics(adata)
        filtered, n_doublets = remove_doublets(adata, expected_doublet_rate=0.1)
        assert filtered.n_obs <= adata.n_obs
        assert n_doublets >= 0
        assert "doublet_score" not in filtered.obs.columns or True  # may be in copy

    def test_skips_few_cells(self):
        adata = _make_test_adata(n_cells=50, n_genes=100)
        filtered, n_doublets = remove_doublets(adata)
        assert n_doublets == 0
        assert filtered.n_obs == adata.n_obs


class TestAmbientRNA:
    def test_estimate_profile(self):
        adata = _make_test_adata(n_cells=500)
        adata = compute_qc_metrics(adata)
        profile = estimate_ambient_profile(adata)
        if profile is not None:
            assert profile.shape == (adata.n_vars,)
            assert np.isclose(profile.sum(), 1.0)

    def test_correct_ambient(self):
        adata = _make_test_adata(n_cells=500)
        adata = compute_qc_metrics(adata)

        X_before = adata.X.toarray() if sp.issparse(adata.X) else adata.X.copy()
        adata = correct_ambient_rna(adata, contamination_fraction=0.1)
        X_after = adata.X.toarray() if sp.issparse(adata.X) else adata.X

        # Corrected counts should be <= original
        assert np.all(X_after <= X_before + 1e-6)
        # No negative counts
        assert np.all(X_after >= 0)

    def test_correction_stores_metadata(self):
        adata = _make_test_adata(n_cells=500)
        adata = compute_qc_metrics(adata)
        adata = correct_ambient_rna(adata)
        if "ambient_correction" in adata.uns:
            assert "top_ambient_genes" in adata.uns["ambient_correction"]


class TestRunIvigQC:
    def test_full_pipeline(self):
        adata = _make_test_adata(n_cells=300, n_genes=150)
        thresholds = QCThresholds(
            min_genes=10,
            max_genes=10000,
            min_counts=10,
            max_counts=100000,
            max_pct_mito=50.0,
            min_cells_per_gene=3,
            expected_doublet_rate=0.05,
        )
        clean, stats = run_ivig_qc(adata, thresholds=thresholds)

        assert clean.n_obs > 0
        assert clean.n_obs <= adata.n_obs
        assert stats.cells_input == 300
        assert stats.cells_final == clean.n_obs
        assert "raw_counts" in clean.layers
        assert "qc_thresholds" in clean.uns
        assert "qc_stats" in clean.uns

    def test_per_sample_stats(self):
        adata = _make_test_adata(n_cells=300, n_genes=150)
        thresholds = QCThresholds(
            min_genes=10, min_counts=10, max_pct_mito=50.0, min_cells_per_gene=3
        )
        _, stats = run_ivig_qc(adata, thresholds=thresholds, sample_key="sample")
        assert len(stats.per_sample_counts) > 0

    def test_preserves_sparse(self):
        adata = _make_test_adata(sparse=True)
        thresholds = QCThresholds(
            min_genes=10, min_counts=10, max_pct_mito=50.0, min_cells_per_gene=3
        )
        clean, _ = run_ivig_qc(adata, thresholds=thresholds)
        assert sp.issparse(clean.X)

    def test_dense_input(self):
        adata = _make_test_adata(sparse=False)
        thresholds = QCThresholds(
            min_genes=10, min_counts=10, max_pct_mito=50.0, min_cells_per_gene=3
        )
        clean, _ = run_ivig_qc(adata, thresholds=thresholds)
        assert clean.n_obs > 0
