"""Tests for bioagentics.data.geo_rnaseq_loader."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.data.geo_rnaseq_loader import (
    _build_anndata,
    _standardize_gene_index,
    normalize_deseq2,
)


def _make_counts(n_genes: int = 100, n_samples: int = 10, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic count matrix (genes x samples)."""
    rng = np.random.default_rng(seed)
    genes = [f"GENE{i}" for i in range(n_genes)]
    samples = [f"SAMPLE{i}" for i in range(n_samples)]
    data = rng.poisson(lam=50, size=(n_genes, n_samples)).astype(float)
    return pd.DataFrame(data, index=genes, columns=samples)


def _make_metadata(n_samples: int = 10) -> pd.DataFrame:
    """Create synthetic sample metadata."""
    samples = [f"SAMPLE{i}" for i in range(n_samples)]
    conditions = ["case" if i % 2 == 0 else "control" for i in range(n_samples)]
    sex = ["M" if i % 3 == 0 else "F" for i in range(n_samples)]
    return pd.DataFrame(
        {"condition": conditions, "sex": sex, "dataset": "GSE_TEST"},
        index=samples,
    )


class TestStandardizeGeneIndex:
    def test_uppercases_symbols(self):
        df = pd.DataFrame({"s1": [1, 2]}, index=["gene1", "Gene2"])
        result = _standardize_gene_index(df)
        assert list(result.index) == ["GENE1", "GENE2"]

    def test_deduplicates_by_max_mean(self):
        df = pd.DataFrame(
            {"s1": [10, 1, 5], "s2": [20, 2, 10]},
            index=["GENE1", "gene1", "GENE2"],
        )
        result = _standardize_gene_index(df)
        assert "GENE1" in result.index
        assert result.loc["GENE1", "s1"] == 10  # kept the higher-mean row

    def test_removes_empty_and_nan(self):
        df = pd.DataFrame({"s1": [1, 2, 3]}, index=["GENE1", "", "nan"])
        result = _standardize_gene_index(df)
        assert len(result) == 1
        assert result.index[0] == "GENE1"


class TestBuildAnndata:
    def test_basic_construction(self):
        counts = _make_counts(n_genes=50, n_samples=5)
        meta = _make_metadata(n_samples=5)
        adata = _build_anndata(counts, meta)
        assert adata.n_obs == 5
        assert adata.n_vars == 50
        assert "condition" in adata.obs.columns

    def test_aligns_samples(self):
        counts = _make_counts(n_genes=10, n_samples=5)
        meta = _make_metadata(n_samples=3)  # fewer samples
        adata = _build_anndata(counts, meta)
        assert adata.n_obs == 3

    def test_missing_metadata(self):
        counts = _make_counts(n_genes=10, n_samples=5)
        meta = pd.DataFrame(index=["NOMATCH1", "NOMATCH2"])
        # Should still work — creates empty metadata
        adata = _build_anndata(counts, meta)
        assert adata.n_obs == 5


class TestNormalizeDeseq2:
    def test_normalization_runs(self):
        counts = _make_counts(n_genes=100, n_samples=10)
        meta = _make_metadata(n_samples=10)
        adata = _build_anndata(counts, meta)
        result = normalize_deseq2(adata)

        assert "counts" in result.layers
        assert "size_factors" in result.obs.columns
        assert result.X.shape == result.layers["counts"].shape
        # Normalized values should differ from raw
        assert not np.allclose(result.X, result.layers["counts"])

    def test_size_factors_positive(self):
        counts = _make_counts(n_genes=100, n_samples=10)
        meta = _make_metadata(n_samples=10)
        adata = _build_anndata(counts, meta)
        result = normalize_deseq2(adata)

        assert (result.obs["size_factors"] > 0).all()

    def test_drops_zero_count_genes(self):
        counts = _make_counts(n_genes=50, n_samples=5)
        # Set one gene to all zeros
        counts.iloc[0, :] = 0
        meta = _make_metadata(n_samples=5)
        adata = _build_anndata(counts, meta)
        result = normalize_deseq2(adata)
        assert result.n_vars == 49
