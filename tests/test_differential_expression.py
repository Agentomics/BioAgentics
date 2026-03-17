"""Tests for bioagentics.models.differential_expression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

from bioagentics.models.differential_expression import (
    run_deseq2,
    run_sex_stratified_de,
    plot_volcano,
)


def _make_de_adata(n_genes: int = 100, n_per_group: int = 8, seed: int = 42) -> ad.AnnData:
    """Create synthetic AnnData with two conditions and sex labels.

    Injects a real signal: first 10 genes are upregulated in 'case' group.
    """
    rng = np.random.default_rng(seed)
    n_total = n_per_group * 2

    baseline = rng.poisson(50, (n_total, n_genes)).astype(float)
    # Inject signal: first 10 genes 3x higher in case samples
    baseline[:n_per_group, :10] = rng.poisson(150, (n_per_group, 10)).astype(float)

    obs = pd.DataFrame({
        "condition": ["case"] * n_per_group + ["control"] * n_per_group,
        "sex": (["M", "F"] * n_total)[:n_total],
    }, index=[f"S{i}" for i in range(n_total)])

    var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])
    adata = ad.AnnData(X=baseline.astype(np.float32), obs=obs, var=var)
    adata.layers["counts"] = baseline.astype(np.float32)
    return adata


class TestRunDeseq2:
    def test_basic_de(self):
        adata = _make_de_adata()
        result = run_deseq2(adata, condition_key="condition", ref_level="control")

        assert "gene" in result.columns
        assert "log2FoldChange" in result.columns
        assert "padj" in result.columns
        assert len(result) > 0

    def test_detects_injected_signal(self):
        adata = _make_de_adata(n_per_group=15)
        result = run_deseq2(adata, condition_key="condition", ref_level="control")

        # Top significant genes should include our injected signal genes
        sig = result[result["padj"] < 0.1]
        sig_genes = set(sig["gene"])
        injected = {f"GENE{i}" for i in range(10)}
        overlap = sig_genes & injected
        # At least some of the 10 injected genes should be detected
        assert len(overlap) >= 3, f"Only {len(overlap)} injected genes detected: {overlap}"

    def test_missing_condition_raises(self):
        adata = _make_de_adata()
        try:
            run_deseq2(adata, condition_key="nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestSexStratified:
    def test_returns_three_modes(self):
        adata = _make_de_adata(n_per_group=10)
        results = run_sex_stratified_de(adata, sex_key="sex")

        assert "combined" in results
        assert "male" in results
        assert "female" in results
        # Combined should always have results
        assert len(results["combined"]) > 0

    def test_missing_sex_key_raises(self):
        adata = _make_de_adata()
        try:
            run_sex_stratified_de(adata, sex_key="nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestVolcanoPlot:
    def test_plot_saves(self, tmp_path):
        adata = _make_de_adata()
        de_df = run_deseq2(adata, condition_key="condition", ref_level="control")

        plot_path = tmp_path / "volcano.png"
        plot_volcano(de_df, title="Test", save_path=plot_path)
        assert plot_path.exists()

    def test_empty_df_no_error(self, tmp_path):
        plot_path = tmp_path / "volcano_empty.png"
        plot_volcano(pd.DataFrame(), title="Empty", save_path=plot_path)
        assert not plot_path.exists()
