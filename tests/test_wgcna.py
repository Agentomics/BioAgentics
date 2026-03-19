"""Tests for bioagentics.models.wgcna (transcriptomic biomarker panel)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

from bioagentics.models.wgcna import (
    select_soft_threshold,
    compute_tom,
    identify_modules,
    compute_module_eigengenes,
    correlate_with_traits,
    find_hub_genes,
    run_wgcna,
    _filter_variable_genes,
)


def _make_wgcna_adata(
    n_samples: int = 60, n_genes: int = 100, n_modules: int = 3, seed: int = 42,
) -> ad.AnnData:
    """Create synthetic AnnData with correlated gene blocks (simulated modules)."""
    rng = np.random.default_rng(seed)
    genes_per_module = n_genes // (n_modules + 1)  # leave some noise genes

    X = rng.normal(0, 1, (n_samples, n_genes)).astype(np.float32)

    # Inject correlated blocks
    for m in range(n_modules):
        start = m * genes_per_module
        end = start + genes_per_module
        shared_signal = rng.normal(0, 1, (n_samples, 1))
        X[:, start:end] += shared_signal * 3

    # Make first block correlate with condition
    condition = np.array(["case"] * (n_samples // 2) + ["control"] * (n_samples // 2))
    case_mask = condition == "case"
    X[case_mask, :genes_per_module] += 2

    obs = pd.DataFrame({
        "condition": condition,
        "sex": (["M", "F"] * n_samples)[:n_samples],
    }, index=[f"S{i}" for i in range(n_samples)])

    var = pd.DataFrame(index=[f"GENE{i}" for i in range(n_genes)])
    return ad.AnnData(X=X, obs=obs, var=var)


class TestSelectSoftThreshold:
    def test_returns_valid_power(self):
        rng = np.random.default_rng(42)
        expr = rng.normal(size=(50, 80)).astype(np.float32)
        genes = [f"G{i}" for i in range(80)]
        power, fit = select_soft_threshold(expr, genes, powers=list(range(1, 11)))
        assert 1 <= power <= 10
        assert "power" in fit.columns
        assert "r2" in fit.columns
        assert len(fit) == 10

    def test_fit_table_has_mean_k(self):
        rng = np.random.default_rng(0)
        expr = rng.normal(size=(30, 50)).astype(np.float32)
        _, fit = select_soft_threshold(expr, [f"G{i}" for i in range(50)])
        assert "mean_k" in fit.columns


class TestComputeTOM:
    def test_tom_is_symmetric(self):
        rng = np.random.default_rng(42)
        n = 20
        data = rng.normal(size=(30, n))
        cor = np.corrcoef(data, rowvar=False)
        tom = compute_tom(cor, power=6)
        assert tom.shape == (n, n)
        assert np.allclose(tom, tom.T, atol=1e-10)

    def test_tom_values_in_range(self):
        rng = np.random.default_rng(42)
        cor = np.corrcoef(rng.normal(size=(30, 15)), rowvar=False)
        tom = compute_tom(cor, power=6)
        assert np.all(tom >= 0)
        assert np.all(tom <= 1)
        assert np.allclose(np.diag(tom), 1.0)


class TestIdentifyModules:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        n = 80
        data = rng.normal(size=(40, n))
        # Add correlated blocks
        for s in range(0, 40, 20):
            block = rng.normal(size=(40, 1))
            data[:, s:s + 20] += block * 2
        cor = np.corrcoef(data, rowvar=False)
        tom = compute_tom(cor, power=6)
        genes = [f"G{i}" for i in range(n)]

        modules = identify_modules(tom, genes, min_module_size=5)
        assert len(modules) == n
        assert "gene_symbol" in modules.columns
        assert "module" in modules.columns

    def test_fixed_cut_height(self):
        rng = np.random.default_rng(42)
        n = 50
        cor = np.corrcoef(rng.normal(size=(30, n)), rowvar=False)
        tom = compute_tom(cor, power=6)
        genes = [f"G{i}" for i in range(n)]

        modules = identify_modules(tom, genes, min_module_size=3, cut_height=0.8)
        assert len(modules) == n


class TestComputeModuleEigengenes:
    def test_returns_eigengenes_for_each_module(self):
        adata = _make_wgcna_adata(n_samples=40, n_genes=60, n_modules=2)
        X = np.array(adata.X)
        genes = list(adata.var_names)
        modules = pd.DataFrame({
            "gene_symbol": genes,
            "module": [1] * 20 + [2] * 20 + [0] * 20,
        })

        eigengenes = compute_module_eigengenes(X, genes, modules)
        assert "ME1" in eigengenes.columns
        assert "ME2" in eigengenes.columns
        assert len(eigengenes) == 40

    def test_skips_small_modules(self):
        genes = [f"G{i}" for i in range(20)]
        rng = np.random.default_rng(42)
        X = rng.normal(size=(30, 20))
        modules = pd.DataFrame({
            "gene_symbol": genes,
            "module": [1] * 15 + [2] * 2 + [0] * 3,  # module 2 too small
        })

        eigengenes = compute_module_eigengenes(X, genes, modules)
        assert "ME1" in eigengenes.columns
        assert "ME2" not in eigengenes.columns


class TestCorrelateWithTraits:
    def test_detects_correlated_trait(self):
        rng = np.random.default_rng(42)
        n = 50
        trait_vals = rng.normal(size=n)
        eigengenes = pd.DataFrame({
            "ME1": trait_vals + rng.normal(0, 0.1, n),  # strongly correlated
            "ME2": rng.normal(size=n),  # uncorrelated
        })
        traits = pd.DataFrame({"phenotype": trait_vals})

        corr = correlate_with_traits(eigengenes, traits)
        me1_row = corr[corr["module"] == "ME1"]
        assert abs(me1_row["correlation"].values[0]) > 0.8
        assert me1_row["pvalue"].values[0] < 0.001

    def test_handles_empty_traits(self):
        eigengenes = pd.DataFrame({"ME1": [1, 2, 3]})
        traits = pd.DataFrame()
        corr = correlate_with_traits(eigengenes, traits)
        assert corr.empty


class TestFindHubGenes:
    def test_returns_sorted_by_membership(self):
        rng = np.random.default_rng(42)
        n_samples, n_genes = 30, 20
        X = rng.normal(size=(n_samples, n_genes))
        genes = [f"G{i}" for i in range(n_genes)]
        modules = pd.DataFrame({
            "gene_symbol": genes,
            "module": [1] * 15 + [0] * 5,
        })

        # Compute eigengene
        eigengenes = compute_module_eigengenes(X, genes, modules)

        hubs = find_hub_genes(X, genes, modules, eigengenes, module_id=1, n_top=5)
        assert len(hubs) == 5
        mms = [h["module_membership"] for h in hubs]
        assert mms == sorted(mms, reverse=True)

    def test_missing_module_returns_empty(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 10))
        genes = [f"G{i}" for i in range(10)]
        modules = pd.DataFrame({"gene_symbol": genes, "module": [1] * 10})
        eigengenes = pd.DataFrame({"ME1": rng.normal(size=20)})

        hubs = find_hub_genes(X, genes, modules, eigengenes, module_id=99)
        assert hubs == []


class TestFilterVariableGenes:
    def test_keeps_most_variable(self):
        rng = np.random.default_rng(42)
        X = rng.normal(0, 1, (20, 50))
        # Make first 5 genes highly variable
        X[:, :5] *= 10
        genes = [f"G{i}" for i in range(50)]

        X_filt, genes_filt = _filter_variable_genes(X, genes, max_genes=10)
        assert X_filt.shape[1] == 10
        assert len(genes_filt) == 10
        # Top variable genes should include the first 5
        assert len(set(genes_filt) & {f"G{i}" for i in range(5)}) == 5

    def test_no_filter_when_under_limit(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(20, 30))
        genes = [f"G{i}" for i in range(30)]

        X_filt, genes_filt = _filter_variable_genes(X, genes, max_genes=50)
        assert X_filt.shape[1] == 30


class TestRunWGCNA:
    def test_full_pipeline(self, tmp_path):
        adata = _make_wgcna_adata(n_samples=40, n_genes=80, n_modules=2)
        result = run_wgcna(
            adata,
            condition_key="condition",
            min_module_size=5,
            dest_dir=tmp_path,
        )

        assert result.n_modules > 0
        assert not result.modules.empty
        assert (tmp_path / "wgcna_modules.csv").exists()
        assert (tmp_path / "wgcna_soft_threshold.csv").exists()

    def test_saves_eigengenes(self, tmp_path):
        adata = _make_wgcna_adata(n_samples=40, n_genes=60, n_modules=2)
        result = run_wgcna(
            adata,
            condition_key="condition",
            min_module_size=5,
            dest_dir=tmp_path,
        )

        if not result.eigengenes.empty:
            assert (tmp_path / "wgcna_eigengenes.csv").exists()

    def test_trait_correlations_computed(self, tmp_path):
        adata = _make_wgcna_adata(n_samples=60, n_genes=80, n_modules=2)
        result = run_wgcna(
            adata,
            condition_key="condition",
            min_module_size=5,
            dest_dir=tmp_path,
        )

        assert isinstance(result.trait_correlations, pd.DataFrame)
        assert result.power > 0
