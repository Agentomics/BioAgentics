"""Tests for IVIG differential abundance analysis."""

import numpy as np
import pandas as pd
import anndata as ad

from bioagentics.pandas_pans.ivig_diff_abundance import (
    DiffAbundanceResult,
    DiffAbundanceSummary,
    compute_cell_type_proportions,
    compute_cell_type_counts,
    define_comparisons,
    run_diff_abundance,
    run_propeller,
    run_fisher,
    run_dirichlet_multinomial,
    _benjamini_hochberg,
    _safe_log2fc,
)


def _make_ivig_adata(n_per_group: int = 100, seed: int = 42) -> ad.AnnData:
    """Create synthetic IVIG dataset with known cell type shifts.

    Creates 3 conditions:
    - control (4 samples): balanced cell types
    - pans_pre (5 samples): elevated monocytes, reduced Tregs
    - pans_post (5 samples): partially normalized proportions
    """
    rng = np.random.default_rng(seed)
    n_genes = 50

    cell_types = ["Classical_Mono", "CD4_Memory", "CD8_Effector", "NK_CD56dim",
                  "B_Naive", "Treg", "pDC"]

    obs_data = []
    X_rows = []

    # Control samples: balanced proportions
    for s in range(4):
        sample_id = f"ctrl_{s}"
        probs = [0.15, 0.20, 0.15, 0.15, 0.15, 0.10, 0.10]
        n_cells = n_per_group
        cts = rng.choice(cell_types, size=n_cells, p=probs)
        for ct in cts:
            obs_data.append({"sample": sample_id, "condition": "control", "cell_type": ct})
            X_rows.append(rng.negative_binomial(2, 0.3, size=n_genes))

    # PANS pre-IVIG: elevated monocytes, depleted Tregs
    for s in range(5):
        sample_id = f"pans_pre_{s}"
        probs = [0.30, 0.18, 0.15, 0.12, 0.12, 0.03, 0.10]
        n_cells = n_per_group
        cts = rng.choice(cell_types, size=n_cells, p=probs)
        for ct in cts:
            obs_data.append({"sample": sample_id, "condition": "pans_pre", "cell_type": ct})
            X_rows.append(rng.negative_binomial(2, 0.3, size=n_genes))

    # PANS post-IVIG: partially normalized
    for s in range(5):
        sample_id = f"pans_post_{s}"
        probs = [0.20, 0.19, 0.15, 0.14, 0.14, 0.08, 0.10]
        n_cells = n_per_group
        cts = rng.choice(cell_types, size=n_cells, p=probs)
        for ct in cts:
            obs_data.append({"sample": sample_id, "condition": "pans_post", "cell_type": ct})
            X_rows.append(rng.negative_binomial(2, 0.3, size=n_genes))

    X = np.array(X_rows, dtype=np.float32)
    obs = pd.DataFrame(obs_data)
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = gene_names
    return adata


class TestProportions:
    def test_compute_proportions_shape(self):
        adata = _make_ivig_adata()
        props = compute_cell_type_proportions(adata, "cell_type", "sample")
        assert props.shape[0] == 14  # 4 ctrl + 5 pre + 5 post
        assert props.shape[1] == 7  # 7 cell types

    def test_proportions_sum_to_one(self):
        adata = _make_ivig_adata()
        props = compute_cell_type_proportions(adata, "cell_type", "sample")
        np.testing.assert_allclose(props.sum(axis=1), 1.0, atol=1e-10)

    def test_compute_counts_shape(self):
        adata = _make_ivig_adata()
        counts = compute_cell_type_counts(adata, "cell_type", "sample")
        assert counts.shape[0] == 14
        assert counts.shape[1] == 7

    def test_counts_match_total(self):
        adata = _make_ivig_adata()
        counts = compute_cell_type_counts(adata, "cell_type", "sample")
        # Each sample has 100 cells
        assert (counts.sum(axis=1) == 100).all()


class TestBHCorrection:
    def test_single_pvalue(self):
        adj = _benjamini_hochberg(np.array([0.05]))
        assert len(adj) == 1
        assert adj[0] == 0.05

    def test_monotonicity(self):
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        adj = _benjamini_hochberg(pvals)
        # Adjusted p-values should be >= raw p-values
        assert (adj >= pvals).all()

    def test_all_significant(self):
        pvals = np.array([0.001, 0.002, 0.003])
        adj = _benjamini_hochberg(pvals)
        assert (adj <= 1.0).all()
        assert (adj >= 0.0).all()

    def test_empty(self):
        adj = _benjamini_hochberg(np.array([]))
        assert len(adj) == 0


class TestSafeLog2FC:
    def test_equal_proportions(self):
        fc = _safe_log2fc(0.1, 0.1)
        assert abs(fc) < 0.01

    def test_doubled(self):
        fc = _safe_log2fc(0.1, 0.2)
        assert abs(fc - 1.0) < 0.01

    def test_zero_handling(self):
        fc = _safe_log2fc(0.0, 0.1)
        assert np.isfinite(fc)
        assert fc > 0

    def test_both_zero(self):
        fc = _safe_log2fc(0.0, 0.0)
        assert np.isfinite(fc)
        assert abs(fc) < 0.01


class TestDefineComparisons:
    def test_standard_conditions(self):
        adata = _make_ivig_adata()
        comps = define_comparisons(adata, "condition", "sample")
        assert len(comps) == 3
        names = [c[0] for c in comps]
        assert "pans_pre_vs_control" in names
        assert "pans_post_vs_control" in names
        assert "pans_pre_vs_post" in names

    def test_missing_post(self):
        adata = _make_ivig_adata()
        # Remove post-IVIG samples
        mask = adata.obs["condition"] != "pans_post"
        adata = adata[mask].copy()
        comps = define_comparisons(adata, "condition", "sample")
        names = [c[0] for c in comps]
        assert "pans_pre_vs_control" in names
        assert "pans_post_vs_control" not in names
        assert "pans_pre_vs_post" not in names


class TestPropeller:
    def test_returns_results(self):
        adata = _make_ivig_adata()
        props = compute_cell_type_proportions(adata, "cell_type", "sample")
        counts = compute_cell_type_counts(adata, "cell_type", "sample")
        g1 = [f"ctrl_{i}" for i in range(4)]
        g2 = [f"pans_pre_{i}" for i in range(5)]
        results = run_propeller(props, counts, g1, g2, "test")
        assert len(results) == 7  # 7 cell types
        assert all(r.method == "propeller" for r in results)

    def test_pvalues_valid(self):
        adata = _make_ivig_adata()
        props = compute_cell_type_proportions(adata, "cell_type", "sample")
        counts = compute_cell_type_counts(adata, "cell_type", "sample")
        g1 = [f"ctrl_{i}" for i in range(4)]
        g2 = [f"pans_pre_{i}" for i in range(5)]
        results = run_propeller(props, counts, g1, g2, "test")
        for r in results:
            assert 0 <= r.pvalue <= 1
            assert 0 <= r.pvalue_adj <= 1

    def test_detects_monocyte_shift(self):
        """Monocytes are inflated in pre-IVIG; should detect this."""
        adata = _make_ivig_adata(n_per_group=200)
        props = compute_cell_type_proportions(adata, "cell_type", "sample")
        counts = compute_cell_type_counts(adata, "cell_type", "sample")
        g1 = [f"ctrl_{i}" for i in range(4)]
        g2 = [f"pans_pre_{i}" for i in range(5)]
        results = run_propeller(props, counts, g1, g2, "test")
        mono = [r for r in results if r.cell_type == "Classical_Mono"][0]
        # Monocytes should be higher in group2 (pre-IVIG)
        assert mono.log2_fold_change > 0

    def test_fallback_to_fisher(self):
        """With only 1 sample per group, should fall back to Fisher."""
        adata = _make_ivig_adata()
        props = compute_cell_type_proportions(adata, "cell_type", "sample")
        counts = compute_cell_type_counts(adata, "cell_type", "sample")
        results = run_propeller(props, counts, ["ctrl_0"], ["pans_pre_0"], "test")
        assert all(r.method == "fisher" for r in results)


class TestFisher:
    def test_returns_results(self):
        adata = _make_ivig_adata()
        counts = compute_cell_type_counts(adata, "cell_type", "sample")
        g1 = [f"ctrl_{i}" for i in range(4)]
        g2 = [f"pans_pre_{i}" for i in range(5)]
        results = run_fisher(counts, g1, g2, "test")
        assert len(results) == 7
        assert all(r.method == "fisher" for r in results)


class TestDirichletMultinomial:
    def test_returns_results(self):
        adata = _make_ivig_adata()
        counts = compute_cell_type_counts(adata, "cell_type", "sample")
        g1 = [f"ctrl_{i}" for i in range(4)]
        g2 = [f"pans_pre_{i}" for i in range(5)]
        results = run_dirichlet_multinomial(counts, g1, g2, "test")
        assert len(results) == 7
        assert all(r.method == "dirichlet_multinomial" for r in results)


class TestDiffAbundanceResult:
    def test_to_dict(self):
        r = DiffAbundanceResult(
            cell_type="Treg", comparison="test",
            prop_group1=0.1, prop_group2=0.05,
            log2_fold_change=-1.0, pvalue=0.01,
        )
        d = r.to_dict()
        assert d["cell_type"] == "Treg"
        assert d["pvalue"] == 0.01


class TestDiffAbundanceSummary:
    def test_to_dataframe(self):
        results = [
            DiffAbundanceResult("A", "comp1", 0.1, 0.2, 1.0, 0.01, 0.02),
            DiffAbundanceResult("B", "comp1", 0.2, 0.1, -1.0, 0.5, 0.6),
        ]
        summary = DiffAbundanceSummary(results=results, comparisons=["comp1"],
                                        n_cell_types=2, n_significant=1)
        df = summary.to_dataframe()
        assert len(df) == 2
        assert "cell_type" in df.columns

    def test_significant_results(self):
        results = [
            DiffAbundanceResult("A", "comp1", 0.1, 0.2, 1.0, 0.01, 0.02),
            DiffAbundanceResult("B", "comp1", 0.2, 0.1, -1.0, 0.5, 0.6),
        ]
        summary = DiffAbundanceSummary(results=results, comparisons=["comp1"],
                                        n_cell_types=2, n_significant=1, alpha=0.05)
        sig = summary.significant_results()
        assert len(sig) == 1
        assert sig.iloc[0]["cell_type"] == "A"

    def test_summary_string(self):
        summary = DiffAbundanceSummary(
            results=[], comparisons=["comp1"],
            n_cell_types=5, n_significant=2, alpha=0.05
        )
        s = summary.summary()
        assert "Differential Abundance" in s


class TestRunDiffAbundance:
    def test_full_pipeline(self):
        adata = _make_ivig_adata(n_per_group=150)
        summary = run_diff_abundance(
            adata, condition_key="condition", sample_key="sample",
            cell_type_key="cell_type", method="propeller",
        )
        assert len(summary.comparisons) == 3
        assert summary.n_cell_types == 7
        assert len(summary.results) == 21  # 7 cell types * 3 comparisons

    def test_fisher_method(self):
        adata = _make_ivig_adata()
        summary = run_diff_abundance(
            adata, condition_key="condition", sample_key="sample",
            cell_type_key="cell_type", method="fisher",
        )
        assert len(summary.results) > 0

    def test_dirichlet_method(self):
        adata = _make_ivig_adata()
        summary = run_diff_abundance(
            adata, condition_key="condition", sample_key="sample",
            cell_type_key="cell_type", method="dirichlet",
        )
        assert len(summary.results) > 0

    def test_custom_comparisons(self):
        adata = _make_ivig_adata()
        comps = [("custom", ["ctrl_0", "ctrl_1"], ["pans_pre_0", "pans_pre_1"])]
        summary = run_diff_abundance(
            adata, condition_key="condition", sample_key="sample",
            cell_type_key="cell_type", comparisons=comps,
        )
        assert len(summary.comparisons) == 1
        assert summary.comparisons[0] == "custom"

    def test_filters_unassigned(self):
        adata = _make_ivig_adata()
        # Add some Unassigned cells
        adata.obs.loc[adata.obs.index[:10], "cell_type"] = "Unassigned"
        summary = run_diff_abundance(
            adata, condition_key="condition", sample_key="sample",
            cell_type_key="cell_type",
        )
        # Unassigned should not be in results
        cell_types_tested = set(r.cell_type for r in summary.results)
        assert "Unassigned" not in cell_types_tested
