"""Tests for IVIG pseudobulk differential expression analysis."""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from bioagentics.pandas_pans.ivig_pseudobulk_de import (
    DEResult,
    PseudobulkDESummary,
    aggregate_pseudobulk,
    define_de_comparisons,
    run_de_wald,
    run_pseudobulk_de,
    _normalize_counts,
    _benjamini_hochberg,
)


def _make_ivig_scrna(n_per_sample: int = 50, n_genes: int = 100, seed: int = 42) -> ad.AnnData:
    """Create synthetic scRNA-seq IVIG dataset with known DE signals.

    Injects upregulation of genes 0-4 in pans_pre monocytes.
    """
    rng = np.random.default_rng(seed)

    cell_types = ["Classical_Mono", "CD4_Memory", "NK_CD56dim", "B_Naive"]
    samples_meta = [
        ("ctrl_0", "control"), ("ctrl_1", "control"), ("ctrl_2", "control"),
        ("pans_pre_0", "pans_pre"), ("pans_pre_1", "pans_pre"), ("pans_pre_2", "pans_pre"),
        ("pans_post_0", "pans_post"), ("pans_post_1", "pans_post"), ("pans_post_2", "pans_post"),
    ]

    obs_data = []
    X_rows = []
    gene_names = [f"GENE{i}" for i in range(n_genes)]

    for sample_id, condition in samples_meta:
        for ct in cell_types:
            for _ in range(n_per_sample):
                counts = rng.negative_binomial(2, 0.3, size=n_genes).astype(np.float32)

                # Inject DE signal: upregulate first 5 genes in pans_pre monocytes
                if condition == "pans_pre" and ct == "Classical_Mono":
                    counts[:5] *= 5

                obs_data.append({
                    "sample": sample_id,
                    "condition": condition,
                    "cell_type": ct,
                })
                X_rows.append(counts)

    X = np.array(X_rows)
    obs = pd.DataFrame(obs_data)
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = gene_names
    return adata


class TestAggregatePseudobulk:
    def test_aggregates_per_cell_type(self):
        adata = _make_ivig_scrna()
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        assert len(pb) == 4  # 4 cell types
        for ct in ["Classical_Mono", "CD4_Memory", "NK_CD56dim", "B_Naive"]:
            assert ct in pb

    def test_sample_count(self):
        adata = _make_ivig_scrna()
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        # Each cell type should have 9 samples
        for ct, pb_adata in pb.items():
            assert pb_adata.n_obs == 9, f"{ct}: expected 9 samples, got {pb_adata.n_obs}"

    def test_gene_count_preserved(self):
        adata = _make_ivig_scrna()
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        for pb_adata in pb.values():
            assert pb_adata.n_vars == 100

    def test_counts_are_summed(self):
        adata = _make_ivig_scrna(n_per_sample=10)
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        mono = pb["Classical_Mono"]
        # Summed counts should be higher than individual cell counts
        assert mono.X.sum() > 0
        # Each pseudobulk value should be sum of ~10 cells
        assert mono.X.mean() > 10

    def test_min_cells_filter(self):
        adata = _make_ivig_scrna(n_per_sample=5)
        # With min_cells=10, and only 5 cells per sample, should get empty
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition", min_cells=10)
        assert len(pb) == 0

    def test_filters_unassigned(self):
        adata = _make_ivig_scrna()
        adata.obs.loc[adata.obs.index[:20], "cell_type"] = "Unassigned"
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        assert "Unassigned" not in pb

    def test_sparse_input(self):
        adata = _make_ivig_scrna(n_per_sample=20)
        adata.X = sp.csr_matrix(adata.X)
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        assert len(pb) == 4

    def test_condition_preserved(self):
        adata = _make_ivig_scrna()
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        mono = pb["Classical_Mono"]
        conditions = set(mono.obs["condition"].values)
        assert "control" in conditions
        assert "pans_pre" in conditions
        assert "pans_post" in conditions


class TestNormalizeCounts:
    def test_shape_preserved(self):
        X = np.array([[10, 20, 30], [5, 10, 15]], dtype=float)
        norm, sf = _normalize_counts(X)
        assert norm.shape == X.shape
        assert len(sf) == 2

    def test_size_factors_positive(self):
        X = np.array([[10, 20, 30], [5, 10, 15]], dtype=float)
        _, sf = _normalize_counts(X)
        assert (sf > 0).all()

    def test_zero_handling(self):
        X = np.array([[0, 0, 0], [5, 10, 15]], dtype=float)
        norm, sf = _normalize_counts(X)
        assert np.all(np.isfinite(norm))


class TestBHCorrection:
    def test_basic(self):
        pvals = np.array([0.001, 0.01, 0.05, 0.1])
        adj = _benjamini_hochberg(pvals)
        assert (adj >= pvals).all()
        assert (adj <= 1.0).all()

    def test_empty(self):
        adj = _benjamini_hochberg(np.array([]))
        assert len(adj) == 0


class TestDefineComparisons:
    def test_standard(self):
        adata = _make_ivig_scrna()
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        comps = define_de_comparisons(pb["Classical_Mono"])
        names = [c[0] for c in comps]
        assert "pans_pre_vs_control" in names
        assert "pans_post_vs_control" in names
        assert "pans_pre_vs_post" in names

    def test_group_assignment(self):
        adata = _make_ivig_scrna()
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        comps = define_de_comparisons(pb["Classical_Mono"])
        # pans_pre_vs_control should have 3 controls and 3 pre samples
        pre_vs_ctrl = [c for c in comps if c[0] == "pans_pre_vs_control"][0]
        assert len(pre_vs_ctrl[1]) == 3  # controls
        assert len(pre_vs_ctrl[2]) == 3  # pans_pre


class TestRunDeWald:
    def test_returns_results(self):
        adata = _make_ivig_scrna()
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        mono = pb["Classical_Mono"]
        g1 = [f"ctrl_{i}" for i in range(3)]
        g2 = [f"pans_pre_{i}" for i in range(3)]
        results = run_de_wald(mono, g1, g2, "Classical_Mono", "test")
        assert len(results) > 0
        assert all(r.cell_type == "Classical_Mono" for r in results)

    def test_pvalues_valid(self):
        adata = _make_ivig_scrna()
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        mono = pb["Classical_Mono"]
        g1 = [f"ctrl_{i}" for i in range(3)]
        g2 = [f"pans_pre_{i}" for i in range(3)]
        results = run_de_wald(mono, g1, g2, "Classical_Mono", "test")
        for r in results:
            assert 0 <= r.pvalue <= 1
            assert 0 <= r.pvalue_adj <= 1

    def test_detects_injected_signal(self):
        """First 5 genes should be upregulated in pans_pre monocytes."""
        adata = _make_ivig_scrna(n_per_sample=80)
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        mono = pb["Classical_Mono"]
        g1 = [f"ctrl_{i}" for i in range(3)]
        g2 = [f"pans_pre_{i}" for i in range(3)]
        results = run_de_wald(mono, g1, g2, "Classical_Mono", "test",
                              min_mean_count=0.5)
        de_genes = {r.gene: r for r in results}
        # GENE0 through GENE4 should have positive log2FC
        for i in range(5):
            gene = f"GENE{i}"
            if gene in de_genes:
                assert de_genes[gene].log2_fold_change > 0

    def test_empty_with_insufficient_samples(self):
        adata = _make_ivig_scrna()
        pb = aggregate_pseudobulk(adata, "cell_type", "sample", "condition")
        mono = pb["Classical_Mono"]
        # Only one sample per group -> empty
        results = run_de_wald(mono, ["ctrl_0"], ["pans_pre_0"], "Classical_Mono", "test")
        assert len(results) == 0


class TestDEResult:
    def test_to_dict(self):
        r = DEResult(gene="TP53", cell_type="Mono", comparison="test",
                     log2_fold_change=2.5, pvalue=0.001, pvalue_adj=0.01)
        d = r.to_dict()
        assert d["gene"] == "TP53"
        assert d["log2_fold_change"] == 2.5


class TestPseudobulkDESummary:
    def test_to_dataframe(self):
        results = [
            DEResult("A", "Mono", "comp1", 1.0, 0.01, 0.02),
            DEResult("B", "Mono", "comp1", -0.5, 0.5, 0.6),
        ]
        summary = PseudobulkDESummary(
            results_by_celltype={"Mono": results},
            comparisons=["comp1"],
            cell_types_tested=["Mono"],
        )
        df = summary.to_dataframe()
        assert len(df) == 2

    def test_get_de_genes(self):
        results = [
            DEResult("A", "Mono", "comp1", 2.0, 0.001, 0.01, method="wald_nb"),
            DEResult("B", "Mono", "comp1", 0.1, 0.5, 0.6, method="wald_nb"),
        ]
        summary = PseudobulkDESummary(
            results_by_celltype={"Mono": results},
            comparisons=["comp1"],
            cell_types_tested=["Mono"],
            alpha=0.05,
            lfc_threshold=0.5,
        )
        sig = summary.get_de_genes("Mono")
        assert len(sig) == 1
        assert sig.iloc[0]["gene"] == "A"

    def test_get_de_genes_empty(self):
        summary = PseudobulkDESummary()
        sig = summary.get_de_genes("NonExistent")
        assert len(sig) == 0

    def test_summary_string(self):
        summary = PseudobulkDESummary(
            comparisons=["comp1"],
            cell_types_tested=["Mono"],
            n_de_genes={"Mono": 42},
        )
        s = summary.summary()
        assert "Pseudobulk DE" in s
        assert "42" in s


class TestRunPseudobulkDE:
    def test_full_pipeline(self):
        adata = _make_ivig_scrna()
        summary = run_pseudobulk_de(
            adata, condition_key="condition", sample_key="sample",
            cell_type_key="cell_type",
        )
        assert len(summary.cell_types_tested) == 4
        assert len(summary.comparisons) == 3

    def test_custom_comparisons(self):
        adata = _make_ivig_scrna()
        comps = [("custom", ["ctrl_0", "ctrl_1", "ctrl_2"],
                  ["pans_pre_0", "pans_pre_1", "pans_pre_2"])]
        summary = run_pseudobulk_de(
            adata, condition_key="condition", sample_key="sample",
            cell_type_key="cell_type", comparisons=comps,
        )
        assert "custom" in summary.comparisons

    def test_min_cells_filter(self):
        adata = _make_ivig_scrna(n_per_sample=8)
        summary = run_pseudobulk_de(
            adata, condition_key="condition", sample_key="sample",
            cell_type_key="cell_type", min_cells=20,
        )
        # With 8 cells per sample and min_cells=20, no cell types should pass
        assert len(summary.cell_types_tested) == 0

    def test_results_have_correct_structure(self):
        adata = _make_ivig_scrna()
        summary = run_pseudobulk_de(
            adata, condition_key="condition", sample_key="sample",
            cell_type_key="cell_type",
        )
        df = summary.to_dataframe()
        assert "gene" in df.columns
        assert "cell_type" in df.columns
        assert "log2_fold_change" in df.columns
        assert "pvalue_adj" in df.columns
