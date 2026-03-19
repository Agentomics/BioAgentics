"""Tests for IVIG pathway enrichment analysis."""

import numpy as np
import pandas as pd
import anndata as ad

from bioagentics.pandas_pans.ivig_pathway_enrichment import (
    ORAResult,
    GSEAResult,
    ModuleScore,
    PathwayEnrichmentSummary,
    get_all_gene_sets,
    run_ora,
    compute_enrichment_score,
    run_gsea_permutation,
    run_gsea,
    score_module,
    score_ivig_modules,
    compare_module_scores,
    run_pathway_enrichment,
    _benjamini_hochberg,
    GENE_SETS,
    IVIG_MODULES,
)


def _make_ivig_scrna(n_per_sample: int = 30, n_genes: int = 80, seed: int = 42) -> ad.AnnData:
    """Create synthetic scRNA-seq data with known pathway signals."""
    rng = np.random.default_rng(seed)

    # Include genes from our curated sets so enrichment can be tested
    pathway_genes = [
        "TNF", "IL1B", "IL6", "IFNG", "CXCL8", "TLR4", "MYD88", "NLRP3",
        "ATG7", "UVRAG", "BECN1", "ATG5", "MAP1LC3B", "SQSTM1", "ULK1",
        "S100A12", "S100A8", "S100A9", "CD14", "IRAK1",
        "HDAC1", "HDAC2", "KDM6B", "EZH2", "SIRT1",
        "GZMB", "PRF1", "GNLY",
        "FOXP3", "IL2RA", "CTLA4",
    ]
    filler_genes = [f"GENE{i}" for i in range(n_genes - len(pathway_genes))]
    gene_names = pathway_genes + filler_genes

    cell_types = ["Classical_Mono", "CD4_Memory", "NK_CD56dim", "B_Naive"]
    samples_meta = [
        ("ctrl_0", "control"), ("ctrl_1", "control"),
        ("pans_pre_0", "pans_pre"), ("pans_pre_1", "pans_pre"),
        ("pans_post_0", "pans_post"), ("pans_post_1", "pans_post"),
    ]

    obs_data = []
    X_rows = []

    for sample_id, condition in samples_meta:
        for ct in cell_types:
            for _ in range(n_per_sample):
                counts = rng.negative_binomial(2, 0.3, size=len(gene_names)).astype(np.float32)

                # Inject autophagy signal in monocytes post-IVIG
                if condition == "pans_post" and ct == "Classical_Mono":
                    for g in ["ATG7", "UVRAG", "BECN1", "ATG5", "MAP1LC3B"]:
                        idx = gene_names.index(g)
                        counts[idx] *= 4

                # Inject S100A12 signal in monocytes pre-IVIG
                if condition == "pans_pre" and ct == "Classical_Mono":
                    for g in ["S100A12", "S100A8", "S100A9"]:
                        idx = gene_names.index(g)
                        counts[idx] *= 5

                # Inject histone modification in NK pre-IVIG
                if condition == "pans_pre" and ct == "NK_CD56dim":
                    for g in ["HDAC1", "HDAC2", "EZH2"]:
                        idx = gene_names.index(g)
                        counts[idx] *= 3

                obs_data.append({
                    "sample": sample_id,
                    "condition": condition,
                    "cell_type": ct,
                })
                X_rows.append(counts)

    X = np.array(X_rows)
    obs = pd.DataFrame(obs_data)
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = pd.Index(gene_names)
    return adata


def _make_de_results() -> tuple[pd.DataFrame, list[str]]:
    """Create synthetic DE results with known enriched pathways."""
    rng = np.random.default_rng(42)

    # Background genes
    pathway_genes = [
        "TNF", "IL1B", "IL6", "IFNG", "CXCL8", "TLR4", "MYD88", "NLRP3",
        "ATG7", "UVRAG", "BECN1", "ATG5", "MAP1LC3B", "SQSTM1",
        "S100A12", "S100A8", "S100A9", "CD14",
        "HDAC1", "HDAC2", "KDM6B", "EZH2",
    ]
    filler = [f"GENE{i}" for i in range(200)]
    all_genes = pathway_genes + filler
    background = all_genes

    rows = []
    for gene in all_genes:
        lfc = rng.normal(0, 0.3)
        pval = rng.uniform(0.01, 1.0)

        # Make inflammatory genes significantly DE
        if gene in ["TNF", "IL1B", "IL6", "CXCL8", "TLR4", "MYD88", "NLRP3"]:
            lfc = rng.uniform(1.0, 3.0)
            pval = rng.uniform(1e-6, 1e-3)

        # Make autophagy genes significantly DE
        if gene in ["ATG7", "UVRAG", "BECN1", "ATG5", "MAP1LC3B", "SQSTM1"]:
            lfc = rng.uniform(0.8, 2.0)
            pval = rng.uniform(1e-5, 1e-2)

        rows.append({
            "gene": gene,
            "cell_type": "Classical_Mono",
            "comparison": "pans_pre_vs_control",
            "log2_fold_change": lfc,
            "pvalue": pval,
            "pvalue_adj": min(pval * len(all_genes) / 10, 1.0),
        })

    df = pd.DataFrame(rows)
    return df, background


# --- Gene set structure tests ---

class TestGeneSets:
    def test_gene_sets_not_empty(self):
        assert len(GENE_SETS) > 0

    def test_all_categories_have_sets(self):
        for cat, sets in GENE_SETS.items():
            assert len(sets) > 0, f"Category {cat} is empty"
            for name, genes in sets.items():
                assert len(genes) > 0, f"Gene set {name} is empty"

    def test_get_all_gene_sets_flat(self):
        flat = get_all_gene_sets()
        assert isinstance(flat, dict)
        assert len(flat) > 0
        # Should be flat (no nested dicts)
        for name, genes in flat.items():
            assert isinstance(genes, list)
            assert all(isinstance(g, str) for g in genes)

    def test_ivig_modules_defined(self):
        assert "autophagy_core" in IVIG_MODULES
        assert "s100a12_tlr4_myd88" in IVIG_MODULES
        assert "histone_modification" in IVIG_MODULES
        assert "defense_response" in IVIG_MODULES


# --- BH correction tests ---

class TestBH:
    def test_empty(self):
        result = _benjamini_hochberg(np.array([]))
        assert len(result) == 0

    def test_single_pvalue(self):
        result = _benjamini_hochberg(np.array([0.03]))
        assert result[0] == 0.03

    def test_monotonic(self):
        pvals = np.array([0.001, 0.01, 0.05, 0.1])
        adj = _benjamini_hochberg(pvals)
        assert all(adj[i] <= adj[i + 1] for i in range(len(adj) - 1))

    def test_capped_at_one(self):
        pvals = np.array([0.8, 0.9, 0.95])
        adj = _benjamini_hochberg(pvals)
        assert all(a <= 1.0 for a in adj)


# --- ORA tests ---

class TestORA:
    def test_basic_ora(self):
        de_genes = ["TNF", "IL1B", "IL6", "TLR4", "MYD88", "NLRP3"]
        bg = de_genes + [f"GENE{i}" for i in range(100)]
        gene_sets = {
            "inflammatory": ["TNF", "IL1B", "IL6", "CXCL8", "CCL2"],
            "unrelated": ["GENE0", "GENE1", "GENE2", "GENE3", "GENE4"],
        }
        results = run_ora(de_genes, bg, gene_sets=gene_sets, min_overlap=2)
        assert len(results) > 0
        # Inflammatory should have overlap
        inf_result = [r for r in results if r.gene_set == "inflammatory"]
        assert len(inf_result) == 1
        assert inf_result[0].n_overlap >= 3

    def test_ora_no_overlap(self):
        de_genes = ["GENE_A", "GENE_B"]
        bg = de_genes + [f"X{i}" for i in range(50)]
        gene_sets = {"set1": ["FOO", "BAR", "BAZ"]}
        results = run_ora(de_genes, bg, gene_sets=gene_sets)
        assert len(results) == 0

    def test_ora_bh_correction(self):
        de_genes = ["TNF", "IL1B", "IL6"]
        bg = de_genes + [f"G{i}" for i in range(50)]
        gene_sets = {
            "set1": ["TNF", "IL1B", "IL6"],
            "set2": ["TNF", "IL1B", "G0"],
        }
        results = run_ora(de_genes, bg, gene_sets=gene_sets)
        # Adjusted p-values should be >= raw
        for r in results:
            assert r.pvalue_adj >= r.pvalue - 1e-10

    def test_ora_result_fields(self):
        de_genes = ["TNF", "IL1B", "IL6"]
        bg = de_genes + [f"G{i}" for i in range(20)]
        gene_sets = {"test_set": ["TNF", "IL1B", "IL6", "CXCL8"]}
        results = run_ora(de_genes, bg, gene_sets=gene_sets)
        assert len(results) == 1
        r = results[0]
        assert r.gene_set == "test_set"
        assert r.n_overlap == 3
        assert r.n_gene_set == 3  # only 3 of 4 are in bg
        d = r.to_dict()
        assert "overlap_genes" in d
        assert "odds_ratio" in d

    def test_ora_with_cell_type_annotation(self):
        results = run_ora(
            ["TNF", "IL1B"], ["TNF", "IL1B", "G1", "G2"],
            gene_sets={"s": ["TNF", "IL1B"]},
            cell_type="Mono", comparison="test",
        )
        assert len(results) == 1
        assert results[0].cell_type == "Mono"
        assert results[0].comparison == "test"

    def test_ora_uses_builtin_sets_when_none(self):
        # Should not crash with built-in sets
        de_genes = ["TNF", "IL1B", "IL6", "IFNG", "CXCL8", "TLR4"]
        bg = de_genes + [f"G{i}" for i in range(200)]
        results = run_ora(de_genes, bg, gene_sets=None)
        # Should find at least some overlap with built-in inflammation sets
        assert len(results) > 0


# --- GSEA tests ---

class TestGSEA:
    def test_enrichment_score_basic(self):
        genes = [f"G{i}" for i in range(20)]
        scores = np.linspace(3, -3, 20)
        gene_set = {"G0", "G1", "G2"}  # top-ranked genes
        es, le = compute_enrichment_score(genes, scores, gene_set)
        assert es > 0  # Should be positive (genes at top)
        assert len(le) > 0

    def test_enrichment_score_bottom(self):
        genes = [f"G{i}" for i in range(20)]
        scores = np.linspace(3, -3, 20)
        gene_set = {"G17", "G18", "G19"}  # bottom-ranked genes
        es, le = compute_enrichment_score(genes, scores, gene_set)
        assert es < 0  # Should be negative (genes at bottom)

    def test_enrichment_score_empty_set(self):
        genes = [f"G{i}" for i in range(10)]
        scores = np.linspace(1, -1, 10)
        es, le = compute_enrichment_score(genes, scores, set())
        assert es == 0.0
        assert len(le) == 0

    def test_gsea_permutation(self):
        genes = [f"G{i}" for i in range(50)]
        scores = np.linspace(5, -5, 50)
        gene_set = {"G0", "G1", "G2", "G3", "G4"}
        es, nes, pval = run_gsea_permutation(genes, scores, gene_set, n_perm=100)
        assert es > 0
        assert 0 < pval <= 1.0

    def test_gsea_on_de_results(self):
        de_df, bg = _make_de_results()
        gene_sets = {
            "inflammatory": ["TNF", "IL1B", "IL6", "CXCL8", "TLR4", "MYD88", "NLRP3"],
        }
        results = run_gsea(
            de_df, gene_sets=gene_sets, n_perm=100,
            cell_type="Mono", comparison="test",
        )
        assert len(results) == 1
        r = results[0]
        assert r.gene_set == "inflammatory"
        assert r.enrichment_score > 0  # These genes are upregulated
        assert r.cell_type == "Mono"

    def test_gsea_result_fields(self):
        de_df, bg = _make_de_results()
        gene_sets = {"test": ["TNF", "IL1B", "IL6", "CXCL8", "TLR4"]}
        results = run_gsea(de_df, gene_sets=gene_sets, n_perm=50)
        assert len(results) == 1
        d = results[0].to_dict()
        assert "enrichment_score" in d
        assert "normalized_es" in d
        assert "leading_edge" in d

    def test_gsea_empty_input(self):
        results = run_gsea(pd.DataFrame())
        assert len(results) == 0

    def test_gsea_min_set_size_filter(self):
        de_df, _ = _make_de_results()
        gene_sets = {"small": ["TNF", "IL1B"]}
        results = run_gsea(de_df, gene_sets=gene_sets, min_set_size=5)
        assert len(results) == 0  # too small


# --- Module scoring tests ---

class TestModuleScoring:
    def test_score_module_basic(self):
        adata = _make_ivig_scrna()
        scores = score_module(
            adata,
            ["ATG7", "UVRAG", "BECN1"],
            "autophagy_test",
        )
        assert len(scores) > 0
        assert all(isinstance(s, ModuleScore) for s in scores)

    def test_score_module_detects_signal(self):
        adata = _make_ivig_scrna()
        scores = score_module(adata, ["ATG7", "UVRAG", "BECN1", "ATG5", "MAP1LC3B"], "autophagy")

        # Find monocyte scores
        mono_post = [s for s in scores if s.cell_type == "Classical_Mono" and s.condition == "pans_post"]
        mono_ctrl = [s for s in scores if s.cell_type == "Classical_Mono" and s.condition == "control"]

        assert len(mono_post) == 1
        assert len(mono_ctrl) == 1
        # Post-IVIG monocytes should have higher autophagy score (we injected 4x signal)
        assert mono_post[0].mean_score > mono_ctrl[0].mean_score

    def test_score_module_missing_genes(self):
        adata = _make_ivig_scrna()
        scores = score_module(adata, ["NONEXISTENT_GENE1", "NONEXISTENT_GENE2"], "missing")
        assert len(scores) == 0

    def test_score_module_to_dict(self):
        adata = _make_ivig_scrna()
        scores = score_module(adata, ["TNF", "IL1B"], "test")
        assert len(scores) > 0
        d = scores[0].to_dict()
        assert "module_name" in d
        assert "mean_score" in d
        assert "genes_detected" in d

    def test_score_ivig_modules(self):
        adata = _make_ivig_scrna()
        df = score_ivig_modules(adata)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "module_name" in df.columns
        assert "cell_type" in df.columns
        assert "condition" in df.columns

    def test_score_ivig_modules_custom(self):
        adata = _make_ivig_scrna()
        custom = {"my_module": ["TNF", "IL1B", "IL6"]}
        df = score_ivig_modules(adata, modules=custom)
        assert len(df) > 0
        assert df["module_name"].unique().tolist() == ["my_module"]

    def test_s100a12_signal_detection(self):
        adata = _make_ivig_scrna()
        scores = score_module(adata, ["S100A12", "S100A8", "S100A9"], "s100")

        mono_pre = [s for s in scores if s.cell_type == "Classical_Mono" and s.condition == "pans_pre"]
        mono_ctrl = [s for s in scores if s.cell_type == "Classical_Mono" and s.condition == "control"]

        assert len(mono_pre) == 1
        assert len(mono_ctrl) == 1
        # Pre-IVIG monocytes should have higher S100 score
        assert mono_pre[0].mean_score > mono_ctrl[0].mean_score

    def test_compare_module_scores(self):
        adata = _make_ivig_scrna()
        df = score_ivig_modules(adata)
        comp = compare_module_scores(df, "autophagy_core", "control", "pans_post")
        assert isinstance(comp, pd.DataFrame)
        if not comp.empty:
            assert "score_diff" in comp.columns


# --- Combined pipeline tests ---

class TestPathwayEnrichmentPipeline:
    def test_run_pathway_enrichment_ora_only(self):
        de_df, bg = _make_de_results()
        summary = run_pathway_enrichment(
            de_df, bg,
            run_ora_analysis=True,
            run_gsea_analysis=False,
        )
        assert isinstance(summary, PathwayEnrichmentSummary)
        assert len(summary.ora_results) > 0
        assert len(summary.gsea_results) == 0

    def test_run_pathway_enrichment_gsea_only(self):
        de_df, bg = _make_de_results()
        summary = run_pathway_enrichment(
            de_df, bg,
            run_ora_analysis=False,
            run_gsea_analysis=True,
            gsea_n_perm=50,
        )
        assert isinstance(summary, PathwayEnrichmentSummary)
        assert len(summary.gsea_results) > 0

    def test_run_pathway_enrichment_with_module_scoring(self):
        adata = _make_ivig_scrna()
        de_df, bg = _make_de_results()
        summary = run_pathway_enrichment(
            de_df, bg,
            run_gsea_analysis=False,
            adata=adata,
        )
        assert not summary.module_scores.empty

    def test_summary_text(self):
        de_df, bg = _make_de_results()
        summary = run_pathway_enrichment(
            de_df, bg,
            run_gsea_analysis=False,
        )
        text = summary.summary()
        assert "Pathway Enrichment" in text

    def test_get_significant_ora(self):
        de_df, bg = _make_de_results()
        summary = run_pathway_enrichment(
            de_df, bg, run_gsea_analysis=False,
        )
        sig = summary.get_significant_ora(alpha=1.0)  # permissive threshold
        assert isinstance(sig, pd.DataFrame)

    def test_to_dataframes(self):
        de_df, bg = _make_de_results()
        summary = run_pathway_enrichment(
            de_df, bg,
            run_gsea_analysis=False,
        )
        ora_df = summary.to_ora_dataframe()
        assert isinstance(ora_df, pd.DataFrame)
        if not ora_df.empty:
            assert "gene_set" in ora_df.columns

    def test_empty_de_results(self):
        summary = run_pathway_enrichment(pd.DataFrame(), [])
        assert len(summary.ora_results) == 0
        assert len(summary.gsea_results) == 0

    def test_custom_gene_sets(self):
        de_df, bg = _make_de_results()
        custom = {"my_set": ["TNF", "IL1B", "IL6", "CXCL8", "TLR4"]}
        summary = run_pathway_enrichment(
            de_df, bg,
            gene_sets=custom,
            run_gsea_analysis=False,
        )
        if summary.ora_results:
            assert summary.ora_results[0].gene_set == "my_set"
