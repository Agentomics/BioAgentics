"""Tests for IVIG cell-cell communication analysis."""

import numpy as np
import pandas as pd
import anndata as ad

from bioagentics.pandas_pans.ivig_cell_communication import (
    CellCommInteraction,
    DiffCommResult,
    CellCommSummary,
    get_lr_dataframe,
    compute_communication_scores,
    compute_differential_communication,
    run_cell_communication,
    _benjamini_hochberg,
    LIGAND_RECEPTOR_DB,
)


def _make_comm_scrna(n_per_sample: int = 30, seed: int = 42) -> ad.AnnData:
    """Create synthetic scRNA-seq data with known LR expression patterns."""
    rng = np.random.default_rng(seed)

    # Include ligand/receptor genes from our DB
    lr_genes = [
        "TNF", "TNFRSF1A", "TNFRSF1B",
        "IL1B", "IL1R1",
        "IL6", "IL6R", "IL6ST",
        "S100A12", "S100A8", "S100A9", "TLR4", "MYD88",
        "CXCL8", "CXCR1", "CXCR2",
        "IFNG", "IFNGR1",
        "CD80", "CD28", "CTLA4",
        "GZMB", "PRF1",
        "C3", "C3AR1",
        "IL10", "IL10RA",
    ]
    filler = [f"GENE{i}" for i in range(50)]
    gene_names = lr_genes + filler

    cell_types = ["Classical_Mono", "CD4_Memory", "NK_CD56dim"]
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

                # Inject S100A12 ligand signal in monocytes pre-IVIG
                if condition == "pans_pre" and ct == "Classical_Mono":
                    for g in ["S100A12", "S100A8", "S100A9"]:
                        counts[gene_names.index(g)] *= 6

                # Inject TNF signal in monocytes pre-IVIG
                if condition == "pans_pre" and ct == "Classical_Mono":
                    counts[gene_names.index("TNF")] *= 4

                # Inject TLR4/MYD88 receptor in monocytes (all conditions)
                if ct == "Classical_Mono":
                    counts[gene_names.index("TLR4")] *= 3
                    counts[gene_names.index("MYD88")] *= 2

                # CD4 T cells express receptors
                if ct == "CD4_Memory":
                    counts[gene_names.index("TNFRSF1A")] *= 3
                    counts[gene_names.index("IL1R1")] *= 2
                    counts[gene_names.index("CD28")] *= 3

                # NK cells express granzyme/perforin
                if ct == "NK_CD56dim":
                    counts[gene_names.index("GZMB")] *= 5
                    counts[gene_names.index("PRF1")] *= 4

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


# --- LR database tests ---

class TestLRDatabase:
    def test_db_not_empty(self):
        assert len(LIGAND_RECEPTOR_DB) > 0

    def test_db_tuple_format(self):
        for entry in LIGAND_RECEPTOR_DB:
            assert len(entry) == 4
            lig, rec, pathway, cat = entry
            assert isinstance(lig, str)
            assert isinstance(rec, str)
            assert isinstance(pathway, str)
            assert isinstance(cat, str)

    def test_db_has_key_categories(self):
        categories = {e[3] for e in LIGAND_RECEPTOR_DB}
        assert "alarmin" in categories
        assert "cytokine" in categories
        assert "coinhibition" in categories

    def test_get_lr_dataframe(self):
        df = get_lr_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"ligand", "receptor", "pathway", "category"}
        assert len(df) == len(LIGAND_RECEPTOR_DB)


# --- Communication scoring tests ---

class TestCommunicationScoring:
    def test_compute_scores_basic(self):
        adata = _make_comm_scrna()
        interactions = compute_communication_scores(adata)
        assert len(interactions) > 0
        assert all(isinstance(i, CellCommInteraction) for i in interactions)

    def test_compute_scores_has_conditions(self):
        adata = _make_comm_scrna()
        interactions = compute_communication_scores(adata)
        conditions = {i.condition for i in interactions}
        assert "control" in conditions
        assert "pans_pre" in conditions

    def test_compute_scores_has_cell_types(self):
        adata = _make_comm_scrna()
        interactions = compute_communication_scores(adata)
        sources = {i.source_cell_type for i in interactions}
        assert len(sources) > 1

    def test_comm_score_positive(self):
        adata = _make_comm_scrna()
        interactions = compute_communication_scores(adata)
        for i in interactions:
            assert i.comm_score >= 0

    def test_min_pct_filter(self):
        adata = _make_comm_scrna()
        strict = compute_communication_scores(adata, min_pct=0.5)
        permissive = compute_communication_scores(adata, min_pct=0.01)
        assert len(strict) <= len(permissive)

    def test_s100a12_mono_signal(self):
        adata = _make_comm_scrna()
        interactions = compute_communication_scores(adata, min_pct=0.05)

        # S100A12 -> TLR4 from monocytes to monocytes should be stronger pre-IVIG
        s100_pre = [
            i for i in interactions
            if i.ligand == "S100A12" and i.receptor == "TLR4"
            and i.source_cell_type == "Classical_Mono"
            and i.target_cell_type == "Classical_Mono"
            and i.condition == "pans_pre"
        ]
        s100_ctrl = [
            i for i in interactions
            if i.ligand == "S100A12" and i.receptor == "TLR4"
            and i.source_cell_type == "Classical_Mono"
            and i.target_cell_type == "Classical_Mono"
            and i.condition == "control"
        ]
        if s100_pre and s100_ctrl:
            assert s100_pre[0].comm_score > s100_ctrl[0].comm_score

    def test_interaction_to_dict(self):
        adata = _make_comm_scrna()
        interactions = compute_communication_scores(adata)
        d = interactions[0].to_dict()
        assert "ligand" in d
        assert "receptor" in d
        assert "comm_score" in d
        assert "ligand_pct" in d

    def test_custom_lr_pairs(self):
        adata = _make_comm_scrna()
        custom = [("TNF", "TNFRSF1A", "TNF", "cytokine")]
        interactions = compute_communication_scores(adata, lr_pairs=custom)
        for i in interactions:
            assert i.ligand == "TNF"
            assert i.receptor == "TNFRSF1A"


# --- Differential communication tests ---

class TestDifferentialCommunication:
    def test_diff_comm_basic(self):
        adata = _make_comm_scrna()
        interactions = compute_communication_scores(adata, min_pct=0.05)
        diff = compute_differential_communication(
            interactions, "control", "pans_pre",
        )
        assert len(diff) > 0
        assert all(isinstance(r, DiffCommResult) for r in diff)

    def test_diff_comm_bh_correction(self):
        adata = _make_comm_scrna()
        interactions = compute_communication_scores(adata, min_pct=0.05)
        diff = compute_differential_communication(interactions, "control", "pans_pre")
        for r in diff:
            assert r.pvalue_adj >= r.pvalue - 1e-10

    def test_diff_comm_result_fields(self):
        adata = _make_comm_scrna()
        interactions = compute_communication_scores(adata, min_pct=0.05)
        diff = compute_differential_communication(interactions, "control", "pans_pre")
        if diff:
            d = diff[0].to_dict()
            assert "log2_fold_change" in d
            assert "pvalue_adj" in d
            assert "condition1" in d
            assert "condition2" in d

    def test_diff_comm_empty(self):
        diff = compute_differential_communication([], "a", "b")
        assert len(diff) == 0


# --- BH correction tests ---

class TestBH:
    def test_empty(self):
        result = _benjamini_hochberg(np.array([]))
        assert len(result) == 0

    def test_adjustment(self):
        pvals = np.array([0.01, 0.04, 0.05])
        adj = _benjamini_hochberg(pvals)
        assert all(a >= p for a, p in zip(adj, pvals))


# --- Summary tests ---

class TestCellCommSummary:
    def test_summary_construction(self):
        summary = CellCommSummary()
        text = summary.summary()
        assert "Cell Communication" in text

    def test_to_dataframes(self):
        adata = _make_comm_scrna()
        summary = run_cell_communication(adata, min_pct=0.05)
        idf = summary.to_interactions_df()
        assert isinstance(idf, pd.DataFrame)
        assert len(idf) > 0

    def test_get_top_interactions(self):
        adata = _make_comm_scrna()
        summary = run_cell_communication(adata, min_pct=0.05)
        top = summary.get_top_interactions(n=5)
        assert len(top) <= 5

    def test_get_top_by_condition(self):
        adata = _make_comm_scrna()
        summary = run_cell_communication(adata, min_pct=0.05)
        top = summary.get_top_interactions(condition="pans_pre", n=10)
        if not top.empty:
            assert all(top["condition"] == "pans_pre")

    def test_get_top_by_category(self):
        adata = _make_comm_scrna()
        summary = run_cell_communication(adata, min_pct=0.05)
        top = summary.get_top_interactions(category="cytokine", n=10)
        if not top.empty:
            assert all(top["category"] == "cytokine")

    def test_pathway_summary(self):
        adata = _make_comm_scrna()
        summary = run_cell_communication(adata, min_pct=0.05)
        pw = summary.get_pathway_summary()
        assert isinstance(pw, pd.DataFrame)
        if not pw.empty:
            assert "total_score" in pw.columns

    def test_significant_changes(self):
        adata = _make_comm_scrna()
        summary = run_cell_communication(adata, min_pct=0.05)
        sig = summary.get_significant_changes(alpha=1.0)
        assert isinstance(sig, pd.DataFrame)


# --- Full pipeline tests ---

class TestFullPipeline:
    def test_run_cell_communication(self):
        adata = _make_comm_scrna()
        summary = run_cell_communication(adata, min_pct=0.05)
        assert isinstance(summary, CellCommSummary)
        assert len(summary.interactions) > 0
        assert len(summary.conditions_analyzed) == 3
        assert len(summary.cell_types_analyzed) == 3

    def test_auto_comparisons(self):
        adata = _make_comm_scrna()
        summary = run_cell_communication(adata, min_pct=0.05)
        # Should auto-detect pre/post/control comparisons
        assert len(summary.diff_results) > 0

    def test_custom_comparisons(self):
        adata = _make_comm_scrna()
        summary = run_cell_communication(
            adata, min_pct=0.05,
            diff_comparisons=[("control", "pans_pre")],
        )
        assert len(summary.diff_results) > 0
        for r in summary.diff_results:
            assert r.condition1 == "control"
            assert r.condition2 == "pans_pre"

    def test_pipeline_summary_text(self):
        adata = _make_comm_scrna()
        summary = run_cell_communication(adata, min_pct=0.05)
        text = summary.summary()
        assert "LR pairs tested" in text
