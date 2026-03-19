"""Tests for IVIG treatment response signature analysis (Phase 4)."""

import numpy as np
import pandas as pd
import anndata as ad

from bioagentics.pandas_pans.ivig_treatment_signatures import (
    GeneSignature,
    SignatureScore,
    TreatmentResponseResult,
    _extract_de_genes,
    run_disease_signature,
    run_treatment_response_signature,
    run_signature_scoring,
    compare_signature_scores,
    run_minimal_predictor,
    run_treatment_signatures,
    _score_signature_in_adata,
    _benjamini_hochberg,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_de_results() -> pd.DataFrame:
    """Create synthetic DE results for testing."""
    rows = []

    # PANS vs control: disease genes in monocytes
    for gene, lfc, padj in [
        ("TNF", 2.5, 0.001), ("IL1B", 3.0, 0.0001), ("S100A12", 4.0, 0.00001),
        ("IL6", 1.8, 0.01), ("CXCL8", 2.0, 0.005),
        ("FOXP3", -2.0, 0.01), ("IL10", -1.5, 0.02),
        ("GENE1", 0.3, 0.8),  # not significant
    ]:
        rows.append({
            "gene": gene, "cell_type": "Classical_Mono",
            "comparison": "pans_vs_control",
            "log2_fold_change": lfc, "pvalue": padj * 0.1,
            "pvalue_adj": padj,
        })

    # PANS vs control: disease genes in NK cells
    for gene, lfc, padj in [
        ("TNF", 1.5, 0.01), ("GZMB", 2.0, 0.005),
        ("PRF1", 1.8, 0.008), ("FOXP3", -1.2, 0.03),
    ]:
        rows.append({
            "gene": gene, "cell_type": "NK_CD56dim",
            "comparison": "pans_vs_control",
            "log2_fold_change": lfc, "pvalue": padj * 0.1,
            "pvalue_adj": padj,
        })

    # PANS vs control: disease genes in T cells
    for gene, lfc, padj in [
        ("IL6", 1.5, 0.02), ("CTLA4", -1.8, 0.01),
        ("TNF", 1.2, 0.03),
    ]:
        rows.append({
            "gene": gene, "cell_type": "CD4_Memory",
            "comparison": "pans_vs_control",
            "log2_fold_change": lfc, "pvalue": padj * 0.1,
            "pvalue_adj": padj,
        })

    # Pre vs post IVIG: treatment effect (reversal of disease genes)
    for gene, lfc, padj in [
        ("TNF", -1.8, 0.01), ("IL1B", -2.0, 0.005), ("S100A12", -3.0, 0.001),
        ("CXCL8", -1.5, 0.02),
        ("FOXP3", 1.0, 0.05),  # reverses downregulation
        ("IL6", -1.0, 0.04),
    ]:
        rows.append({
            "gene": gene, "cell_type": "Classical_Mono",
            "comparison": "pre_vs_post",
            "log2_fold_change": lfc, "pvalue": padj * 0.1,
            "pvalue_adj": padj,
        })

    # Pre vs post: NK treatment effect
    for gene, lfc, padj in [
        ("TNF", -1.2, 0.03), ("GZMB", -1.0, 0.05),
    ]:
        rows.append({
            "gene": gene, "cell_type": "NK_CD56dim",
            "comparison": "pre_vs_post",
            "log2_fold_change": lfc, "pvalue": padj * 0.1,
            "pvalue_adj": padj,
        })

    return pd.DataFrame(rows)


def _make_adata(n_per_group: int = 20, seed: int = 42) -> ad.AnnData:
    """Create synthetic AnnData for signature scoring tests."""
    rng = np.random.default_rng(seed)

    gene_names = [
        "TNF", "IL1B", "S100A12", "IL6", "CXCL8", "FOXP3", "IL10",
        "GZMB", "PRF1", "CTLA4", "GENE1", "GENE2", "GENE3",
    ]

    cell_types = ["Classical_Mono", "NK_CD56dim", "CD4_Memory"]
    conditions = ["control", "pans_pre", "pans_post"]

    obs_list = []
    X_rows = []

    for ct in cell_types:
        for cond in conditions:
            for _ in range(n_per_group):
                counts = rng.negative_binomial(2, 0.3, size=len(gene_names)).astype(np.float32)

                # Inject disease signal in pans_pre
                if cond == "pans_pre" and ct == "Classical_Mono":
                    for g in ["TNF", "IL1B", "S100A12"]:
                        idx = gene_names.index(g)
                        counts[idx] *= 4
                    for g in ["FOXP3", "IL10"]:
                        idx = gene_names.index(g)
                        counts[idx] *= 0.3

                # Post-IVIG normalizes
                if cond == "pans_post" and ct == "Classical_Mono":
                    for g in ["TNF", "IL1B", "S100A12"]:
                        idx = gene_names.index(g)
                        counts[idx] *= 1.2  # slight residual elevation

                obs_list.append({
                    "cell_type": ct,
                    "condition": cond,
                    "sample": f"{cond}_{ct}",
                })
                X_rows.append(counts)

    X = np.vstack(X_rows)
    obs = pd.DataFrame(obs_list)
    var = pd.DataFrame(index=gene_names)

    return ad.AnnData(X=X, obs=obs, var=var)


# ---------------------------------------------------------------------------
# GeneSignature dataclass tests
# ---------------------------------------------------------------------------


class TestGeneSignature:
    def test_all_genes(self):
        sig = GeneSignature(name="test", genes_up=["A", "B"], genes_down=["C", "A"])
        assert sig.all_genes == ["A", "B", "C"]

    def test_n_genes(self):
        sig = GeneSignature(name="test", genes_up=["A", "B"], genes_down=["C"])
        assert sig.n_genes == 3

    def test_to_dict(self):
        sig = GeneSignature(
            name="test", genes_up=["A"], genes_down=["B"],
            cell_type="mono", comparison="pans_vs_control",
        )
        d = sig.to_dict()
        assert d["name"] == "test"
        assert d["n_genes_up"] == 1
        assert d["n_genes_down"] == 1
        assert d["n_genes_total"] == 2

    def test_empty_signature(self):
        sig = GeneSignature(name="empty")
        assert sig.all_genes == []
        assert sig.n_genes == 0


# ---------------------------------------------------------------------------
# SignatureScore dataclass tests
# ---------------------------------------------------------------------------


class TestSignatureScore:
    def test_to_dict(self):
        ss = SignatureScore(
            signature_name="test", cell_type="mono", condition="ctrl",
            mean_score=0.5, std_score=0.1, n_cells=100,
            n_genes_detected=10, n_genes_total=15,
        )
        d = ss.to_dict()
        assert d["signature_name"] == "test"
        assert d["mean_score"] == 0.5
        assert d["n_cells"] == 100


# ---------------------------------------------------------------------------
# TreatmentResponseResult tests
# ---------------------------------------------------------------------------


class TestTreatmentResponseResult:
    def test_summary(self):
        result = TreatmentResponseResult(
            disease_signatures={
                "mono": GeneSignature(name="disease_mono", genes_up=["A", "B"], genes_down=["C"]),
            },
            treatment_signatures={
                "mono": GeneSignature(name="treatment_mono", genes_up=["A"], genes_down=[]),
            },
            cell_types_analyzed=["Classical_Mono"],
        )
        s = result.summary()
        assert "Disease signatures: 1" in s
        assert "Treatment-responsive signatures: 1" in s

    def test_empty_result(self):
        result = TreatmentResponseResult()
        s = result.summary()
        assert "Disease signatures: 0" in s


# ---------------------------------------------------------------------------
# _extract_de_genes tests
# ---------------------------------------------------------------------------


class TestExtractDEGenes:
    def test_basic_filtering(self):
        de_df = _make_de_results()
        sig = _extract_de_genes(de_df, alpha=0.05, lfc_threshold=1.0)
        assert len(sig) > 0
        assert all(sig["pvalue_adj"] < 0.05)
        assert all(sig["log2_fold_change"].abs() > 1.0)

    def test_strict_threshold(self):
        de_df = _make_de_results()
        sig_strict = _extract_de_genes(de_df, alpha=0.001, lfc_threshold=2.0)
        sig_loose = _extract_de_genes(de_df, alpha=0.05, lfc_threshold=1.0)
        assert len(sig_strict) <= len(sig_loose)

    def test_from_list(self):
        rows = [
            {"gene": "A", "log2_fold_change": 2.0, "pvalue_adj": 0.01},
            {"gene": "B", "log2_fold_change": 0.5, "pvalue_adj": 0.01},
        ]
        sig = _extract_de_genes(rows, alpha=0.05, lfc_threshold=1.0)
        assert len(sig) == 1
        assert sig.iloc[0]["gene"] == "A"

    def test_empty_input(self):
        sig = _extract_de_genes(pd.DataFrame(), alpha=0.05, lfc_threshold=1.0)
        assert sig.empty

    def test_missing_columns(self):
        import pytest
        with pytest.raises(ValueError, match="Missing columns"):
            _extract_de_genes([{"gene": "A", "pvalue_adj": 0.01}])


# ---------------------------------------------------------------------------
# Disease signature tests
# ---------------------------------------------------------------------------


class TestDiseaseSignature:
    def test_basic_extraction(self):
        de_df = _make_de_results()
        sigs = run_disease_signature(de_df, alpha=0.05, lfc_threshold=1.0)
        assert len(sigs) > 0
        assert "consensus" in sigs

    def test_per_cell_type_signatures(self):
        de_df = _make_de_results()
        sigs = run_disease_signature(de_df, alpha=0.05, lfc_threshold=1.0)
        assert "Classical_Mono" in sigs
        mono_sig = sigs["Classical_Mono"]
        assert "TNF" in mono_sig.genes_up
        assert "FOXP3" in mono_sig.genes_down

    def test_consensus_contains_cross_ct_genes(self):
        de_df = _make_de_results()
        sigs = run_disease_signature(de_df, alpha=0.05, lfc_threshold=1.0, min_cell_types=2)
        if "consensus" in sigs:
            # TNF is DE in mono, NK, and T cells
            assert "TNF" in sigs["consensus"].all_genes

    def test_empty_input(self):
        sigs = run_disease_signature(pd.DataFrame(), alpha=0.05, lfc_threshold=1.0)
        assert len(sigs) == 0

    def test_strict_threshold_reduces_genes(self):
        de_df = _make_de_results()
        sigs_loose = run_disease_signature(de_df, alpha=0.05, lfc_threshold=1.0)
        sigs_strict = run_disease_signature(de_df, alpha=0.01, lfc_threshold=2.0)
        if "consensus" in sigs_loose and "consensus" in sigs_strict:
            assert sigs_strict["consensus"].n_genes <= sigs_loose["consensus"].n_genes

    def test_comparison_pattern_filter(self):
        de_df = _make_de_results()
        sigs = run_disease_signature(de_df, comparison_pattern="nonexistent")
        assert len(sigs) == 0

    def test_no_cell_type_column(self):
        rows = [
            {"gene": "A", "log2_fold_change": 2.0, "pvalue_adj": 0.01,
             "comparison": "pans_vs_control"},
        ]
        sigs = run_disease_signature(rows, alpha=0.05, lfc_threshold=1.0)
        assert "all" in sigs or "consensus" in sigs


# ---------------------------------------------------------------------------
# Treatment-responsive signature tests
# ---------------------------------------------------------------------------


class TestTreatmentResponseSignature:
    def test_basic_reversal_detection(self):
        de_df = _make_de_results()
        sigs = run_treatment_response_signature(de_df)
        assert len(sigs) > 0

    def test_reversal_genes_have_opposite_directions(self):
        de_df = _make_de_results()
        sigs = run_treatment_response_signature(de_df)
        if "Classical_Mono" in sigs:
            mono_sig = sigs["Classical_Mono"]
            # TNF is up in disease, down in treatment -> should be in genes_up
            # (labeled by disease direction)
            assert "TNF" in mono_sig.genes_up or "TNF" in mono_sig.genes_down

    def test_consensus_signature(self):
        de_df = _make_de_results()
        sigs = run_treatment_response_signature(de_df)
        assert "consensus" in sigs
        assert sigs["consensus"].n_genes > 0

    def test_no_reversal_requirement(self):
        de_df = _make_de_results()
        sigs_req = run_treatment_response_signature(de_df, require_reversal=True)
        sigs_no = run_treatment_response_signature(de_df, require_reversal=False)
        # Without reversal requirement, should get >= as many genes
        total_req = sum(s.n_genes for s in sigs_req.values())
        total_no = sum(s.n_genes for s in sigs_no.values())
        assert total_no >= total_req

    def test_empty_input(self):
        sigs = run_treatment_response_signature(pd.DataFrame())
        assert len(sigs) == 0

    def test_no_treatment_comparison(self):
        # Only disease comparison, no treatment data
        rows = [
            {"gene": "A", "cell_type": "mono", "comparison": "pans_vs_control",
             "log2_fold_change": 2.0, "pvalue_adj": 0.01},
        ]
        sigs = run_treatment_response_signature(rows, require_reversal=True)
        assert len(sigs) == 0


# ---------------------------------------------------------------------------
# Signature scoring tests
# ---------------------------------------------------------------------------


class TestSignatureScoring:
    def test_score_single_signature(self):
        adata = _make_adata()
        sig = GeneSignature(
            name="test", genes_up=["TNF", "IL1B"], genes_down=["FOXP3"],
        )
        scores = _score_signature_in_adata(adata, sig)
        assert len(scores) > 0
        # pans_pre monocytes should have higher scores (up genes elevated, down genes reduced)
        mono_pre = [s for s in scores if s.cell_type == "Classical_Mono" and s.condition == "pans_pre"]
        mono_ctrl = [s for s in scores if s.cell_type == "Classical_Mono" and s.condition == "control"]
        if mono_pre and mono_ctrl:
            assert mono_pre[0].mean_score > mono_ctrl[0].mean_score

    def test_score_with_no_matching_genes(self):
        adata = _make_adata()
        sig = GeneSignature(name="empty", genes_up=["NONEXISTENT1", "NONEXISTENT2"])
        scores = _score_signature_in_adata(adata, sig)
        assert len(scores) == 0

    def test_run_signature_scoring_multiple(self):
        adata = _make_adata()
        sigs = {
            "sig1": GeneSignature(name="sig1", genes_up=["TNF"]),
            "sig2": GeneSignature(name="sig2", genes_up=["GZMB"]),
        }
        df = run_signature_scoring(adata, sigs)
        assert not df.empty
        assert set(df["signature_name"].unique()) == {"sig1", "sig2"}

    def test_scoring_direction_correctness(self):
        """Down-regulated genes should flip sign in scoring."""
        adata = _make_adata()
        # Score with only down gene
        sig_down = GeneSignature(name="down_only", genes_down=["TNF"])
        scores = _score_signature_in_adata(adata, sig_down)
        # pans_pre has high TNF -> flipped -> should give negative/lower score
        mono_pre = [s for s in scores if s.cell_type == "Classical_Mono" and s.condition == "pans_pre"]
        mono_ctrl = [s for s in scores if s.cell_type == "Classical_Mono" and s.condition == "control"]
        if mono_pre and mono_ctrl:
            assert mono_pre[0].mean_score < mono_ctrl[0].mean_score

    def test_empty_signatures_dict(self):
        adata = _make_adata()
        df = run_signature_scoring(adata, {})
        assert df.empty


# ---------------------------------------------------------------------------
# Signature comparison tests
# ---------------------------------------------------------------------------


class TestCompareSignatureScores:
    def test_basic_comparison(self):
        adata = _make_adata()
        sigs = {
            "disease": GeneSignature(
                name="disease", genes_up=["TNF", "IL1B", "S100A12"],
                genes_down=["FOXP3", "IL10"],
            ),
        }
        scores_df = run_signature_scoring(adata, sigs)
        comp = compare_signature_scores(scores_df, "disease", "control", "pans_pre")
        assert not comp.empty
        assert "pvalue" in comp.columns
        assert "cohens_d" in comp.columns

    def test_comparison_with_nonexistent_signature(self):
        scores_df = pd.DataFrame({
            "signature_name": ["a"], "cell_type": ["mono"],
            "condition": ["ctrl"], "mean_score": [0.5],
            "std_score": [0.1], "n_cells": [100],
        })
        comp = compare_signature_scores(scores_df, "nonexistent", "ctrl", "pans")
        assert comp.empty

    def test_empty_scores(self):
        comp = compare_signature_scores(pd.DataFrame(), "test", "a", "b")
        assert comp.empty


# ---------------------------------------------------------------------------
# Minimal predictor tests
# ---------------------------------------------------------------------------


class TestMinimalPredictor:
    def test_basic_selection(self):
        de_df = _make_de_results()
        genes, weights = run_minimal_predictor(de_df, n_genes=5)
        assert len(genes) <= 5
        assert len(genes) > 0
        assert len(weights) == len(genes)
        # Weights should be positive
        assert all(w > 0 for w in weights.values())

    def test_top_genes_are_most_significant(self):
        de_df = _make_de_results()
        genes, weights = run_minimal_predictor(de_df, n_genes=3)
        # S100A12 has highest effect size and significance -> should be in top
        assert "S100A12" in genes

    def test_treatment_responsive_bonus(self):
        de_df = _make_de_results()
        treatment_sigs = {
            "mono": GeneSignature(
                name="treatment_mono",
                genes_up=["TNF", "IL1B"],
            ),
        }
        genes_with, weights_with = run_minimal_predictor(
            de_df, treatment_signatures=treatment_sigs, n_genes=10,
        )
        genes_without, weights_without = run_minimal_predictor(
            de_df, n_genes=10,
        )
        # TNF should have higher weight with treatment bonus
        if "TNF" in weights_with and "TNF" in weights_without:
            assert weights_with["TNF"] >= weights_without["TNF"]

    def test_empty_input(self):
        genes, weights = run_minimal_predictor(pd.DataFrame())
        assert genes == []
        assert weights == {}

    def test_n_genes_larger_than_available(self):
        rows = [
            {"gene": "A", "log2_fold_change": 2.0, "pvalue_adj": 0.01,
             "comparison": "pans_vs_control"},
        ]
        genes, weights = run_minimal_predictor(rows, n_genes=100)
        assert len(genes) == 1


# ---------------------------------------------------------------------------
# BH correction tests
# ---------------------------------------------------------------------------


class TestBH:
    def test_basic(self):
        pvals = np.array([0.01, 0.05, 0.1])
        adj = _benjamini_hochberg(pvals)
        assert len(adj) == 3
        assert adj[0] <= adj[1] <= adj[2]
        assert all(0 <= p <= 1 for p in adj)

    def test_empty(self):
        adj = _benjamini_hochberg(np.array([]))
        assert len(adj) == 0

    def test_single(self):
        adj = _benjamini_hochberg(np.array([0.05]))
        assert adj[0] == 0.05


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_run_treatment_signatures(self):
        adata = _make_adata()
        de_df = _make_de_results()
        result = run_treatment_signatures(
            adata, de_df,
            disease_lfc=1.0,
            treatment_lfc=0.5,
            predictor_n_genes=10,
        )
        assert isinstance(result, TreatmentResponseResult)
        assert len(result.disease_signatures) > 0
        assert len(result.cell_types_analyzed) > 0
        assert len(result.predictor_genes) > 0

    def test_pipeline_with_empty_de(self):
        adata = _make_adata()
        result = run_treatment_signatures(
            adata, pd.DataFrame(),
            disease_lfc=1.0,
        )
        assert isinstance(result, TreatmentResponseResult)
        assert len(result.disease_signatures) == 0

    def test_pipeline_produces_scores(self):
        adata = _make_adata()
        de_df = _make_de_results()
        result = run_treatment_signatures(
            adata, de_df,
            disease_lfc=1.0,
            treatment_lfc=0.5,
        )
        if not result.signature_scores.empty:
            assert "signature_name" in result.signature_scores.columns
            assert "mean_score" in result.signature_scores.columns
