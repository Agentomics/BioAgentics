"""Tests for causal gene prioritization module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.tourettes.ts_gwas_functional_annotation.causal_gene_prioritization import (
    GeneEvidence,
    KNOWN_TS_GENES,
    MAGMA_GENE_HITS,
    SCORE_EQTL_CAP,
    SCORE_EQTL_PER_TISSUE,
    SCORE_HIC,
    SCORE_LITERATURE,
    SCORE_PATHWAY_CAP,
    SCORE_PATHWAY_PER_HIT,
    SCORE_POSITIONAL,
    assign_genes_to_loci,
    build_gene_evidence,
    compute_gene_score,
    evidence_to_dataframe,
    rank_genes_per_locus,
    write_prioritization_results,
    write_prioritization_summary,
)


# --- Fixtures ---


@pytest.fixture
def snp_gene_df():
    """Integrated SNP-to-gene mapping with multi-modal evidence."""
    return pd.DataFrame({
        "SNP": [
            "rs1", "rs1", "rs2", "rs3", "rs4", "rs5",
            "rs6", "rs7", "rs8",
        ],
        "GENE": [
            "DRD1", "DRD2", "BCL11B", "NRXN1", "SEMA6D", "NDFIP2",
            "FOXP2", "RBM26", "GRIN2A",
        ],
        "CHR": [5, 11, 14, 2, 15, 5, 7, 14, 16],
        "BP": [
            174850000, 113300000, 99650000, 50500000, 46100000, 174900000,
            114100000, 99700000, 10000000,
        ],
        "GENE_START": [
            174800000, 113280000, 99600000, 50000000, 46000000, 174850000,
            114000000, 99650000, 9900000,
        ],
        "GENE_END": [
            174900000, 113400000, 99700000, 51000000, 46200000, 174950000,
            114200000, 99750000, 10100000,
        ],
        "DISTANCE_KB": [0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0],
        "POSITIONAL": [True, True, True, True, True, True, False, True, True],
        "EQTL": [True, True, False, True, False, False, False, False, True],
        "EQTL_TISSUES": [
            "Brain_Caudate_basal_ganglia;Brain_Putamen_basal_ganglia",
            "Brain_Frontal_Cortex_BA9",
            "",
            "Brain_Caudate_basal_ganglia;Brain_Cortex;Brain_Cerebellum",
            "",
            "",
            "",
            "",
            "Brain_Hippocampus",
        ],
        "EQTL_BEST_P": [1e-6, 1e-4, 1.0, 1e-8, 1.0, 1.0, 1.0, 1.0, 1e-3],
        "HIC": [False, False, True, True, False, False, True, False, False],
        "HIC_TISSUES": [
            "", "", "PsychENCODE_brain", "PsychENCODE_brain", "", "",
            "PsychENCODE_brain", "", "",
        ],
        "N_EVIDENCE": [2, 2, 2, 3, 1, 1, 1, 1, 2],
        "IS_CANDIDATE": [False, False, True, True, True, False, False, False, False],
    })


@pytest.fixture
def gene_results_df():
    """Gene-level MAGMA results."""
    return pd.DataFrame({
        "GENE": ["DRD1", "DRD2", "BCL11B", "NRXN1", "SEMA6D", "NDFIP2",
                 "FOXP2", "RBM26", "GRIN2A"],
        "CHR": [5, 11, 14, 2, 15, 5, 7, 14, 16],
        "START": [174800000, 113280000, 99600000, 50000000, 46000000,
                  174850000, 114000000, 99650000, 9900000],
        "STOP": [174900000, 113400000, 99700000, 51000000, 46200000,
                 174950000, 114200000, 99750000, 10100000],
        "P": [0.001, 0.01, 0.0005, 0.002, 0.03, 0.008, 0.025, 0.012, 0.02],
    })


@pytest.fixture
def convergence_df():
    """Pathway convergence results."""
    return pd.DataFrame({
        "PATHWAY": ["dopamine_signaling", "synaptic_adhesion", "axon_guidance"],
        "SOURCE": ["builtin"] * 3,
        "CONVERGENCE_FDR": [0.01, 0.03, 0.04],
        "CONVERGENCE_P": [0.001, 0.005, 0.01],
        "GENES": ["DRD1;DRD2", "NRXN1;CNTN6", "SEMA6D;SLIT2"],
    })


# --- Scoring tests ---


class TestComputeGeneScore:
    def test_positional_only(self):
        ev = GeneEvidence(gene="X", positional=True)
        assert compute_gene_score(ev) == SCORE_POSITIONAL

    def test_eqtl_single_tissue(self):
        ev = GeneEvidence(gene="X", eqtl=True, eqtl_n_tissues=1)
        assert compute_gene_score(ev) == SCORE_EQTL_PER_TISSUE

    def test_eqtl_multiple_tissues(self):
        ev = GeneEvidence(gene="X", eqtl=True, eqtl_n_tissues=3)
        assert compute_gene_score(ev) == min(3 * SCORE_EQTL_PER_TISSUE, SCORE_EQTL_CAP)

    def test_eqtl_capped(self):
        ev = GeneEvidence(gene="X", eqtl=True, eqtl_n_tissues=10)
        assert compute_gene_score(ev) == SCORE_EQTL_CAP

    def test_hic(self):
        ev = GeneEvidence(gene="X", hic=True)
        assert compute_gene_score(ev) == SCORE_HIC

    def test_pathway_convergence(self):
        ev = GeneEvidence(gene="X", n_convergent_pathways=2)
        assert compute_gene_score(ev) == 2 * SCORE_PATHWAY_PER_HIT

    def test_pathway_capped(self):
        ev = GeneEvidence(gene="X", n_convergent_pathways=10)
        assert compute_gene_score(ev) == SCORE_PATHWAY_CAP

    def test_literature(self):
        ev = GeneEvidence(gene="X", is_known_ts_gene=True)
        assert compute_gene_score(ev) == SCORE_LITERATURE

    def test_all_evidence(self):
        ev = GeneEvidence(
            gene="X",
            positional=True,
            eqtl=True, eqtl_n_tissues=2,
            hic=True,
            n_convergent_pathways=1,
            is_known_ts_gene=True,
        )
        expected = (
            SCORE_POSITIONAL
            + min(2 * SCORE_EQTL_PER_TISSUE, SCORE_EQTL_CAP)
            + SCORE_HIC
            + SCORE_PATHWAY_PER_HIT
            + SCORE_LITERATURE
        )
        assert compute_gene_score(ev) == expected

    def test_no_evidence(self):
        ev = GeneEvidence(gene="X")
        assert compute_gene_score(ev) == 0.0


# --- Locus assignment tests ---


class TestAssignGenesToLoci:
    def test_basic_assignment(self, snp_gene_df):
        loci = assign_genes_to_loci(snp_gene_df)
        assert len(loci) > 0
        # All genes should be assigned
        all_assigned = set()
        for genes in loci.values():
            all_assigned.update(genes)
        input_genes = set(snp_gene_df["GENE"].unique())
        assert input_genes == all_assigned

    def test_empty_input(self):
        loci = assign_genes_to_loci(pd.DataFrame())
        assert loci == {}

    def test_missing_columns(self):
        df = pd.DataFrame({"FOO": [1], "BAR": [2]})
        loci = assign_genes_to_loci(df)
        assert loci == {}

    def test_nearby_genes_merge(self):
        """Genes within merge_distance_kb should be in same locus."""
        df = pd.DataFrame({
            "GENE": ["A", "B"],
            "CHR": [1, 1],
            "BP": [100000, 200000],
            "GENE_START": [95000, 195000],
            "GENE_END": [105000, 205000],
        })
        loci = assign_genes_to_loci(df, merge_distance_kb=1000)
        # Both genes within 1Mb => same locus
        assert len(loci) == 1
        assert set(loci[1]) == {"A", "B"}

    def test_distant_genes_separate(self):
        """Genes far apart should be in different loci."""
        df = pd.DataFrame({
            "GENE": ["A", "B"],
            "CHR": [1, 1],
            "BP": [100000, 100000000],
            "GENE_START": [95000, 99000000],
            "GENE_END": [105000, 101000000],
        })
        loci = assign_genes_to_loci(df, merge_distance_kb=1000)
        assert len(loci) == 2


# --- Evidence building tests ---


class TestBuildGeneEvidence:
    def test_basic_evidence(self, snp_gene_df, gene_results_df):
        evidence = build_gene_evidence(snp_gene_df, gene_results_df)
        assert len(evidence) > 0
        genes = {ev.gene for ev in evidence}
        assert "DRD1" in genes
        assert "BCL11B" in genes

    def test_positional_evidence(self, snp_gene_df):
        evidence = build_gene_evidence(snp_gene_df)
        drd1 = next(ev for ev in evidence if ev.gene == "DRD1")
        assert drd1.positional is True

    def test_eqtl_evidence(self, snp_gene_df):
        evidence = build_gene_evidence(snp_gene_df)
        drd1 = next(ev for ev in evidence if ev.gene == "DRD1")
        assert drd1.eqtl is True
        assert drd1.eqtl_n_tissues == 2
        assert drd1.eqtl_best_p == 1e-6

    def test_hic_evidence(self, snp_gene_df):
        evidence = build_gene_evidence(snp_gene_df)
        bcl11b = next(ev for ev in evidence if ev.gene == "BCL11B")
        assert bcl11b.hic is True

    def test_nrxn1_multi_evidence(self, snp_gene_df):
        """NRXN1 has positional + eQTL + Hi-C = high score."""
        evidence = build_gene_evidence(snp_gene_df)
        nrxn1 = next(ev for ev in evidence if ev.gene == "NRXN1")
        assert nrxn1.positional is True
        assert nrxn1.eqtl is True
        assert nrxn1.hic is True
        assert nrxn1.is_known_ts_gene is True  # in TS_CANDIDATE_GENES
        assert nrxn1.total_score > 0

    def test_magma_hit_flag(self, snp_gene_df):
        evidence = build_gene_evidence(snp_gene_df)
        bcl11b = next(ev for ev in evidence if ev.gene == "BCL11B")
        ndfip2 = next(ev for ev in evidence if ev.gene == "NDFIP2")
        rbm26 = next(ev for ev in evidence if ev.gene == "RBM26")
        assert bcl11b.is_magma_hit is True
        assert ndfip2.is_magma_hit is True
        assert rbm26.is_magma_hit is True

    def test_literature_flag(self, snp_gene_df):
        evidence = build_gene_evidence(snp_gene_df)
        nrxn1 = next(ev for ev in evidence if ev.gene == "NRXN1")
        assert nrxn1.is_known_ts_gene is True  # NRXN1 is in TS_CANDIDATE_GENES
        sema6d = next(ev for ev in evidence if ev.gene == "SEMA6D")
        assert sema6d.is_known_ts_gene is True  # SEMA6D is in TS_CANDIDATE_GENES

    def test_convergence_integration(self, snp_gene_df, convergence_df):
        evidence = build_gene_evidence(snp_gene_df, convergence_df=convergence_df)
        drd1 = next(ev for ev in evidence if ev.gene == "DRD1")
        assert drd1.n_convergent_pathways >= 1
        assert "dopamine_signaling" in drd1.convergent_pathways

    def test_empty_input(self):
        evidence = build_gene_evidence(pd.DataFrame())
        assert evidence == []

    def test_gene_p_integration(self, snp_gene_df, gene_results_df):
        evidence = build_gene_evidence(snp_gene_df, gene_results_df)
        bcl11b = next(ev for ev in evidence if ev.gene == "BCL11B")
        assert bcl11b.magma_gene_p == 0.0005


# --- Ranking tests ---


class TestRankGenesPerLocus:
    def test_ranking_by_score(self):
        evidence = [
            GeneEvidence(gene="A", locus_id=1, total_score=5),
            GeneEvidence(gene="B", locus_id=1, total_score=10),
            GeneEvidence(gene="C", locus_id=1, total_score=3),
        ]
        ranked = rank_genes_per_locus(evidence)
        locus1 = [ev for ev in ranked if ev.locus_id == 1]
        assert locus1[0].gene == "B"  # highest score
        assert locus1[0].rank_in_locus == 1
        assert locus1[1].gene == "A"
        assert locus1[1].rank_in_locus == 2
        assert locus1[2].gene == "C"
        assert locus1[2].rank_in_locus == 3

    def test_tiebreak_by_magma_p(self):
        evidence = [
            GeneEvidence(gene="A", locus_id=1, total_score=5, magma_gene_p=0.01),
            GeneEvidence(gene="B", locus_id=1, total_score=5, magma_gene_p=0.001),
        ]
        ranked = rank_genes_per_locus(evidence)
        locus1 = [ev for ev in ranked if ev.locus_id == 1]
        assert locus1[0].gene == "B"  # lower P wins
        assert locus1[0].rank_in_locus == 1

    def test_multi_locus_independent(self):
        evidence = [
            GeneEvidence(gene="A", locus_id=1, total_score=5),
            GeneEvidence(gene="B", locus_id=2, total_score=10),
            GeneEvidence(gene="C", locus_id=2, total_score=3),
        ]
        ranked = rank_genes_per_locus(evidence)
        a = next(ev for ev in ranked if ev.gene == "A")
        assert a.rank_in_locus == 1  # only gene in locus 1
        b = next(ev for ev in ranked if ev.gene == "B")
        assert b.rank_in_locus == 1  # top in locus 2

    def test_empty_input(self):
        assert rank_genes_per_locus([]) == []


# --- Output tests ---


class TestEvidenceToDataframe:
    def test_basic_conversion(self):
        evidence = [
            GeneEvidence(gene="A", chr=1, locus_id=1, total_score=5, rank_in_locus=1),
            GeneEvidence(gene="B", chr=2, locus_id=2, total_score=3, rank_in_locus=1),
        ]
        df = evidence_to_dataframe(evidence)
        assert len(df) == 2
        assert "GENE" in df.columns
        assert "TOTAL_SCORE" in df.columns
        assert "IS_MAGMA_HIT" in df.columns
        assert df.iloc[0]["GENE"] == "A"
        assert df.iloc[0]["TOTAL_SCORE"] == 5

    def test_empty_input(self):
        df = evidence_to_dataframe([])
        assert df.empty

    def test_inf_distance_becomes_nan(self):
        evidence = [GeneEvidence(gene="X", distance_kb=float("inf"))]
        df = evidence_to_dataframe(evidence)
        assert pd.isna(df.iloc[0]["DISTANCE_KB"])


class TestWriteOutputs:
    def test_write_prioritization(self, tmp_path):
        evidence = [
            GeneEvidence(gene="A", chr=1, locus_id=1, total_score=5, rank_in_locus=1),
        ]
        path = tmp_path / "test.tsv"
        write_prioritization_results(evidence, path)
        assert path.exists()
        df = pd.read_csv(path, sep="\t")
        assert len(df) == 1
        assert df.iloc[0]["GENE"] == "A"

    def test_write_summary(self, tmp_path):
        evidence = [
            GeneEvidence(
                gene="BCL11B", chr=14, locus_id=1, total_score=8,
                rank_in_locus=1, is_magma_hit=True, magma_gene_p=0.0005,
                positional=True, eqtl=True, eqtl_n_tissues=2, hic=True,
            ),
        ]
        path = write_prioritization_summary(evidence, tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "BCL11B" in content
        assert "MAGMA Gene-Based Hits" in content

    def test_write_empty(self, tmp_path):
        path = tmp_path / "empty.tsv"
        write_prioritization_results([], path)
        # Should not crash


# --- Integration test ---


class TestIntegration:
    def test_full_pipeline(self, snp_gene_df, gene_results_df, convergence_df):
        """End-to-end test of evidence building + ranking."""
        loci = assign_genes_to_loci(snp_gene_df, gene_results_df)
        evidence = build_gene_evidence(
            snp_gene_df, gene_results_df, convergence_df, loci,
        )
        evidence = rank_genes_per_locus(evidence)

        # Should have evidence for all genes
        assert len(evidence) == snp_gene_df["GENE"].nunique()

        # Top-scored gene should have rank 1 in its locus
        top = max(evidence, key=lambda e: e.total_score)
        assert top.rank_in_locus == 1

        # NRXN1 should score high (positional + eQTL(3 tissues) + Hi-C + literature)
        nrxn1 = next(ev for ev in evidence if ev.gene == "NRXN1")
        expected_min = SCORE_POSITIONAL + SCORE_EQTL_CAP + SCORE_HIC + SCORE_LITERATURE
        assert nrxn1.total_score >= expected_min

        # MAGMA hits should be flagged
        for gene in MAGMA_GENE_HITS:
            if gene in {ev.gene for ev in evidence}:
                ev = next(e for e in evidence if e.gene == gene)
                assert ev.is_magma_hit is True

    def test_output_round_trip(self, snp_gene_df, gene_results_df, tmp_path):
        """Test that results can be written and read back."""
        evidence = build_gene_evidence(snp_gene_df, gene_results_df)
        evidence = rank_genes_per_locus(evidence)

        path = tmp_path / "results.tsv"
        write_prioritization_results(evidence, path)

        df = pd.read_csv(path, sep="\t")
        assert len(df) == len(evidence)
        assert set(df["GENE"]) == {ev.gene for ev in evidence}
