"""Tests for pathway convergence analysis module."""

from __future__ import annotations

import pandas as pd
import pytest

from bioagentics.tourettes.ts_gwas_functional_annotation.pathway_convergence import (
    DomainConvergenceResult,
    Locus,
    PathwayConvergenceResult,
    _classify_domain,
    define_loci,
    domain_convergence,
    pathway_convergence_analysis,
    write_convergence_results,
    write_convergence_summary,
    write_domain_results,
)


# --- Fixtures ---


@pytest.fixture
def gene_results_df():
    """Gene-level results spanning 3 chromosomes / 4 loci."""
    return pd.DataFrame({
        "GENE": [
            "DRD1", "DRD2", "NRXN1", "SLIT2",
            "GAD1", "GRIN2A", "SEMA6D", "BCL11B",
            "FOXP2", "TH",
        ],
        "CHR": [5, 11, 2, 4, 2, 16, 15, 14, 7, 11],
        "START": [
            174800000, 113280000, 50000000, 86000000,
            170800000, 9900000, 46000000, 99600000,
            114000000, 113200000,
        ],
        "STOP": [
            174900000, 113400000, 51000000, 86100000,
            170900000, 10100000, 46200000, 99700000,
            114200000, 113250000,
        ],
        "P": [0.001, 0.01, 0.005, 0.04, 0.02, 0.03, 0.008, 0.015, 0.025, 0.012],
    })


@pytest.fixture
def gene_set_results_df():
    """MAGMA gene-set enrichment results."""
    return pd.DataFrame({
        "GENE_SET": [
            "dopamine_signaling", "synaptic_adhesion", "axon_guidance",
            "neurodevelopment", "unrelated_pathway",
        ],
        "SOURCE": ["builtin"] * 5,
        "P": [0.002, 0.01, 0.03, 0.015, 0.8],
        "FDR_Q": [0.01, 0.04, 0.10, 0.06, 0.95],
    })


@pytest.fixture
def gene_sets():
    """Gene set definitions matching test genes."""
    return {
        "dopamine_signaling": ("builtin", ["DRD1", "DRD2", "TH", "SLC6A3", "COMT"]),
        "synaptic_adhesion": ("builtin", ["NRXN1", "NLGN1", "CNTN6", "SLITRK1"]),
        "axon_guidance": ("builtin", ["SLIT2", "SEMA6D", "NTN4", "ROBO1"]),
        "neurodevelopment": ("builtin", ["BCL11B", "FOXP2", "TBR1", "SATB2"]),
        "unrelated_pathway": ("custom", ["ZZZ1", "ZZZ2", "ZZZ3"]),
    }


# --- Locus definition tests ---


class TestDefineLoci:
    def test_basic_locus_definition(self, gene_results_df):
        loci = define_loci(gene_results_df, p_threshold=0.05)
        assert len(loci) > 0
        for loc in loci:
            assert len(loc.genes) >= 1
            assert loc.lead_gene_p > 0

    def test_all_genes_assigned(self, gene_results_df):
        loci = define_loci(gene_results_df, p_threshold=0.05)
        all_genes = set()
        for loc in loci:
            all_genes.update(loc.genes)
        sig_genes = set(gene_results_df[gene_results_df["P"] < 0.05]["GENE"])
        assert all_genes == sig_genes

    def test_nearby_genes_merge(self):
        """Genes within merge distance on same chr should form one locus."""
        df = pd.DataFrame({
            "GENE": ["A", "B"],
            "CHR": [1, 1],
            "START": [100000, 200000],
            "STOP": [110000, 210000],
            "P": [0.01, 0.02],
        })
        loci = define_loci(df, p_threshold=0.05, merge_distance_kb=200)
        assert len(loci) == 1
        assert set(loci[0].genes) == {"A", "B"}

    def test_distant_genes_separate(self):
        """Genes far apart should form separate loci."""
        df = pd.DataFrame({
            "GENE": ["A", "B"],
            "CHR": [1, 1],
            "START": [1000000, 50000000],
            "STOP": [1100000, 50100000],
            "P": [0.01, 0.02],
        })
        loci = define_loci(df, p_threshold=0.05, merge_distance_kb=1000)
        assert len(loci) == 2

    def test_different_chromosomes(self):
        """Genes on different chromosomes are always separate loci."""
        df = pd.DataFrame({
            "GENE": ["A", "B"],
            "CHR": [1, 2],
            "START": [100000, 100000],
            "STOP": [110000, 110000],
            "P": [0.01, 0.02],
        })
        loci = define_loci(df, p_threshold=0.05)
        assert len(loci) == 2

    def test_p_threshold_filters(self, gene_results_df):
        """Stricter threshold should yield fewer genes in loci."""
        loci_loose = define_loci(gene_results_df, p_threshold=0.05)
        loci_strict = define_loci(gene_results_df, p_threshold=0.005)
        genes_loose = sum(len(loc.genes) for loc in loci_loose)
        genes_strict = sum(len(loc.genes) for loc in loci_strict)
        assert genes_strict <= genes_loose

    def test_empty_input(self):
        df = pd.DataFrame(columns=["GENE", "CHR", "START", "STOP", "P"])
        loci = define_loci(df, p_threshold=0.05)
        assert loci == []

    def test_no_significant_genes(self):
        df = pd.DataFrame({
            "GENE": ["A"], "CHR": [1], "START": [100], "STOP": [200], "P": [0.9],
        })
        loci = define_loci(df, p_threshold=0.05)
        assert loci == []

    def test_lead_gene_is_most_significant(self):
        df = pd.DataFrame({
            "GENE": ["A", "B", "C"],
            "CHR": [1, 1, 1],
            "START": [100000, 150000, 200000],
            "STOP": [110000, 160000, 210000],
            "P": [0.03, 0.001, 0.02],
        })
        loci = define_loci(df, p_threshold=0.05, merge_distance_kb=200)
        assert len(loci) == 1
        assert loci[0].lead_gene == "B"
        assert loci[0].lead_gene_p == pytest.approx(0.001)

    def test_missing_columns(self):
        df = pd.DataFrame({"GENE": ["A"], "P": [0.01]})
        loci = define_loci(df, p_threshold=0.05)
        assert loci == []


# --- Domain classification tests ---


class TestClassifyDomain:
    def test_dopamine(self):
        assert _classify_domain("dopamine_signaling") == "dopaminergic_signaling"

    def test_synaptic(self):
        assert _classify_domain("synaptic_adhesion_molecules") == "synaptic_adhesion"

    def test_axon(self):
        assert _classify_domain("axon_guidance_signaling") == "axon_guidance"

    def test_neurodevelopment(self):
        assert _classify_domain("neurodevelopment_processes") == "neurodevelopment"

    def test_gaba(self):
        assert _classify_domain("GABA_ergic_transmission") == "gaba_glutamate"

    def test_unknown(self):
        assert _classify_domain("completely_unrelated") == "other"


# --- Pathway convergence tests ---


class TestPathwayConvergence:
    def test_basic_convergence(self, gene_results_df, gene_set_results_df, gene_sets):
        loci = define_loci(gene_results_df, p_threshold=0.05)
        results = pathway_convergence_analysis(
            loci, gene_set_results_df, gene_sets, enrichment_p_threshold=0.05,
        )
        assert len(results) > 0
        for r in results:
            assert r.n_loci_hit >= 1
            assert r.n_loci_hit <= r.n_loci_total
            assert 0 <= r.convergence_p <= 1
            assert 0 <= r.convergence_fdr <= 1

    def test_multi_loci_pathway_detected(self, gene_results_df, gene_set_results_df, gene_sets):
        """Dopamine signaling has genes on chr5 and chr11 — should hit 2 loci."""
        loci = define_loci(gene_results_df, p_threshold=0.05)
        results = pathway_convergence_analysis(
            loci, gene_set_results_df, gene_sets, enrichment_p_threshold=0.05,
        )
        dopamine = [r for r in results if r.pathway == "dopamine_signaling"]
        assert len(dopamine) == 1
        assert dopamine[0].n_loci_hit >= 2

    def test_fdr_correction_applied(self, gene_results_df, gene_set_results_df, gene_sets):
        loci = define_loci(gene_results_df, p_threshold=0.05)
        results = pathway_convergence_analysis(
            loci, gene_set_results_df, gene_sets, enrichment_p_threshold=0.05,
        )
        if len(results) >= 2:
            # FDR should be >= raw P for most results
            for r in results:
                assert r.convergence_fdr >= r.convergence_p - 1e-10

    def test_unenriched_pathways_excluded(
        self, gene_results_df, gene_set_results_df, gene_sets,
    ):
        """Pathways with P > threshold should be excluded."""
        loci = define_loci(gene_results_df, p_threshold=0.05)
        results = pathway_convergence_analysis(
            loci, gene_set_results_df, gene_sets, enrichment_p_threshold=0.05,
        )
        names = {r.pathway for r in results}
        assert "unrelated_pathway" not in names

    def test_empty_loci(self, gene_set_results_df, gene_sets):
        results = pathway_convergence_analysis(
            [], gene_set_results_df, gene_sets,
        )
        assert results == []

    def test_empty_gene_sets(self, gene_results_df, gene_set_results_df):
        loci = define_loci(gene_results_df, p_threshold=0.05)
        results = pathway_convergence_analysis(
            loci, gene_set_results_df, {},
        )
        assert results == []

    def test_domain_assigned(self, gene_results_df, gene_set_results_df, gene_sets):
        loci = define_loci(gene_results_df, p_threshold=0.05)
        results = pathway_convergence_analysis(
            loci, gene_set_results_df, gene_sets, enrichment_p_threshold=0.05,
        )
        for r in results:
            assert r.domain != ""

    def test_enrichment_p_carried_through(
        self, gene_results_df, gene_set_results_df, gene_sets,
    ):
        loci = define_loci(gene_results_df, p_threshold=0.05)
        results = pathway_convergence_analysis(
            loci, gene_set_results_df, gene_sets, enrichment_p_threshold=0.05,
        )
        dopamine = [r for r in results if r.pathway == "dopamine_signaling"]
        if dopamine:
            assert dopamine[0].enrichment_p == pytest.approx(0.002)


# --- Domain convergence tests ---


class TestDomainConvergence:
    def test_basic_domain(self, gene_results_df, gene_set_results_df, gene_sets):
        loci = define_loci(gene_results_df, p_threshold=0.05)
        pw_results = pathway_convergence_analysis(
            loci, gene_set_results_df, gene_sets, enrichment_p_threshold=0.05,
        )
        dom_results = domain_convergence(pw_results)
        assert len(dom_results) > 0
        for d in dom_results:
            assert d.n_pathways >= 1
            assert 0 <= d.fisher_p <= 1

    def test_empty_input(self):
        assert domain_convergence([]) == []

    def test_fdr_monotone(self, gene_results_df, gene_set_results_df, gene_sets):
        loci = define_loci(gene_results_df, p_threshold=0.05)
        pw_results = pathway_convergence_analysis(
            loci, gene_set_results_df, gene_sets, enrichment_p_threshold=0.05,
        )
        dom_results = domain_convergence(pw_results)
        fdrs = [d.fisher_fdr for d in dom_results]
        for i in range(len(fdrs) - 1):
            assert fdrs[i] <= fdrs[i + 1] + 1e-10


# --- Output tests ---


class TestOutputWriters:
    def test_write_convergence_results(self, tmp_path):
        results = [
            PathwayConvergenceResult(
                pathway="test_pathway",
                source="builtin",
                n_loci_total=5,
                n_loci_hit=3,
                loci_hit=[1, 2, 4],
                genes_per_locus={1: ["A"], 2: ["B"], 4: ["C"]},
                n_pathway_genes_in_data=3,
                convergence_p=0.01,
                convergence_fdr=0.05,
                domain="dopaminergic_signaling",
                enrichment_p=0.001,
                enrichment_fdr=0.01,
            ),
        ]
        path = write_convergence_results(results, tmp_path)
        assert path.exists()
        df = pd.read_csv(path, sep="\t")
        assert len(df) == 1
        assert df.iloc[0]["PATHWAY"] == "test_pathway"
        assert df.iloc[0]["N_LOCI_HIT"] == 3

    def test_write_domain_results(self, tmp_path):
        results = [
            DomainConvergenceResult(
                domain="dopaminergic_signaling",
                n_pathways=3,
                n_pathways_convergent=2,
                median_loci_hit=2.5,
                top_pathways=["a", "b"],
                fisher_p=0.01,
                fisher_fdr=0.03,
            ),
        ]
        path = write_domain_results(results, tmp_path)
        assert path.exists()
        df = pd.read_csv(path, sep="\t")
        assert len(df) == 1
        assert df.iloc[0]["DOMAIN"] == "dopaminergic_signaling"

    def test_write_summary(self, tmp_path):
        loci = [
            Locus(1, 1, 100, 200, ["A", "B"], "A", 0.001),
        ]
        pw_results = [
            PathwayConvergenceResult(
                pathway="test", source="builtin", n_loci_total=1,
                n_loci_hit=1, loci_hit=[1],
                genes_per_locus={1: ["A"]},
                n_pathway_genes_in_data=1,
                convergence_p=0.05, convergence_fdr=0.1,
                domain="other", enrichment_p=0.01,
            ),
        ]
        dom_results = [
            DomainConvergenceResult(
                domain="other", n_pathways=1, n_pathways_convergent=0,
                median_loci_hit=1.0, top_pathways=["test"],
                fisher_p=0.05, fisher_fdr=0.1,
            ),
        ]
        path = write_convergence_summary(pw_results, dom_results, loci, tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "Pathway Convergence Summary" in content
        assert "test" in content
