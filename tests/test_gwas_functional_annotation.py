"""Tests for TS GWAS functional annotation pipeline.

Tests GWAS preprocessing, SNP-to-gene mapping, and integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.tourettes.ts_gwas_functional_annotation.preprocess_gwas import (
    apply_qc_filters,
    clean_chromosome,
    compute_derived_fields,
    compute_qc_summary,
    standardize_columns,
)
from bioagentics.tourettes.ts_gwas_functional_annotation.magma_analysis import (
    GeneResult,
    GeneSetResult,
    gene_analysis,
    gene_set_analysis,
    load_all_gene_sets,
    load_gmt,
    map_snps_to_genes,
    write_gene_results,
    write_gene_set_results,
)
from bioagentics.tourettes.ts_gwas_functional_annotation.snp_to_gene import (
    GeneMapping,
    integrate_mappings,
    load_gene_annotations,
    positional_mapping,
)


# --- Fixtures ---


@pytest.fixture
def raw_gwas_df():
    """Minimal GWAS DataFrame with non-standard column names."""
    return pd.DataFrame({
        "RSID": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "CHROMOSOME": [1, 1, 2, 2, 3],
        "POSITION": [100000, 200000, 150000, 300000, 50000],
        "PVAL": [1e-8, 0.05, 1e-5, 0.5, 1e-3],
        "EFFECT": [0.3, -0.1, 0.2, 0.01, -0.15],
        "STDERR": [0.05, 0.08, 0.04, 0.05, 0.06],
        "MAF": [0.15, 0.02, 0.30, 0.005, 0.10],
        "INFO": [0.95, 0.80, 0.70, 0.50, 0.90],
    })


@pytest.fixture
def std_gwas_df():
    """GWAS DataFrame with standard column names."""
    return pd.DataFrame({
        "SNP": ["rs1", "rs2", "rs3", "rs4", "rs5"],
        "CHR": [1, 1, 2, 2, 3],
        "BP": [100000, 200000, 150000, 300000, 50000],
        "P": [1e-8, 0.05, 1e-5, 0.5, 1e-3],
        "BETA": [0.3, -0.1, 0.2, 0.01, -0.15],
        "SE": [0.05, 0.08, 0.04, 0.05, 0.06],
        "FRQ": [0.15, 0.02, 0.30, 0.005, 0.10],
        "INFO": [0.95, 0.80, 0.70, 0.50, 0.90],
    })


@pytest.fixture
def gene_annot_df():
    """Gene annotation DataFrame."""
    return pd.DataFrame({
        "GENE": ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
        "CHR": [1, 1, 2, 3],
        "START": [95000, 250000, 140000, 45000],
        "STOP": [105000, 260000, 160000, 55000],
    })


# --- Preprocessing Tests ---


class TestStandardizeColumns:
    def test_renames_common_formats(self, raw_gwas_df):
        df = standardize_columns(raw_gwas_df)
        assert "SNP" in df.columns
        assert "CHR" in df.columns
        assert "BP" in df.columns
        assert "P" in df.columns
        assert "BETA" in df.columns
        assert "SE" in df.columns

    def test_preserves_standard_columns(self, std_gwas_df):
        df = standardize_columns(std_gwas_df)
        assert list(df.columns) == list(std_gwas_df.columns)


class TestCleanChromosome:
    def test_removes_chr_prefix(self):
        df = pd.DataFrame({"CHR": ["chr1", "chr2", "chrX"], "P": [0.1, 0.2, 0.3]})
        result = clean_chromosome(df)
        assert list(result["CHR"]) == [1, 2]  # X removed (autosomes only)

    def test_keeps_autosomes_only(self):
        df = pd.DataFrame({"CHR": [1, 2, 23, 24], "P": [0.1, 0.2, 0.3, 0.4]})
        result = clean_chromosome(df)
        assert len(result) == 2
        assert set(result["CHR"]) == {1, 2}


class TestComputeDerivedFields:
    def test_z_from_beta_se(self):
        df = pd.DataFrame({"BETA": [0.5, -0.3], "SE": [0.1, 0.1]})
        result = compute_derived_fields(df)
        assert "Z" in result.columns
        assert np.isclose(result["Z"].iloc[0], 5.0)
        assert np.isclose(result["Z"].iloc[1], -3.0)

    def test_p_from_z(self):
        df = pd.DataFrame({"Z": [5.0, -3.0]})
        result = compute_derived_fields(df)
        assert "P" in result.columns
        assert result["P"].iloc[0] < 1e-5
        assert result["P"].iloc[1] < 0.01

    def test_beta_from_or(self):
        df = pd.DataFrame({"OR": [2.0, 0.5], "P": [0.01, 0.05]})
        result = compute_derived_fields(df)
        assert "BETA" in result.columns
        assert np.isclose(result["BETA"].iloc[0], np.log(2.0))


class TestApplyQCFilters:
    def test_removes_low_maf(self, std_gwas_df):
        result = apply_qc_filters(std_gwas_df, min_maf=0.01)
        # rs4 has MAF=0.005, should be removed
        assert "rs4" not in result["SNP"].values
        assert len(result) == 4

    def test_removes_low_info(self, std_gwas_df):
        result = apply_qc_filters(std_gwas_df, min_info=0.6)
        # rs4 has INFO=0.50, should be removed
        assert "rs4" not in result["SNP"].values

    def test_removes_duplicates_keeps_lowest_p(self):
        df = pd.DataFrame({
            "SNP": ["rs1", "rs1", "rs2"],
            "P": [0.05, 0.01, 0.1],
            "CHR": [1, 1, 2],
        })
        result = apply_qc_filters(df)
        assert len(result[result["SNP"] == "rs1"]) == 1
        assert result[result["SNP"] == "rs1"]["P"].iloc[0] == 0.01


class TestComputeQCSummary:
    def test_summary_fields(self, std_gwas_df):
        summary = compute_qc_summary(std_gwas_df)
        assert summary["n_snps"] == 5
        assert summary["n_chromosomes"] == 3
        assert summary["n_gw_sig"] == 1  # rs1 at 1e-8
        assert summary["n_suggestive"] >= 1  # at least rs1
        assert "lambda_gc" in summary

    def test_lambda_gc_near_one_for_uniform(self):
        np.random.seed(42)
        df = pd.DataFrame({
            "P": np.random.uniform(0, 1, 10000),
            "CHR": np.ones(10000, dtype=int),
        })
        summary = compute_qc_summary(df)
        assert 0.9 < summary["lambda_gc"] < 1.1


# --- SNP-to-Gene Mapping Tests ---


class TestLoadGeneAnnotations:
    def test_returns_empty_for_missing_file(self, tmp_path):
        path = tmp_path / "nonexistent.tsv"
        result = load_gene_annotations(path)
        assert result.empty
        assert set(result.columns) == {"GENE", "CHR", "START", "STOP"}

    def test_loads_valid_file(self, tmp_path):
        path = tmp_path / "genes.tsv"
        df = pd.DataFrame({
            "GENE": ["A", "B"],
            "CHR": [1, 2],
            "START": [100, 200],
            "STOP": [500, 600],
        })
        df.to_csv(path, sep="\t", index=False)
        result = load_gene_annotations(path)
        assert len(result) == 2

    def test_deduplicates_genes(self, tmp_path):
        path = tmp_path / "genes.tsv"
        df = pd.DataFrame({
            "GENE": ["A", "A", "B"],
            "CHR": [1, 1, 2],
            "START": [100, 100, 200],
            "STOP": [500, 300, 600],  # First A is longer
        })
        df.to_csv(path, sep="\t", index=False)
        result = load_gene_annotations(path)
        # Should keep longest span for gene A
        assert len(result[result["GENE"] == "A"]) == 1
        a_row = result[result["GENE"] == "A"].iloc[0]
        assert a_row["STOP"] == 500


class TestPositionalMapping:
    def test_maps_snp_within_gene(self, std_gwas_df, gene_annot_df):
        mappings = positional_mapping(std_gwas_df, gene_annot_df, window_kb=10)
        # rs1 at chr1:100000 should map to GENE_A (95000-105000)
        gene_a_maps = [m for m in mappings if m.gene == "GENE_A" and m.snp == "rs1"]
        assert len(gene_a_maps) == 1
        assert gene_a_maps[0].distance_kb == 0.0  # within gene body
        assert gene_a_maps[0].positional is True

    def test_maps_snp_within_window(self, std_gwas_df, gene_annot_df):
        mappings = positional_mapping(std_gwas_df, gene_annot_df, window_kb=10)
        # rs3 at chr2:150000 should map to GENE_C (140000-160000)
        gene_c_maps = [m for m in mappings if m.gene == "GENE_C" and m.snp == "rs3"]
        assert len(gene_c_maps) == 1

    def test_no_mapping_outside_window(self, std_gwas_df, gene_annot_df):
        mappings = positional_mapping(std_gwas_df, gene_annot_df, window_kb=1)
        # rs2 at chr1:200000 is far from GENE_A (95k-105k) and GENE_B (250k-260k)
        gene_b_rs2 = [m for m in mappings if m.gene == "GENE_B" and m.snp == "rs2"]
        assert len(gene_b_rs2) == 0

    def test_empty_annotations(self, std_gwas_df):
        empty = pd.DataFrame(columns=["GENE", "CHR", "START", "STOP"])
        mappings = positional_mapping(std_gwas_df, empty)
        assert len(mappings) == 0


class TestIntegrateMappings:
    def test_merges_multi_modal_evidence(self):
        pos = [GeneMapping(
            snp="rs1", gene="GENE_A", chr=1, snp_bp=100000,
            gene_start=95000, gene_end=105000, distance_kb=0.0,
            positional=True,
        )]
        eqtl = [GeneMapping(
            snp="rs1", gene="GENE_A", chr=1, snp_bp=100000,
            gene_start=95000, gene_end=105000, distance_kb=0.0,
            eqtl=True, eqtl_tissues=["Brain_Caudate_basal_ganglia"],
            eqtl_best_p=1e-6,
        )]
        hic = []

        result = integrate_mappings(pos, eqtl, hic)
        assert len(result) == 1
        row = result.iloc[0]
        assert bool(row["POSITIONAL"]) is True
        assert bool(row["EQTL"]) is True
        assert bool(row["HIC"]) is False
        assert row["N_EVIDENCE"] == 2

    def test_separate_snp_gene_pairs(self):
        pos = [
            GeneMapping(snp="rs1", gene="GENE_A", chr=1, snp_bp=100000,
                       gene_start=95000, gene_end=105000, distance_kb=0.0,
                       positional=True),
            GeneMapping(snp="rs1", gene="GENE_B", chr=1, snp_bp=100000,
                       gene_start=250000, gene_end=260000, distance_kb=145.0,
                       positional=True),
        ]
        result = integrate_mappings(pos, [], [])
        assert len(result) == 2

    def test_candidate_gene_flagged(self):
        pos = [GeneMapping(
            snp="rs1", gene="FLT3", chr=1, snp_bp=100000,
            gene_start=95000, gene_end=105000, distance_kb=0.0,
            positional=True,
        )]
        result = integrate_mappings(pos, [], [])
        assert bool(result.iloc[0]["IS_CANDIDATE"]) is True

    def test_empty_input(self):
        result = integrate_mappings([], [], [])
        assert result.empty


# --- MAGMA Analysis Tests ---


@pytest.fixture
def magma_gwas_df():
    """GWAS DataFrame with multiple SNPs per gene region for MAGMA testing."""
    return pd.DataFrame({
        "SNP": [f"rs{i}" for i in range(1, 21)],
        "CHR": [1]*5 + [1]*5 + [2]*5 + [3]*5,
        "BP": (
            [100000, 101000, 102000, 103000, 104000]  # near GENE_A
            + [500000, 501000, 502000, 503000, 504000]  # near GENE_B
            + [200000, 201000, 202000, 203000, 204000]  # near GENE_C
            + [300000, 301000, 302000, 303000, 304000]  # near GENE_D
        ),
        "P": (
            [1e-8, 1e-6, 1e-4, 0.01, 0.1]  # GENE_A: strong signal
            + [0.5, 0.3, 0.8, 0.6, 0.9]  # GENE_B: no signal
            + [1e-5, 1e-3, 0.05, 0.2, 0.4]  # GENE_C: moderate signal
            + [0.01, 0.02, 0.05, 0.1, 0.3]  # GENE_D: weak signal
        ),
        "BETA": [0.3, 0.2, 0.1, 0.05, 0.01] * 4,
        "SE": [0.05] * 20,
    })


@pytest.fixture
def magma_gene_annot():
    """Gene annotations for MAGMA testing."""
    return pd.DataFrame({
        "GENE": ["GENE_A", "GENE_B", "GENE_C", "GENE_D"],
        "CHR": [1, 1, 2, 3],
        "START": [99000, 499000, 199000, 299000],
        "STOP": [105000, 505000, 205000, 305000],
    })


class TestMapSnpsToGenes:
    def test_maps_snps_within_window(self, magma_gwas_df, magma_gene_annot):
        gene_snps = map_snps_to_genes(magma_gwas_df, magma_gene_annot, window_kb=10)
        assert "GENE_A" in gene_snps
        assert len(gene_snps["GENE_A"]) == 5

    def test_all_genes_mapped(self, magma_gwas_df, magma_gene_annot):
        gene_snps = map_snps_to_genes(magma_gwas_df, magma_gene_annot, window_kb=10)
        assert len(gene_snps) == 4

    def test_empty_annotations(self, magma_gwas_df):
        empty = pd.DataFrame(columns=["GENE", "CHR", "START", "STOP"])
        gene_snps = map_snps_to_genes(magma_gwas_df, empty)
        assert len(gene_snps) == 0


class TestGeneAnalysis:
    def test_returns_sorted_by_pvalue(self, magma_gwas_df, magma_gene_annot):
        results = gene_analysis(magma_gwas_df, magma_gene_annot)
        p_values = [r.p_value for r in results]
        assert p_values == sorted(p_values)

    def test_gene_a_most_significant(self, magma_gwas_df, magma_gene_annot):
        results = gene_analysis(magma_gwas_df, magma_gene_annot)
        assert results[0].gene == "GENE_A"
        assert results[0].p_value < 0.01

    def test_gene_b_least_significant(self, magma_gwas_df, magma_gene_annot):
        results = gene_analysis(magma_gwas_df, magma_gene_annot)
        gene_b = [r for r in results if r.gene == "GENE_B"][0]
        assert gene_b.p_value > 0.1

    def test_top_snp_recorded(self, magma_gwas_df, magma_gene_annot):
        results = gene_analysis(magma_gwas_df, magma_gene_annot)
        gene_a = results[0]
        assert gene_a.top_snp == "rs1"
        assert gene_a.top_snp_p == pytest.approx(1e-8)

    def test_n_snps_correct(self, magma_gwas_df, magma_gene_annot):
        results = gene_analysis(magma_gwas_df, magma_gene_annot)
        for r in results:
            assert r.n_snps == 5

    def test_candidate_flagged(self, magma_gwas_df):
        annot = pd.DataFrame({
            "GENE": ["BCL11B"],
            "CHR": [1],
            "START": [99000],
            "STOP": [105000],
        })
        results = gene_analysis(magma_gwas_df, annot)
        if results:
            assert results[0].is_candidate is True

    def test_empty_gwas(self, magma_gene_annot):
        empty = pd.DataFrame(columns=["SNP", "CHR", "BP", "P"])
        results = gene_analysis(empty, magma_gene_annot)
        assert len(results) == 0


class TestGeneSetAnalysis:
    def _make_gene_results(self):
        """Create gene results with known Z-score pattern."""
        genes_high = ["G1", "G2", "G3", "G4", "G5"]
        genes_low = [f"BG{i}" for i in range(50)]
        results = []
        for g in genes_high:
            results.append(GeneResult(
                gene=g, chr=1, start=0, stop=1000, n_snps=10,
                top_snp="rs1", top_snp_p=1e-6, stat=30.0,
                p_value=1e-4, z_score=3.7,
            ))
        for g in genes_low:
            results.append(GeneResult(
                gene=g, chr=1, start=0, stop=1000, n_snps=10,
                top_snp="rs1", top_snp_p=0.5, stat=5.0,
                p_value=0.5, z_score=0.0,
            ))
        return results

    def test_enriched_set_detected(self):
        gene_results = self._make_gene_results()
        gene_sets = {
            "enriched_set": ("builtin", ["G1", "G2", "G3", "G4", "G5"]),
            "random_set": ("builtin", ["BG0", "BG1", "BG2", "BG3", "BG4"]),
        }
        gs_results = gene_set_analysis(gene_results, gene_sets)
        enriched = [r for r in gs_results if r.gene_set == "enriched_set"][0]
        random_r = [r for r in gs_results if r.gene_set == "random_set"][0]
        assert enriched.p_value < random_r.p_value
        assert enriched.beta > 0

    def test_fdr_correction_applied(self):
        gene_results = self._make_gene_results()
        gene_sets = {
            f"set_{i}": ("builtin", [f"BG{j}" for j in range(i*5, i*5+5)])
            for i in range(10)
        }
        gs_results = gene_set_analysis(gene_results, gene_sets)
        for r in gs_results:
            assert r.fdr_q >= r.p_value  # FDR >= raw p

    def test_fdr_monotonic(self):
        gene_results = self._make_gene_results()
        gene_sets = {
            f"set_{i}": ("builtin", [f"BG{j}" for j in range(i*5, i*5+5)])
            for i in range(10)
        }
        gs_results = gene_set_analysis(gene_results, gene_sets)
        fdr_values = [r.fdr_q for r in gs_results]
        # After sorting by p-value, FDR should be non-decreasing
        for i in range(len(fdr_values) - 1):
            assert fdr_values[i] <= fdr_values[i + 1] + 1e-10

    def test_min_genes_filter(self):
        gene_results = self._make_gene_results()
        gene_sets = {"tiny_set": ("builtin", ["G1", "G2"])}  # only 2 genes
        gs_results = gene_set_analysis(gene_results, gene_sets, min_genes=5)
        assert len(gs_results) == 0

    def test_top_genes_reported(self):
        gene_results = self._make_gene_results()
        gene_sets = {"test_set": ("builtin", ["G1", "G2", "G3", "G4", "G5"])}
        gs_results = gene_set_analysis(gene_results, gene_sets)
        assert len(gs_results[0].top_genes) > 0

    def test_empty_gene_results(self):
        gs_results = gene_set_analysis([], {"set1": ("builtin", ["A", "B", "C"])})
        assert len(gs_results) == 0


class TestLoadGMT:
    def test_loads_gmt_file(self, tmp_path):
        gmt_path = tmp_path / "test.gmt"
        gmt_path.write_text(
            "SET_A\tdescription\tGENE1\tGENE2\tGENE3\n"
            "SET_B\tdescription\tGENE4\tGENE5\n"
        )
        gene_sets = load_gmt(gmt_path)
        assert "SET_A" in gene_sets
        assert len(gene_sets["SET_A"]) == 3
        assert "SET_B" in gene_sets
        assert len(gene_sets["SET_B"]) == 2

    def test_skips_short_lines(self, tmp_path):
        gmt_path = tmp_path / "test.gmt"
        gmt_path.write_text("SHORT\tdesc\n")  # too few fields
        gene_sets = load_gmt(gmt_path)
        assert len(gene_sets) == 0


class TestLoadAllGeneSets:
    def test_loads_builtin_when_no_dir(self, tmp_path):
        result = load_all_gene_sets(tmp_path / "nonexistent")
        # Should have built-in sets
        assert len(result) > 0
        assert "dopamine_signaling" in result
        source, genes = result["dopamine_signaling"]
        assert source == "builtin"
        assert "DRD1" in genes

    def test_loads_gmt_from_dir(self, tmp_path):
        gmt_path = tmp_path / "c2_curated.gmt"
        gmt_path.write_text("MY_PATHWAY\tdesc\tGENE1\tGENE2\tGENE3\n")
        result = load_all_gene_sets(tmp_path)
        assert "MY_PATHWAY" in result
        source, _ = result["MY_PATHWAY"]
        assert source == "msigdb_c2"


class TestWriteResults:
    def test_write_gene_results(self, tmp_path):
        results = [GeneResult(
            gene="GENE_A", chr=1, start=100, stop=200, n_snps=10,
            top_snp="rs1", top_snp_p=1e-8, stat=50.0,
            p_value=1e-5, z_score=4.2,
        )]
        path = write_gene_results(results, tmp_path)
        assert path.exists()
        df = pd.read_csv(path, sep="\t")
        assert len(df) == 1
        assert df.iloc[0]["GENE"] == "GENE_A"

    def test_write_gene_set_results(self, tmp_path):
        results = [GeneSetResult(
            gene_set="test_set", source="builtin",
            n_genes_defined=10, n_genes_tested=8,
            beta=0.5, beta_se=0.1, z_score=5.0,
            p_value=1e-6, fdr_q=1e-5, top_genes=["A", "B"],
        )]
        path = write_gene_set_results(results, tmp_path)
        assert path.exists()
        df = pd.read_csv(path, sep="\t")
        assert len(df) == 1
        assert df.iloc[0]["GENE_SET"] == "test_set"
