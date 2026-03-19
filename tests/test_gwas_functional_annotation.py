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
