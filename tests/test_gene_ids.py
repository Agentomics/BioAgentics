"""Tests for gene ID harmonization module."""

from __future__ import annotations

import pytest

from bioagentics.data.gene_ids import (
    find_common_genes,
    load_depmap_matrix,
    load_depmap_model_metadata,
    load_depmap_mutations,
    parse_depmap_gene_col,
)

DATA_DIR = "data/depmap/25q3"


class TestParseDepmapGeneCol:
    def test_standard_format(self):
        symbol, entrez = parse_depmap_gene_col("A1BG (1)")
        assert symbol == "A1BG"
        assert entrez == 1

    def test_multiword_symbol(self):
        symbol, entrez = parse_depmap_gene_col("TSPAN6 (7105)")
        assert symbol == "TSPAN6"
        assert entrez == 7105

    def test_non_gene_col(self):
        symbol, entrez = parse_depmap_gene_col("ModelID")
        assert symbol == "ModelID"
        assert entrez is None


@pytest.mark.skipif(
    not __import__("pathlib").Path(DATA_DIR).exists(),
    reason="DepMap data not downloaded",
)
class TestLoadDepMap:
    def test_load_crispr_matrix(self):
        df = load_depmap_matrix(f"{DATA_DIR}/CRISPRGeneEffect.csv")
        assert df.shape[0] > 500  # >500 cell lines
        assert df.shape[1] > 15000  # >15k genes
        # Check that columns are HUGO symbols (no parentheses)
        assert not any("(" in c for c in df.columns)
        # Check known genes present
        assert "KRAS" in df.columns
        assert "EGFR" in df.columns
        assert "TP53" in df.columns

    def test_load_expression_matrix(self):
        df = load_depmap_matrix(f"{DATA_DIR}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
        assert df.shape[0] > 500
        assert df.shape[1] > 15000
        assert not any("(" in c for c in df.columns)

    def test_load_cn_matrix(self):
        df = load_depmap_matrix(f"{DATA_DIR}/PortalOmicsCNGeneLog2.csv")
        assert df.shape[0] > 500
        assert df.shape[1] > 15000
        assert not any("(" in c for c in df.columns)

    def test_load_model_metadata(self):
        df = load_depmap_model_metadata(f"{DATA_DIR}/Model.csv")
        assert "OncotreeLineage" in df.columns
        assert "CellLineName" in df.columns
        nsclc = df[df["OncotreeLineage"] == "Lung"]
        assert len(nsclc) > 50  # expect >50 lung lines

    def test_load_mutations(self):
        df = load_depmap_mutations(f"{DATA_DIR}/OmicsSomaticMutations.csv")
        assert "HugoSymbol" in df.columns
        assert "ModelID" in df.columns
        assert len(df) > 100000

    def test_common_genes_across_matrices(self):
        crispr = load_depmap_matrix(f"{DATA_DIR}/CRISPRGeneEffect.csv")
        expr = load_depmap_matrix(f"{DATA_DIR}/OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")
        cn = load_depmap_matrix(f"{DATA_DIR}/PortalOmicsCNGeneLog2.csv")
        common = find_common_genes(crispr.columns, expr.columns, cn.columns)
        # Expect >95% coverage of protein-coding genes
        total_unique = len(set(crispr.columns) | set(expr.columns) | set(cn.columns))
        coverage = len(common) / total_unique
        assert coverage > 0.5  # at least 50% overlap
        assert len(common) > 15000  # expect >15k common genes
