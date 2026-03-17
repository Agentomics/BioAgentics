"""Tests for PANS neuroinflammation cross-reference module."""

import pandas as pd

from bioagentics.models.pans_neuroinflammation_xref import (
    DDR_HIGHLIGHT_GENES,
    classify_de_status,
    compute_overlap_enrichment,
    cross_reference,
)


def test_classify_de_status():
    assert classify_de_status(2.0, 0.001) == "upregulated"
    assert classify_de_status(-1.5, 0.01) == "downregulated"
    assert classify_de_status(0.1, 0.001) == "unchanged"
    assert classify_de_status(2.0, 0.1) == "unchanged"  # not significant


def test_cross_reference():
    pans_df = pd.DataFrame({
        "gene_symbol": ["TREX1", "SAMHD1", "ADNP"],
        "pathway_axis": ["DDR", "DDR", "Chromatin"],
        "variant_type": ["P", "P", "P"],
    })
    de_df = pd.DataFrame({
        "gene_symbol": ["TREX1", "SAMHD1", "GAPDH"],
        "mouse_symbol": ["Trex1", "Samhd1", "Gapdh"],
        "log2fc": [1.5, -0.8, 0.3],
        "pvalue": [0.001, 0.005, 0.5],
        "padj": [0.01, 0.02, 0.7],
    })

    xref = cross_reference(pans_df, de_df)
    assert len(xref) == 3
    assert xref[xref["gene_symbol"] == "TREX1"].iloc[0]["de_status"] == "upregulated"
    assert xref[xref["gene_symbol"] == "SAMHD1"].iloc[0]["de_status"] == "downregulated"
    assert xref[xref["gene_symbol"] == "ADNP"].iloc[0]["de_status"] == "not_detected"
    assert xref[xref["gene_symbol"] == "TREX1"].iloc[0]["is_ddr_gene"]


def test_compute_overlap_enrichment():
    de_df = pd.DataFrame({
        "gene_symbol": [f"Gene{i}" for i in range(100)],
        "padj": [0.01] * 20 + [0.5] * 80,
    })
    # 3 PANS genes, 2 of which are in the DE set
    pans_genes = ["Gene0", "Gene1", "Gene99"]

    result = compute_overlap_enrichment(pans_genes, de_df)
    assert result["pans_in_de"] == 2
    assert result["pans_not_in_de"] == 1
    assert result["total_de"] == 20
    assert result["fisher_pvalue"] <= 1.0
    assert result["odds_ratio"] > 0


def test_ddr_highlight_genes():
    assert "TREX1" in DDR_HIGHLIGHT_GENES
    assert "SAMHD1" in DDR_HIGHLIGHT_GENES
    assert "ADNP" in DDR_HIGHLIGHT_GENES
