"""Tests for PANS mTOR convergence analysis module."""

import pandas as pd

from bioagentics.models.pans_mtor_convergence import (
    IVIG_AUTOPHAGY_GENES,
    MTOR_PATHWAY_GENES,
    compute_mtor_enrichment,
    compute_mtor_overlap,
)


def test_mtor_pathway_genes_nonempty():
    assert len(MTOR_PATHWAY_GENES) >= 20


def test_ivig_genes_in_mtor():
    assert IVIG_AUTOPHAGY_GENES.issubset(MTOR_PATHWAY_GENES)


def test_compute_mtor_overlap():
    # Most PANS genes won't directly overlap with mTOR
    pans_genes = ["MTOR", "TREX1", "SAMHD1"]  # only MTOR is in the set
    result = compute_mtor_overlap(pans_genes)

    assert result["direct_overlap_count"] == 1
    assert "MTOR" in result["direct_overlap_genes"]
    assert result["pans_gene_count"] == 3


def test_compute_mtor_overlap_no_overlap():
    result = compute_mtor_overlap(["TREX1", "SAMHD1"])
    assert result["direct_overlap_count"] == 0


def test_compute_mtor_enrichment_with_de():
    de_df = pd.DataFrame({
        "gene_symbol": ["MTOR", "AKT1", "GAPDH", "TP53", "BRCA1"],
        "padj": [0.001, 0.01, 0.5, 0.6, 0.8],
    })

    result = compute_mtor_enrichment(["TREX1"], de_df)
    assert result["enrichment_tested"]
    assert result["mtor_de_count"] >= 1


def test_compute_mtor_enrichment_no_de():
    result = compute_mtor_enrichment(["TREX1"], None)
    assert not result["enrichment_tested"]
