"""Tests for iron homeostasis pathway spatial profiling module."""

import numpy as np
import pandas as pd

from bioagentics.analysis.tourettes.iron_pathway import (
    IRON_CSTC_STRUCTURES,
    MRI_IRON_DEPLETION_RANK,
    compute_iron_enrichment,
    concordance_test,
)


def _make_iron_expr_df() -> pd.DataFrame:
    """Create synthetic iron pathway expression data."""
    rng = np.random.default_rng(42)
    genes = ["TF", "TFRC", "FTH1", "FTL", "ACO1", "IREB2", "HAMP", "SLC40A1"]
    regions = list(IRON_CSTC_STRUCTURES.keys())
    records = []
    for gene in genes:
        for region in regions:
            for donor_id in [9861, 10021]:
                records.append({
                    "gene_symbol": gene,
                    "cstc_region": region,
                    "donor_id": donor_id,
                    "mean_zscore": rng.normal(0, 1),
                    "n_samples": 3,
                    "n_probes": 2,
                })
    return pd.DataFrame(records)


def test_iron_cstc_structures():
    assert "caudate" in IRON_CSTC_STRUCTURES
    assert "substantia_nigra" in IRON_CSTC_STRUCTURES
    assert "red_nucleus" in IRON_CSTC_STRUCTURES
    assert len(IRON_CSTC_STRUCTURES) == 8


def test_mri_depletion_rank_matches_structures():
    """MRI ranks should cover same regions as iron CSTC structures."""
    assert set(MRI_IRON_DEPLETION_RANK.keys()) == set(IRON_CSTC_STRUCTURES.keys())


def test_compute_iron_enrichment():
    df = _make_iron_expr_df()
    enrichment = compute_iron_enrichment(df)
    assert enrichment.shape[0] == 8  # 8 genes
    assert enrichment.shape[1] == 8  # 8 regions


def test_compute_iron_enrichment_empty():
    result = compute_iron_enrichment(pd.DataFrame())
    assert result.empty


def test_concordance_test_returns_stats():
    df = _make_iron_expr_df()
    enrichment = compute_iron_enrichment(df)
    result = concordance_test(enrichment)
    assert "overall_spearman_rho" in result
    assert "overall_p_value" in result
    assert "concordant" in result
    assert "per_gene_results" in result
    assert len(result["per_gene_results"]) == 8
    assert result["n_regions"] == 8


def test_concordance_test_empty():
    result = concordance_test(pd.DataFrame())
    assert "error" in result


def test_concordance_per_gene_fields():
    df = _make_iron_expr_df()
    enrichment = compute_iron_enrichment(df)
    result = concordance_test(enrichment)
    for gene_result in result["per_gene_results"]:
        assert "gene_symbol" in gene_result
        assert "spearman_rho" in gene_result
        assert "p_value" in gene_result
        assert -1.0 <= gene_result["spearman_rho"] <= 1.0
