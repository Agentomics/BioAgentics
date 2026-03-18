"""Tests for AHBA spatial mapping module.

Unit tests that don't require network access (mock the API).
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bioagentics.analysis.tourettes.ahba_spatial import (
    CSTC_STRUCTURES,
    compute_enrichment,
    map_samples_to_cstc,
    check_regional_specificity,
)


def _make_expression_df() -> pd.DataFrame:
    """Create a synthetic expression DataFrame for testing."""
    rng = np.random.default_rng(42)
    regions = list(CSTC_STRUCTURES.keys())
    genes = ["FLT3", "MEIS1", "HDC", "SLITRK1", "WWC1"]
    records = []
    for gene in genes:
        for region in regions:
            for donor_id in [9861, 10021]:
                records.append({
                    "gene_symbol": gene,
                    "cstc_region": region,
                    "donor_id": donor_id,
                    "mean_zscore": rng.normal(0, 1),
                    "n_samples": 5,
                    "n_probes": 2,
                })
    return pd.DataFrame(records)


def test_cstc_structures_defined():
    expected = {"prefrontal_cortex", "motor_cortex", "caudate", "putamen",
                "GPe", "GPi", "STN", "thalamus"}
    assert set(CSTC_STRUCTURES.keys()) == expected


def test_map_samples_to_cstc():
    sid = list(CSTC_STRUCTURES.values())[0][0]
    region_name = list(CSTC_STRUCTURES.keys())[0]
    df = pd.DataFrame([
        {"structure_id": sid, "value": 1.0},
        {"structure_id": 99999, "value": 2.0},
    ])
    result = map_samples_to_cstc(df)
    assert result.iloc[0]["cstc_region"] == region_name
    assert result.iloc[1]["cstc_region"] == "other"


def test_compute_enrichment_shape():
    df = _make_expression_df()
    enrichment = compute_enrichment(df)
    assert enrichment.shape[0] == 5  # 5 genes
    assert enrichment.shape[1] == 8  # 8 CSTC regions


def test_compute_enrichment_zscore_properties():
    """Enrichment z-scores should have mean ~0 and std ~1 per gene (row)."""
    df = _make_expression_df()
    enrichment = compute_enrichment(df)
    for _, row in enrichment.iterrows():
        assert abs(row.mean()) < 0.01
        assert abs(row.std() - 1.0) < 0.01


def test_compute_enrichment_empty():
    result = compute_enrichment(pd.DataFrame())
    assert result.empty


def test_check_regional_specificity_returns_stats():
    df = _make_expression_df()
    enrichment = compute_enrichment(df)
    stats = check_regional_specificity(enrichment)
    assert "f_statistic" in stats
    assert "p_value" in stats
    assert "significant" in stats
    assert "top_regions" in stats
    assert stats["n_genes"] == 5
    assert stats["n_regions"] == 8


def test_check_regional_specificity_empty():
    result = check_regional_specificity(pd.DataFrame())
    assert "error" in result


def test_resolve_probes_mock():
    """Test probe resolution with mocked API."""
    mock_response = {
        "msg": [
            {"id": 1001, "name": "A_probe", "gene_id": 100, "genes": {"acronym": "FLT3"}},
            {"id": 1002, "name": "B_probe", "gene_id": 100, "genes": {"acronym": "FLT3"}},
        ]
    }
    with patch("bioagentics.analysis.tourettes.ahba_spatial._api_get", return_value=mock_response):
        from bioagentics.analysis.tourettes.ahba_spatial import resolve_probes
        probes = resolve_probes("FLT3")
        assert len(probes) == 2
        assert probes[0]["gene_symbol"] == "FLT3"
        assert probes[0]["id"] == 1001
