"""Tests for DBS fiber tract expression profiling module."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.analysis.tourettes.dbs_tracts import (
    DBS_TRACTS,
    NON_TRACT_BG_REGIONS,
    THALAMIC_RESPONSE_MAP,
    compare_tract_vs_nontract,
    extract_tract_expression,
    get_tract_regions,
    profile_all_tracts,
    analyze_thalamic_gene_subsets,
)


def _make_expr_df() -> pd.DataFrame:
    """Create synthetic expression data spanning all CSTC regions."""
    rng = np.random.default_rng(42)
    regions = [
        "prefrontal_cortex", "motor_cortex", "caudate", "putamen",
        "GPe", "GPi", "STN", "thalamus",
    ]
    genes = ["FLT3", "MEIS1", "HDC", "SLITRK1", "WWC1", "TF", "TFRC", "FTH1"]
    records = []
    for gene in genes:
        for region in regions:
            for donor_id in [9861, 10021, 12876]:
                records.append({
                    "gene_symbol": gene,
                    "cstc_region": region,
                    "donor_id": donor_id,
                    "mean_zscore": rng.normal(0, 1),
                    "n_samples": 5,
                    "n_probes": 2,
                })
    return pd.DataFrame(records)


class TestDBSTracts:
    def test_tracts_defined(self):
        assert len(DBS_TRACTS) == 3
        expected = {"ansa_lenticularis", "fasciculus_lenticularis",
                    "posterior_intralaminar_lentiform"}
        assert set(DBS_TRACTS.keys()) == expected

    def test_tract_regions_valid(self):
        """All tract regions should be valid CSTC structure names."""
        from bioagentics.analysis.tourettes.ahba_spatial import CSTC_STRUCTURES
        valid_regions = set(CSTC_STRUCTURES.keys())
        for tract_name, info in DBS_TRACTS.items():
            for region in info["regions"]:
                assert region in valid_regions, (
                    f"Tract {tract_name} references unknown region {region}"
                )

    def test_get_tract_regions(self):
        regions = get_tract_regions("ansa_lenticularis")
        assert "GPi" in regions
        assert "thalamus" in regions

    def test_get_tract_regions_unknown(self):
        with pytest.raises(KeyError):
            get_tract_regions("nonexistent_tract")

    def test_thalamic_response_map(self):
        assert "tic_responsive" in THALAMIC_RESPONSE_MAP
        assert "ocd_responsive" in THALAMIC_RESPONSE_MAP


class TestExtractTractExpression:
    def test_extract_filters_correctly(self):
        df = _make_expr_df()
        result = extract_tract_expression(df, "ansa_lenticularis")
        assert set(result["cstc_region"].unique()) <= {"GPi", "thalamus"}
        assert "tract" in result.columns
        assert (result["tract"] == "ansa_lenticularis").all()

    def test_extract_empty_input(self):
        result = extract_tract_expression(pd.DataFrame(columns=[
            "gene_symbol", "cstc_region", "donor_id", "mean_zscore",
        ]), "ansa_lenticularis")
        assert result.empty


class TestProfileAllTracts:
    def test_profile_shape(self):
        df = _make_expr_df()
        profiles = profile_all_tracts(df)
        assert not profiles.empty
        assert "gene_symbol" in profiles.columns
        assert "tract" in profiles.columns
        assert "mean_zscore" in profiles.columns

    def test_profile_all_tracts_present(self):
        df = _make_expr_df()
        profiles = profile_all_tracts(df)
        assert set(profiles["tract"].unique()) == set(DBS_TRACTS.keys())

    def test_profile_has_abbreviations(self):
        df = _make_expr_df()
        profiles = profile_all_tracts(df)
        assert set(profiles["tract_abbrev"].unique()) == {"AL", "FL", "PIL"}


class TestCompareTractVsNontract:
    def test_comparison_output(self):
        df = _make_expr_df()
        result = compare_tract_vs_nontract(df)
        assert not result.empty
        assert "gene_symbol" in result.columns
        assert "p_value" in result.columns
        assert "significant" in result.columns
        assert "tract_mean" in result.columns
        assert "nontract_mean" in result.columns

    def test_comparison_sorted_by_pvalue(self):
        df = _make_expr_df()
        result = compare_tract_vs_nontract(df)
        if len(result) > 1:
            assert result["p_value"].is_monotonic_increasing

    def test_comparison_empty_input(self):
        empty = pd.DataFrame(columns=[
            "gene_symbol", "cstc_region", "donor_id", "mean_zscore",
        ])
        result = compare_tract_vs_nontract(empty)
        assert result.empty


class TestThalamicAnalysis:
    def test_analysis_structure(self):
        df = _make_expr_df()
        result = analyze_thalamic_gene_subsets(df)
        assert "subset_stats" in result
        assert "pairwise_comparisons" in result
        assert "interpretation" in result

    def test_analysis_with_custom_sets(self):
        df = _make_expr_df()
        result = analyze_thalamic_gene_subsets(
            df, gene_set_names=["tsaicg_gwas", "iron_homeostasis"],
        )
        assert "subset_stats" in result

    def test_analysis_empty_thalamus(self):
        df = _make_expr_df()
        df = df[df["cstc_region"] != "thalamus"]
        result = analyze_thalamic_gene_subsets(df)
        assert "error" in result
