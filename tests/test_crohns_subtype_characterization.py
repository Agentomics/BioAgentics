"""Tests for subtype characterization module."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.crohns_subtype_characterization import (
    SubtypeCharacterization,
    build_biomarker_panel,
    clinical_associations,
    map_subtypes_to_axes,
    subtype_correlation_network,
    top_features_per_subtype,
)


@pytest.fixture
def subtyped_data():
    """Synthetic data with 2 subtypes and known feature differences."""
    rng = np.random.default_rng(42)
    n = 30
    subtypes = pd.Series(
        [0] * 15 + [1] * 15,
        index=[f"P{i}" for i in range(n)],
        name="subtype",
    )

    # Species: subtype 0 has high Bifidobacterium, subtype 1 has high tryptophan taxa
    species_data = rng.normal(0, 1, (n, 8))
    species_data[:15, 0] += 3  # Bifidobacterium_longum high in subtype 0
    species_data[15:, 1] += 3  # Tryptophan_taxa high in subtype 1
    species = pd.DataFrame(
        species_data,
        index=subtypes.index,
        columns=[
            "Bifidobacterium_longum", "Ruminococcus_gnavus",
            "Faecalibacterium_prausnitzii", "Escherichia_coli",
            "sp_4", "sp_5", "sp_6", "sp_7",
        ],
    )

    # Metabolomics: subtype 0 has high TCDCA, subtype 1 has high tryptophan
    metab_data = rng.normal(0, 1, (n, 10))
    metab_data[:15, 0] += 2.5  # TCDCA high in subtype 0
    metab_data[15:, 1] += 2.5  # tryptophan high in subtype 1
    metabolomics = pd.DataFrame(
        metab_data,
        index=subtypes.index,
        columns=[
            "taurochenodeoxycholic_acid", "tryptophan", "leucine",
            "butyrate", "mb_4", "mb_5", "mb_6", "mb_7", "mb_8", "mb_9",
        ],
    )

    # Metadata
    metadata = pd.DataFrame(
        {
            "diagnosis": ["CD"] * n,
            "Montreal_location": (["L1"] * 8 + ["L2"] * 7 + ["L2"] * 5 + ["L3"] * 10),
            "crp": rng.exponential(5, n),
            "fecal_calprotectin": rng.exponential(200, n),
            "age": rng.integers(20, 60, n),
        },
        index=subtypes.index,
    )

    return subtypes, species, metabolomics, metadata


# ── Top Features ──


def test_top_features_per_subtype(subtyped_data):
    subtypes, species, _, _ = subtyped_data
    top = top_features_per_subtype(species, subtypes, n_top=5)
    assert 0 in top
    assert 1 in top
    assert len(top[0]) <= 5


def test_top_features_includes_known(subtyped_data):
    subtypes, species, _, _ = subtyped_data
    top = top_features_per_subtype(species, subtypes)
    # Bifidobacterium should be top for subtype 0
    sub0_features = list(top[0]["feature"])
    assert "Bifidobacterium_longum" in sub0_features


# ── Clinical Associations ──


def test_clinical_associations_returns_both(subtyped_data):
    subtypes, _, _, metadata = subtyped_data
    results = clinical_associations(subtypes, metadata)
    assert "categorical" in results
    assert "continuous" in results


def test_clinical_continuous_tests(subtyped_data):
    subtypes, _, _, metadata = subtyped_data
    results = clinical_associations(
        subtypes, metadata,
        continuous_cols=["crp", "fecal_calprotectin", "age"],
    )
    cont = results["continuous"]
    assert len(cont) > 0
    assert "p_value" in cont.columns


# ── Metabolic Axis Mapping ──


def test_map_subtypes_to_axes(subtyped_data):
    subtypes, species, metabolomics, _ = subtyped_data
    axes = map_subtypes_to_axes(subtypes, species, metabolomics)
    assert 0 in axes
    assert 1 in axes
    assert "Bifidobacterium-TCDCA" in axes[0]
    assert "Tryptophan-NAD" in axes[0]
    # Subtype 0 should have higher Bifidobacterium-TCDCA score
    assert axes[0]["Bifidobacterium-TCDCA"] > axes[1]["Bifidobacterium-TCDCA"]


# ── Correlation Network ──


def test_correlation_network(subtyped_data):
    subtypes, species, metabolomics, _ = subtyped_data
    net = subtype_correlation_network(
        species, metabolomics, subtypes,
        subtype=0, min_corr=0.0, fdr_threshold=1.0,
    )
    # Should return a DataFrame with correlation edges
    if len(net) > 0:
        assert "species" in net.columns
        assert "metabolite" in net.columns
        assert "correlation" in net.columns


def test_correlation_network_too_few_samples():
    subtypes = pd.Series([0, 0, 1], index=["P0", "P1", "P2"])
    species = pd.DataFrame({"sp": [1, 2, 3]}, index=["P0", "P1", "P2"])
    metab = pd.DataFrame({"mb": [4, 5, 6]}, index=["P0", "P1", "P2"])
    net = subtype_correlation_network(species, metab, subtypes, subtype=0)
    assert len(net) == 0  # Too few samples


# ── Biomarker Panel ──


def test_build_biomarker_panel(subtyped_data):
    subtypes, species, metabolomics, _ = subtyped_data
    top_features = top_features_per_subtype(species, subtypes)
    axis_map = map_subtypes_to_axes(subtypes, species, metabolomics)

    panels = build_biomarker_panel(top_features, axis_map)
    assert 0 in panels
    assert "markers" in panels[0]
    assert "dominant_axis" in panels[0]
    assert "therapeutic_suggestion" in panels[0]


# ── Full Pipeline ──


def test_subtype_characterization_full(subtyped_data, tmp_path):
    subtypes, species, metabolomics, metadata = subtyped_data
    charact = SubtypeCharacterization()
    results = charact.characterize(
        subtypes, species, metabolomics, metadata, output_dir=tmp_path
    )

    assert "top_species" in results
    assert "top_metabolites" in results
    assert "clinical_associations" in results
    assert "axis_mapping" in results
    assert "networks" in results
    assert "biomarker_panels" in results

    # Check files saved
    assert (tmp_path / "metabolic_axis_mapping.csv").exists()
    assert (tmp_path / "biomarker_panels.csv").exists()
