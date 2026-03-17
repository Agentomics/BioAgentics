"""Tests for PANS variant gene data module."""

from bioagentics.data.pans_variants import (
    AXIS_GENE_COUNTS,
    EXPECTED_GENE_COUNT,
    PATHWAY_AXES,
    get_pans_gene_symbols,
    get_pans_variant_genes,
)


def test_gene_count():
    df = get_pans_variant_genes()
    assert len(df) == EXPECTED_GENE_COUNT


def test_gene_symbols_count():
    symbols = get_pans_gene_symbols()
    assert len(symbols) == EXPECTED_GENE_COUNT


def test_no_duplicate_genes():
    symbols = get_pans_gene_symbols()
    assert len(symbols) == len(set(symbols))


def test_dataframe_columns():
    df = get_pans_variant_genes()
    expected_cols = {"gene_symbol", "pathway_axis", "variant_type", "functional_annotation"}
    assert set(df.columns) == expected_cols


def test_axis_groupings():
    df = get_pans_variant_genes()
    for axis, expected_count in AXIS_GENE_COUNTS.items():
        actual = len(df[df["pathway_axis"] == axis])
        assert actual == expected_count, f"{axis}: expected {expected_count}, got {actual}"


def test_all_axes_present():
    df = get_pans_variant_genes()
    axes_in_data = set(df["pathway_axis"].unique())
    assert axes_in_data == set(PATHWAY_AXES)


def test_variant_types_valid():
    df = get_pans_variant_genes()
    valid_types = {"P", "LP", "VUS"}
    assert set(df["variant_type"].unique()).issubset(valid_types)


def test_symbols_match_dataframe():
    symbols = get_pans_gene_symbols()
    df = get_pans_variant_genes()
    assert symbols == df["gene_symbol"].tolist()
