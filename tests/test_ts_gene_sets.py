"""Tests for TS risk gene sets module (CSTC circuit expression atlas)."""

from bioagentics.data.tourettes.gene_sets import (
    TSAICG_GWAS,
    gene_symbols,
    get_gene_set,
    list_gene_sets,
)


def test_list_gene_sets_returns_all():
    names = list_gene_sets()
    assert "tsaicg_gwas" in names
    assert "rare_variant" in names
    assert "iron_homeostasis" in names
    assert "hippo_signaling" in names
    assert "ts_combined" in names
    assert len(names) == 5


def test_get_gene_set_gwas():
    gs = get_gene_set("tsaicg_gwas")
    assert isinstance(gs, dict)
    assert "FLT3" in gs
    assert "MEIS1" in gs
    assert len(gs) == len(TSAICG_GWAS)


def test_get_gene_set_rare_variant():
    gs = get_gene_set("rare_variant")
    for gene in ["SLITRK1", "HDC", "NRXN1", "CNTN6", "WWC1"]:
        assert gene in gs, f"{gene} missing from rare_variant set"
    assert len(gs) == 5


def test_get_gene_set_iron():
    gs = get_gene_set("iron_homeostasis")
    for gene in ["TF", "TFRC", "FTH1", "FTL", "ACO1", "IREB2", "HAMP", "SLC40A1"]:
        assert gene in gs, f"{gene} missing from iron_homeostasis set"
    assert len(gs) == 8


def test_get_gene_set_hippo():
    gs = get_gene_set("hippo_signaling")
    for gene in ["WWC1", "YAP1", "WWTR1", "LATS1", "LATS2", "STK4", "STK3"]:
        assert gene in gs, f"{gene} missing from hippo_signaling set"
    assert len(gs) == 7


def test_combined_is_union():
    combined = get_gene_set("ts_combined")
    for name in list_gene_sets():
        if name == "ts_combined":
            continue
        gs = get_gene_set(name)
        for symbol in gs:
            assert symbol in combined, f"{symbol} from {name} missing in ts_combined"


def test_combined_no_duplicates():
    """Combined set should have fewer genes than the sum of all sets due to overlaps."""
    individual_total = sum(
        len(get_gene_set(n)) for n in list_gene_sets() if n != "ts_combined"
    )
    combined = get_gene_set("ts_combined")
    # There are known overlaps (e.g., NRXN1 in GWAS + rare_variant, WWC1 in rare + hippo)
    assert len(combined) < individual_total
    assert len(combined) > 0


def test_get_gene_set_returns_copy():
    gs1 = get_gene_set("tsaicg_gwas")
    gs1["FAKE_GENE"] = "should not persist"
    gs2 = get_gene_set("tsaicg_gwas")
    assert "FAKE_GENE" not in gs2


def test_get_gene_set_unknown_raises():
    import pytest

    with pytest.raises(KeyError, match="Unknown gene set"):
        get_gene_set("nonexistent")


def test_gene_symbols_sorted():
    syms = gene_symbols("rare_variant")
    assert syms == sorted(syms)
    assert "HDC" in syms


def test_descriptions_are_nonempty():
    for name in list_gene_sets():
        gs = get_gene_set(name)
        for symbol, desc in gs.items():
            assert desc, f"{symbol} in {name} has empty description"
