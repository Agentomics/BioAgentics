"""Tests for PANS GEO expression module (unit tests, no network calls)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

from bioagentics.data.pans_geo_expression import (
    MOUSE_TO_HUMAN,
    _build_probe_to_gene_map,
    classify_samples,
    compute_de,
    normalize_expression,
    _uppercase_symbol_map,
)


def test_mouse_to_human_mapping_covers_pans_genes():
    """All 22 PANS variant genes should have explicit mouse mappings."""
    from bioagentics.data.pans_variants import get_pans_gene_symbols

    human_symbols = set(MOUSE_TO_HUMAN.values())
    pans_genes = set(get_pans_gene_symbols())
    missing = pans_genes - human_symbols
    assert not missing, f"Missing mouse-to-human mappings: {missing}"


def test_uppercase_symbol_map():
    assert _uppercase_symbol_map("Trex1") == "TREX1"
    assert _uppercase_symbol_map("Samhd1") == "SAMHD1"
    assert _uppercase_symbol_map("Park2") == "PRKN"  # alias
    assert _uppercase_symbol_map("SomeGene") == "SOMEGENE"  # fallback


def test_classify_samples():
    meta = pd.DataFrame({
        "title": ["LPS 4h rep1", "LPS 4h rep2", "Control rep1", "PBS rep1"],
        "source": ["microglia", "microglia", "microglia", "microglia"],
        "characteristics": [["treatment: LPS"], ["treatment: LPS"],
                            ["treatment: control"], ["treatment: PBS"]],
    }, index=["GSM1", "GSM2", "GSM3", "GSM4"])

    lps, ctrl = classify_samples(meta)
    assert set(lps) == {"GSM1", "GSM2"}
    assert set(ctrl) == {"GSM3", "GSM4"}


def test_normalize_expression():
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        rng.exponential(100, size=(50, 4)),
        index=[f"Gene{i}" for i in range(50)],
        columns=["S1", "S2", "S3", "S4"],
    )
    normed = normalize_expression(df)
    assert normed.shape == df.shape
    assert not normed.isna().any().any()


def test_compute_de():
    rng = np.random.default_rng(42)
    n_genes = 20
    genes = [f"Gene{i}" for i in range(n_genes)]

    # Create data where Gene0 is clearly upregulated in LPS
    ctrl_data = rng.normal(5, 0.5, size=(n_genes, 3))
    lps_data = rng.normal(5, 0.5, size=(n_genes, 3))
    lps_data[0, :] += 5  # make Gene0 strongly upregulated

    df = pd.DataFrame(
        np.hstack([ctrl_data, lps_data]),
        index=genes,
        columns=["C1", "C2", "C3", "L1", "L2", "L3"],
    )

    de = compute_de(df, ["L1", "L2", "L3"], ["C1", "C2", "C3"])
    assert len(de) == n_genes
    assert "gene_symbol" in de.columns
    assert "log2fc" in de.columns
    assert "pvalue" in de.columns
    assert "padj" in de.columns

    # Gene0 should be the most significant
    top_gene = de.iloc[0]["gene_symbol"]
    assert top_gene == "Gene0"
    assert de.iloc[0]["log2fc"] > 0


def _make_fake_gse(annot_df):
    """Build a minimal GEO-like object with a GPL annotation table."""
    gpl = SimpleNamespace(table=annot_df)
    gse = SimpleNamespace(gpls={"GPL1234": gpl})
    return gse


def test_build_probe_to_gene_map_gene_symbol_column():
    annot = pd.DataFrame({
        "ID": ["probe_1", "probe_2", "probe_3"],
        "Gene Symbol": ["Trex1", "Samhd1", "---"],
    })
    gse = _make_fake_gse(annot)
    mapping = _build_probe_to_gene_map(gse, "GPL1234")
    assert mapping == {"probe_1": "Trex1", "probe_2": "Samhd1"}


def test_build_probe_to_gene_map_gene_assignment_column():
    annot = pd.DataFrame({
        "ID": ["TC001", "TC002", "TC003"],
        "gene_assignment": [
            "NM_001 // Ep300 // E1A binding protein p300 // 15 E1 // 328572",
            "NM_002 // Prkn // parkin // 17 A1 // 50873",
            "---",
        ],
    })
    gse = _make_fake_gse(annot)
    mapping = _build_probe_to_gene_map(gse, "GPL1234")
    assert mapping == {"TC001": "Ep300", "TC002": "Prkn"}


def test_build_probe_to_gene_map_multi_symbol():
    """When Gene Symbol has /// separated entries, take the first."""
    annot = pd.DataFrame({
        "ID": ["p1"],
        "Gene Symbol": ["Abc /// Def"],
    })
    gse = _make_fake_gse(annot)
    mapping = _build_probe_to_gene_map(gse, "GPL1234")
    assert mapping == {"p1": "Abc"}


def test_build_probe_to_gene_map_no_platform():
    gse = SimpleNamespace(gpls={})
    assert _build_probe_to_gene_map(gse, None) == {}
    assert _build_probe_to_gene_map(gse, "MISSING") == {}
