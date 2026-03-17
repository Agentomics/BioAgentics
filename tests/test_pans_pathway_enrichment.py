"""Tests for PANS pathway enrichment module (unit tests, no network calls)."""

import pandas as pd

from bioagentics.models.pans_pathway_enrichment import (
    ENRICHR_LIBRARIES,
    IMMUNE_FOCUS_TERMS,
    filter_immune_pathways,
)


def test_enrichr_libraries_nonempty():
    assert len(ENRICHR_LIBRARIES) >= 3


def test_immune_focus_terms_nonempty():
    assert len(IMMUNE_FOCUS_TERMS) >= 5
    assert any("Toll" in t for t in IMMUNE_FOCUS_TERMS)
    assert any("cGAS" in t or "STING" in t for t in IMMUNE_FOCUS_TERMS)


def test_filter_immune_pathways():
    df = pd.DataFrame({
        "term": [
            "Toll-like receptor signaling pathway",
            "Ribosome biogenesis",
            "Type I interferon signaling",
            "Cell cycle",
        ],
        "adj_p_value": [0.01, 0.5, 0.03, 0.8],
        "p_value": [0.001, 0.3, 0.002, 0.6],
        "library": ["KEGG", "KEGG", "Reactome", "GO"],
    })
    immune = filter_immune_pathways(df)
    assert len(immune) == 2
    assert set(immune["term"]) == {
        "Toll-like receptor signaling pathway",
        "Type I interferon signaling",
    }


def test_filter_immune_empty_df():
    result = filter_immune_pathways(pd.DataFrame())
    assert result.empty
