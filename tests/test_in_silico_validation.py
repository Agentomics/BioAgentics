"""Tests for in silico validation module."""

import pandas as pd
import pytest

from bioagentics.data.cd_fibrosis.in_silico_validation import (
    FIBROSIS_MARKERS,
    KNOWN_MARKER_EFFECTS,
    build_network_edges,
    compute_pathway_enrichment,
    generate_validation_report,
    predict_marker_effects,
)


def _make_candidates() -> pd.DataFrame:
    return pd.DataFrame([
        {"compound": "vorinostat", "composite_score": 0.70},
        {"compound": "pirfenidone", "composite_score": 0.65},
        {"compound": "ontunisertib", "composite_score": 0.55},
    ])


class TestPredictMarkerEffects:
    def test_returns_dataframe(self):
        df = predict_marker_effects(_make_candidates())
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_has_marker_columns(self):
        df = predict_marker_effects(_make_candidates())
        for marker in FIBROSIS_MARKERS:
            assert marker in df.columns

    def test_known_effects_populated(self):
        df = predict_marker_effects(_make_candidates())
        vor = df[df["compound"] == "vorinostat"].iloc[0]
        assert vor["COL1A1"] < 0  # known downregulator

    def test_unknown_compound_gets_zeros(self):
        cands = pd.DataFrame([{"compound": "unknown-drug"}])
        df = predict_marker_effects(cands)
        for marker in FIBROSIS_MARKERS:
            assert df.iloc[0][marker] == 0.0


class TestBuildNetworkEdges:
    def test_returns_edges(self):
        edges = build_network_edges(_make_candidates())
        assert isinstance(edges, pd.DataFrame)
        assert len(edges) > 0

    def test_edge_types(self):
        edges = build_network_edges(_make_candidates())
        assert "compound_target" in edges["edge_type"].values
        assert "target_pathway" in edges["edge_type"].values

    def test_columns(self):
        edges = build_network_edges(_make_candidates())
        assert set(edges.columns) == {"source", "target", "edge_type", "weight"}

    def test_empty_candidates(self):
        cands = pd.DataFrame(columns=["compound"])
        edges = build_network_edges(cands)
        assert len(edges) == 0


class TestComputePathwayEnrichment:
    def test_returns_all_pathways(self):
        enrichment = compute_pathway_enrichment(_make_candidates())
        assert isinstance(enrichment, pd.DataFrame)
        assert len(enrichment) > 0

    def test_tgf_beta_targeted(self):
        enrichment = compute_pathway_enrichment(_make_candidates())
        tgfb = enrichment[enrichment["pathway"] == "TGF-beta"]
        assert len(tgfb) == 1
        assert tgfb.iloc[0]["n_compounds_targeting"] >= 1

    def test_has_novelty_score(self):
        enrichment = compute_pathway_enrichment(_make_candidates())
        assert "novelty_score" in enrichment.columns


class TestGenerateReport:
    def test_report_produces_files(self, tmp_path):
        results = generate_validation_report(
            _make_candidates(), output_dir=tmp_path, top_n=3,
        )
        assert "marker_effects" in results
        assert "network_edges" in results
        assert "pathway_enrichment" in results
        assert (tmp_path / "predicted_marker_effects.tsv").exists()
        assert (tmp_path / "network_edges.tsv").exists()
        assert (tmp_path / "pathway_enrichment.tsv").exists()

    def test_report_with_single_candidate(self, tmp_path):
        cands = pd.DataFrame([{"compound": "pirfenidone", "composite_score": 0.7}])
        results = generate_validation_report(cands, output_dir=tmp_path, top_n=1)
        assert len(results["marker_effects"]) == 1
