"""Tests for cytokine_network module."""

from __future__ import annotations

from bioagentics.cytokine_meta_analysis import MetaAnalysisResult
from bioagentics.cytokine_network import (
    build_network,
    export_network,
    get_modules,
    hub_analysis,
    overlay_meta_results,
    visualize_network,
)


def test_build_network():
    G = build_network()
    assert G.number_of_nodes() > 10
    assert G.number_of_edges() > 15
    assert "IL-6" in G.nodes
    assert G.nodes["IL-6"].get("is_hub") is True


def test_il6_hub_connectivity():
    """IL-6 should be the most connected node."""
    G = build_network()
    degrees = dict(G.degree())
    max_node = max(degrees, key=degrees.get)
    assert max_node == "IL-6"


def test_nlrp3_independent_arm():
    """NLRP3 → IL-1β edge should exist and be marked as inflammasome."""
    G = build_network()
    assert G.has_edge("NLRP3", "IL-1β")
    assert G.edges["NLRP3", "IL-1β"]["module"] == "inflammasome"
    assert G.nodes["NLRP3"].get("is_independent_arm") is True


def test_bbb_module():
    """IL-17A → BBB_disruption → S100B pathway should exist."""
    G = build_network()
    assert G.has_edge("IL-17A", "BBB_disruption")
    assert G.has_edge("BBB_disruption", "S100B")


def test_cirs_module():
    """CIRS module nodes should be present."""
    G = build_network()
    cirs_nodes = {"TGF-β1", "MMP-9", "C4a", "α-MSH", "C3"}
    for node in cirs_nodes:
        assert node in G.nodes, f"{node} missing from network"


def test_epigenetic_persistence_node():
    """IFNγ epigenetic persistence node should exist with correct attributes."""
    G = build_network()
    assert "IFNγ_epigenetic_persistence" in G.nodes
    assert G.nodes["IFNγ_epigenetic_persistence"]["timescale"] == "weeks-months"
    assert G.has_edge("IFN-γ", "IFNγ_epigenetic_persistence")
    edge_data = G.edges["IFN-γ", "IFNγ_epigenetic_persistence"]
    assert edge_data["type"] == "epigenetic_damage"


def test_overlay_meta_results():
    G = build_network()
    results = [
        MetaAnalysisResult(analyte="IL-6", k=4, pooled_effect=1.5, p_value=0.001),
        MetaAnalysisResult(analyte="IL-10", k=3, pooled_effect=-0.8, p_value=0.02),
        MetaAnalysisResult(analyte="IL-4", k=3, pooled_effect=0.2, p_value=0.4),
    ]
    overlay_meta_results(G, results)
    assert G.nodes["IL-6"]["meta_color"] == "red"
    assert G.nodes["IL-6"]["meta_direction"] == "up"
    assert G.nodes["IL-10"]["meta_color"] == "blue"
    assert G.nodes["IL-10"]["meta_direction"] == "down"
    assert G.nodes["IL-4"]["meta_color"] == "grey"
    assert G.nodes["IL-4"]["meta_direction"] == "ns"
    # Unmeasured node
    assert G.nodes["NLRP3"]["meta_direction"] == "unmeasured"


def test_get_modules():
    G = build_network()
    modules = get_modules(G)
    assert "innate" in modules
    assert "IL-6" in modules["innate"]


def test_hub_analysis():
    G = build_network()
    centrality = hub_analysis(G)
    assert "IL-6" in centrality
    assert centrality["IL-6"]["degree_centrality"] > 0


def test_visualize_network(tmp_path):
    G = build_network()
    results = [
        MetaAnalysisResult(analyte="IL-6", k=4, pooled_effect=1.5, p_value=0.001),
    ]
    overlay_meta_results(G, results)
    out = visualize_network(G, output_path=tmp_path / "test_network.png")
    assert out.exists()


def test_export_network(tmp_path):
    G = build_network()
    out = export_network(G, output_path=tmp_path / "network.json")
    assert out.exists()
