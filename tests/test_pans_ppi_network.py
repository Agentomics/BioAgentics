"""Tests for PANS PPI network module (unit tests, no network calls)."""

import networkx as nx

from bioagentics.models.pans_ppi_network import (
    AXIS_COLORS,
    build_network,
    compute_metrics,
    detect_modules,
    identify_hub_genes,
)


def _make_test_interactions():
    """Create mock STRING interactions for testing."""
    return [
        {"preferredName_A": "TREX1", "preferredName_B": "SAMHD1", "score": 900},
        {"preferredName_A": "TREX1", "preferredName_B": "EP300", "score": 500},
        {"preferredName_A": "SAMHD1", "preferredName_B": "FANCD2", "score": 600},
        {"preferredName_A": "EP300", "preferredName_B": "CUX1", "score": 700},
        {"preferredName_A": "MASP1", "preferredName_B": "MASP2", "score": 950},
        {"preferredName_A": "MBL2", "preferredName_B": "MASP1", "score": 980},
        {"preferredName_A": "MBL2", "preferredName_B": "MASP2", "score": 970},
    ]


def _make_gene_axis_map():
    return {
        "TREX1": "DDR-cGAS-STING/AIM2 inflammasome",
        "SAMHD1": "DDR-cGAS-STING/AIM2 inflammasome",
        "EP300": "DDR-cGAS-STING/AIM2 inflammasome",
        "FANCD2": "DDR-cGAS-STING/AIM2 inflammasome",
        "CUX1": "DDR-cGAS-STING/AIM2 inflammasome",
        "MBL2": "Lectin complement",
        "MASP1": "Lectin complement",
        "MASP2": "Lectin complement",
        "ADNP": "Chromatin/neuroprotection",
    }


def test_build_network():
    interactions = _make_test_interactions()
    gene_axis = _make_gene_axis_map()
    G = build_network(interactions, gene_axis)

    assert isinstance(G, nx.Graph)
    assert G.number_of_nodes() >= 8
    assert G.number_of_edges() == 7
    assert G.nodes["TREX1"]["pathway_axis"] == "DDR-cGAS-STING/AIM2 inflammasome"


def test_compute_metrics():
    interactions = _make_test_interactions()
    G = build_network(interactions, _make_gene_axis_map())
    metrics = compute_metrics(G)

    assert not metrics.empty
    assert "gene_symbol" in metrics.columns
    assert "degree_centrality" in metrics.columns
    assert "betweenness_centrality" in metrics.columns
    assert "clustering_coefficient" in metrics.columns


def test_detect_modules():
    interactions = _make_test_interactions()
    G = build_network(interactions, _make_gene_axis_map())
    modules = detect_modules(G)

    assert isinstance(modules, dict)
    assert "TREX1" in modules
    assert "MBL2" in modules


def test_identify_hub_genes():
    interactions = _make_test_interactions()
    G = build_network(interactions, _make_gene_axis_map())
    metrics = compute_metrics(G)
    hubs = identify_hub_genes(metrics, top_n=3)

    assert len(hubs) <= 3
    assert "gene_symbol" in hubs.columns


def test_axis_colors_cover_all_axes():
    from bioagentics.data.pans_variants import PATHWAY_AXES
    for axis in PATHWAY_AXES:
        assert axis in AXIS_COLORS, f"Missing color for axis: {axis}"
