"""Cytokine interaction network constructor for PANDAS/PANS flare analysis.

Builds a networkx-based cytokine interaction network with:
- IL-6 as the central hub node (Pozzilli et al.)
- NLRP3→IL-1β independent parallel arm (Luo et al.)
- CIRS/complement module (TGF-β1, MMP-9, C4a, α-MSH, C3)
- BBB module (IL-17A → BBB disruption → S100B)
- IFNγ epigenetic persistence node (Shammas 2026, PMID 41448185)
- IFNγ peripheral priming node — reversible arm (Gorin 2026, PMID 40599159)
- IL-12 Th1-polarizing node
- Meta-analysis overlay (upregulated=red, downregulated=blue, grey=ns)

Usage::

    from bioagentics.cytokine_network import build_network, overlay_meta_results, visualize_network

    G = build_network()
    overlay_meta_results(G, meta_results)
    visualize_network(G, output_path="network.png")
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from bioagentics.config import REPO_ROOT
from bioagentics.cytokine_meta_analysis import MetaAnalysisResult

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "cytokine-network-flare-prediction"


# ---------------------------------------------------------------------------
# Curated seed interactions
# ---------------------------------------------------------------------------

# Node types for visual/analytical grouping
NODE_TYPES = {
    # Th1 axis
    "IFN-γ": "th1",
    "TNF-α": "th1",
    # Th2 axis
    "IL-4": "th2",
    "IL-13": "th2",
    # Th17 axis
    "IL-17A": "th17",
    "IL-22": "th17",
    "IL-23": "th17",
    # Regulatory
    "IL-10": "regulatory",
    "TGF-β": "regulatory",
    # Innate / pro-inflammatory
    "IL-1β": "innate",
    "IL-6": "innate",  # central hub
    "IL-8": "innate",
    "TNF-α": "innate",
    "GM-CSF": "innate",
    # NLRP3 inflammasome
    "NLRP3": "inflammasome",
    # CIRS / complement module
    "TGF-β1": "cirs",
    "MMP-9": "cirs",
    "C4a": "cirs",
    "α-MSH": "cirs",
    "C3": "cirs",
    # BBB module
    "BBB_disruption": "bbb",
    "S100B": "bbb",
    # Epigenetic persistence (Shammas 2026)
    "IFNγ_epigenetic_persistence": "epigenetic",
    # IFNγ peripheral priming — reversible arm (Gorin 2026, PMID 40599159)
    "IFNg_peripheral_priming": "peripheral_priming",
    # IL-12: Th1-polarizing cytokine driving IFN-γ production
    "IL-12": "th1",
}

# Curated cytokine-cytokine interactions (source → target)
# Based on ImmunoGlobe/InnateDB + literature curation
SEED_INTERACTIONS = [
    # IL-6 hub — no cytokine elevated without concurrent IL-6 (Pozzilli et al.)
    ("IL-6", "IL-1β", {"type": "amplification", "module": "hub"}),
    ("IL-6", "TNF-α", {"type": "amplification", "module": "hub"}),
    ("IL-6", "IL-17A", {"type": "induction", "module": "hub"}),
    ("IL-6", "IL-8", {"type": "induction", "module": "hub"}),
    ("IL-6", "IL-10", {"type": "feedback", "module": "hub"}),
    ("IL-6", "IL-4", {"type": "modulation", "module": "hub"}),
    ("IL-6", "IFN-γ", {"type": "cross-regulation", "module": "hub"}),
    ("IL-6", "IL-22", {"type": "induction", "module": "hub"}),
    ("IL-6", "GM-CSF", {"type": "induction", "module": "hub"}),

    # NLRP3 → IL-1β independent arm (Luo et al.)
    ("NLRP3", "IL-1β", {"type": "activation", "module": "inflammasome"}),

    # IL-1β → IL-6 positive feedback
    ("IL-1β", "IL-6", {"type": "induction", "module": "innate"}),
    ("TNF-α", "IL-6", {"type": "induction", "module": "innate"}),
    ("TNF-α", "IL-1β", {"type": "amplification", "module": "innate"}),

    # Th17 interactions
    ("IL-23", "IL-17A", {"type": "differentiation", "module": "th17"}),
    ("IL-6", "IL-23", {"type": "induction", "module": "th17"}),
    ("IL-17A", "IL-6", {"type": "induction", "module": "th17"}),
    ("IL-17A", "IL-8", {"type": "induction", "module": "th17"}),

    # BBB module: IL-17A → BBB disruption → S100B (task #338)
    ("IL-17A", "BBB_disruption", {"type": "disruption", "module": "bbb"}),
    ("BBB_disruption", "S100B", {"type": "release", "module": "bbb"}),
    ("TNF-α", "BBB_disruption", {"type": "disruption", "module": "bbb"}),

    # CIRS / complement module (task #281)
    ("C3", "C4a", {"type": "complement_cascade", "module": "cirs"}),
    ("C4a", "MMP-9", {"type": "activation", "module": "cirs"}),
    ("TGF-β1", "MMP-9", {"type": "induction", "module": "cirs"}),
    ("α-MSH", "IL-6", {"type": "inhibition", "module": "cirs"}),
    ("α-MSH", "TNF-α", {"type": "inhibition", "module": "cirs"}),
    ("IL-6", "C3", {"type": "induction", "module": "cirs"}),

    # Regulatory feedback
    ("IL-10", "IL-6", {"type": "inhibition", "module": "regulatory"}),
    ("IL-10", "TNF-α", {"type": "inhibition", "module": "regulatory"}),
    ("IL-10", "IL-1β", {"type": "inhibition", "module": "regulatory"}),
    ("TGF-β", "IL-17A", {"type": "modulation", "module": "regulatory"}),

    # Th1/Th2 cross-regulation
    ("IFN-γ", "IL-4", {"type": "inhibition", "module": "cross_regulation"}),
    ("IL-4", "IFN-γ", {"type": "inhibition", "module": "cross_regulation"}),
    ("IFN-γ", "TNF-α", {"type": "synergy", "module": "th1"}),

    # IFNγ epigenetic persistence (Shammas 2026, PMID 41448185)
    # IFNγ drives persistent chromatin closing in neurons — synaptopathy
    # outlasts the immune response
    ("IFN-γ", "IFNγ_epigenetic_persistence", {"type": "epigenetic_damage", "module": "epigenetic",
     "timescale": "weeks-months", "mechanism": "chromatin_closing"}),

    # IFNγ peripheral priming — reversible arm (Gorin 2026, PMID 40599159)
    # JAK/STAT-dependent; decays when IFNγ signaling stops
    ("IFN-γ", "IFNg_peripheral_priming", {"type": "reversible_priming", "module": "peripheral_priming",
     "timescale": "hours", "mechanism": "JAK_STAT"}),

    # IL-12: Th1-polarizing cytokine
    ("IL-12", "IFN-γ", {"type": "differentiation", "module": "th1"}),
    ("IL-6", "IL-12", {"type": "modulation", "module": "hub"}),
]


# ---------------------------------------------------------------------------
# Network construction
# ---------------------------------------------------------------------------


def build_network() -> nx.DiGraph:
    """Build the curated cytokine interaction network.

    Returns a directed graph with node attributes (type, module) and
    edge attributes (type, module, timescale where applicable).
    """
    G = nx.DiGraph()

    # Add nodes with type attributes
    for node, ntype in NODE_TYPES.items():
        G.add_node(node, node_type=ntype)

    # Add edges from seed interactions
    for src, tgt, attrs in SEED_INTERACTIONS:
        G.add_edge(src, tgt, **attrs)

    # Ensure IL-6 is marked as hub
    G.nodes["IL-6"]["is_hub"] = True

    # Mark NLRP3 as independent arm
    G.nodes["NLRP3"]["is_independent_arm"] = True

    # Mark epigenetic persistence node (irreversible arm)
    if "IFNγ_epigenetic_persistence" in G.nodes:
        G.nodes["IFNγ_epigenetic_persistence"]["timescale"] = "weeks-months"
        G.nodes["IFNγ_epigenetic_persistence"]["mechanism"] = "chromatin_closing_neurons"

    # Mark peripheral priming node (reversible arm)
    if "IFNg_peripheral_priming" in G.nodes:
        G.nodes["IFNg_peripheral_priming"]["timescale"] = "hours"
        G.nodes["IFNg_peripheral_priming"]["mechanism"] = "JAK_STAT_signaling"

    logger.info("Built network: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


# ---------------------------------------------------------------------------
# Meta-analysis overlay
# ---------------------------------------------------------------------------


def overlay_meta_results(
    G: nx.DiGraph,
    results: list[MetaAnalysisResult],
    alpha: float = 0.05,
) -> nx.DiGraph:
    """Overlay meta-analysis results onto the network.

    Adds node attributes:
    - meta_effect: pooled Hedges' g
    - meta_p: p-value
    - meta_direction: "up", "down", or "ns"
    - meta_color: "red" (up), "blue" (down), "grey" (ns)
    """
    result_map = {r.analyte: r for r in results}

    for node in G.nodes:
        if node in result_map:
            r = result_map[node]
            G.nodes[node]["meta_effect"] = r.pooled_effect
            G.nodes[node]["meta_p"] = r.p_value
            if r.p_value < alpha:
                direction = "up" if r.pooled_effect > 0 else "down"
                color = "red" if direction == "up" else "blue"
            else:
                direction = "ns"
                color = "grey"
            G.nodes[node]["meta_direction"] = direction
            G.nodes[node]["meta_color"] = color
        else:
            G.nodes[node]["meta_direction"] = "unmeasured"
            G.nodes[node]["meta_color"] = "lightgrey"

    return G


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

# Module-based position layout seeds
_MODULE_CENTERS = {
    "hub": (0.0, 0.0),
    "innate": (-0.3, 0.3),
    "inflammasome": (-0.5, 0.0),
    "th1": (0.4, 0.4),
    "th2": (0.4, -0.3),
    "th17": (0.0, 0.5),
    "regulatory": (-0.3, -0.4),
    "cirs": (-0.5, -0.3),
    "bbb": (0.3, 0.6),
    "epigenetic": (0.6, 0.3),
    "peripheral_priming": (0.6, 0.1),
    "cross_regulation": (0.5, 0.0),
}


def _layout(G: nx.DiGraph) -> dict:
    """Compute a spring layout with module-based initial positions."""
    pos_init = {}
    for node in G.nodes:
        ntype = G.nodes[node].get("node_type", "innate")
        center = _MODULE_CENTERS.get(ntype, (0, 0))
        pos_init[node] = (center[0] + np.random.uniform(-0.1, 0.1),
                          center[1] + np.random.uniform(-0.1, 0.1))
    return nx.spring_layout(G, pos=pos_init, k=0.5, iterations=50, seed=42)


def visualize_network(
    G: nx.DiGraph,
    output_path: Path | str | None = None,
    title: str = "PANDAS/PANS Cytokine Network — Flare Signature",
) -> Path:
    """Generate a publication-quality network visualization.

    Node colors encode meta-analysis direction (red=up, blue=down, grey=ns).
    Node sizes scale with degree centrality. IL-6 hub is emphasized.
    Dashed edges indicate inhibition; special styling for epigenetic edges.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "cytokine_network.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = _layout(G)

    # Node colors from meta overlay
    node_colors = [G.nodes[n].get("meta_color", "lightgrey") for n in G.nodes]

    # Node sizes: hub gets extra emphasis
    degrees = dict(G.degree())
    node_sizes = []
    for n in G.nodes:
        base = 300 + degrees.get(n, 1) * 80
        if G.nodes[n].get("is_hub"):
            base *= 1.8
        node_sizes.append(base)

    # Edge styles
    edge_colors = []
    edge_styles = []
    for u, v, d in G.edges(data=True):
        etype = d.get("type", "")
        if etype == "inhibition":
            edge_colors.append("red")
            edge_styles.append("dashed")
        elif etype == "epigenetic_damage":
            edge_colors.append("purple")
            edge_styles.append("dotted")
        elif etype == "reversible_priming":
            edge_colors.append("green")
            edge_styles.append("dashed")
        else:
            edge_colors.append("grey")
            edge_styles.append("solid")

    # Draw edges grouped by style
    for style in ["solid", "dashed", "dotted"]:
        edge_list = [(u, v) for (u, v, _), s in zip(G.edges(data=True), edge_styles) if s == style]
        colors = [c for c, s in zip(edge_colors, edge_styles) if s == style]
        if edge_list:
            nx.draw_networkx_edges(
                G, pos, edgelist=edge_list, edge_color=colors,
                style=style, alpha=0.5, arrows=True, arrowsize=12,
                connectionstyle="arc3,rad=0.1", ax=ax,
            )

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85, ax=ax)

    # Labels with clean names
    label_map = {n: n.replace("_", "\n") for n in G.nodes}
    nx.draw_networkx_labels(G, pos, labels=label_map, font_size=7, font_weight="bold", ax=ax)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="red", alpha=0.7, label="Upregulated in flare"),
        Patch(facecolor="blue", alpha=0.7, label="Downregulated in flare"),
        Patch(facecolor="grey", alpha=0.7, label="Not significant"),
        Patch(facecolor="lightgrey", alpha=0.7, label="Unmeasured"),
    ]
    ax.legend(handles=legend_elements, loc="lower left", fontsize=8)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved network visualization: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Network analysis helpers
# ---------------------------------------------------------------------------


def get_modules(G: nx.DiGraph) -> dict[str, list[str]]:
    """Group nodes by their module (node_type) attribute."""
    modules: dict[str, list[str]] = {}
    for node, data in G.nodes(data=True):
        mod = data.get("node_type", "unknown")
        modules.setdefault(mod, []).append(node)
    return modules


def hub_analysis(G: nx.DiGraph) -> dict[str, float]:
    """Return degree centrality and betweenness centrality for all nodes."""
    deg = nx.degree_centrality(G)
    bet = nx.betweenness_centrality(G)
    return {
        node: {"degree_centrality": deg[node], "betweenness_centrality": bet[node]}
        for node in G.nodes
    }


def export_network(G: nx.DiGraph, output_path: Path | str | None = None) -> Path:
    """Export network to JSON (node-link format) for downstream use."""
    if output_path is None:
        output_path = OUTPUT_DIR / "cytokine_network.json"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Exported network to %s", output_path)
    return output_path
