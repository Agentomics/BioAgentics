"""Network visualization for PANDAS autoantibody target network.

Generates publication-quality network visualizations using networkx + matplotlib:
  1. Symptom-domain node coloring
  2. Pathway annotations as node groups/communities
  3. Brain-region expression overlay (node size)
  4. Patient subgroup stratification views
  5. Cytokine layer visually distinguished
  6. FOLR1 as distinct hub with folate pathway

Usage:
    uv run python -m pandas_pans.autoantibody_target_network_mapping.network_visualization

Output:
    output/pandas_pans/autoantibody-target-network-mapping/*.png
"""

import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from pandas_pans.autoantibody_target_network_mapping.cytokine_layer import (
    get_cytokine_gene_symbols,
)
from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    SEED_PROTEINS,
    get_gene_symbols,
    get_seed_dict,
)

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/pandas_pans/autoantibody_network")
OUTPUT_DIR = Path("output/pandas_pans/autoantibody-target-network-mapping")

# Color palettes
SYMPTOM_COLORS = {
    "ocd_compulsive": "#e41a1c",
    "tic_motor": "#377eb8",
    "anxiety": "#4daf4a",
    "eating_restriction": "#984ea3",
    "cognitive": "#ff7f00",
    "emotional_lability": "#a65628",
    "autoimmune_neuropsych": "#f781bf",
    "dopaminergic": "#999999",
}

LAYER_COLORS = {
    "autoantibody_ppi": "#2196F3",
    "cytokine_amplification": "#FF5722",
    "cross_layer": "#9C27B0",
}

MECHANISM_COLORS = {
    "dopaminergic": "#e41a1c",
    "calcium": "#377eb8",
    "metabolic": "#4daf4a",
    "structural": "#984ea3",
    "folate": "#ff7f00",
}

SUBGROUP_COLORS = {
    "dopaminergic_dominant": "#e41a1c",
    "calcium_signaling": "#377eb8",
    "metabolic_surface": "#4daf4a",
    "folate_disruption": "#ff7f00",
    "broad_autoimmunity": "#984ea3",
    "cunningham_classic": "#a65628",
    "drd1_camkii": "#f781bf",
    "drd2_folr1": "#999999",
}


def load_network_for_viz(max_nodes: int = 200) -> tuple[nx.Graph, pd.DataFrame]:
    """Load network and convergence data, limiting to top nodes for readability.

    Returns (graph, convergence_df) limited to the most important nodes.
    """
    conv_path = DATA_DIR / "convergence_analysis.tsv"
    conv_df = pd.read_csv(conv_path, sep="\t")

    # Take top nodes by convergence score
    top_nodes = set(conv_df.nlargest(max_nodes, "convergence_score")["gene_symbol"])

    # Always include seeds and cytokines
    seed_set = set(get_gene_symbols())
    cytokine_set = set(get_cytokine_gene_symbols())
    top_nodes = top_nodes | seed_set | cytokine_set

    # Build subgraph from extended network
    network_path = DATA_DIR / "extended_network.tsv"
    G = nx.Graph()
    chunks = pd.read_csv(
        network_path, sep="\t",
        usecols=["source", "target", "combined_score", "layer"],
        chunksize=5000,
    )
    for chunk in chunks:
        for _, row in chunk.iterrows():
            src, tgt = str(row["source"]), str(row["target"])
            if src in top_nodes and tgt in top_nodes:
                score = float(row["combined_score"]) if pd.notna(row["combined_score"]) else 0.5
                layer = str(row.get("layer", "autoantibody_ppi"))
                if not G.has_edge(src, tgt):
                    G.add_edge(src, tgt, combined_score=score, layer=layer)

    # Filter to connected component containing seeds
    if G.number_of_nodes() > 0:
        components = list(nx.connected_components(G))
        main_comp = max(components, key=len)
        G = G.subgraph(main_comp).copy()

    conv_df = conv_df[conv_df["gene_symbol"].isin(G.nodes())]
    logger.info("Visualization subgraph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G, conv_df


def _node_sizes(G: nx.Graph, conv_df: pd.DataFrame) -> list[float]:
    """Compute node sizes based on convergence score."""
    score_map = dict(zip(conv_df["gene_symbol"], conv_df["convergence_score"]))
    sizes = []
    for n in G.nodes():
        score = score_map.get(n, 0.01)
        sizes.append(80 + 800 * score)
    return sizes


def _get_layout(G: nx.Graph) -> dict:
    """Compute spring layout for the graph."""
    return nx.spring_layout(G, k=1.5 / (G.number_of_nodes() ** 0.5), iterations=50, seed=42)


def plot_overview(G: nx.Graph, conv_df: pd.DataFrame, pos: dict) -> None:
    """Plot 1: Overview network with node types colored."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    seed_set = set(get_gene_symbols())
    cytokine_set = set(get_cytokine_gene_symbols())
    sizes = _node_sizes(G, conv_df)

    # Color by node type
    colors = []
    for n in G.nodes():
        if n in seed_set:
            colors.append("#e41a1c")
        elif n in cytokine_set:
            colors.append("#FF5722")
        else:
            colors.append("#2196F3")

    # Edge colors by layer
    edge_colors = []
    for u, v in G.edges():
        layer = G[u][v].get("layer", "autoantibody_ppi")
        edge_colors.append(LAYER_COLORS.get(layer, "#cccccc"))

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color=edge_colors, width=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.8, edgecolors="white", linewidths=0.5)

    # Label seeds and cytokines
    labels = {n: n for n in G.nodes() if n in seed_set or n in cytokine_set}
    # Also label top 10 novel hubs
    novel_top = conv_df[~conv_df["gene_symbol"].isin(seed_set | cytokine_set)].nlargest(10, "convergence_score")
    for _, row in novel_top.iterrows():
        if row["gene_symbol"] in G.nodes():
            labels[row["gene_symbol"]] = row["gene_symbol"]

    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e41a1c", markersize=10, label="Seed protein"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722", markersize=10, label="Cytokine layer"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3", markersize=10, label="Interactor"),
        Line2D([0], [0], color=LAYER_COLORS["autoantibody_ppi"], linewidth=2, label="Autoantibody PPI edge"),
        Line2D([0], [0], color=LAYER_COLORS["cytokine_amplification"], linewidth=2, label="Cytokine edge"),
        Line2D([0], [0], color=LAYER_COLORS["cross_layer"], linewidth=2, label="Cross-layer edge"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title("PANDAS Autoantibody Target Network\n(Two-Layer: PPI + Cytokine Amplification)", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    path = OUTPUT_DIR / "01_network_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved overview plot to %s", path)


def plot_symptom_domains(G: nx.Graph, conv_df: pd.DataFrame, pos: dict) -> None:
    """Plot 2: Nodes colored by symptom domain (primary domain)."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    domain_map = dict(zip(conv_df["gene_symbol"], conv_df["symptom_domains"]))
    sizes = _node_sizes(G, conv_df)

    colors = []
    for n in G.nodes():
        domains = domain_map.get(n, "")
        if domains and isinstance(domains, str) and domains.strip():
            primary = domains.split(",")[0]
            colors.append(SYMPTOM_COLORS.get(primary, "#dddddd"))
        else:
            colors.append("#dddddd")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1, edge_color="#cccccc", width=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.8, edgecolors="white", linewidths=0.5)

    # Label seed proteins
    seed_set = set(get_gene_symbols())
    labels = {n: n for n in G.nodes() if n in seed_set}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=d.replace("_", " ").title())
        for d, c in SYMPTOM_COLORS.items()
    ]
    legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="#dddddd", markersize=10, label="No mapping"))
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.9, ncol=2)

    ax.set_title("PANDAS Network: Symptom Domain Mapping\n(Node color = primary symptom domain)", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    path = OUTPUT_DIR / "02_symptom_domain_coloring.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved symptom domain plot to %s", path)


def plot_pathway_communities(G: nx.Graph, conv_df: pd.DataFrame, pos: dict) -> None:
    """Plot 3: Nodes sized by enriched pathway count, colored by mechanism."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    enriched_map = dict(zip(conv_df["gene_symbol"], conv_df["total_enriched_pathways"]))
    seed_dict = get_seed_dict()
    seed_set = set(get_gene_symbols())
    cytokine_set = set(get_cytokine_gene_symbols())

    # Size by enriched pathway count
    sizes = []
    for n in G.nodes():
        pw_count = enriched_map.get(n, 0)
        sizes.append(30 + 5 * pw_count)

    # Color by mechanism category for seeds, neutral for others
    colors = []
    for n in G.nodes():
        if n in seed_set:
            mech = seed_dict.get(n, {}).get("mechanism_category", "")
            colors.append(MECHANISM_COLORS.get(mech, "#666666"))
        elif n in cytokine_set:
            colors.append("#FF5722")
        else:
            pw = enriched_map.get(n, 0)
            if pw > 50:
                colors.append("#1a237e")
            elif pw > 20:
                colors.append("#42a5f5")
            else:
                colors.append("#bbdefb")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1, edge_color="#cccccc", width=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.8, edgecolors="white", linewidths=0.5)

    # Label high-pathway nodes
    labels = {}
    for n in G.nodes():
        pw = enriched_map.get(n, 0)
        if pw > 80 or n in seed_set:
            labels[n] = n
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight="bold", ax=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=f"Seed: {m}")
        for m, c in MECHANISM_COLORS.items()
    ]
    legend_elements.extend([
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722", markersize=10, label="Cytokine"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1a237e", markersize=10, label=">50 enriched pathways"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#42a5f5", markersize=10, label="20-50 enriched pathways"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#bbdefb", markersize=10, label="<20 enriched pathways"),
    ])
    ax.legend(handles=legend_elements, loc="upper left", fontsize=7, framealpha=0.9)

    ax.set_title("PANDAS Network: Pathway Convergence\n(Node size = enriched pathway count)", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    path = OUTPUT_DIR / "03_pathway_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved pathway convergence plot to %s", path)


def plot_brain_expression(G: nx.Graph, conv_df: pd.DataFrame, pos: dict) -> None:
    """Plot 4: Node size by brain region expression breadth."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    brain_map = dict(zip(conv_df["gene_symbol"], conv_df["brain_regions"]))
    seed_set = set(get_gene_symbols())

    sizes = []
    colors = []
    for n in G.nodes():
        raw = brain_map.get(n, "")
        regions = raw if isinstance(raw, str) else ""
        n_regions = len(regions.split(",")) if regions.strip() else 0
        sizes.append(30 + 80 * n_regions)

        if "basal_ganglia" in regions:
            colors.append("#d32f2f")
        elif "prefrontal_cortex" in regions:
            colors.append("#1565c0")
        elif "hippocampus" in regions:
            colors.append("#2e7d32")
        elif "thalamus" in regions:
            colors.append("#f57f17")
        elif n_regions > 0:
            colors.append("#90a4ae")
        else:
            colors.append("#e0e0e0")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1, edge_color="#cccccc", width=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.8, edgecolors="white", linewidths=0.5)

    labels = {n: n for n in G.nodes() if n in seed_set}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#d32f2f", markersize=10, label="Basal ganglia enriched"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1565c0", markersize=10, label="Prefrontal cortex enriched"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2e7d32", markersize=10, label="Hippocampus enriched"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f57f17", markersize=10, label="Thalamus enriched"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#90a4ae", markersize=10, label="Other brain region"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e0e0e0", markersize=10, label="No expression data"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title("PANDAS Network: Brain Region Expression\n(Node size = number of brain regions; color = primary region)", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    path = OUTPUT_DIR / "04_brain_expression_overlay.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved brain expression plot to %s", path)


def plot_cytokine_layer(G: nx.Graph, conv_df: pd.DataFrame, pos: dict) -> None:
    """Plot 5: Cytokine amplification layer highlighted."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    seed_set = set(get_gene_symbols())
    cytokine_set = set(get_cytokine_gene_symbols())
    cross_layer_map = dict(zip(conv_df["gene_symbol"], conv_df["is_cross_layer"]))
    sizes = _node_sizes(G, conv_df)

    # Draw edges first, color by layer
    for u, v in G.edges():
        layer = G[u][v].get("layer", "autoantibody_ppi")
        color = LAYER_COLORS.get(layer, "#cccccc")
        width = 2.0 if layer in ("cytokine_amplification", "cross_layer") else 0.3
        alpha = 0.5 if layer in ("cytokine_amplification", "cross_layer") else 0.08
        ax.plot(
            [pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
            color=color, alpha=alpha, linewidth=width, zorder=1,
        )

    # Node colors
    colors = []
    borders = []
    for n in G.nodes():
        if n in cytokine_set:
            colors.append("#FF5722")
            borders.append("#BF360C")
        elif n in seed_set:
            colors.append("#e41a1c")
            borders.append("#b71c1c")
        elif cross_layer_map.get(n, False):
            colors.append("#9C27B0")
            borders.append("#4A148C")
        else:
            colors.append("#90CAF9")
            borders.append("white")

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors,
                           alpha=0.85, edgecolors=borders, linewidths=1.0)

    # Label cytokines and seeds
    labels = {n: n for n in G.nodes() if n in seed_set or n in cytokine_set}
    # Also label cross-layer hubs
    cross_nodes = conv_df[conv_df["is_cross_layer"] == True].nlargest(5, "convergence_score")
    for _, row in cross_nodes.iterrows():
        if row["gene_symbol"] in G.nodes() and row["gene_symbol"] not in labels:
            labels[row["gene_symbol"]] = row["gene_symbol"]

    nx.draw_networkx_labels(G, pos, labels, font_size=7, font_weight="bold", ax=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e41a1c", markersize=10, label="Seed protein (Layer 1)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722", markersize=10, label="Cytokine protein (Layer 2)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9C27B0", markersize=10, label="Cross-layer node"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#90CAF9", markersize=10, label="PPI interactor"),
        Line2D([0], [0], color=LAYER_COLORS["autoantibody_ppi"], linewidth=2, label="Autoantibody PPI"),
        Line2D([0], [0], color=LAYER_COLORS["cytokine_amplification"], linewidth=2, label="Cytokine amplification"),
        Line2D([0], [0], color=LAYER_COLORS["cross_layer"], linewidth=2, label="Cross-layer"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title("PANDAS Network: Two-Hit Framework\n(Layer 1: Autoantibody PPI + Layer 2: Cytokine Amplification)", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    path = OUTPUT_DIR / "05_cytokine_layer.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved cytokine layer plot to %s", path)


def plot_folr1_neighborhood(conv_df: pd.DataFrame) -> None:
    """Plot 6: FOLR1 local neighborhood with folate pathway highlighted."""
    network_path = DATA_DIR / "extended_network.tsv"
    chunks = pd.read_csv(
        network_path, sep="\t",
        usecols=["source", "target", "combined_score", "layer"],
        chunksize=5000,
    )

    # Get FOLR1 first-degree neighbors
    folr1_edges = []
    for chunk in chunks:
        mask = (chunk["source"] == "FOLR1") | (chunk["target"] == "FOLR1")
        folr1_edges.append(chunk[mask])

    if not folr1_edges:
        logger.warning("FOLR1 not found in network")
        return

    folr1_df = pd.concat(folr1_edges)
    neighbors = set(folr1_df["source"]) | set(folr1_df["target"])

    # Build local graph including inter-neighbor edges
    G_local = nx.Graph()
    chunks = pd.read_csv(
        network_path, sep="\t",
        usecols=["source", "target", "combined_score", "layer"],
        chunksize=5000,
    )
    for chunk in chunks:
        for _, row in chunk.iterrows():
            src, tgt = str(row["source"]), str(row["target"])
            if src in neighbors and tgt in neighbors:
                G_local.add_edge(src, tgt,
                                 combined_score=float(row["combined_score"]) if pd.notna(row["combined_score"]) else 0.5,
                                 layer=str(row.get("layer", "")))

    if G_local.number_of_nodes() == 0:
        logger.warning("No FOLR1 neighborhood found")
        return

    pos = nx.spring_layout(G_local, k=2.0 / (G_local.number_of_nodes() ** 0.5), iterations=50, seed=42)
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    seed_set = set(get_gene_symbols())
    cytokine_set = set(get_cytokine_gene_symbols())

    # Load pathway info for folate-related pathways
    kegg_path = DATA_DIR / "kegg_node_pathway_mapping.tsv"
    folate_genes: set[str] = set()
    if kegg_path.exists():
        kegg_df = pd.read_csv(kegg_path, sep="\t")
        folate_mask = kegg_df["pathway_name"].str.contains("folate|Folate|one carbon|One carbon", case=False, na=False)
        folate_genes = set(kegg_df[folate_mask]["gene_symbol"])

    colors = []
    sizes = []
    for n in G_local.nodes():
        if n == "FOLR1":
            colors.append("#ff7f00")
            sizes.append(600)
        elif n in seed_set:
            colors.append("#e41a1c")
            sizes.append(300)
        elif n in cytokine_set:
            colors.append("#FF5722")
            sizes.append(200)
        elif n in folate_genes:
            colors.append("#FFA726")
            sizes.append(200)
        else:
            colors.append("#90CAF9")
            sizes.append(100)

    nx.draw_networkx_edges(G_local, pos, ax=ax, alpha=0.2, edge_color="#cccccc", width=0.5)
    nx.draw_networkx_nodes(G_local, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.85, edgecolors="white", linewidths=0.5)

    # Label all nodes in small neighborhood
    if G_local.number_of_nodes() <= 60:
        labels = {n: n for n in G_local.nodes()}
    else:
        labels = {n: n for n in G_local.nodes() if n in seed_set or n in cytokine_set or n in folate_genes or n == "FOLR1"}
    nx.draw_networkx_labels(G_local, pos, labels, font_size=6, font_weight="bold", ax=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f00", markersize=12, label="FOLR1 (folate receptor)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FFA726", markersize=10, label="Folate pathway gene"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e41a1c", markersize=10, label="Other seed protein"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722", markersize=10, label="Cytokine protein"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#90CAF9", markersize=10, label="Interactor"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8, framealpha=0.9)

    ax.set_title(f"FOLR1 Neighborhood ({G_local.number_of_nodes()} nodes)\nFolate Transport Disruption Hub", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    path = OUTPUT_DIR / "06_folr1_folate_hub.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved FOLR1 neighborhood plot to %s", path)


def plot_patient_subgroups(conv_df: pd.DataFrame) -> None:
    """Plot 7: Patient subgroup stratification views (2x2 grid)."""
    subgroups_path = DATA_DIR / "patient_subgroups.json"
    if not subgroups_path.exists():
        logger.warning("Patient subgroups file not found")
        return

    with open(subgroups_path) as f:
        subgroup_data = json.load(f)

    subnetwork_path = DATA_DIR / "subgroup_subnetworks.tsv"
    if not subnetwork_path.exists():
        return

    subnetwork_df = pd.read_csv(subnetwork_path, sep="\t")
    seed_set = set(get_gene_symbols())

    # Pick 4 representative subgroups
    display_subgroups = [
        "dopaminergic_dominant", "calcium_signaling",
        "metabolic_surface", "folate_disruption",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    for idx, sg_name in enumerate(display_subgroups):
        ax = axes[idx]
        sg_info = subgroup_data.get("subgroups", {}).get(sg_name, {})
        sg_edges = subnetwork_df[subnetwork_df["subgroup"] == sg_name]

        if sg_edges.empty:
            ax.text(0.5, 0.5, f"{sg_name}\nNo edges", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            continue

        # Build local graph (limit to 100 nodes)
        G_sub = nx.Graph()
        for _, row in sg_edges.iterrows():
            src, tgt = str(row["source"]), str(row["target"])
            G_sub.add_edge(src, tgt)

        # Limit nodes if too many
        if G_sub.number_of_nodes() > 100:
            sg_seeds = set(sg_info.get("seed_genes", []))
            # Keep seeds + their highest-degree neighbors
            keep = set(sg_seeds)
            for s in sg_seeds:
                if s in G_sub:
                    neighbors = sorted(G_sub.neighbors(s), key=lambda x: G_sub.degree(x), reverse=True)
                    keep.update(neighbors[:15])
            # Fill to 100 with highest-degree nodes
            remaining = sorted(G_sub.nodes() - keep, key=lambda x: G_sub.degree(x), reverse=True)
            keep.update(remaining[:100 - len(keep)])
            G_sub = G_sub.subgraph(keep).copy()

        pos_sub = nx.spring_layout(G_sub, k=2.0 / max(1, G_sub.number_of_nodes() ** 0.5), iterations=40, seed=42)

        sg_seeds = set(sg_info.get("seed_genes", []))
        colors = []
        sizes = []
        for n in G_sub.nodes():
            if n in sg_seeds:
                colors.append(SUBGROUP_COLORS.get(sg_name, "#e41a1c"))
                sizes.append(300)
            elif n in seed_set:
                colors.append("#e41a1c")
                sizes.append(150)
            else:
                colors.append("#90CAF9")
                sizes.append(50)

        nx.draw_networkx_edges(G_sub, pos_sub, ax=ax, alpha=0.15, edge_color="#cccccc", width=0.3)
        nx.draw_networkx_nodes(G_sub, pos_sub, ax=ax, node_size=sizes, node_color=colors, alpha=0.8, edgecolors="white", linewidths=0.3)

        # Label seeds
        labels = {n: n for n in G_sub.nodes() if n in sg_seeds}
        nx.draw_networkx_labels(G_sub, pos_sub, labels, font_size=7, font_weight="bold", ax=ax)

        phenotype = sg_info.get("clinical_phenotype", "")
        ax.set_title(f"{sg_name.replace('_', ' ').title()}\n{phenotype}\n({G_sub.number_of_nodes()} nodes)", fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle("PANDAS Patient Subgroup Network Stratification", fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    path = OUTPUT_DIR / "07_patient_subgroups.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved patient subgroup plot to %s", path)


def plot_novel_targets(G: nx.Graph, conv_df: pd.DataFrame, pos: dict) -> None:
    """Plot 8: Novel therapeutic targets highlighted."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))

    seed_set = set(get_gene_symbols())
    cytokine_set = set(get_cytokine_gene_symbols())
    novel_df = conv_df[conv_df["node_type"] == "interactor"].nlargest(20, "convergence_score")
    novel_set = set(novel_df["gene_symbol"])
    druggable_map = dict(zip(conv_df["gene_symbol"], conv_df["is_druggable"]))
    sizes = _node_sizes(G, conv_df)

    colors = []
    borders = []
    for n in G.nodes():
        if n in novel_set:
            colors.append("#FFD600")
            borders.append("#F57F17")
        elif n in seed_set:
            colors.append("#e41a1c")
            borders.append("white")
        elif n in cytokine_set:
            colors.append("#FF5722")
            borders.append("white")
        else:
            colors.append("#e0e0e0")
            borders.append("white")

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.08, edge_color="#cccccc", width=0.3)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=colors, alpha=0.85,
                           edgecolors=borders, linewidths=1.0)

    labels = {n: n for n in G.nodes() if n in novel_set or n in seed_set}
    nx.draw_networkx_labels(G, pos, labels, font_size=6, font_weight="bold", ax=ax)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#FFD600", markeredgecolor="#F57F17",
               markersize=14, label="Novel therapeutic target"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e41a1c", markersize=10, label="Seed protein"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#FF5722", markersize=10, label="Cytokine protein"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e0e0e0", markersize=10, label="Other interactor"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9, framealpha=0.9)

    ax.set_title("PANDAS Network: Novel Therapeutic Targets\n(Top 20 convergence hubs not previously proposed for PANDAS)", fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()

    path = OUTPUT_DIR / "08_novel_targets.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved novel targets plot to %s", path)


def run() -> None:
    """Generate all network visualizations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading network and convergence data for visualization...")
    G, conv_df = load_network_for_viz(max_nodes=200)
    if G.number_of_nodes() == 0:
        logger.error("No nodes in visualization subgraph")
        return

    pos = _get_layout(G)

    # Generate all plots
    logger.info("Generating visualizations...")
    plot_overview(G, conv_df, pos)
    plot_symptom_domains(G, conv_df, pos)
    plot_pathway_communities(G, conv_df, pos)
    plot_brain_expression(G, conv_df, pos)
    plot_cytokine_layer(G, conv_df, pos)
    plot_folr1_neighborhood(conv_df)
    plot_patient_subgroups(conv_df)
    plot_novel_targets(G, conv_df, pos)

    logger.info("=== All visualizations saved to %s ===", OUTPUT_DIR)


if __name__ == "__main__":
    run()
