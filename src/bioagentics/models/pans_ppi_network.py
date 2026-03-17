"""Protein-protein interaction network analysis for PANS variant genes.

Queries STRING database for interactions among PANS variant genes, constructs
a networkx graph, computes centrality metrics, identifies hub genes and network
modules, and generates a visualization colored by pathway axis.

Usage:
    uv run python -m bioagentics.models.pans_ppi_network [--dest DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import requests

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/pandas_pans/pans-genetic-variant-pathway-analysis")

STRING_API = "https://string-db.org/api"
SPECIES_HUMAN = 9606
MIN_CONFIDENCE = 400  # STRING confidence score threshold (0-1000)
TIMEOUT = 60

# Colors for pathway axes
AXIS_COLORS = {
    "DDR-cGAS-STING/AIM2 inflammasome": "#E74C3C",
    "Mitochondrial-innate immunity": "#3498DB",
    "Gut-immune": "#2ECC71",
    "Lectin complement": "#F39C12",
    "Chromatin/neuroprotection": "#9B59B6",
}


def query_string_interactions(gene_symbols: list[str],
                              species: int = SPECIES_HUMAN,
                              min_score: int = MIN_CONFIDENCE) -> list[dict]:
    """Query STRING API for protein-protein interactions.

    Args:
        gene_symbols: List of human gene symbols.
        species: NCBI taxonomy ID (default: 9606 for human).
        min_score: Minimum combined confidence score (0-1000).

    Returns:
        List of interaction dicts with preferredName_A, preferredName_B, score.
    """
    url = f"{STRING_API}/json/network"
    params = {
        "identifiers": "%0d".join(gene_symbols),
        "species": species,
        "required_score": min_score,
        "caller_identity": "bioagentics",
    }

    logger.info("Querying STRING for %d genes (score >= %d)...",
                len(gene_symbols), min_score)

    resp = requests.post(url, data=params, timeout=TIMEOUT)
    resp.raise_for_status()
    interactions = resp.json()

    logger.info("STRING returned %d interactions", len(interactions))
    return interactions


def build_network(interactions: list[dict],
                  gene_axis_map: dict[str, str]) -> nx.Graph:
    """Build networkx graph from STRING interactions.

    Nodes are annotated with pathway_axis. Edges have combined_score weight.
    """
    G = nx.Graph()

    # Add all PANS genes as nodes (even if no interactions)
    for gene, axis in gene_axis_map.items():
        G.add_node(gene, pathway_axis=axis)

    # Add edges from STRING
    for interaction in interactions:
        gene_a = interaction.get("preferredName_A", "")
        gene_b = interaction.get("preferredName_B", "")
        score = interaction.get("score", 0)

        if gene_a and gene_b:
            G.add_edge(gene_a, gene_b, combined_score=score)
            # Ensure nodes have axis annotation
            for g in (gene_a, gene_b):
                if "pathway_axis" not in G.nodes[g]:
                    G.nodes[g]["pathway_axis"] = gene_axis_map.get(g, "unknown")

    logger.info("Network: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def compute_metrics(G: nx.Graph) -> pd.DataFrame:
    """Compute centrality metrics for all nodes.

    Returns DataFrame with columns: gene_symbol, pathway_axis, degree,
    degree_centrality, betweenness_centrality, clustering_coefficient.
    """
    if G.number_of_nodes() == 0:
        return pd.DataFrame()

    degree_cent = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)

    records = []
    for node in G.nodes():
        records.append({
            "gene_symbol": node,
            "pathway_axis": G.nodes[node].get("pathway_axis", "unknown"),
            "degree": G.degree(node),
            "degree_centrality": degree_cent[node],
            "betweenness_centrality": betweenness[node],
            "clustering_coefficient": clustering[node],
        })

    df = pd.DataFrame(records)
    return df.sort_values("degree_centrality", ascending=False).reset_index(drop=True)


def detect_modules(G: nx.Graph) -> dict[str, int]:
    """Detect network modules using greedy modularity communities.

    Returns dict mapping gene_symbol -> module_id.
    """
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return {n: 0 for n in G.nodes()}

    communities = nx.community.greedy_modularity_communities(G)
    module_map = {}
    for i, community in enumerate(communities):
        for node in community:
            module_map[node] = i

    logger.info("Detected %d modules", len(communities))
    return module_map


def identify_hub_genes(metrics_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Identify hub genes by degree and betweenness centrality."""
    if metrics_df.empty:
        return metrics_df

    # Score by combined rank of degree and betweenness
    df = metrics_df.copy()
    df["degree_rank"] = df["degree_centrality"].rank(ascending=False)
    df["betweenness_rank"] = df["betweenness_centrality"].rank(ascending=False)
    df["hub_score"] = (df["degree_rank"] + df["betweenness_rank"]) / 2
    df = df.sort_values("hub_score")

    return df.head(top_n).drop(columns=["degree_rank", "betweenness_rank", "hub_score"])


def plot_network(G: nx.Graph, metrics_df: pd.DataFrame,
                 dest: Path, title: str = "PANS Variant Gene PPI Network") -> None:
    """Generate network visualization colored by pathway axis, sized by centrality."""
    if G.number_of_nodes() == 0:
        logger.warning("Empty network, skipping plot")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    # Layout
    if G.number_of_edges() > 0:
        pos = nx.spring_layout(G, k=2, seed=42)
    else:
        pos = nx.circular_layout(G)

    # Node colors by pathway axis
    node_colors = []
    for node in G.nodes():
        axis = G.nodes[node].get("pathway_axis", "unknown")
        node_colors.append(AXIS_COLORS.get(axis, "#95A5A6"))

    # Node sizes by degree centrality
    if not metrics_df.empty:
        cent_map = dict(zip(metrics_df["gene_symbol"], metrics_df["degree_centrality"]))
        node_sizes = [300 + 2000 * cent_map.get(n, 0) for n in G.nodes()]
    else:
        node_sizes = [500] * G.number_of_nodes()

    # Draw
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color="gray")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.8,
                           edgecolors="black", linewidths=0.5)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight="bold")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color, label=axis)
                       for axis, color in AXIS_COLORS.items()]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved network plot: %s", dest)


def save_network_json(G: nx.Graph, metrics_df: pd.DataFrame,
                      module_map: dict[str, int], dest: Path) -> None:
    """Save network data as JSON for downstream analysis."""
    data = {
        "nodes": [],
        "edges": [],
        "modules": module_map,
    }

    for node in G.nodes():
        node_data = {
            "gene_symbol": node,
            "pathway_axis": G.nodes[node].get("pathway_axis", "unknown"),
            "module": module_map.get(node, -1),
        }
        if not metrics_df.empty:
            row = metrics_df[metrics_df["gene_symbol"] == node]
            if not row.empty:
                node_data["degree"] = int(row.iloc[0]["degree"])
                node_data["degree_centrality"] = float(row.iloc[0]["degree_centrality"])
                node_data["betweenness_centrality"] = float(row.iloc[0]["betweenness_centrality"])
        data["nodes"].append(node_data)

    for u, v, edge_data in G.edges(data=True):
        data["edges"].append({
            "source": u,
            "target": v,
            "combined_score": edge_data.get("combined_score", 0),
        })

    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Saved network JSON: %s", dest)


def run_pans_ppi_analysis(dest_dir: Path | None = None) -> tuple[nx.Graph, pd.DataFrame]:
    """Run the full PPI network analysis pipeline.

    Returns (networkx Graph, metrics DataFrame).
    """
    from bioagentics.data.pans_variants import get_pans_variant_genes

    if dest_dir is None:
        dest_dir = OUTPUT_DIR

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Get gene data
    gene_df = get_pans_variant_genes()
    gene_symbols = gene_df["gene_symbol"].tolist()
    gene_axis_map = dict(zip(gene_df["gene_symbol"], gene_df["pathway_axis"]))

    # Query STRING
    interactions = query_string_interactions(gene_symbols)

    # Build network
    G = build_network(interactions, gene_axis_map)

    # Compute metrics
    metrics_df = compute_metrics(G)
    if not metrics_df.empty:
        metrics_df.to_csv(dest_dir / "ppi_network_metrics.csv", index=False)

    # Detect modules
    module_map = detect_modules(G)

    # Identify hubs
    hubs = identify_hub_genes(metrics_df)
    if not hubs.empty:
        logger.info("Hub genes: %s", ", ".join(hubs["gene_symbol"].tolist()))
        hubs.to_csv(dest_dir / "ppi_hub_genes.csv", index=False)

    # Save outputs
    save_network_json(G, metrics_df, module_map, dest_dir / "ppi_network.json")
    plot_network(G, metrics_df, dest_dir / "ppi_network.png")

    return G, metrics_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PANS variant gene PPI network analysis via STRING"
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    G, metrics_df = run_pans_ppi_analysis(dest_dir=args.dest)

    print(f"\nNetwork: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    if not metrics_df.empty:
        print("\nTop genes by degree centrality:")
        for _, row in metrics_df.head(5).iterrows():
            print(f"  {row['gene_symbol']} ({row['pathway_axis']}): "
                  f"degree={row['degree']}, centrality={row['degree_centrality']:.3f}")


if __name__ == "__main__":
    main()
