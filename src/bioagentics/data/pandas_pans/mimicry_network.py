"""Network visualization of GAS-human molecular mimicry relationships.

Generates network graphs showing GAS protein → human neuronal protein mimicry.
Nodes colored by type (GAS by serotype, human by brain region).
Edges weighted by composite score.

Usage:
    uv run python -m bioagentics.data.pandas_pans.mimicry_network [--dest DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path("data/pandas_pans/gas-molecular-mimicry-mapping")
SCREEN_DIR = PROJECT_DIR / "mimicry_screen"
OUTPUT_BASE = Path("output/pandas_pans/gas-molecular-mimicry-mapping")
SCORING_DIR = OUTPUT_BASE / "target_scoring"
OUTPUT_DIR = OUTPUT_BASE / "network_visualization"

# Serotype colors
SEROTYPE_COLORS = {
    "M1": "#E74C3C",   # red
    "M3": "#E67E22",   # orange
    "M5": "#F1C40F",   # yellow
    "M12": "#2ECC71",  # green
    "M18": "#3498DB",  # blue
    "M49": "#9B59B6",  # purple
}

HUMAN_COLOR = "#1ABC9C"  # teal for human targets
KNOWN_COLOR = "#E91E63"  # pink for known PANDAS targets

# GAS organism codes to serotypes
ORGANISM_TO_SEROTYPE = {
    "STRP1": "M1",
    "STRP3": "M3",
    "STRPC": "M5",   # M5 Manfredo
    "STRPZ": "M49",  # M49 NZ131
}


def get_serotype(qseqid: str) -> str:
    """Extract serotype from GAS protein ID."""
    # sp|P0C0G7|G3P_STRP1 → STRP1 → M1
    parts = qseqid.split("|")
    if len(parts) >= 3:
        org = parts[2].split("_")[-1]
        return ORGANISM_TO_SEROTYPE.get(org, "unknown")
    return "unknown"


def get_short_name(qseqid: str) -> str:
    """Get a short readable name for a GAS protein."""
    parts = qseqid.split("|")
    if len(parts) >= 3:
        return parts[2]  # e.g., G3P_STRP1
    return qseqid[:20]


def build_network(hits_df: pd.DataFrame, scored_df: pd.DataFrame | None) -> nx.Graph:
    """Build a networkx graph from mimicry hits."""
    G = nx.Graph()

    # Score lookup
    scores = {}
    if scored_df is not None and not scored_df.empty:
        for _, row in scored_df.iterrows():
            scores[row["human_accession"]] = row.get("composite_score", 0)

    # Add edges from hits (deduplicated by protein pair)
    seen = set()
    for _, row in hits_df.iterrows():
        gas_id = row["qseqid"]
        human_acc = row["human_accession"]
        human_gene = str(row.get("human_gene", "")) if pd.notna(row.get("human_gene")) else ""
        known = bool(row.get("known_target", False))

        pair_key = (gas_id, human_acc)
        if pair_key in seen:
            continue
        seen.add(pair_key)

        serotype = get_serotype(gas_id)
        short_name = get_short_name(gas_id)

        # Add GAS node
        if gas_id not in G:
            G.add_node(gas_id, type="gas", serotype=serotype, label=short_name)

        # Add human node
        display_name = human_gene if human_gene else human_acc
        if human_acc not in G:
            G.add_node(human_acc, type="human", label=display_name,
                       known_target=known,
                       score=scores.get(human_acc, 0))

        # Add edge
        G.add_edge(gas_id, human_acc,
                   pident=row["pident"],
                   bitscore=row["bitscore"],
                   weight=row["bitscore"] / 100)  # scale for layout

    return G


def draw_network(G: nx.Graph, dest: Path) -> None:
    """Draw publication-quality network figure."""
    if len(G) == 0:
        logger.warning("Empty graph, skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(16, 12))

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=100, seed=42)

    # Separate node types
    gas_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "gas"]
    human_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "human"]

    # Draw edges with variable width based on bitscore
    edge_widths = [G[u][v].get("weight", 1.0) for u, v in G.edges()]
    max_width = max(edge_widths) if edge_widths else 1
    edge_widths_norm = [w / max_width * 3 + 0.5 for w in edge_widths]

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3,
                           width=edge_widths_norm, edge_color="#888888")

    # Draw GAS nodes colored by serotype
    for serotype, color in SEROTYPE_COLORS.items():
        nodes = [n for n in gas_nodes
                 if G.nodes[n].get("serotype") == serotype]
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax,
                                   node_color=color, node_size=200,
                                   node_shape="s", alpha=0.8)

    # Draw unknown serotype GAS nodes
    unknown_gas = [n for n in gas_nodes
                   if G.nodes[n].get("serotype") not in SEROTYPE_COLORS]
    if unknown_gas:
        nx.draw_networkx_nodes(G, pos, nodelist=unknown_gas, ax=ax,
                               node_color="#95A5A6", node_size=200,
                               node_shape="s", alpha=0.8)

    # Draw human nodes — known targets highlighted
    known_human = [n for n in human_nodes if G.nodes[n].get("known_target")]
    novel_human = [n for n in human_nodes if not G.nodes[n].get("known_target")]

    if novel_human:
        scores = [G.nodes[n].get("score", 0) * 800 + 400 for n in novel_human]
        nx.draw_networkx_nodes(G, pos, nodelist=novel_human, ax=ax,
                               node_color=HUMAN_COLOR, node_size=scores,
                               node_shape="o", alpha=0.9, edgecolors="black",
                               linewidths=1.5)
    if known_human:
        scores = [G.nodes[n].get("score", 0) * 800 + 400 for n in known_human]
        nx.draw_networkx_nodes(G, pos, nodelist=known_human, ax=ax,
                               node_color=KNOWN_COLOR, node_size=scores,
                               node_shape="o", alpha=0.9, edgecolors="black",
                               linewidths=2.5)

    # Labels for human targets only (GAS nodes are too dense)
    human_labels = {n: G.nodes[n].get("label", n) for n in human_nodes}
    nx.draw_networkx_labels(G, pos, labels=human_labels, ax=ax,
                            font_size=10, font_weight="bold")

    # Legend
    legend_handles = []
    for serotype, color in SEROTYPE_COLORS.items():
        legend_handles.append(mpatches.Patch(color=color, label=f"GAS {serotype}"))
    legend_handles.append(mpatches.Patch(color=HUMAN_COLOR, label="Human target (novel)"))
    legend_handles.append(mpatches.Patch(color=KNOWN_COLOR, label="Human target (known PANDAS)"))

    ax.legend(handles=legend_handles, loc="upper left", fontsize=9,
              framealpha=0.9, title="Node Type")

    ax.set_title("GAS → Human Molecular Mimicry Network\n"
                 "Edge width ∝ DIAMOND bitscore · Node size ∝ composite score",
                 fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(dest / "mimicry_network.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Network figure saved: mimicry_network.png")


def export_graph_data(G: nx.Graph, dest: Path) -> None:
    """Export graph data as TSV for downstream use."""
    # Nodes
    node_rows = []
    for n, d in G.nodes(data=True):
        node_rows.append({
            "id": n,
            "type": d.get("type", ""),
            "label": d.get("label", ""),
            "serotype": d.get("serotype", ""),
            "known_target": d.get("known_target", ""),
            "score": d.get("score", ""),
        })
    pd.DataFrame(node_rows).to_csv(dest / "network_nodes.tsv", sep="\t", index=False)

    # Edges
    edge_rows = []
    for u, v, d in G.edges(data=True):
        edge_rows.append({
            "source": u,
            "target": v,
            "pident": d.get("pident", ""),
            "bitscore": d.get("bitscore", ""),
        })
    pd.DataFrame(edge_rows).to_csv(dest / "network_edges.tsv", sep="\t", index=False)

    logger.info("Graph data exported: network_nodes.tsv, network_edges.tsv")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate network visualization of mimicry relationships",
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.dest.mkdir(parents=True, exist_ok=True)

    # Load filtered hits
    hits_path = SCREEN_DIR / "hits_filtered.tsv"
    if not hits_path.exists():
        raise FileNotFoundError(f"Run mimicry_screen.py first: {hits_path}")
    hits_df = pd.read_csv(hits_path, sep="\t")

    # Load scoring (optional)
    scored_df = None
    scored_path = SCORING_DIR / "ranked_targets.tsv"
    if scored_path.exists():
        scored_df = pd.read_csv(scored_path, sep="\t")
        logger.info("Loaded scoring data: %d targets", len(scored_df))

    # Build graph
    logger.info("Building network graph...")
    G = build_network(hits_df, scored_df)
    logger.info("Graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    gas_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "gas")
    human_nodes = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "human")
    logger.info("  GAS proteins: %d, Human targets: %d", gas_nodes, human_nodes)

    # Draw network
    logger.info("Generating network visualization...")
    draw_network(G, args.dest)

    # Export data
    export_graph_data(G, args.dest)

    logger.info("Done. Network visualization in %s", args.dest)


if __name__ == "__main__":
    main()
