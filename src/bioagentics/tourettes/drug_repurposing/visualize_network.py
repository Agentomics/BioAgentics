"""Generate network visualizations for TS drug-target-disease module interactions.

Creates publication-quality visualizations:
1. TS disease PPI network colored by functional modules
2. Drug-target overlay showing top candidates and their target connections
3. Pathway-level summary showing which TS pathways each candidate targets

Output: output/tourettes/ts-drug-repurposing-network/figures/

Usage:
    uv run python -m bioagentics.tourettes.drug_repurposing.visualize_network
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-drug-repurposing-network"
FIGURES_DIR = OUTPUT_DIR / "figures"

NETWORK_PATH = OUTPUT_DIR / "ts_disease_network.graphml"
MODULES_PATH = OUTPUT_DIR / "ts_network_modules.csv"
RANKED_PATH = OUTPUT_DIR / "ranked_candidates.csv"

# Module color palette
MODULE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#a65628", "#f781bf", "#999999", "#66c2a5", "#fc8d62",
    "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494",
]


def plot_disease_network(G: nx.Graph, modules_df: pd.DataFrame) -> None:
    """Plot TS disease PPI network colored by functional modules."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    # Get module assignments
    module_map = dict(zip(modules_df["gene"], modules_df["module_id"]))
    seed_set = set(modules_df[modules_df["is_seed"]]["gene"])

    # Assign colors by module
    node_colors = []
    node_sizes = []
    for node in G.nodes():
        mod_id = module_map.get(node, -1)
        color = MODULE_COLORS[mod_id % len(MODULE_COLORS)] if mod_id >= 0 else "#cccccc"
        node_colors.append(color)
        node_sizes.append(80 if node in seed_set else 15)

    # Layout — use spring layout with seed gene emphasis
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # Draw edges (thin, light)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.05, width=0.3, edge_color="#888888")

    # Draw non-seed nodes first
    non_seed_nodes = [n for n in G.nodes() if n not in seed_set]
    non_seed_colors = [node_colors[i] for i, n in enumerate(G.nodes()) if n not in seed_set]
    non_seed_sizes = [node_sizes[i] for i, n in enumerate(G.nodes()) if n not in seed_set]
    nx.draw_networkx_nodes(G, pos, nodelist=non_seed_nodes, node_color=non_seed_colors,
                           node_size=non_seed_sizes, ax=ax, alpha=0.4, linewidths=0)

    # Draw seed nodes on top
    seed_nodes = [n for n in G.nodes() if n in seed_set]
    seed_colors = [node_colors[i] for i, n in enumerate(G.nodes()) if n in seed_set]
    seed_sizes = [node_sizes[i] for i, n in enumerate(G.nodes()) if n in seed_set]
    nx.draw_networkx_nodes(G, pos, nodelist=seed_nodes, node_color=seed_colors,
                           node_size=seed_sizes, ax=ax, alpha=0.9, linewidths=1,
                           edgecolors="black")

    # Label seed genes
    seed_labels = {n: n for n in seed_nodes}
    nx.draw_networkx_labels(G, pos, labels=seed_labels, ax=ax, font_size=5,
                            font_weight="bold", alpha=0.8)

    # Legend
    unique_modules = sorted(modules_df["module_id"].unique())
    # Get pathway annotations per module
    module_labels = {}
    for mod_id in unique_modules:
        mod_df = modules_df[modules_df["module_id"] == mod_id]
        pathways = set()
        for p_str in mod_df["pathway_annotations"].dropna():
            if p_str:
                for p in str(p_str).split(";"):
                    pathways.add(p)
        top_pathway = sorted(pathways)[0] if pathways else f"module_{mod_id}"
        n_seeds = mod_df["is_seed"].sum()
        module_labels[mod_id] = f"M{mod_id}: {top_pathway} ({n_seeds} seeds)"

    patches = [
        mpatches.Patch(color=MODULE_COLORS[m % len(MODULE_COLORS)],
                       label=module_labels.get(m, f"Module {m}"))
        for m in unique_modules[:10]
    ]
    ax.legend(handles=patches, loc="upper left", fontsize=7, framealpha=0.9)

    ax.set_title("Tourette Syndrome Disease PPI Network\n"
                 f"({G.number_of_nodes()} proteins, {G.number_of_edges()} interactions, "
                 f"{len(unique_modules)} modules)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "ts_disease_network.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'ts_disease_network.png'}")


def plot_candidate_ranking(ranked_df: pd.DataFrame) -> None:
    """Plot horizontal bar chart of ranked candidates with evidence breakdown."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    n = len(ranked_df)
    y_pos = np.arange(n)

    # Stacked bar components
    components = [
        ("network_proximity_score", "Network Proximity", "#2196F3"),
        ("signature_score", "Signature Match", "#4CAF50"),
        ("pathway_relevance", "Pathway Relevance", "#FF9800"),
        ("safety_score", "Safety Profile", "#9C27B0"),
        ("clinical_precedent", "Clinical Precedent", "#F44336"),
    ]

    weights = {
        "network_proximity_score": 0.30,
        "signature_score": 0.20,
        "pathway_relevance": 0.20,
        "safety_score": 0.15,
        "clinical_precedent": 0.15,
    }

    left = np.zeros(n)
    for col, label, color in components:
        values = ranked_df[col].values * weights[col]
        ax.barh(y_pos, values, left=left, color=color, label=label, alpha=0.85, height=0.7)
        left += values

    # Add penalty markers
    for i, (_, row) in enumerate(ranked_df.iterrows()):
        if row["vmat2_penalty"] > 0:
            ax.annotate(f"-{row['vmat2_penalty']:.0%}",
                        (row["final_score"] + 0.02, i),
                        fontsize=7, color="red", fontweight="bold", va="center")

    # Labels
    labels = []
    for _, row in ranked_df.iterrows():
        suffix = ""
        if row["is_positive_control"]:
            suffix = " [+ctrl]"
        elif row["is_negative_control"]:
            suffix = " [-ctrl]"
        labels.append(f"#{int(row['rank'])} {row['drug_name']}{suffix}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Weighted Score", fontsize=11)
    ax.set_title("TS Drug Repurposing Candidates — Multi-Criteria Ranking",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1.0)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "candidate_ranking.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'candidate_ranking.png'}")


def plot_pathway_heatmap(ranked_df: pd.DataFrame) -> None:
    """Plot heatmap showing which TS pathways each candidate drug targets."""
    all_pathways = set()
    drug_pathway_map: dict[str, set[str]] = {}
    for _, row in ranked_df.iterrows():
        pathways = set(str(row["pathway_annotations"]).split(";")) if row["pathway_annotations"] else set()
        pathways.discard("")
        drug_pathway_map[row["drug_name"]] = pathways
        all_pathways.update(pathways)

    if not all_pathways:
        print("  No pathway annotations available for heatmap")
        return

    pathways = sorted(all_pathways)
    drugs = list(ranked_df["drug_name"])

    # Build matrix
    matrix = np.zeros((len(drugs), len(pathways)))
    for i, drug in enumerate(drugs):
        for j, pathway in enumerate(pathways):
            if pathway in drug_pathway_map.get(drug, set()):
                matrix[i, j] = 1.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", interpolation="nearest")

    ax.set_xticks(range(len(pathways)))
    ax.set_xticklabels([p.replace("_", "\n") for p in pathways],
                       rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(drugs)))
    ax.set_yticklabels([f"#{i+1} {d}" for i, d in enumerate(drugs)], fontsize=9)

    ax.set_title("Drug-Pathway Target Matrix\n(TS-Relevant Pathways)",
                 fontsize=13, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Targets pathway", shrink=0.6)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "pathway_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {FIGURES_DIR / 'pathway_heatmap.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TS network visualizations")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating network visualizations...")

    # 1. Disease network
    if NETWORK_PATH.exists() and MODULES_PATH.exists():
        print("\n1. Disease PPI network...")
        G = nx.read_graphml(str(NETWORK_PATH))
        modules_df = pd.read_csv(MODULES_PATH)
        plot_disease_network(G, modules_df)
    else:
        print("  Skipping disease network (files not found)")

    # 2. Candidate ranking
    if RANKED_PATH.exists():
        ranked_df = pd.read_csv(RANKED_PATH)
        print("\n2. Candidate ranking chart...")
        plot_candidate_ranking(ranked_df)

        print("\n3. Pathway heatmap...")
        plot_pathway_heatmap(ranked_df)
    else:
        print("  Skipping ranking/pathway plots (ranked_candidates.csv not found)")

    print("\nVisualization generation complete.")


if __name__ == "__main__":
    main()
