"""Phase 4 — Network model: innate deficiency → autoantibody diversification.

Task 109: Model how lectin complement failure → prolonged GAS exposure →
epitope spreading → autoantibody diversification. Connects innate deficiency
model outputs (Phase 1-2) with autoantibody-target-network-mapping results.

Approach:
  1. Load autoantibody PPI network and community structure
  2. Identify innate immunity bridging nodes in the autoantibody network
  3. Compute community-level innate connectivity (which autoantibody
     communities are most reachable from innate immune hubs)
  4. Model epitope spreading: network distance from innate immune genes
     predicts order of autoantibody target acquisition under prolonged exposure
  5. Output: ranked predictions of autoantibody targets under innate deficiency

Usage::

    uv run python -m pandas_pans.innate_immunity_deficiency_model.innate_autoantibody_network
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from bioagentics.config import DATA_DIR, REPO_ROOT
from bioagentics.pandas_pans.innate_immunity_modules import (
    INNATE_MODULES,
    LECTIN_COMPLEMENT_GENES,
    LECTIN_COMPLEMENT_DOWNSTREAM,
    get_all_innate_genes,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "innate-immunity-deficiency-model"
AUTOAB_DIR = DATA_DIR / "pandas_pans" / "autoantibody_network"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_autoantibody_network() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load autoantibody PPI edges, community assignments, and convergence data."""
    edges = pd.read_csv(AUTOAB_DIR / "combined_ppi_edgelist.tsv", sep="\t")
    communities = pd.read_csv(AUTOAB_DIR / "community_assignments.tsv", sep="\t")
    convergence = pd.read_csv(AUTOAB_DIR / "convergence_analysis.tsv", sep="\t")
    return edges, communities, convergence


# ---------------------------------------------------------------------------
# 1. Bridging node identification
# ---------------------------------------------------------------------------


def find_bridging_nodes(
    communities: pd.DataFrame,
    convergence: pd.DataFrame,
) -> pd.DataFrame:
    """Find innate immunity genes present in the autoantibody network.

    These are bridging nodes — genes that participate in both innate
    immune defense and the autoantibody target PPI network. They represent
    molecular links between innate failure and autoantibody production.
    """
    all_innate = set(get_all_innate_genes())
    network_genes = set(communities["gene_symbol"].unique())

    overlap = all_innate & network_genes
    if not overlap:
        logger.warning("No innate immunity genes found in autoantibody network")
        return pd.DataFrame()

    # Annotate each bridging node with its innate module(s) and network properties
    rows: list[dict] = []
    for gene in sorted(overlap):
        # Which innate modules does this gene belong to?
        modules = [
            mod_name
            for mod_name, mod_genes in INNATE_MODULES.items()
            if gene in mod_genes
        ]

        # Network properties from convergence analysis
        conv_row = convergence[convergence["gene_symbol"] == gene]
        comm_row = communities[communities["gene_symbol"] == gene]

        row: dict = {
            "gene_symbol": gene,
            "innate_modules": "; ".join(modules),
            "n_innate_modules": len(modules),
            "community": int(comm_row["community"].iloc[0]) if not comm_row.empty else -1,
            "is_seed": bool(comm_row["is_seed"].iloc[0]) if not comm_row.empty else False,
        }

        if not conv_row.empty:
            cr = conv_row.iloc[0]
            row["degree"] = int(cr.get("degree", 0))
            row["betweenness"] = float(cr.get("betweenness_centrality", 0))
            row["pagerank"] = float(cr.get("pagerank", 0))
            row["convergence_score"] = float(cr.get("convergence_score", 0))
            row["is_druggable"] = bool(cr.get("is_druggable", False))
            row["n_drug_interactions"] = int(cr.get("n_drug_interactions", 0))

        rows.append(row)

    df = pd.DataFrame(rows)
    if "degree" in df.columns:
        df = df.sort_values("degree", ascending=False)
    return df


# ---------------------------------------------------------------------------
# 2. Community innate connectivity
# ---------------------------------------------------------------------------


def compute_community_innate_connectivity(
    edges: pd.DataFrame,
    communities: pd.DataFrame,
) -> pd.DataFrame:
    """Compute how connected each autoantibody community is to innate immune genes.

    For each community, count:
    - Number of innate bridging nodes in the community
    - Number of edges from innate genes to community members
    - Fraction of community members reachable from innate genes in 1 hop
    """
    all_innate = set(get_all_innate_genes())
    lectin_genes = set(LECTIN_COMPLEMENT_GENES + LECTIN_COMPLEMENT_DOWNSTREAM)

    # Build adjacency from edges
    adjacency: dict[str, set[str]] = {}
    for _, row in edges.iterrows():
        s, t = str(row.get("source", "")), str(row.get("target", ""))
        if s and t:
            adjacency.setdefault(s, set()).add(t)
            adjacency.setdefault(t, set()).add(s)

    # Per-community analysis
    comm_groups = communities.groupby("community")
    rows: list[dict] = []

    for comm_id, group in comm_groups:
        members = set(group["gene_symbol"].values)
        seeds = set(group[group["is_seed"] == True]["gene_symbol"].values)  # noqa: E712
        n_members = len(members)

        # Innate genes in this community
        innate_in_comm = members & all_innate
        lectin_in_comm = members & lectin_genes

        # Edges from innate genes to this community (1-hop reachability)
        innate_neighbors_in_comm: set[str] = set()
        for gene in all_innate:
            neighbors = adjacency.get(gene, set())
            innate_neighbors_in_comm |= (neighbors & members)

        # Lectin-specific reachability
        lectin_neighbors_in_comm: set[str] = set()
        for gene in lectin_genes:
            neighbors = adjacency.get(gene, set())
            lectin_neighbors_in_comm |= (neighbors & members)

        rows.append({
            "community": int(comm_id),
            "n_members": n_members,
            "n_seeds": len(seeds),
            "n_innate_bridging": len(innate_in_comm),
            "innate_bridging_genes": "; ".join(sorted(innate_in_comm)) if innate_in_comm else "",
            "n_lectin_bridging": len(lectin_in_comm),
            "lectin_bridging_genes": "; ".join(sorted(lectin_in_comm)) if lectin_in_comm else "",
            "n_innate_1hop_reachable": len(innate_neighbors_in_comm),
            "frac_innate_reachable": len(innate_neighbors_in_comm) / n_members if n_members > 0 else 0,
            "n_lectin_1hop_reachable": len(lectin_neighbors_in_comm),
            "frac_lectin_reachable": len(lectin_neighbors_in_comm) / n_members if n_members > 0 else 0,
        })

    df = pd.DataFrame(rows)
    # Score: weighted combination of bridging presence and reachability
    df["innate_connectivity_score"] = (
        0.4 * (df["n_innate_bridging"] / df["n_innate_bridging"].max().clip(1))
        + 0.3 * df["frac_innate_reachable"]
        + 0.3 * (df["n_lectin_bridging"] / df["n_lectin_bridging"].max().clip(1))
    )
    return df.sort_values("innate_connectivity_score", ascending=False)


# ---------------------------------------------------------------------------
# 3. Epitope spreading model
# ---------------------------------------------------------------------------


def model_epitope_spreading(
    edges: pd.DataFrame,
    communities: pd.DataFrame,
    convergence: pd.DataFrame,
) -> pd.DataFrame:
    """Model epitope spreading order under innate deficiency.

    The hypothesis: lectin complement failure → prolonged GAS antigen exposure →
    immune system has more time to diversify autoantibody targets. Targets closer
    (in network distance) to initial immune response nodes are hit first; under
    prolonged exposure, more distant targets are also engaged.

    Uses BFS from innate immune bridging nodes to rank autoantibody seeds
    by predicted order of targeting.
    """
    all_innate = set(get_all_innate_genes())
    network_genes = set(communities["gene_symbol"].unique())
    seeds = set(communities[communities["is_seed"] == True]["gene_symbol"].values)  # noqa: E712

    # Build adjacency
    adjacency: dict[str, set[str]] = {}
    for _, row in edges.iterrows():
        s, t = str(row.get("source", "")), str(row.get("target", ""))
        if s and t:
            adjacency.setdefault(s, set()).add(t)
            adjacency.setdefault(t, set()).add(s)

    # BFS from all innate bridging nodes simultaneously
    start_nodes = all_innate & network_genes
    if not start_nodes:
        logger.warning("No innate nodes in network for BFS")
        return pd.DataFrame()

    # Multi-source BFS
    distances: dict[str, int] = {n: 0 for n in start_nodes}
    queue = list(start_nodes)
    visited = set(start_nodes)

    idx = 0
    while idx < len(queue):
        node = queue[idx]
        idx += 1
        for neighbor in adjacency.get(node, set()):
            if neighbor not in visited:
                visited.add(neighbor)
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)

    # Rank autoantibody seed targets by distance from innate nodes
    rows: list[dict] = []
    for seed_gene in sorted(seeds):
        dist = distances.get(seed_gene, -1)
        comm_row = communities[communities["gene_symbol"] == seed_gene]
        conv_row = convergence[convergence["gene_symbol"] == seed_gene]

        row: dict = {
            "autoantibody_target": seed_gene,
            "distance_from_innate": dist,
            "community": int(comm_row["community"].iloc[0]) if not comm_row.empty else -1,
        }

        if not conv_row.empty:
            cr = conv_row.iloc[0]
            row["node_type"] = str(cr.get("node_type", ""))
            row["mechanism_category"] = str(cr.get("mechanism_category", ""))
            row["degree"] = int(cr.get("degree", 0))
            row["convergence_score"] = float(cr.get("convergence_score", 0))
            row["symptom_domains"] = str(cr.get("symptom_domains", ""))

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Predicted targeting order: closer targets hit first, higher degree = more likely
    max_dist = df["distance_from_innate"].replace(-1, np.nan).max()
    if pd.notna(max_dist) and max_dist > 0:
        df["proximity_score"] = 1.0 - (
            df["distance_from_innate"].replace(-1, max_dist + 1) / (max_dist + 1)
        )
    else:
        df["proximity_score"] = 0.0

    if "degree" in df.columns and df["degree"].max() > 0:
        df["hub_score"] = df["degree"] / df["degree"].max()
    else:
        df["hub_score"] = 0.0

    # Combined: targets close to innate nodes AND highly connected are most likely
    df["predicted_targeting_score"] = 0.6 * df["proximity_score"] + 0.4 * df["hub_score"]
    df["predicted_order"] = (
        df["predicted_targeting_score"].rank(ascending=False, method="min").astype(int)
    )

    return df.sort_values("predicted_order")


# ---------------------------------------------------------------------------
# 4. Summary statistics
# ---------------------------------------------------------------------------


def compute_summary(
    bridging: pd.DataFrame,
    community_conn: pd.DataFrame,
    spreading: pd.DataFrame,
) -> dict:
    """Generate summary statistics for the network model."""
    summary: dict = {
        "n_bridging_nodes": len(bridging),
        "n_seed_bridging": int(bridging["is_seed"].sum()) if "is_seed" in bridging.columns else 0,
        "n_communities": len(community_conn),
        "n_communities_with_innate_bridge": int(
            (community_conn["n_innate_bridging"] > 0).sum()
        ),
        "n_communities_with_lectin_bridge": int(
            (community_conn["n_lectin_bridging"] > 0).sum()
        ),
        "n_autoantibody_targets": len(spreading),
    }

    if not spreading.empty and "distance_from_innate" in spreading.columns:
        reachable = spreading[spreading["distance_from_innate"] >= 0]
        summary["n_targets_reachable_from_innate"] = len(reachable)
        if not reachable.empty:
            summary["mean_distance_to_targets"] = float(reachable["distance_from_innate"].mean())
            summary["median_distance_to_targets"] = float(reachable["distance_from_innate"].median())
            # Targets within 2 hops of innate genes — predicted "early" targets
            close = reachable[reachable["distance_from_innate"] <= 2]
            summary["n_early_targets_leq2_hops"] = len(close)
            summary["early_targets"] = list(close["autoantibody_target"].values)

    if not bridging.empty and "innate_modules" in bridging.columns:
        # Which innate modules have bridging nodes
        all_modules: set[str] = set()
        for mods in bridging["innate_modules"]:
            for m in str(mods).split("; "):
                if m:
                    all_modules.add(m)
        summary["innate_modules_with_bridges"] = sorted(all_modules)

    return summary


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_innate_autoantibody_network() -> dict[str, Path]:
    """Run the innate deficiency → autoantibody network model."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # Load data
    logger.info("Loading autoantibody network data...")
    edges, communities, convergence = load_autoantibody_network()
    logger.info(
        "Network: %d edges, %d nodes, %d communities",
        len(edges),
        len(communities),
        communities["community"].nunique(),
    )

    # 1. Bridging nodes
    logger.info("\n=== Innate immunity bridging nodes ===")
    bridging = find_bridging_nodes(communities, convergence)
    if not bridging.empty:
        path = OUTPUT_DIR / "innate_autoantibody_bridging_nodes.csv"
        bridging.to_csv(path, index=False)
        outputs["bridging_nodes"] = path
        logger.info(
            "Found %d innate genes in autoantibody network (seed targets: %d)",
            len(bridging),
            int(bridging["is_seed"].sum()) if "is_seed" in bridging.columns else 0,
        )
        logger.info("Top bridging nodes:\n%s", bridging.head(10).to_string(index=False))

    # 2. Community connectivity
    logger.info("\n=== Community innate connectivity ===")
    community_conn = compute_community_innate_connectivity(edges, communities)
    if not community_conn.empty:
        path = OUTPUT_DIR / "community_innate_connectivity.csv"
        community_conn.to_csv(path, index=False)
        outputs["community_connectivity"] = path
        top = community_conn[community_conn["innate_connectivity_score"] > 0].head(10)
        logger.info("Communities with innate connections:\n%s", top.to_string(index=False))

    # 3. Epitope spreading model
    logger.info("\n=== Epitope spreading model ===")
    spreading = model_epitope_spreading(edges, communities, convergence)
    if not spreading.empty:
        path = OUTPUT_DIR / "epitope_spreading_predictions.csv"
        spreading.to_csv(path, index=False)
        outputs["epitope_spreading"] = path
        logger.info(
            "Ranked %d autoantibody targets by predicted targeting order",
            len(spreading),
        )
        logger.info("Top predicted early targets:\n%s", spreading.head(10).to_string(index=False))

    # 4. Summary
    logger.info("\n=== Summary ===")
    summary = compute_summary(bridging, community_conn, spreading)
    path = OUTPUT_DIR / "innate_autoantibody_network_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    outputs["summary"] = path
    for k, v in summary.items():
        logger.info("  %s: %s", k, v)

    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_innate_autoantibody_network()
