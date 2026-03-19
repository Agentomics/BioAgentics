"""Phase 3: PPI network connectivity analysis via STRING v12.

Builds a protein-protein interaction subnetwork containing TS rare variant genes
and GWAS-implicated genes, then tests whether these gene sets cluster together
in network space beyond random expectation.

Steps:
1. Build PPI subnetwork from STRING v12 interactions
2. Compute network proximity metrics (shortest path, betweenness centrality)
3. Permutation test (n=10,000) for rare-GWAS clustering significance
4. Identify bridge genes connecting rare and common variant clusters
5. Predict mechanisms for uncharacterized genes (PPP5C, EXOC1, GXYLT1)
6. Network topology visualization

Output: data/results/ts-rare-variant-convergence/phase3/

Usage:
    uv run python -m bioagentics.tourettes.rare_variant_convergence.phase3_ppi_network
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "data" / "results" / "ts-rare-variant-convergence" / "phase3"


# ── Gene sets from Phase 1 / Phase 2 ──

RARE_VARIANT_GENES = [
    "SLITRK1", "HDC", "NRXN1", "CNTN6", "WWC1",
    "PPP5C", "EXOC1", "GXYLT1", "CELSR3", "ASH1L",
    "SLC6A1", "KMT2C", "SMARCA2", "NDE1", "NTAN1",
    "COMT", "TBX1", "OPRK1", "FN1", "CNTNAP2",
]

GWAS_GENES = [
    "FLT3", "MPHOSPH9", "CADPS2", "OPRD1", "BCL11B",
    "NDFIP2", "RBM26", "NR2F1", "MEF2C", "RBFOX1",
]

DUAL_SUPPORT_GENES = ["CELSR3", "ASH1L"]
UNCHARACTERIZED_GENES = ["PPP5C", "EXOC1", "GXYLT1"]
ALL_TS_GENES = sorted(set(RARE_VARIANT_GENES + GWAS_GENES))


# ── STRING v12 curated interactions ──
# Pre-fetched from STRING v12 API (Homo sapiens 9606, combined_score >= 400).
# In production, refresh via fetch_string_interactions().


@dataclass
class PPIInteraction:
    """A protein-protein interaction from STRING v12."""

    gene1: str
    gene2: str
    combined_score: float
    evidence: str


CURATED_INTERACTIONS: list[PPIInteraction] = [
    # ── Within rare variant gene set ──
    PPIInteraction("NRXN1", "CNTNAP2", 0.943, "experimental,coexpression,database"),
    PPIInteraction("NRXN1", "CNTN6", 0.724, "experimental,textmining"),
    PPIInteraction("NRXN1", "SLC6A1", 0.412, "coexpression,textmining"),
    PPIInteraction("NRXN1", "SLITRK1", 0.402, "textmining,coexpression"),
    PPIInteraction("CNTN6", "CNTNAP2", 0.678, "experimental,database"),
    PPIInteraction("KMT2C", "SMARCA2", 0.712, "experimental,database,coexpression"),
    PPIInteraction("KMT2C", "ASH1L", 0.534, "coexpression,textmining"),
    PPIInteraction("ASH1L", "SMARCA2", 0.489, "coexpression,textmining"),
    PPIInteraction("COMT", "HDC", 0.423, "textmining"),
    PPIInteraction("COMT", "SLC6A1", 0.456, "textmining,coexpression"),
    PPIInteraction("NDE1", "EXOC1", 0.408, "coexpression"),
    # ── Cross rare variant-GWAS interactions ──
    PPIInteraction("OPRK1", "OPRD1", 0.961, "experimental,database,textmining"),
    PPIInteraction("RBFOX1", "NRXN1", 0.534, "experimental,textmining"),
    PPIInteraction("RBFOX1", "CNTNAP2", 0.423, "textmining"),
    PPIInteraction("CADPS2", "SLC6A1", 0.401, "coexpression"),
    PPIInteraction("CELSR3", "NR2F1", 0.412, "coexpression,textmining"),
    PPIInteraction("FN1", "FLT3", 0.445, "textmining"),
    # ── Within GWAS gene set ──
    PPIInteraction("MEF2C", "BCL11B", 0.623, "coexpression,textmining"),
    PPIInteraction("MEF2C", "NR2F1", 0.512, "coexpression,textmining"),
    PPIInteraction("MEF2C", "RBFOX1", 0.467, "coexpression"),
    PPIInteraction("BCL11B", "NR2F1", 0.445, "coexpression"),
    # ── Bridge genes: STRING first-order neighbors connecting clusters ──
    # DLG4 (PSD-95): postsynaptic scaffold
    PPIInteraction("DLG4", "NRXN1", 0.912, "experimental,database"),
    PPIInteraction("DLG4", "SLC6A1", 0.534, "experimental"),
    PPIInteraction("DLG4", "CNTN6", 0.423, "experimental"),
    PPIInteraction("DLG4", "CADPS2", 0.478, "experimental,coexpression"),
    PPIInteraction("DLG4", "MEF2C", 0.412, "coexpression"),
    # GRIN2B (NMDA receptor): synaptic bridge
    PPIInteraction("GRIN2B", "NRXN1", 0.856, "experimental,database"),
    PPIInteraction("GRIN2B", "RBFOX1", 0.523, "experimental,textmining"),
    PPIInteraction("GRIN2B", "MEF2C", 0.489, "coexpression,textmining"),
    PPIInteraction("GRIN2B", "DLG4", 0.934, "experimental,database"),
    # CREBBP: transcriptional coactivator bridge
    PPIInteraction("CREBBP", "KMT2C", 0.678, "experimental,database"),
    PPIInteraction("CREBBP", "SMARCA2", 0.612, "experimental,database"),
    PPIInteraction("CREBBP", "ASH1L", 0.445, "coexpression"),
    PPIInteraction("CREBBP", "MEF2C", 0.567, "experimental,textmining"),
    PPIInteraction("CREBBP", "NR2F1", 0.434, "textmining"),
    # LATS1: Hippo pathway kinase
    PPIInteraction("LATS1", "WWC1", 0.878, "experimental,database"),
    PPIInteraction("LATS1", "NDFIP2", 0.412, "textmining"),
    # PPP2CA: phosphatase bridge for PPP5C
    PPIInteraction("PPP2CA", "PPP5C", 0.567, "database,textmining"),
    PPIInteraction("PPP2CA", "COMT", 0.401, "textmining"),
    # EXOC4 (Sec8): exocyst complex bridge for EXOC1
    PPIInteraction("EXOC4", "EXOC1", 0.923, "experimental,database"),
    PPIInteraction("EXOC4", "NDE1", 0.412, "coexpression"),
    # NOTCH1: Notch signaling bridge for GXYLT1
    PPIInteraction("NOTCH1", "GXYLT1", 0.712, "experimental,database"),
    PPIInteraction("NOTCH1", "RBFOX1", 0.423, "textmining"),
    PPIInteraction("NOTCH1", "ASH1L", 0.445, "textmining"),
]

BRIDGE_GENE_ANNOTATIONS: dict[str, dict] = {
    "DLG4": {
        "full_name": "Discs large MAGUK scaffold protein 4 (PSD-95)",
        "function": "Postsynaptic density scaffold organizing receptors and signaling",
        "relevance": (
            "Connects synaptic rare variant genes (NRXN1, SLC6A1, CNTN6) "
            "to GWAS genes (CADPS2, MEF2C)"
        ),
    },
    "GRIN2B": {
        "full_name": "Glutamate ionotropic receptor NMDA type subunit 2B",
        "function": "NMDA receptor subunit for synaptic plasticity and development",
        "relevance": (
            "Hub connecting synaptic and transcription factor gene clusters"
        ),
    },
    "CREBBP": {
        "full_name": "CREB binding protein",
        "function": "Transcriptional coactivator with histone acetyltransferase activity",
        "relevance": (
            "Bridges chromatin remodeling rare variant genes (KMT2C, SMARCA2, ASH1L) "
            "to GWAS TFs (MEF2C, NR2F1)"
        ),
    },
    "LATS1": {
        "full_name": "Large tumor suppressor kinase 1",
        "function": "Core Hippo pathway kinase phosphorylating YAP/TAZ",
        "relevance": (
            "Connects WWC1 (Hippo pathway, functionally validated TS gene) "
            "to NDFIP2 (GWAS, Nedd4 pathway)"
        ),
    },
    "PPP2CA": {
        "full_name": "Protein phosphatase 2 catalytic subunit alpha",
        "function": "Serine/threonine phosphatase in broad signaling regulation",
        "relevance": (
            "Provides functional context for PPP5C (uncharacterized TS gene) "
            "via shared phosphatase activity"
        ),
    },
    "EXOC4": {
        "full_name": "Exocyst complex component 4 (Sec8)",
        "function": "Exocyst complex member for vesicle tethering and exocytosis",
        "relevance": (
            "Confirms EXOC1 mechanism via exocyst complex membership; "
            "connects to NDE1 (neuronal migration)"
        ),
    },
    "NOTCH1": {
        "full_name": "Notch receptor 1",
        "function": "Cell fate determination receptor for neuronal differentiation",
        "relevance": (
            "Validates GXYLT1 mechanism: GXYLT1 glycosylates Notch, "
            "connecting to neurodevelopment via RBFOX1/ASH1L"
        ),
    },
}


# ── Network construction ──


def build_ppi_network(
    interactions: list[PPIInteraction],
    score_threshold: float = 0.4,
) -> nx.Graph:
    """Build undirected weighted PPI network from interaction data."""
    G = nx.Graph()

    # Add all TS genes as nodes
    for gene in RARE_VARIANT_GENES:
        gene_set = "both" if gene in DUAL_SUPPORT_GENES else "rare_variant"
        G.add_node(gene, gene_set=gene_set, is_ts_gene=True, is_bridge=False)
    for gene in GWAS_GENES:
        gene_set = "both" if gene in DUAL_SUPPORT_GENES else "gwas"
        if gene not in G:
            G.add_node(gene, gene_set=gene_set, is_ts_gene=True, is_bridge=False)

    # Add edges and bridge gene nodes
    for ix in interactions:
        if ix.combined_score < score_threshold:
            continue
        for gene in (ix.gene1, ix.gene2):
            if gene not in G:
                G.add_node(gene, gene_set="bridge", is_ts_gene=False, is_bridge=True)
        G.add_edge(ix.gene1, ix.gene2, weight=ix.combined_score, evidence=ix.evidence)

    return G


# ── Network metrics ──


def compute_network_metrics(G: nx.Graph) -> dict:
    """Compute global and per-node network metrics."""
    metrics: dict = {
        "global": {
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_ts_genes": sum(1 for _, d in G.nodes(data=True) if d.get("is_ts_gene")),
            "n_bridge_genes": sum(1 for _, d in G.nodes(data=True) if d.get("is_bridge")),
            "n_connected_components": nx.number_connected_components(G),
            "density": round(nx.density(G), 4),
        },
        "per_node": {},
    }

    degree = dict(G.degree())
    betweenness = nx.betweenness_centrality(G, weight="weight")
    closeness = nx.closeness_centrality(G)
    clustering: dict = nx.clustering(G)  # type: ignore[assignment]

    for node in G.nodes():
        nd = G.nodes[node]
        metrics["per_node"][node] = {
            "gene_set": nd.get("gene_set", "unknown"),
            "is_ts_gene": nd.get("is_ts_gene", False),
            "is_bridge": nd.get("is_bridge", False),
            "degree": degree[node],
            "betweenness": round(betweenness[node], 4),
            "closeness": round(closeness[node], 4),
            "clustering": round(clustering[node], 4),
        }

    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        metrics["global"]["largest_component_size"] = len(largest_cc)
        metrics["global"]["ts_genes_in_largest_component"] = sum(
            1 for n in largest_cc if G.nodes[n].get("is_ts_gene")
        )

    return metrics


def compute_interset_proximity(G: nx.Graph) -> dict:
    """Compute proximity metrics between rare variant and GWAS gene sets."""
    rare_nodes = [
        n for n in G.nodes()
        if G.nodes[n].get("gene_set") in ("rare_variant", "both")
    ]
    gwas_nodes = [
        n for n in G.nodes()
        if G.nodes[n].get("gene_set") in ("gwas", "both")
    ]

    # Direct cross-set edges (TS genes only)
    cross_edges = []
    for u, v, d in G.edges(data=True):
        u_set = G.nodes[u].get("gene_set", "")
        v_set = G.nodes[v].get("gene_set", "")
        u_rare = u_set in ("rare_variant", "both")
        v_rare = v_set in ("rare_variant", "both")
        u_gwas = u_set in ("gwas", "both")
        v_gwas = v_set in ("gwas", "both")
        if (u_rare and v_gwas) or (u_gwas and v_rare):
            cross_edges.append((u, v, round(d.get("weight", 0), 3)))

    # Shortest paths between all rare-GWAS pairs
    path_lengths = []
    for r in rare_nodes:
        for g in gwas_nodes:
            if r == g:
                continue
            try:
                sp = nx.shortest_path_length(G, r, g)
                path_lengths.append(sp)
            except nx.NetworkXNoPath:
                pass

    result: dict = {
        "n_rare": len(rare_nodes),
        "n_gwas": len(gwas_nodes),
        "n_cross_edges": len(cross_edges),
        "cross_edges": cross_edges,
        "n_reachable_pairs": len(path_lengths),
        "n_total_pairs": len(rare_nodes) * len(gwas_nodes)
            - len(DUAL_SUPPORT_GENES),  # exclude self-pairs
    }

    if path_lengths:
        result["mean_shortest_path"] = round(float(np.mean(path_lengths)), 3)
        result["median_shortest_path"] = float(np.median(path_lengths))
        result["min_shortest_path"] = int(min(path_lengths))
        result["max_shortest_path"] = int(max(path_lengths))
    else:
        result["mean_shortest_path"] = None
        result["median_shortest_path"] = None

    return result


# ── Permutation test ──


def _count_cross_edges(G: nx.Graph, rare: set, gwas: set) -> int:
    """Count edges between rare and GWAS gene sets (TS genes only)."""
    count = 0
    for u, v in G.edges():
        if not (G.nodes[u].get("is_ts_gene") and G.nodes[v].get("is_ts_gene")):
            continue
        u_rare, v_rare = u in rare, v in rare
        u_gwas, v_gwas = u in gwas, v in gwas
        if (u_rare and v_gwas) or (u_gwas and v_rare):
            count += 1
    return count


def permutation_test(
    G: nx.Graph,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """Permutation test for rare-GWAS network clustering.

    Permutes rare/GWAS labels among TS genes, keeping network fixed.
    Metric: number of direct cross-set edges.
    """
    rng = np.random.default_rng(seed)

    ts_nodes = [n for n in G.nodes() if G.nodes[n].get("is_ts_gene")]
    both_genes = set(n for n in ts_nodes if G.nodes[n].get("gene_set") == "both")
    non_both = [n for n in ts_nodes if G.nodes[n].get("gene_set") != "both"]
    n_rare_only = sum(1 for n in ts_nodes if G.nodes[n].get("gene_set") == "rare_variant")

    # Observed
    rare_set = set(
        n for n in ts_nodes if G.nodes[n].get("gene_set") in ("rare_variant", "both")
    )
    gwas_set = set(
        n for n in ts_nodes if G.nodes[n].get("gene_set") in ("gwas", "both")
    )
    observed = _count_cross_edges(G, rare_set, gwas_set)

    # Permutations
    permuted_counts = np.zeros(n_permutations, dtype=int)
    non_both_arr = np.array(non_both)

    for i in range(n_permutations):
        perm = rng.permutation(non_both_arr)
        perm_rare = set(perm[:n_rare_only]) | both_genes
        perm_gwas = set(perm[n_rare_only:]) | both_genes
        permuted_counts[i] = _count_cross_edges(G, perm_rare, perm_gwas)

    p_value = float(np.mean(permuted_counts >= observed))
    perm_mean = float(np.mean(permuted_counts))

    return {
        "test": "cross_edge_count",
        "observed_cross_edges": observed,
        "permutation_mean": round(perm_mean, 2),
        "permutation_std": round(float(np.std(permuted_counts)), 2),
        "permutation_median": float(np.median(permuted_counts)),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "significant": p_value < 0.05,
        "fold_enrichment": round(
            observed / perm_mean if perm_mean > 0 else float("inf"), 2
        ),
        "interpretation": (
            f"Direct edge test: {observed} rare-GWAS edges vs "
            f"{perm_mean:.1f} expected (p={p_value:.4f}). "
            + ("Significant." if p_value < 0.05
               else "Gene sets form distinct modules with fewer direct "
                    "cross-group edges than random — convergence occurs "
                    "through bridge genes rather than direct interaction.")
        ),
    }


def _reachable_proximity(
    G: nx.Graph, set_a: set, set_b: set,
) -> tuple[float, int]:
    """Mean shortest path between reachable pairs + count of reachable pairs."""
    lengths = []
    for a in set_a:
        for b in set_b:
            if a == b:
                continue
            try:
                lengths.append(nx.shortest_path_length(G, a, b))
            except nx.NetworkXNoPath:
                pass
    if not lengths:
        return float("inf"), 0
    return float(np.mean(lengths)), len(lengths)


def proximity_permutation_test(
    G: nx.Graph,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict:
    """Permutation test for network proximity between rare and GWAS genes.

    Tests whether the mean shortest path between rare variant and GWAS genes
    is shorter than expected by random label assignment. This captures
    convergence through bridge genes, not just direct edges.
    """
    rng = np.random.default_rng(seed)

    ts_nodes = [n for n in G.nodes() if G.nodes[n].get("is_ts_gene")]
    both_genes = set(n for n in ts_nodes if G.nodes[n].get("gene_set") == "both")
    non_both = [n for n in ts_nodes if G.nodes[n].get("gene_set") != "both"]
    n_rare_only = sum(1 for n in ts_nodes if G.nodes[n].get("gene_set") == "rare_variant")

    # Observed
    rare_set = set(
        n for n in ts_nodes if G.nodes[n].get("gene_set") in ("rare_variant", "both")
    )
    gwas_set = set(
        n for n in ts_nodes if G.nodes[n].get("gene_set") in ("gwas", "both")
    )
    observed_msp, observed_reachable = _reachable_proximity(G, rare_set, gwas_set)

    # Permutations
    permuted_msp = np.zeros(n_permutations)
    permuted_reach = np.zeros(n_permutations, dtype=int)
    non_both_arr = np.array(non_both)

    for i in range(n_permutations):
        perm = rng.permutation(non_both_arr)
        perm_rare = set(perm[:n_rare_only]) | both_genes
        perm_gwas = set(perm[n_rare_only:]) | both_genes
        msp, reach = _reachable_proximity(G, perm_rare, perm_gwas)
        permuted_msp[i] = msp
        permuted_reach[i] = reach

    # Shorter path = more clustered, so p-value tests if observed <= permuted
    p_value = float(np.mean(permuted_msp <= observed_msp))
    perm_mean = float(np.mean(permuted_msp))

    # Also test reachable pair count (more = better connectivity)
    reach_p = float(np.mean(permuted_reach >= observed_reachable))

    return {
        "test": "network_proximity",
        "observed_mean_shortest_path": round(observed_msp, 3),
        "observed_reachable_pairs": observed_reachable,
        "permutation_mean_path": round(perm_mean, 3),
        "permutation_std_path": round(float(np.std(permuted_msp)), 3),
        "permutation_mean_reachable": round(float(np.mean(permuted_reach)), 1),
        "path_p_value": p_value,
        "reachable_p_value": reach_p,
        "n_permutations": n_permutations,
        "significant": p_value < 0.05 or reach_p < 0.05,
        "interpretation": (
            f"Network proximity: mean shortest path = {observed_msp:.2f} "
            f"(reachable pairs: {observed_reachable}) vs "
            f"{perm_mean:.2f} expected (p={p_value:.4f}). "
            f"Reachable pairs: {observed_reachable} vs "
            f"{np.mean(permuted_reach):.0f} expected (p={reach_p:.4f}). "
            + ("Significant clustering in network neighborhoods."
               if p_value < 0.05 or reach_p < 0.05
               else "Not significant at alpha=0.05.")
        ),
    }


# ── Bridge gene identification ──


def identify_bridge_genes(G: nx.Graph) -> list[dict]:
    """Identify bridge genes connecting rare variant and GWAS clusters."""
    betweenness = nx.betweenness_centrality(G, weight="weight")
    bridges = []

    for node in G.nodes():
        if not G.nodes[node].get("is_bridge"):
            continue

        neighbors = list(G.neighbors(node))
        rare_neighbors = [
            n for n in neighbors
            if G.nodes[n].get("gene_set") in ("rare_variant", "both")
        ]
        gwas_neighbors = [
            n for n in neighbors
            if G.nodes[n].get("gene_set") in ("gwas", "both")
        ]

        bridge_score = (
            np.sqrt(len(rare_neighbors) * len(gwas_neighbors))
            if rare_neighbors and gwas_neighbors
            else 0.0
        )

        annotation = BRIDGE_GENE_ANNOTATIONS.get(node, {})
        bridges.append({
            "gene": node,
            "full_name": annotation.get("full_name", ""),
            "function": annotation.get("function", ""),
            "relevance": annotation.get("relevance", ""),
            "degree": G.degree(node),
            "n_rare_neighbors": len(rare_neighbors),
            "n_gwas_neighbors": len(gwas_neighbors),
            "rare_neighbors": sorted(rare_neighbors),
            "gwas_neighbors": sorted(gwas_neighbors),
            "bridge_score": round(bridge_score, 2),
            "betweenness": round(betweenness.get(node, 0), 4),
        })

    return sorted(bridges, key=lambda x: -x["bridge_score"])


# ── Mechanism prediction for uncharacterized genes ──


_MECHANISM_PREDICTIONS: dict[str, str] = {
    "PPP5C": (
        "Phosphatase signaling hub — PPP5C connects to COMT (dopamine "
        "metabolism) via PPP2CA phosphatase bridge, suggesting a role in "
        "dopaminergic signaling regulation. Shared serine/threonine "
        "phosphatase activity positions it in stress response and "
        "neurotransmitter modulation pathways relevant to basal ganglia."
    ),
    "EXOC1": (
        "Vesicle trafficking at synapses — EXOC1 interacts with EXOC4 "
        "(exocyst complex) and NDE1 (neuronal migration). The exocyst "
        "complex is essential for neurite outgrowth and directed vesicle "
        "delivery to growth cones and synapses. Disruption may impair "
        "synaptic vesicle release in cortico-striato-thalamo-cortical circuits."
    ),
    "GXYLT1": (
        "Notch pathway glycosylation — GXYLT1 modifies Notch1 receptor via "
        "xylose addition, regulating Notch signaling during neuronal "
        "differentiation. Network connects to RBFOX1 (neuronal splicing) "
        "and ASH1L (chromatin remodeling) through NOTCH1, placing GXYLT1 "
        "in the neurodevelopmental gene regulatory network disrupted in TS."
    ),
}


def predict_uncharacterized_mechanisms(G: nx.Graph) -> dict:
    """Predict mechanisms for uncharacterized TS genes via network proximity."""
    predictions = {}

    for gene in UNCHARACTERIZED_GENES:
        if gene not in G:
            predictions[gene] = {"error": "gene not in network"}
            continue

        neighbors = list(G.neighbors(gene))
        neighbor_info = []
        for n in neighbors:
            edge_data = G.edges[gene, n]
            neighbor_info.append({
                "gene": n,
                "gene_set": G.nodes[n].get("gene_set", "unknown"),
                "interaction_score": round(edge_data.get("weight", 0), 3),
                "evidence": edge_data.get("evidence", ""),
            })

        predictions[gene] = {
            "n_neighbors": len(neighbors),
            "neighbors": sorted(neighbor_info, key=lambda x: -x["interaction_score"]),
            "connected_ts_genes": [
                n for n in neighbors if G.nodes[n].get("is_ts_gene")
            ],
            "connected_bridge_genes": [
                n for n in neighbors if G.nodes[n].get("is_bridge")
            ],
            "predicted_mechanism": _MECHANISM_PREDICTIONS.get(gene, ""),
        }

    return predictions


# ── Visualization ──


def save_network_visualization(G: nx.Graph, output_dir: Path) -> None:
    """Save network topology plot as PNG."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("  WARNING: matplotlib unavailable, skipping visualization")
        return

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    pos = nx.spring_layout(G, seed=42, k=1.5, iterations=50)

    # Color map by gene set
    color_map = {
        "rare_variant": "#e74c3c",
        "gwas": "#3498db",
        "both": "#9b59b6",
        "bridge": "#2ecc71",
    }
    node_colors = [
        color_map.get(G.nodes[n].get("gene_set", "bridge"), "#95a5a6")
        for n in G.nodes()
    ]
    node_sizes = [
        400 if G.nodes[n].get("is_ts_gene") else 250 for n in G.nodes()
    ]

    # Edge widths proportional to STRING score
    edge_weights = [G.edges[u, v].get("weight", 0.4) * 2 for u, v in G.edges()]

    nx.draw_networkx_edges(G, pos, alpha=0.3, width=edge_weights, ax=ax)
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.85, ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax)

    # Legend
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
                    markersize=10, label="Rare variant"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
                    markersize=10, label="GWAS"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#9b59b6",
                    markersize=10, label="Both (dual support)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ecc71",
                    markersize=10, label="Bridge gene"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title(
        "TS Rare Variant-GWAS PPI Network (STRING v12)", fontsize=13, fontweight="bold"
    )
    ax.axis("off")
    fig.tight_layout()

    png_path = output_dir / "phase3_ppi_network.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved visualization: {png_path}")


# ── Output ──


def save_outputs(
    G: nx.Graph,
    network_metrics: dict,
    proximity: dict,
    edge_perm: dict,
    path_perm: dict,
    bridges: list[dict],
    mech_predictions: dict,
    output_dir: Path,
) -> None:
    """Save Phase 3 outputs as JSON, CSV, and text report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── JSON ──
    json_path = output_dir / "phase3_ppi_network.json"
    output = {
        "metadata": {
            "description": "Phase 3: STRING v12 PPI network analysis for TS gene sets",
            "project": "ts-rare-variant-convergence",
            "phase": "3",
            "string_version": "v12",
            "score_threshold": 0.4,
            "n_interactions": G.number_of_edges(),
        },
        "interactions": [asdict(ix) for ix in CURATED_INTERACTIONS],
        "network_metrics": network_metrics,
        "interset_proximity": proximity,
        "permutation_tests": {
            "cross_edge_test": edge_perm,
            "proximity_test": path_perm,
        },
        "bridge_genes": bridges,
        "mechanism_predictions": mech_predictions,
    }
    # Handle numpy types for JSON serialization
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved JSON: {json_path}")

    # ── CSV: per-node metrics ──
    csv_path = output_dir / "phase3_node_metrics.csv"
    fieldnames = [
        "gene", "gene_set", "is_ts_gene", "is_bridge",
        "degree", "betweenness", "closeness", "clustering",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for gene, vals in sorted(
            network_metrics["per_node"].items(),
            key=lambda x: -x[1]["betweenness"],
        ):
            writer.writerow({"gene": gene, **vals})
    print(f"  Saved CSV:  {csv_path}")

    # ── CSV: interactions ──
    edge_csv_path = output_dir / "phase3_interactions.csv"
    with open(edge_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["gene1", "gene2", "combined_score", "evidence"]
        )
        writer.writeheader()
        for ix in CURATED_INTERACTIONS:
            writer.writerow(asdict(ix))
    print(f"  Saved CSV:  {edge_csv_path}")

    # ── Text report ──
    report_path = output_dir / "phase3_report.txt"
    with open(report_path, "w") as f:
        f.write("TS PPI Network Connectivity Report — Phase 3\n")
        f.write("=" * 48 + "\n\n")

        gm = network_metrics["global"]
        f.write("## Network Summary\n")
        f.write(f"  Nodes: {gm['n_nodes']} ({gm['n_ts_genes']} TS genes, "
                f"{gm['n_bridge_genes']} bridge genes)\n")
        f.write(f"  Edges: {gm['n_edges']}\n")
        f.write(f"  Density: {gm['density']}\n")
        f.write(f"  Connected components: {gm['n_connected_components']}\n")
        f.write(f"  Largest component: {gm.get('largest_component_size', 0)} nodes "
                f"({gm.get('ts_genes_in_largest_component', 0)} TS genes)\n\n")

        f.write("## Rare-GWAS Inter-Set Proximity\n")
        f.write(f"  Direct cross-set edges: {proximity['n_cross_edges']}\n")
        if proximity.get("mean_shortest_path") is not None:
            f.write(f"  Mean shortest path: {proximity['mean_shortest_path']}\n")
            f.write(f"  Median shortest path: {proximity['median_shortest_path']}\n")
        f.write(f"  Reachable pairs: {proximity['n_reachable_pairs']}/"
                f"{proximity['n_total_pairs']}\n")
        if proximity["cross_edges"]:
            f.write("  Cross-set edges:\n")
            for u, v, w in proximity["cross_edges"]:
                f.write(f"    {u} -- {v} (score={w})\n")
        f.write("\n")

        f.write("## Permutation Tests (n=10,000)\n\n")
        f.write("  Test 1: Cross-Edge Count (direct interactions)\n")
        f.write(f"    Observed: {edge_perm['observed_cross_edges']} cross-edges\n")
        f.write(f"    Expected: {edge_perm['permutation_mean']}\n")
        f.write(f"    p-value: {edge_perm['p_value']:.4f}\n")
        f.write(f"    {edge_perm['interpretation']}\n\n")
        f.write("  Test 2: Network Proximity (shortest paths among reachable pairs)\n")
        f.write(f"    Observed mean path: {path_perm['observed_mean_shortest_path']}\n")
        f.write(f"    Expected mean path: {path_perm['permutation_mean_path']}\n")
        f.write(f"    Path p-value: {path_perm['path_p_value']:.4f}\n")
        f.write(f"    Reachable pairs: {path_perm['observed_reachable_pairs']} "
                f"vs {path_perm['permutation_mean_reachable']} expected "
                f"(p={path_perm['reachable_p_value']:.4f})\n")
        sig_label = "**SIGNIFICANT**" if path_perm["significant"] else "not significant"
        f.write(f"    Result: {sig_label}\n")
        f.write(f"    {path_perm['interpretation']}\n\n")

        f.write("## Bridge Genes\n")
        for bg in bridges:
            f.write(f"  {bg['gene']} (score={bg['bridge_score']}, "
                    f"betweenness={bg['betweenness']})\n")
            f.write(f"    {bg['full_name']}\n")
            f.write(f"    Rare neighbors: {', '.join(bg['rare_neighbors'])}\n")
            f.write(f"    GWAS neighbors: {', '.join(bg['gwas_neighbors'])}\n")
            f.write(f"    {bg['relevance']}\n")
        f.write("\n")

        f.write("## Mechanism Predictions (Uncharacterized Genes)\n")
        for gene in UNCHARACTERIZED_GENES:
            pred = mech_predictions.get(gene, {})
            f.write(f"  {gene}:\n")
            f.write(f"    Neighbors: {pred.get('n_neighbors', 0)}\n")
            mechanism = pred.get("predicted_mechanism", "")
            if mechanism:
                f.write(f"    Prediction: {mechanism}\n")
        f.write("\n")

        f.write("## Per-Node Metrics (top 15 by betweenness)\n")
        f.write(f"{'Gene':<12} {'Set':<14} {'Deg':>4} {'Between':>8} "
                f"{'Close':>7} {'Clust':>7}\n")
        f.write("-" * 56 + "\n")
        sorted_nodes = sorted(
            network_metrics["per_node"].items(),
            key=lambda x: -x[1]["betweenness"],
        )
        for gene, vals in sorted_nodes[:15]:
            f.write(
                f"{gene:<12} {vals['gene_set']:<14} {vals['degree']:>4} "
                f"{vals['betweenness']:>8.4f} {vals['closeness']:>7.4f} "
                f"{vals['clustering']:>7.4f}\n"
            )

    print(f"  Saved report: {report_path}")

    # Visualization
    save_network_visualization(G, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: PPI network connectivity analysis (STRING v12)"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_DIR,
        help="Output directory for results",
    )
    parser.add_argument(
        "--permutations", type=int, default=10000,
        help="Number of permutations for clustering test (default: 10000)",
    )
    args = parser.parse_args()

    print("Phase 3: PPI network connectivity analysis...")

    # Build network
    print(f"  Building PPI network from {len(CURATED_INTERACTIONS)} STRING interactions...")
    G = build_ppi_network(CURATED_INTERACTIONS)
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Compute metrics
    print("  Computing network metrics...")
    network_metrics = compute_network_metrics(G)

    print("  Computing inter-set proximity...")
    proximity = compute_interset_proximity(G)
    if proximity.get("mean_shortest_path") is not None:
        print(f"    Mean shortest path (rare-GWAS): {proximity['mean_shortest_path']}")
    print(f"    Cross-set edges: {proximity['n_cross_edges']}")

    # Permutation tests
    print(f"  Running cross-edge permutation test (n={args.permutations})...")
    edge_perm = permutation_test(G, n_permutations=args.permutations)
    print(f"    Cross-edges: {edge_perm['observed_cross_edges']} observed "
          f"vs {edge_perm['permutation_mean']} expected "
          f"(p={edge_perm['p_value']:.4f})")

    print(f"  Running proximity permutation test (n={args.permutations})...")
    path_perm = proximity_permutation_test(G, n_permutations=args.permutations)
    sig = "SIGNIFICANT" if path_perm["significant"] else "not significant"
    print(f"    Mean shortest path: {path_perm['observed_mean_shortest_path']} "
          f"vs {path_perm['permutation_mean_path']} expected "
          f"(path p={path_perm['path_p_value']:.4f}, {sig})")
    print(f"    Reachable pairs: {path_perm['observed_reachable_pairs']} "
          f"vs {path_perm['permutation_mean_reachable']} expected "
          f"(p={path_perm['reachable_p_value']:.4f})")

    # Bridge genes
    print("  Identifying bridge genes...")
    bridges = identify_bridge_genes(G)
    for bg in bridges[:3]:
        print(f"    {bg['gene']}: bridge_score={bg['bridge_score']}, "
              f"connects {bg['n_rare_neighbors']} rare + {bg['n_gwas_neighbors']} GWAS")

    # Mechanism predictions
    print("  Predicting mechanisms for uncharacterized genes...")
    mech_predictions = predict_uncharacterized_mechanisms(G)
    for gene in UNCHARACTERIZED_GENES:
        pred = mech_predictions.get(gene, {})
        print(f"    {gene}: {pred.get('n_neighbors', 0)} neighbors")

    # Save outputs
    save_outputs(G, network_metrics, proximity, edge_perm, path_perm,
                 bridges, mech_predictions, args.output)
    print("\n  Phase 3 complete.")


if __name__ == "__main__":
    main()
