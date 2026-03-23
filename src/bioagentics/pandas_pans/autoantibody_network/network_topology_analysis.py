"""Task #1083: Network topology and community structure analysis.

Performs community detection, modularity analysis, hub significance testing,
and network robustness analysis on the PANDAS autoantibody interactome.
"""

import json
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from scipy import stats
from collections import Counter
from pathlib import Path

DATA_DIR = Path("data/pandas_pans/autoantibody_network")

SEED_PROTEINS = {"DRD1", "DRD2", "CAMK2A", "TUBB3", "PKM", "ALDOC", "ENO1", "ENO2", "ENO3"}


def build_graph():
    """Build networkx graph from extended network edge list."""
    edges = pd.read_csv(DATA_DIR / "extended_network.tsv", sep="\t")
    G = nx.Graph()
    for _, row in edges.iterrows():
        w = float(row.get("combined_score", 0.7))
        G.add_edge(row["source"], row["target"], weight=w, layer=row.get("layer", "unknown"))
    return G


def community_detection(G):
    """Louvain community detection with modularity scoring."""
    partition = community_louvain.best_partition(G, weight="weight", random_state=42)
    modularity = community_louvain.modularity(partition, G, weight="weight")

    # Community sizes
    comm_sizes = Counter(partition.values())

    # Seed protein community assignments
    seed_communities = {}
    for seed in SEED_PROTEINS:
        if seed in partition:
            seed_communities[seed] = partition[seed]

    # Check if seeds cluster together or spread across communities
    seed_comm_ids = list(seed_communities.values())
    unique_seed_comms = set(seed_comm_ids)

    return partition, modularity, comm_sizes, seed_communities, unique_seed_comms


def hub_significance_test(G, hub_metrics_df, n_random=1000):
    """Test whether observed hub centrality is significant vs random networks."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    # Get observed metrics for top 20 hubs
    top_hubs = hub_metrics_df.nlargest(20, "hub_score")

    observed_degrees = top_hubs.set_index("gene_symbol")["degree"].to_dict()
    observed_betweenness = top_hubs.set_index("gene_symbol")["betweenness_centrality"].to_dict()
    observed_pagerank = top_hubs.set_index("gene_symbol")["pagerank"].to_dict()

    # Degree distribution of the real network
    real_degrees = [d for _, d in G.degree()]
    real_degree_mean = np.mean(real_degrees)
    real_degree_std = np.std(real_degrees)

    # Generate random network degree distributions for comparison
    # Using configuration model to preserve degree sequence
    random_max_degrees = []
    random_max_betweenness = []
    for i in range(n_random):
        degree_seq = list(dict(G.degree()).values())
        # Make degree sequence graphical
        if sum(degree_seq) % 2 != 0:
            degree_seq[0] += 1
        try:
            G_rand = nx.configuration_model(degree_seq, seed=i)
            G_rand = nx.Graph(G_rand)  # remove multi-edges
            G_rand.remove_edges_from(nx.selfloop_edges(G_rand))
            random_max_degrees.append(max(dict(G_rand.degree()).values()))
            # Approximate betweenness on a sample for speed
            bc = nx.betweenness_centrality(G_rand, k=min(100, n_nodes))
            random_max_betweenness.append(max(bc.values()))
        except Exception:
            continue

    # Z-scores for top hub degree
    top_hub_name = top_hubs.iloc[0]["gene_symbol"]
    top_hub_degree = top_hubs.iloc[0]["degree"]
    degree_zscore = (top_hub_degree - real_degree_mean) / real_degree_std if real_degree_std > 0 else 0

    # Empirical p-value: fraction of random networks with max degree >= observed top degree
    if random_max_degrees:
        degree_pval = np.mean([d >= top_hub_degree for d in random_max_degrees])
    else:
        degree_pval = np.nan

    if random_max_betweenness:
        top_hub_bc = top_hubs.iloc[0]["betweenness_centrality"]
        bc_pval = np.mean([b >= top_hub_bc for b in random_max_betweenness])
    else:
        bc_pval = np.nan

    # Power-law test on degree distribution
    degrees = np.array(real_degrees)
    degrees_nonzero = degrees[degrees > 0]
    # Fit power law via log-log linear regression
    log_bins = np.logspace(np.log10(1), np.log10(max(degrees_nonzero)), 30)
    hist, bin_edges = np.histogram(degrees_nonzero, bins=log_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = hist > 0
    if mask.sum() > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            np.log10(bin_centers[mask]), np.log10(hist[mask])
        )
        power_law_exponent = -slope
        power_law_r2 = r_value ** 2
    else:
        power_law_exponent = np.nan
        power_law_r2 = np.nan

    return {
        "top_hub": top_hub_name,
        "top_hub_degree": int(top_hub_degree),
        "degree_zscore": round(degree_zscore, 2),
        "degree_empirical_pval": round(degree_pval, 4) if not np.isnan(degree_pval) else None,
        "betweenness_empirical_pval": round(bc_pval, 4) if not np.isnan(bc_pval) else None,
        "n_random_networks": len(random_max_degrees),
        "random_max_degree_mean": round(np.mean(random_max_degrees), 1) if random_max_degrees else None,
        "random_max_degree_std": round(np.std(random_max_degrees), 1) if random_max_degrees else None,
        "power_law_exponent": round(power_law_exponent, 3) if not np.isnan(power_law_exponent) else None,
        "power_law_r2": round(power_law_r2, 3) if not np.isnan(power_law_r2) else None,
        "network_degree_mean": round(real_degree_mean, 2),
        "network_degree_std": round(real_degree_std, 2),
        "network_degree_max": int(max(real_degrees)),
        "network_degree_median": int(np.median(real_degrees)),
    }


def robustness_analysis(G, hub_metrics_df, n_removals=50):
    """Test network robustness by sequential removal of top hubs."""
    top_hubs = hub_metrics_df.nlargest(n_removals, "hub_score")["gene_symbol"].tolist()

    G_copy = G.copy()
    initial_components = nx.number_connected_components(G_copy)
    initial_largest_cc = len(max(nx.connected_components(G_copy), key=len))

    results = [{
        "hubs_removed": 0,
        "nodes_remaining": G_copy.number_of_nodes(),
        "edges_remaining": G_copy.number_of_edges(),
        "n_components": initial_components,
        "largest_cc_size": initial_largest_cc,
        "largest_cc_fraction": 1.0,
    }]

    for i, hub in enumerate(top_hubs):
        if hub in G_copy:
            G_copy.remove_node(hub)
        n_comp = nx.number_connected_components(G_copy)
        largest_cc = len(max(nx.connected_components(G_copy), key=len)) if G_copy.number_of_nodes() > 0 else 0

        results.append({
            "hubs_removed": i + 1,
            "hub_removed": hub,
            "nodes_remaining": G_copy.number_of_nodes(),
            "edges_remaining": G_copy.number_of_edges(),
            "n_components": n_comp,
            "largest_cc_size": largest_cc,
            "largest_cc_fraction": round(largest_cc / initial_largest_cc, 4),
        })

    # Compare: random removal baseline
    G_copy2 = G.copy()
    random_nodes = list(G_copy2.nodes())
    rng = np.random.default_rng(42)
    rng.shuffle(random_nodes)

    random_results = []
    initial_largest_cc2 = len(max(nx.connected_components(G_copy2), key=len))
    for i in range(n_removals):
        if i < len(random_nodes) and random_nodes[i] in G_copy2:
            G_copy2.remove_node(random_nodes[i])
        largest_cc2 = len(max(nx.connected_components(G_copy2), key=len)) if G_copy2.number_of_nodes() > 0 else 0
        random_results.append(round(largest_cc2 / initial_largest_cc2, 4))

    return results, random_results


def main():
    print("=" * 60)
    print("Task #1083: Network Topology and Community Structure")
    print("=" * 60)

    # Build graph
    print("\n[1/5] Building graph...")
    G = build_graph()
    print(f"  Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.6f}")
    print(f"  Connected components: {nx.number_connected_components(G)}")

    # Community detection
    print("\n[2/5] Running Louvain community detection...")
    partition, modularity, comm_sizes, seed_comms, unique_seed_comms = community_detection(G)
    n_communities = len(comm_sizes)
    print(f"  Communities detected: {n_communities}")
    print(f"  Modularity: {modularity:.4f}")
    print(f"  Top 5 community sizes: {sorted(comm_sizes.values(), reverse=True)[:5]}")
    print(f"  Seed protein communities: {seed_comms}")
    print(f"  Seeds span {len(unique_seed_comms)} distinct communities")

    # Save community assignments
    comm_df = pd.DataFrame([
        {"gene_symbol": node, "community": comm, "is_seed": node in SEED_PROTEINS}
        for node, comm in partition.items()
    ])
    comm_df.to_csv(DATA_DIR / "community_assignments.tsv", sep="\t", index=False)

    # Community-level stats
    comm_stats = []
    for comm_id in sorted(comm_sizes.keys()):
        members = [n for n, c in partition.items() if c == comm_id]
        seeds_in = [n for n in members if n in SEED_PROTEINS]
        subgraph = G.subgraph(members)
        comm_stats.append({
            "community_id": comm_id,
            "size": len(members),
            "seed_proteins": "; ".join(seeds_in) if seeds_in else "",
            "n_seeds": len(seeds_in),
            "internal_edges": subgraph.number_of_edges(),
            "density": round(nx.density(subgraph), 6) if len(members) > 1 else 0,
        })
    comm_stats_df = pd.DataFrame(comm_stats)
    comm_stats_df.to_csv(DATA_DIR / "community_stats.tsv", sep="\t", index=False)

    # Hub significance testing
    print("\n[3/5] Testing hub significance (1000 random networks)...")
    hub_metrics = pd.read_csv(DATA_DIR / "hub_centrality_metrics.tsv", sep="\t")
    hub_sig = hub_significance_test(G, hub_metrics, n_random=1000)
    print(f"  Top hub: {hub_sig['top_hub']} (degree={hub_sig['top_hub_degree']})")
    print(f"  Degree z-score: {hub_sig['degree_zscore']}")
    print(f"  Degree empirical p-value: {hub_sig['degree_empirical_pval']}")
    print(f"  Betweenness empirical p-value: {hub_sig['betweenness_empirical_pval']}")
    print(f"  Power-law exponent: {hub_sig['power_law_exponent']}, R²: {hub_sig['power_law_r2']}")

    # Robustness analysis
    print("\n[4/5] Running robustness analysis (top 50 hub removal)...")
    robustness, random_baseline = robustness_analysis(G, hub_metrics, n_removals=50)
    rob_df = pd.DataFrame(robustness)
    rob_df.to_csv(DATA_DIR / "robustness_analysis.tsv", sep="\t", index=False)

    # Key robustness metrics (guard for networks with fewer than 50 hubs)
    after_10 = robustness[min(10, len(robustness) - 1)]["largest_cc_fraction"]
    after_20 = robustness[min(20, len(robustness) - 1)]["largest_cc_fraction"]
    after_50 = robustness[min(50, len(robustness) - 1)]["largest_cc_fraction"]
    random_after_10 = random_baseline[min(9, len(random_baseline) - 1)]
    random_after_20 = random_baseline[min(19, len(random_baseline) - 1)]
    random_after_50 = random_baseline[min(49, len(random_baseline) - 1)]

    print(f"  After removing top 10 hubs: {after_10:.1%} of largest CC remains (random: {random_after_10:.1%})")
    print(f"  After removing top 20 hubs: {after_20:.1%} of largest CC remains (random: {random_after_20:.1%})")
    print(f"  After removing top 50 hubs: {after_50:.1%} of largest CC remains (random: {random_after_50:.1%})")

    # Compile summary
    print("\n[5/5] Compiling results...")
    summary = {
        "network_properties": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "density": round(nx.density(G), 6),
            "connected_components": nx.number_connected_components(G),
            "largest_cc_size": len(max(nx.connected_components(G), key=len)),
        },
        "community_detection": {
            "algorithm": "Louvain (python-louvain, random_state=42)",
            "n_communities": n_communities,
            "modularity": round(modularity, 4),
            "top_5_sizes": sorted(comm_sizes.values(), reverse=True)[:5],
            "seed_community_assignments": seed_comms,
            "n_unique_seed_communities": len(unique_seed_comms),
            "seeds_share_community": len(unique_seed_comms) < len(seed_comms),
        },
        "hub_significance": hub_sig,
        "robustness": {
            "targeted_removal_10hubs": after_10,
            "targeted_removal_20hubs": after_20,
            "targeted_removal_50hubs": after_50,
            "random_removal_10nodes": random_after_10,
            "random_removal_20nodes": random_after_20,
            "random_removal_50nodes": random_after_50,
            "network_is_hub_dependent": after_50 < random_after_50,
        },
    }

    with open(DATA_DIR / "topology_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("TOPOLOGY ANALYSIS COMPLETE")
    print(f"Output: community_assignments.tsv, community_stats.tsv,")
    print(f"        robustness_analysis.tsv, topology_analysis_summary.json")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    summary = main()
