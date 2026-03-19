"""Patient heterogeneity subgroup framework for PANDAS autoantibody network.

Models patient-to-patient autoantibody heterogeneity per McGregor 2025 findings.
Defines patient subgroups by autoantibody profile combinations, constructs
subgroup-specific subnetworks, and identifies shared vs private network
disruption patterns. Supports temporal expansion of target sets (epitope spreading).

Usage:
    uv run python -m pandas_pans.autoantibody_target_network_mapping.patient_heterogeneity

Output:
    data/pandas_pans/autoantibody_network/patient_subgroups.json
    data/pandas_pans/autoantibody_network/subgroup_subnetworks.tsv
    data/pandas_pans/autoantibody_network/shared_vs_private_analysis.tsv
    data/pandas_pans/autoantibody_network/heterogeneity_stats.json
"""

import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import pandas as pd

from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    SEED_PROTEINS,
    SeedProtein,
    get_gene_symbols,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/pandas_pans/autoantibody_network")


# Patient subgroup definitions based on known autoantibody profiles.
# These reflect hypothetical but clinically plausible combinations based on
# Cunningham Panel results and literature-reported autoantibody patterns.
# In real patients, autoantibody profiles vary significantly (McGregor 2025).
PATIENT_SUBGROUPS = {
    "dopaminergic_dominant": {
        "description": "Patients with primarily anti-dopamine receptor autoantibodies",
        "seed_genes": ["DRD1", "DRD2"],
        "clinical_phenotype": "Tics, OCD, chorea-like movements",
    },
    "calcium_signaling": {
        "description": "Patients with anti-CaMKII + dopamine receptor autoantibodies",
        "seed_genes": ["CAMK2A", "DRD1", "DRD2"],
        "clinical_phenotype": "Tics, OCD, emotional lability, cognitive changes",
    },
    "metabolic_surface": {
        "description": "Patients with anti-glycolytic enzyme autoantibodies",
        "seed_genes": ["PKM", "ALDOC", "ENO1", "ENO2"],
        "clinical_phenotype": "Cognitive regression, neuronal metabolic disruption",
    },
    "folate_disruption": {
        "description": "Patients with anti-FOLR1 (folate receptor) autoantibodies",
        "seed_genes": ["FOLR1"],
        "clinical_phenotype": "Cognitive/developmental regression, folate deficiency symptoms",
    },
    "broad_autoimmunity": {
        "description": "Patients with multiple autoantibody targets (epitope spreading)",
        "seed_genes": ["DRD1", "DRD2", "CAMK2A", "TUBB3", "PKM", "ENO1", "FOLR1"],
        "clinical_phenotype": "Severe, multi-domain symptoms",
    },
    "cunningham_classic": {
        "description": "Classic Cunningham Panel positive profile",
        "seed_genes": ["DRD1", "DRD2", "TUBB3", "CAMK2A"],
        "clinical_phenotype": "Classic PANDAS presentation: tics + OCD",
    },
    "drd1_camkii": {
        "description": "DRD1 + CaMKII combination (common pattern)",
        "seed_genes": ["DRD1", "CAMK2A"],
        "clinical_phenotype": "Movement abnormalities, calcium signaling disruption",
    },
    "drd2_folr1": {
        "description": "DRD2 + FOLR1 combination",
        "seed_genes": ["DRD2", "FOLR1"],
        "clinical_phenotype": "Dopaminergic + folate transport disruption",
    },
}


def load_network() -> pd.DataFrame:
    """Load the combined PPI edge list."""
    path = DATA_DIR / "combined_ppi_edgelist.tsv"
    if not path.exists():
        logger.error("Combined PPI edgelist not found: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def extract_subnetwork(
    network_df: pd.DataFrame,
    seed_genes: list[str],
    max_depth: int = 1,
) -> pd.DataFrame:
    """Extract subnetwork around specific seed genes.

    Returns edges where at least one node is a seed gene or a first-degree
    interactor of a seed gene.
    """
    if network_df.empty:
        return pd.DataFrame()

    # Depth 0: edges directly involving seed genes
    seed_set = set(seed_genes)
    direct_edges = network_df[
        network_df["source"].isin(seed_set) | network_df["target"].isin(seed_set)
    ]

    if max_depth == 0:
        return direct_edges

    # Depth 1: include edges between first-degree interactors
    interactors = set(direct_edges["source"]) | set(direct_edges["target"])
    subnetwork = network_df[
        network_df["source"].isin(interactors) & network_df["target"].isin(interactors)
    ]

    return subnetwork


def analyze_subgroup_overlap(
    subgroup_nodes: dict[str, set[str]],
) -> pd.DataFrame:
    """Analyze node overlap between all pairs of subgroups.

    Returns DataFrame with pairwise overlap metrics.
    """
    rows = []
    subgroup_names = sorted(subgroup_nodes.keys())

    for sg1, sg2 in combinations(subgroup_names, 2):
        nodes1 = subgroup_nodes[sg1]
        nodes2 = subgroup_nodes[sg2]
        shared = nodes1 & nodes2
        private1 = nodes1 - nodes2
        private2 = nodes2 - nodes1
        union = nodes1 | nodes2
        jaccard = len(shared) / len(union) if union else 0

        rows.append({
            "subgroup_1": sg1,
            "subgroup_2": sg2,
            "nodes_1": len(nodes1),
            "nodes_2": len(nodes2),
            "shared_nodes": len(shared),
            "private_to_1": len(private1),
            "private_to_2": len(private2),
            "jaccard_index": round(jaccard, 4),
            "shared_genes_sample": ",".join(sorted(shared)[:10]),
        })

    return pd.DataFrame(rows)


def compute_shared_vs_private(
    subgroup_subnetworks: dict[str, pd.DataFrame],
    all_seed_genes: set[str],
) -> pd.DataFrame:
    """Identify shared vs private disruption patterns across subgroups.

    A node is 'shared' if it appears in >=2 subgroup subnetworks,
    'private' if it appears in exactly one.
    """
    # Count in how many subgroups each node appears
    node_subgroup_count: dict[str, set[str]] = defaultdict(set)
    for sg_name, sg_df in subgroup_subnetworks.items():
        if sg_df.empty:
            continue
        nodes = set(sg_df["source"]) | set(sg_df["target"])
        for n in nodes:
            node_subgroup_count[n].add(sg_name)

    rows = []
    for node, subgroups in node_subgroup_count.items():
        n_subgroups = len(subgroups)
        rows.append({
            "gene_symbol": node,
            "n_subgroups": n_subgroups,
            "pattern": "shared" if n_subgroups >= 2 else "private",
            "subgroups": ",".join(sorted(subgroups)),
            "is_seed": node in all_seed_genes,
            "is_universal": n_subgroups >= len(subgroup_subnetworks) - 1,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("n_subgroups", ascending=False)
    return df


def check_ppp1r12b(network_df: pd.DataFrame) -> dict:
    """Check PPP1R12B against PANDAS target lists per task requirement."""
    all_nodes = set(network_df["source"]) | set(network_df["target"])
    seed_genes = set(get_gene_symbols())

    result = {
        "gene": "PPP1R12B",
        "in_network": "PPP1R12B" in all_nodes,
        "is_seed": "PPP1R12B" in seed_genes,
    }

    if "PPP1R12B" in all_nodes:
        ppp_edges = network_df[
            (network_df["source"] == "PPP1R12B") | (network_df["target"] == "PPP1R12B")
        ]
        neighbors = set(ppp_edges["source"]) | set(ppp_edges["target"]) - {"PPP1R12B"}
        seed_neighbors = neighbors & seed_genes
        result["n_interactions"] = len(ppp_edges)
        result["interacts_with_seeds"] = sorted(seed_neighbors)
        result["total_neighbors"] = len(neighbors)

    return result


def model_epitope_spreading(
    network_df: pd.DataFrame,
    initial_seeds: list[str],
    n_flares: int = 3,
) -> list[dict]:
    """Model temporal epitope spreading over successive flares.

    Simulates how autoantibody targets may diversify over time by
    expanding to first-degree interactors of current targets in each flare.
    """
    current_targets = set(initial_seeds)
    history = []

    for flare in range(n_flares + 1):
        # Get all first-degree neighbors of current targets
        target_edges = network_df[
            network_df["source"].isin(current_targets) | network_df["target"].isin(current_targets)
        ]
        neighbors = set(target_edges["source"]) | set(target_edges["target"])
        new_targets = neighbors - current_targets

        history.append({
            "flare": flare,
            "n_targets": len(current_targets),
            "n_new_targets": len(new_targets) if flare > 0 else 0,
            "targets": sorted(current_targets),
            "new_targets_sample": sorted(new_targets)[:20],
        })

        if flare < n_flares:
            # Expand: add some proportion of first-degree interactors
            # In reality, only a subset would become new autoantibody targets
            # Model conservative expansion: top interactors by edge score
            if not target_edges.empty and "combined_score" in target_edges.columns:
                # Add top 10% of new neighbors by interaction score
                new_target_edges = target_edges[
                    target_edges["source"].isin(new_targets) | target_edges["target"].isin(new_targets)
                ]
                if not new_target_edges.empty:
                    top_n = max(3, len(new_targets) // 10)
                    top_new = set()
                    for _, edge in new_target_edges.nlargest(top_n, "combined_score").iterrows():
                        if edge["source"] in new_targets:
                            top_new.add(edge["source"])
                        if edge["target"] in new_targets:
                            top_new.add(edge["target"])
                    current_targets = current_targets | top_new

    return history


def run() -> None:
    """Execute the patient heterogeneity subgroup analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    all_seed_genes = set(get_gene_symbols())

    # Step 1: Load network
    logger.info("Step 1: Loading PPI network")
    network_df = load_network()
    if network_df.empty:
        logger.error("Failed to load network")
        return
    logger.info("Loaded network: %d edges", len(network_df))

    # Step 2: Extract subgroup subnetworks
    logger.info("Step 2: Extracting subgroup subnetworks")
    subgroup_subnetworks: dict[str, pd.DataFrame] = {}
    subgroup_nodes: dict[str, set[str]] = {}
    subnetwork_rows = []

    for sg_name, sg_def in PATIENT_SUBGROUPS.items():
        sg_subnet = extract_subnetwork(network_df, sg_def["seed_genes"])
        subgroup_subnetworks[sg_name] = sg_subnet
        nodes = set(sg_subnet["source"]) | set(sg_subnet["target"]) if not sg_subnet.empty else set()
        subgroup_nodes[sg_name] = nodes

        # Tag edges with subgroup
        if not sg_subnet.empty:
            tagged = sg_subnet.copy()
            tagged["subgroup"] = sg_name
            subnetwork_rows.append(tagged)

        logger.info(
            "  %s: %d seed genes -> %d edges, %d nodes",
            sg_name,
            len(sg_def["seed_genes"]),
            len(sg_subnet),
            len(nodes),
        )

    # Save subnetwork edges
    if subnetwork_rows:
        all_subnetworks = pd.concat(subnetwork_rows, ignore_index=True)
        subnetwork_path = DATA_DIR / "subgroup_subnetworks.tsv"
        all_subnetworks.to_csv(subnetwork_path, sep="\t", index=False)
        logger.info("Saved subgroup subnetworks to %s", subnetwork_path)

    # Step 3: Overlap analysis
    logger.info("Step 3: Analyzing subgroup overlap")
    overlap_df = analyze_subgroup_overlap(subgroup_nodes)

    # Step 4: Shared vs private disruption patterns
    logger.info("Step 4: Computing shared vs private disruption patterns")
    shared_private_df = compute_shared_vs_private(subgroup_subnetworks, all_seed_genes)

    if not shared_private_df.empty:
        sp_path = DATA_DIR / "shared_vs_private_analysis.tsv"
        shared_private_df.to_csv(sp_path, sep="\t", index=False)
        logger.info("Saved shared/private analysis to %s", sp_path)

    # Step 5: Check PPP1R12B
    logger.info("Step 5: Checking PPP1R12B against network")
    ppp1r12b_result = check_ppp1r12b(network_df)
    logger.info("PPP1R12B: in_network=%s, is_seed=%s",
                ppp1r12b_result["in_network"], ppp1r12b_result["is_seed"])

    # Step 6: Model epitope spreading for a few subgroups
    logger.info("Step 6: Modeling epitope spreading")
    spreading_results = {}
    for sg_name in ["dopaminergic_dominant", "folate_disruption", "drd1_camkii"]:
        seeds = PATIENT_SUBGROUPS[sg_name]["seed_genes"]
        history = model_epitope_spreading(network_df, seeds, n_flares=3)
        spreading_results[sg_name] = history
        logger.info(
            "  %s: flare 0=%d targets -> flare 3=%d targets",
            sg_name,
            history[0]["n_targets"],
            history[-1]["n_targets"],
        )

    # Step 7: Save all results
    subgroup_info = {}
    for sg_name, sg_def in PATIENT_SUBGROUPS.items():
        nodes = subgroup_nodes.get(sg_name, set())
        seed_set = set(sg_def["seed_genes"])
        interactors = nodes - seed_set

        subgroup_info[sg_name] = {
            "description": sg_def["description"],
            "clinical_phenotype": sg_def["clinical_phenotype"],
            "seed_genes": sg_def["seed_genes"],
            "n_nodes": len(nodes),
            "n_interactors": len(interactors),
            "n_edges": len(subgroup_subnetworks.get(sg_name, pd.DataFrame())),
        }

    results = {
        "subgroups": subgroup_info,
        "ppp1r12b_check": ppp1r12b_result,
        "epitope_spreading": {
            sg: [{"flare": h["flare"], "n_targets": h["n_targets"],
                  "n_new": h["n_new_targets"]} for h in hist]
            for sg, hist in spreading_results.items()
        },
    }

    subgroups_path = DATA_DIR / "patient_subgroups.json"
    with open(subgroups_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved subgroup definitions to %s", subgroups_path)

    # Stats
    shared_count = len(shared_private_df[shared_private_df["pattern"] == "shared"]) if not shared_private_df.empty else 0
    private_count = len(shared_private_df[shared_private_df["pattern"] == "private"]) if not shared_private_df.empty else 0
    universal = len(shared_private_df[shared_private_df["is_universal"]]) if not shared_private_df.empty else 0

    stats = {
        "n_subgroups": len(PATIENT_SUBGROUPS),
        "shared_nodes": shared_count,
        "private_nodes": private_count,
        "universal_nodes": universal,
        "ppp1r12b_in_network": ppp1r12b_result["in_network"],
        "subgroup_sizes": {
            sg: len(nodes) for sg, nodes in subgroup_nodes.items()
        },
    }

    # Add overlap summary
    if not overlap_df.empty:
        stats["mean_jaccard"] = round(float(overlap_df["jaccard_index"].mean()), 4)
        most_overlapping = overlap_df.nlargest(3, "jaccard_index")
        stats["most_overlapping_pairs"] = [
            {
                "pair": f"{row['subgroup_1']} - {row['subgroup_2']}",
                "jaccard": float(row["jaccard_index"]),
                "shared": int(row["shared_nodes"]),
            }
            for _, row in most_overlapping.iterrows()
        ]

    stats_path = DATA_DIR / "heterogeneity_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    logger.info("=== Patient Heterogeneity Framework Summary ===")
    logger.info("Defined %d patient subgroups", len(PATIENT_SUBGROUPS))
    logger.info("Shared nodes: %d, Private nodes: %d, Universal: %d",
                shared_count, private_count, universal)
    for sg_name, info in subgroup_info.items():
        logger.info("  %s: %d nodes, %d edges",
                    sg_name, info["n_nodes"], info["n_edges"])


if __name__ == "__main__":
    run()
