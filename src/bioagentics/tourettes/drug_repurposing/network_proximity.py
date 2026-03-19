"""Network proximity analysis for drug-disease module distances.

For each FDA-approved drug (from DrugBank), computes network proximity
between drug targets and TS disease modules:
1. Shortest path distance between drug targets and each TS module
2. Z-score normalization against 1000 random drug-target sets of same size
3. Aggregate proximity score across modules

Can also use ChEMBL bioactivity data as a fallback drug-target source
when DrugBank XML is not available.

Output: output/tourettes/ts-drug-repurposing-network/network_proximity_scores.csv

Usage:
    uv run python -m bioagentics.tourettes.drug_repurposing.network_proximity
    uv run python -m bioagentics.tourettes.drug_repurposing.network_proximity --use-chembl
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "tourettes" / "ts-drug-repurposing-network"
OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-drug-repurposing-network"

NETWORK_PATH = OUTPUT_DIR / "ts_disease_network.graphml"
MODULES_PATH = OUTPUT_DIR / "ts_network_modules.csv"
DRUGBANK_PATH = DATA_DIR / "drugbank_targets.tsv"
CHEMBL_PATH = DATA_DIR / "chembl_bioactivity.tsv"
OUTPUT_PATH = OUTPUT_DIR / "network_proximity_scores.csv"


def load_disease_network(path: Path) -> nx.Graph:
    """Load TS disease network from GraphML."""
    G = nx.read_graphml(str(path))
    print(f"  Disease network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def load_modules(path: Path) -> dict[int, set[str]]:
    """Load module assignments: module_id -> set of gene symbols."""
    df = pd.read_csv(path)
    modules: dict[int, set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        modules[int(row["module_id"])].add(row["gene"])
    print(f"  Loaded {len(modules)} modules")
    return dict(modules)


def load_drug_targets_drugbank(path: Path) -> dict[str, dict]:
    """Load drug-target pairs from DrugBank TSV.

    Returns: {drug_name: {"targets": set[str], "drugbank_id": str, "label": str}}
    """
    drugs: dict[str, dict] = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name = row["drug_name"]
            if name not in drugs:
                drugs[name] = {
                    "targets": set(),
                    "drugbank_id": row["drugbank_id"],
                    "label": row.get("ts_control_label", ""),
                }
            if row["target_gene"]:
                drugs[name]["targets"].add(row["target_gene"])
    return drugs


def load_drug_targets_chembl(path: Path, pchembl_min: float = 6.0) -> dict[str, dict]:
    """Load drug-target pairs from ChEMBL bioactivity data.

    Filters to compounds with pChEMBL >= 6.0 (sub-micromolar activity).
    Returns: {compound_name: {"targets": set[str], "chembl_id": str}}
    """
    drugs: dict[str, dict] = {}
    df = pd.read_csv(path, sep="\t")

    for _, row in df.iterrows():
        name = str(row.get("compound_name", "")).strip()
        if not name or name == "nan":
            name = str(row.get("chembl_id", ""))
        pchembl = row.get("pchembl_value", "")
        try:
            if pchembl and float(pchembl) < pchembl_min:
                continue
        except (ValueError, TypeError):
            continue

        if name not in drugs:
            drugs[name] = {
                "targets": set(),
                "chembl_id": str(row.get("chembl_id", "")),
                "label": "",
            }
        drugs[name]["targets"].add(row["target_gene"])

    return drugs


def compute_module_distance(
    G: nx.Graph,
    drug_targets: set[str],
    module_genes: set[str],
    network_nodes: set[str] | None = None,
) -> float:
    """Compute closest shortest-path distance between drug targets and module genes.

    Uses the 'closest' measure: d_c(S,T) = 1/|T| * sum_{t in T} min_{s in S} d(s,t)

    Parameters
    ----------
    network_nodes : set[str] | None
        Pre-computed set(G.nodes()) to avoid recomputing in hot loops.
    """
    if network_nodes is None:
        network_nodes = set(G.nodes())

    targets_in_net = drug_targets & network_nodes
    module_in_net = module_genes & network_nodes

    if not targets_in_net or not module_in_net:
        return float("inf")

    # Multi-source Dijkstra from all targets at once (much faster than per-pair shortest paths)
    distances = nx.multi_source_dijkstra_path_length(G, targets_in_net)

    total_dist = 0.0
    for module_gene in module_in_net:
        d = distances.get(module_gene, float("inf"))
        total_dist += d

    return total_dist / len(module_in_net)


def compute_proximity_zscore(
    G: nx.Graph,
    drug_targets: set[str],
    module_genes: set[str],
    n_random: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute network proximity z-score for drug-module pair.

    Returns: (raw_distance, z_score, p_value)
    """
    # Pre-compute node set once for the entire permutation loop
    network_nodes = set(G.nodes())

    # Observed distance
    d_obs = compute_module_distance(G, drug_targets, module_genes, network_nodes)
    if d_obs == float("inf"):
        return float("inf"), 0.0, 1.0

    # Random background: sample random node sets of same size as drug targets
    rng = random.Random(seed)
    all_nodes = list(G.nodes())
    n_targets = len(drug_targets & network_nodes)

    random_distances = []
    for _ in range(n_random):
        random_targets = set(rng.sample(all_nodes, min(n_targets, len(all_nodes))))
        d_rand = compute_module_distance(G, random_targets, module_genes, network_nodes)
        if d_rand != float("inf"):
            random_distances.append(d_rand)

    if len(random_distances) < 10:
        return d_obs, 0.0, 1.0

    mu = np.mean(random_distances)
    sigma = np.std(random_distances)
    if sigma == 0:
        return d_obs, 0.0, 1.0

    z_score = (d_obs - mu) / sigma
    # Approximate p-value from z-score (one-tailed, lower is better)
    from scipy.stats import norm
    p_value = norm.cdf(z_score)

    return d_obs, z_score, p_value


def run_proximity_analysis(
    G: nx.Graph,
    modules: dict[int, set[str]],
    drugs: dict[str, dict],
    n_random: int = 1000,
    min_targets_in_network: int = 1,
) -> list[dict]:
    """Run proximity analysis for all drugs against all modules.

    Returns list of results sorted by best z-score.
    """
    network_nodes = set(G.nodes())
    results = []

    # Pre-filter drugs with targets in network
    eligible_drugs = {}
    for name, info in drugs.items():
        targets_in_net = info["targets"] & network_nodes
        if len(targets_in_net) >= min_targets_in_network:
            eligible_drugs[name] = info

    print(f"  Eligible drugs (>= {min_targets_in_network} targets in network): {len(eligible_drugs)}")

    for i, (name, info) in enumerate(eligible_drugs.items()):
        if (i + 1) % 50 == 0:
            print(f"    Processing drug {i + 1}/{len(eligible_drugs)}...")

        best_z = float("inf")
        best_module = -1
        best_raw = float("inf")
        best_p = 1.0

        for mod_id, mod_genes in modules.items():
            raw_d, z, p = compute_proximity_zscore(
                G, info["targets"], mod_genes, n_random=n_random
            )
            if z < best_z:
                best_z = z
                best_module = mod_id
                best_raw = raw_d
                best_p = p

        drug_id = info.get("drugbank_id", info.get("chembl_id", ""))
        results.append({
            "drug_id": drug_id,
            "drug_name": name,
            "n_targets": len(info["targets"]),
            "n_targets_in_network": len(info["targets"] & network_nodes),
            "closest_module": best_module,
            "raw_distance": round(best_raw, 4),
            "z_score": round(best_z, 4),
            "p_value": round(best_p, 6),
            "ts_control_label": info.get("label", ""),
        })

    results.sort(key=lambda x: x["z_score"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Network proximity analysis")
    parser.add_argument("--use-chembl", action="store_true",
                        help="Use ChEMBL data instead of DrugBank")
    parser.add_argument("--n-random", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load network and modules
    G = load_disease_network(NETWORK_PATH)
    modules = load_modules(MODULES_PATH)

    # Load drug-target data
    if args.use_chembl:
        if not CHEMBL_PATH.exists():
            print("ChEMBL data not found. Run download_chembl.py first.")
            return
        print("  Loading drug targets from ChEMBL...")
        drugs = load_drug_targets_chembl(CHEMBL_PATH)
    else:
        if not DRUGBANK_PATH.exists():
            print("DrugBank data not found. Falling back to ChEMBL...")
            if CHEMBL_PATH.exists():
                drugs = load_drug_targets_chembl(CHEMBL_PATH)
            else:
                print("No drug-target data available. Run download_drugbank.py or download_chembl.py first.")
                return
        else:
            print("  Loading drug targets from DrugBank...")
            drugs = load_drug_targets_drugbank(DRUGBANK_PATH)

    print(f"  Total drugs loaded: {len(drugs)}")

    # Run proximity analysis
    print(f"\nRunning network proximity analysis (n_random={args.n_random})...")
    results = run_proximity_analysis(G, modules, drugs, n_random=args.n_random)

    # Save results
    if results:
        fieldnames = [
            "drug_id", "drug_name", "n_targets", "n_targets_in_network",
            "closest_module", "raw_distance", "z_score", "p_value",
            "ts_control_label",
        ]
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved {len(results)} drug proximity scores to {args.output}")

    # Summary
    sig_drugs = [r for r in results if r["z_score"] < -2.0]
    print(f"\nSignificant drugs (z < -2.0): {len(sig_drugs)}")
    print("\nTop 20 closest drugs:")
    for r in results[:20]:
        label = f" [{r['ts_control_label']}]" if r["ts_control_label"] else ""
        print(f"  {r['drug_name']}: z={r['z_score']:.3f}, "
              f"module={r['closest_module']}, targets_in_net={r['n_targets_in_network']}{label}")


if __name__ == "__main__":
    main()
