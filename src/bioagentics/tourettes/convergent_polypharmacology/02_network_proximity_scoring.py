"""Phase 1b: Network proximity scoring for drug-pathway coverage.

Computes weighted pathway coverage using network proximity: for each drug,
measures the shortest-path distance between its targets and the genes in
each convergent pathway within the PPI network. Drugs with targets that are
network-close to pathway genes (even if not direct members) get credit.

This extends the binary coverage from Phase 1a to capture indirect pathway
effects through the PPI network, addressing the 7 druggability gaps.

Inputs:
  - output/tourettes/ts-drug-repurposing-network/ts_disease_network.graphml
  - data/results/ts-rare-variant-convergence/phase4/phase4_pathway_convergence.json
  - data/tourettes/ts-drug-repurposing-network/chembl_bioactivity.tsv

Outputs:
  - output/tourettes/ts-convergent-polypharmacology/network_proximity_by_pathway.csv
  - output/tourettes/ts-convergent-polypharmacology/drug_pathway_weighted_coverage.csv

Usage:
    uv run python -m bioagentics.tourettes.convergent_polypharmacology.02_network_proximity_scoring
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

# Supplementary curated drugs not in ChEMBL TSV (must match 01_drug_pathway_mapping.py)
SUPPLEMENTARY_DRUGS: dict[str, dict] = {
    "PIMAVANSERIN": {
        "targets": {"HTR2A", "HTR2C"},
        "chembl_id": "CHEMBL1214124",
        "max_phase": 4,
    },
    "ARIPIPRAZOLE": {
        "targets": {"DRD2", "DRD3", "HTR2A", "HTR2C"},
        "chembl_id": "CHEMBL1112",
        "max_phase": 4,
    },
    "GUANFACINE": {
        "targets": {"ADRA2A"},
        "chembl_id": "CHEMBL1383",
        "max_phase": 4,
    },
}

# --- Paths ---
NETWORK_PATH = (
    REPO_ROOT / "output" / "tourettes" / "ts-drug-repurposing-network" / "ts_disease_network.graphml"
)
CONVERGENCE_PATH = (
    REPO_ROOT
    / "data"
    / "results"
    / "ts-rare-variant-convergence"
    / "phase4"
    / "phase4_pathway_convergence.json"
)
CHEMBL_PATH = (
    REPO_ROOT / "data" / "tourettes" / "ts-drug-repurposing-network" / "chembl_bioactivity.tsv"
)
RANKED_PATH = (
    REPO_ROOT / "output" / "tourettes" / "ts-drug-repurposing-network" / "ranked_candidates.csv"
)
OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-convergent-polypharmacology"

# Network proximity distance threshold for pathway coverage
MAX_PROXIMITY_DISTANCE = 3  # hops in PPI network
N_RANDOM = 500  # permutations for z-score (reduced for 8GB RAM)


def load_convergent_pathways(path: Path) -> list[dict]:
    """Load the 11 convergent pathways from Phase 4."""
    with open(path) as f:
        data = json.load(f)
    pathways = []
    for entry in data["convergence_results"]:
        if entry.get("convergence_significant"):
            pathways.append({
                "pathway_id": entry["pathway_id"],
                "pathway_name": entry["pathway_name"],
                "all_genes": sorted(set(entry["rare_genes"] + entry["gwas_genes"])),
            })
    pathways.sort(key=lambda x: x.get("combined_p", 1.0))
    return pathways


def load_chembl_drugs(path: Path, pchembl_min: float = 6.0) -> dict[str, set[str]]:
    """Load drug -> target set from ChEMBL. Named compounds only."""
    df = pd.read_csv(path, sep="\t")
    drugs: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        name = str(row.get("compound_name", "")).strip()
        if not name or name == "nan":
            continue
        pchembl = row.get("pchembl_value", "")
        try:
            if pchembl and float(pchembl) < pchembl_min:
                continue
        except (ValueError, TypeError):
            continue
        drugs.setdefault(name, set()).add(str(row["target_gene"]))
    return drugs


def load_ranked_candidates(path: Path) -> dict[str, dict]:
    """Load previously ranked candidates for annotation."""
    if not path.exists():
        return {}
    candidates = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            candidates[row["drug_name"]] = {
                "rank": int(row["rank"]),
                "final_score": float(row["final_score"]),
                "safety_tier": row["safety_tier"],
                "bbb_penetrant": row["bbb_penetrant"] == "True",
            }
    return candidates


def compute_pathway_proximity(
    G: nx.Graph,
    drug_targets: set[str],
    pathway_genes: list[str],
) -> float:
    """Compute closest-distance proximity between drug targets and pathway genes.

    Returns the mean of min shortest-path distances from each pathway gene
    to the nearest drug target. Lower = closer.
    """
    net_nodes = set(G.nodes())
    targets_in_net = drug_targets & net_nodes
    pathway_in_net = set(pathway_genes) & net_nodes

    if not targets_in_net or not pathway_in_net:
        return float("inf")

    distances = []
    for pg in pathway_in_net:
        min_d = float("inf")
        for t in targets_in_net:
            try:
                d = nx.shortest_path_length(G, t, pg)
                min_d = min(min_d, d)
            except nx.NetworkXNoPath:
                pass
        distances.append(min_d)

    finite = [d for d in distances if d != float("inf")]
    if not finite:
        return float("inf")
    return float(np.mean(finite))


def compute_proximity_zscore(
    G: nx.Graph,
    drug_targets: set[str],
    pathway_genes: list[str],
    n_random: int = N_RANDOM,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute z-scored network proximity for a drug-pathway pair.

    Returns: (raw_distance, z_score, p_value)
    """
    d_obs = compute_pathway_proximity(G, drug_targets, pathway_genes)
    if d_obs == float("inf"):
        return float("inf"), 0.0, 1.0

    rng = random.Random(seed)
    all_nodes = list(G.nodes())
    n_targets = len(drug_targets & set(G.nodes()))

    random_dists = []
    for _ in range(n_random):
        rand_targets = set(rng.sample(all_nodes, min(n_targets, len(all_nodes))))
        d_rand = compute_pathway_proximity(G, rand_targets, pathway_genes)
        if d_rand != float("inf"):
            random_dists.append(d_rand)

    if len(random_dists) < 10:
        return d_obs, 0.0, 1.0

    mu = np.mean(random_dists)
    sigma = np.std(random_dists)
    if sigma == 0:
        return d_obs, 0.0, 1.0

    z = (d_obs - mu) / sigma
    from scipy.stats import norm
    p = float(norm.cdf(z))
    return d_obs, float(z), p


def proximity_to_score(z_score: float) -> float:
    """Convert z-score to a 0-1 proximity score. More negative z = higher score."""
    if z_score >= 0:
        return 0.0
    # Clamp to [-5, 0] range, normalize to [0, 1]
    return min(abs(z_score) / 5.0, 1.0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 1b: Network Proximity Scoring")
    print("=" * 60)

    # Load PPI network
    print("\n1. Loading PPI disease network...")
    G = nx.read_graphml(str(NETWORK_PATH))
    print(f"   {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Load convergent pathways
    print("\n2. Loading convergent pathways...")
    pathways = load_convergent_pathways(CONVERGENCE_PATH)
    print(f"   {len(pathways)} pathways")

    # Check which pathway genes are in the network
    net_nodes = set(G.nodes())
    for pw in pathways:
        in_net = [g for g in pw["all_genes"] if g in net_nodes]
        print(f"   {pw['pathway_name']}: {len(in_net)}/{len(pw['all_genes'])} genes in network")

    # Load drugs
    print("\n3. Loading ChEMBL drugs...")
    drugs = load_chembl_drugs(CHEMBL_PATH)
    # Merge supplementary curated drugs
    for name, info in SUPPLEMENTARY_DRUGS.items():
        if name not in drugs:
            drugs[name] = set(info["targets"])
    print(f"   {len(drugs)} named compounds")

    # Filter to drugs with at least one target in the network
    eligible = {
        name: targets
        for name, targets in drugs.items()
        if targets & net_nodes
    }
    print(f"   {len(eligible)} with targets in PPI network")

    # Load ranked candidates
    ranked = load_ranked_candidates(RANKED_PATH)

    # Compute proximity for each drug-pathway pair
    print(f"\n4. Computing network proximity (n_random={N_RANDOM})...")
    print(f"   This may take a few minutes...")

    proximity_rows = []
    weighted_coverage: dict[str, dict] = {}

    for i, (drug_name, targets) in enumerate(sorted(eligible.items())):
        if (i + 1) % 50 == 0:
            print(f"   Processing drug {i + 1}/{len(eligible)}...")

        drug_row = {
            "drug_name": drug_name,
            "targets": ";".join(sorted(targets)),
            "n_targets": len(targets),
        }

        total_score = 0.0
        n_significant = 0

        for pw in pathways:
            raw_d, z, p = compute_proximity_zscore(
                G, targets, pw["all_genes"], n_random=N_RANDOM
            )

            score = proximity_to_score(z)
            total_score += score
            if z < -2.0:
                n_significant += 1

            # Per-pathway proximity detail
            proximity_rows.append({
                "drug_name": drug_name,
                "targets": ";".join(sorted(targets)),
                "pathway_id": pw["pathway_id"],
                "pathway_name": pw["pathway_name"],
                "raw_distance": round(raw_d, 4) if raw_d != float("inf") else "",
                "z_score": round(z, 4),
                "p_value": round(p, 6),
                "proximity_score": round(score, 4),
                "significant": z < -2.0,
            })

        rinfo = ranked.get(drug_name, {})
        drug_row["total_proximity_score"] = round(total_score, 4)
        drug_row["n_pathways_significant"] = n_significant
        drug_row["mean_proximity_score"] = round(total_score / len(pathways), 4)
        drug_row["repurposing_rank"] = rinfo.get("rank", "")
        drug_row["repurposing_score"] = rinfo.get("final_score", "")
        drug_row["safety_tier"] = rinfo.get("safety_tier", "")
        drug_row["bbb_penetrant"] = rinfo.get("bbb_penetrant", "")

        # Add per-pathway scores as columns
        for pw in pathways:
            matching = [r for r in proximity_rows
                       if r["drug_name"] == drug_name and r["pathway_id"] == pw["pathway_id"]]
            if matching:
                drug_row[f"score_{pw['pathway_id']}"] = matching[0]["proximity_score"]

        weighted_coverage[drug_name] = drug_row

    # Save detailed proximity results
    print("\n5. Saving results...")

    prox_path = OUTPUT_DIR / "network_proximity_by_pathway.csv"
    prox_fields = [
        "drug_name", "targets", "pathway_id", "pathway_name",
        "raw_distance", "z_score", "p_value", "proximity_score", "significant",
    ]
    with open(prox_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=prox_fields)
        writer.writeheader()
        writer.writerows(proximity_rows)
    print(f"   Detailed proximity: {prox_path}")

    # Save weighted coverage summary
    coverage_path = OUTPUT_DIR / "drug_pathway_weighted_coverage.csv"
    score_cols = [f"score_{pw['pathway_id']}" for pw in pathways]
    cov_fields = [
        "drug_name", "targets", "n_targets",
        "total_proximity_score", "n_pathways_significant", "mean_proximity_score",
    ] + score_cols + [
        "repurposing_rank", "repurposing_score", "safety_tier", "bbb_penetrant",
    ]

    sorted_drugs = sorted(
        weighted_coverage.values(),
        key=lambda x: -x["total_proximity_score"],
    )

    with open(coverage_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cov_fields)
        writer.writeheader()
        writer.writerows(sorted_drugs)
    print(f"   Weighted coverage: {coverage_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nTop 20 drugs by total proximity score:")
    for row in sorted_drugs[:20]:
        rank_info = f" [rank #{row['repurposing_rank']}]" if row['repurposing_rank'] else ""
        print(f"  {row['drug_name']}: score={row['total_proximity_score']:.3f}, "
              f"sig_pathways={row['n_pathways_significant']}, "
              f"targets={row['targets']}{rank_info}")

    # Check which pathways are reachable via network proximity
    print(f"\nPathway reachability (drugs with significant proximity z < -2):")
    for pw in pathways:
        sig_count = sum(
            1 for r in proximity_rows
            if r["pathway_id"] == pw["pathway_id"] and r["significant"]
        )
        print(f"  {pw['pathway_name']}: {sig_count} drugs significantly proximal")


if __name__ == "__main__":
    main()
