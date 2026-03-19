"""Phase 1c: Identify polypharmacological agents for convergent TS pathways.

Combines binary pathway coverage (Phase 1a) and network proximity scores
(Phase 1b) to identify drugs that cover >=2 convergent pathways through
multiple targets. Ranks agents by a composite polypharmacology score
integrating direct coverage, network proximity, safety, and clinical data.

Inputs:
  - output/tourettes/ts-convergent-polypharmacology/drug_pathway_coverage.csv
  - output/tourettes/ts-convergent-polypharmacology/drug_pathway_weighted_coverage.csv
  - output/tourettes/ts-drug-repurposing-network/ranked_candidates.csv

Outputs:
  - output/tourettes/ts-convergent-polypharmacology/polypharmacology_agents.csv
  - output/tourettes/ts-convergent-polypharmacology/phase1_summary.csv

Usage:
    uv run python -m bioagentics.tourettes.convergent_polypharmacology.03_polypharmacology_agents
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-convergent-polypharmacology"
RANKED_PATH = (
    REPO_ROOT / "output" / "tourettes" / "ts-drug-repurposing-network" / "ranked_candidates.csv"
)

# Convergent pathway IDs in significance order
CONVERGENT_PATHWAY_IDS = [
    "R-HSA-112316",  # neuronal system
    "GO:0007399",    # nervous system development
    "GO:0007268",    # chemical synaptic transmission
    "hsa04728",      # dopaminergic synapse
    "GO:0007409",    # axonogenesis
    "GO:0021987",    # cerebral cortex development
    "GO:0030182",    # neuron differentiation
    "GO:0001764",    # neuron migration
    "GO:0050804",    # modulation of chemical synaptic transmission
    "GO:0048699",    # generation of neurons
    "hsa04330",      # Notch signaling pathway
]

PATHWAY_NAMES = {
    "R-HSA-112316": "neuronal system",
    "GO:0007399": "nervous system development",
    "GO:0007268": "chemical synaptic transmission",
    "hsa04728": "dopaminergic synapse",
    "GO:0007409": "axonogenesis",
    "GO:0021987": "cerebral cortex development",
    "GO:0030182": "neuron differentiation",
    "GO:0001764": "neuron migration",
    "GO:0050804": "modulation of synaptic transmission",
    "GO:0048699": "generation of neurons",
    "hsa04330": "Notch signaling",
}

# Target diversity: drugs hitting targets in different receptor classes
# score higher than drugs hitting multiple subtypes of the same receptor
TARGET_CLASS = {
    "DRD1": "dopamine", "DRD2": "dopamine", "DRD3": "dopamine", "DRD4": "dopamine",
    "HTR2A": "serotonin", "HTR2C": "serotonin",
    "ADRA2A": "adrenergic",
    "GABRA1": "gaba",
    "HRH3": "histamine",
    "CHRM1": "cholinergic", "CHRM4": "cholinergic",
    "CNR1": "cannabinoid", "CNR2": "cannabinoid",
    "PDE10A": "phosphodiesterase",
    "SLC18A2": "transporter", "SLC6A4": "transporter",
}


def load_binary_coverage(path: Path) -> pd.DataFrame:
    """Load Phase 1a binary pathway coverage."""
    return pd.read_csv(path)


def load_weighted_coverage(path: Path) -> pd.DataFrame:
    """Load Phase 1b weighted proximity scores."""
    return pd.read_csv(path)


def load_ranked_candidates(path: Path) -> dict[str, dict]:
    """Load drug repurposing ranked candidates."""
    if not path.exists():
        return {}
    candidates = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            candidates[row["drug_name"].upper()] = {
                "rank": int(row["rank"]),
                "final_score": float(row["final_score"]),
                "safety_tier": row["safety_tier"],
                "bbb_penetrant": row["bbb_penetrant"] == "True",
                "in_ts_trials": row["in_ts_trials"] == "True",
                "pathway_annotations": row.get("pathway_annotations", ""),
            }
    return candidates


def compute_target_diversity(targets_str: str) -> tuple[int, list[str]]:
    """Count distinct pharmacological classes in a drug's target profile."""
    targets = [t.strip() for t in targets_str.split(";") if t.strip()]
    classes = set()
    for t in targets:
        cls = TARGET_CLASS.get(t, "other")
        classes.add(cls)
    return len(classes), sorted(classes)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 1c: Polypharmacological Agent Identification")
    print("=" * 60)

    # Load Phase 1a binary coverage
    binary_path = OUTPUT_DIR / "drug_pathway_coverage.csv"
    print("\n1. Loading binary pathway coverage (Phase 1a)...")
    binary_df = load_binary_coverage(binary_path)
    print(f"   {len(binary_df)} drugs with pathway coverage data")

    # Load Phase 1b weighted coverage
    weighted_path = OUTPUT_DIR / "drug_pathway_weighted_coverage.csv"
    print("\n2. Loading network proximity scores (Phase 1b)...")
    weighted_df = load_weighted_coverage(weighted_path)
    print(f"   {len(weighted_df)} drugs with proximity scores")

    # Load ranked candidates
    ranked = load_ranked_candidates(RANKED_PATH)

    # Merge binary and weighted data
    print("\n3. Identifying polypharmacological agents...")

    # Build composite scores
    results = []
    for _, row in binary_df.iterrows():
        drug_name = row["drug_name"]

        # Binary coverage count
        n_binary = int(row["n_pathways_covered"])
        if n_binary < 1:
            continue

        # Get weighted score from Phase 1b
        w_match = weighted_df[weighted_df["drug_name"] == drug_name]
        if not w_match.empty:
            total_prox = float(w_match.iloc[0]["total_proximity_score"])
            n_sig_pathways = int(w_match.iloc[0]["n_pathways_significant"])
        else:
            total_prox = 0.0
            n_sig_pathways = 0

        # Target diversity
        targets = str(row["targets"])
        n_targets = int(row["n_targets"])
        n_classes, classes = compute_target_diversity(targets)

        # Effective pathway coverage = binary + proximity-significant
        # (some pathways may be covered via proximity but not directly)
        effective_coverage = max(n_binary, n_binary + n_sig_pathways)

        # Composite polypharmacology score:
        # 1. Binary coverage (0-4 max, normalized)
        binary_score = min(n_binary / 4.0, 1.0)
        # 2. Network proximity (normalized)
        prox_score = min(total_prox / 5.0, 1.0)
        # 3. Target diversity bonus
        diversity_score = min(n_classes / 4.0, 1.0)

        # Weighted composite
        composite = (
            0.40 * binary_score
            + 0.35 * prox_score
            + 0.25 * diversity_score
        )

        # Ranked candidate info
        rinfo = ranked.get(drug_name.upper(), {})

        covered_pathways = []
        for pw_id in CONVERGENT_PATHWAY_IDS:
            if pw_id in binary_df.columns and row.get(pw_id, 0) == 1:
                covered_pathways.append(PATHWAY_NAMES.get(pw_id, pw_id))

        results.append({
            "drug_name": drug_name,
            "targets": targets,
            "n_targets": n_targets,
            "n_target_classes": n_classes,
            "target_classes": ";".join(classes),
            "n_pathways_binary": n_binary,
            "n_pathways_proximity": n_sig_pathways,
            "effective_coverage": effective_coverage,
            "covered_pathways": ";".join(covered_pathways),
            "total_proximity_score": round(total_prox, 4),
            "binary_score": round(binary_score, 4),
            "proximity_score_norm": round(prox_score, 4),
            "diversity_score": round(diversity_score, 4),
            "composite_score": round(composite, 4),
            "repurposing_rank": rinfo.get("rank", ""),
            "repurposing_score": rinfo.get("final_score", ""),
            "safety_tier": rinfo.get("safety_tier", ""),
            "bbb_penetrant": rinfo.get("bbb_penetrant", ""),
            "in_ts_trials": rinfo.get("in_ts_trials", ""),
        })

    # Sort by composite score
    results.sort(key=lambda x: -x["composite_score"])

    # Save polypharmacological agents
    agents_path = OUTPUT_DIR / "polypharmacology_agents.csv"
    fieldnames = [
        "drug_name", "targets", "n_targets", "n_target_classes", "target_classes",
        "n_pathways_binary", "n_pathways_proximity", "effective_coverage",
        "covered_pathways", "total_proximity_score",
        "binary_score", "proximity_score_norm", "diversity_score", "composite_score",
        "repurposing_rank", "repurposing_score", "safety_tier",
        "bbb_penetrant", "in_ts_trials",
    ]
    with open(agents_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"   Saved {len(results)} agents to {agents_path}")

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    multi_target = [r for r in results if r["n_targets"] >= 2]
    multi_class = [r for r in results if r["n_target_classes"] >= 2]
    print(f"\nTotal drugs with pathway coverage: {len(results)}")
    print(f"Multi-target drugs (>=2 targets): {len(multi_target)}")
    print(f"Multi-class drugs (>=2 pharmacological classes): {len(multi_class)}")

    # Top polypharmacological agents
    print(f"\nTop 15 polypharmacological agents:")
    print(f"{'Rank':<5} {'Drug':<25} {'Score':<7} {'BinPW':<6} {'ProxPW':<7} "
          f"{'Targets':<7} {'Classes':<8} {'Safety':<15}")
    print("-" * 80)
    for i, r in enumerate(results[:15], 1):
        safety = r["safety_tier"] if r["safety_tier"] else "n/a"
        print(f"{i:<5} {r['drug_name']:<25} {r['composite_score']:.3f}  "
              f"{r['n_pathways_binary']:<6} {r['n_pathways_proximity']:<7} "
              f"{r['n_targets']:<7} {r['n_target_classes']:<8} {safety}")

    # Success criteria: >=5 agents covering >=3 pathways
    agents_3pw = [r for r in results if r["effective_coverage"] >= 3]
    print(f"\nSuccess criterion: >=5 agents covering >=3 pathways")
    print(f"  Found: {len(agents_3pw)} agents with effective coverage >=3")
    if agents_3pw:
        for r in agents_3pw[:10]:
            print(f"    {r['drug_name']}: {r['effective_coverage']} pathways "
                  f"({r['target_classes']})")

    # Create Phase 1 summary
    summary_path = OUTPUT_DIR / "phase1_summary.csv"
    summary_rows = []
    for i, r in enumerate(results[:30], 1):
        summary_rows.append({
            "rank": i,
            "drug_name": r["drug_name"],
            "composite_score": r["composite_score"],
            "n_pathways": r["effective_coverage"],
            "covered_pathways": r["covered_pathways"],
            "targets": r["targets"],
            "n_target_classes": r["n_target_classes"],
            "target_classes": r["target_classes"],
            "safety_tier": r["safety_tier"],
            "bbb_penetrant": r["bbb_penetrant"],
            "in_ts_trials": r["in_ts_trials"],
        })
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nPhase 1 summary (top 30) saved to {summary_path}")


if __name__ == "__main__":
    main()
