"""Phase 2a: Combination screening for drug pairs.

For all drug pairs from Phase 1 candidates (constrained to FDA-approved,
BBB-penetrant where known), compute union pathway coverage and rank by:
  (a) number of convergent pathways covered
  (b) sum of pathway proximity scores
  (c) target diversity bonus

Uses greedy set cover to identify coverage-optimal pairs: minimum drug
count for maximum pathway coverage.

Inputs:
  - output/tourettes/ts-convergent-polypharmacology/polypharmacology_agents.csv
  - output/tourettes/ts-convergent-polypharmacology/drug_pathway_coverage.csv
  - output/tourettes/ts-convergent-polypharmacology/drug_pathway_weighted_coverage.csv

Outputs:
  - output/tourettes/ts-convergent-polypharmacology/drug_combinations.csv
  - output/tourettes/ts-convergent-polypharmacology/top_combinations.csv

Usage:
    uv run python -m bioagentics.tourettes.convergent_polypharmacology.04_combination_screening
"""

from __future__ import annotations

import csv
from itertools import combinations
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-convergent-polypharmacology"

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

# Drugs prioritized for combination screening:
# Previously ranked TS drug candidates with known safety/BBB data
PRIORITY_DRUGS = {
    "ARIPIPRAZOLE", "ECOPIPAM", "RISPERIDONE", "CLONIDINE", "GUANFACINE",
    "HALOPERIDOL", "PIMOZIDE", "TOPIRAMATE", "FLUPHENAZINE", "XANOMELINE",
    "PITOLISANT", "CANNABIDIOL", "N-ACETYLCYSTEINE", "RILUZOLE", "FLUVOXAMINE",
    "PIMAVANSERIN",  # 5-HT2A inverse agonist; cortical dual-circuit candidate
}

# Extended set: well-known CNS drugs from ChEMBL data
KNOWN_CNS_DRUGS = PRIORITY_DRUGS | {
    "CLOZAPINE", "OLANZAPINE", "QUETIAPINE", "ZIPRASIDONE",
    "BROMOCRIPTINE", "CABERGOLINE", "PRAMIPEXOLE", "ROPINIROLE",
    "BUSPIRONE", "KETANSERIN", "NEFAZODONE",
    "FLUOXETINE", "SERTRALINE", "PAROXETINE", "CITALOPRAM",
    "DULOXETINE", "VENLAFAXINE", "ATOMOXETINE",
    "APOMORPHINE", "DOPAMINE", "SEROTONIN",
}

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


def load_agents() -> pd.DataFrame:
    """Load polypharmacological agents from Phase 1c."""
    return pd.read_csv(OUTPUT_DIR / "polypharmacology_agents.csv")


def load_binary_coverage() -> pd.DataFrame:
    """Load binary pathway coverage from Phase 1a."""
    return pd.read_csv(OUTPUT_DIR / "drug_pathway_coverage.csv")


def load_weighted_coverage() -> pd.DataFrame:
    """Load weighted proximity scores from Phase 1b."""
    return pd.read_csv(OUTPUT_DIR / "drug_pathway_weighted_coverage.csv")


def get_drug_data(binary_df: pd.DataFrame, weighted_df: pd.DataFrame) -> dict[str, dict]:
    """Build drug data dictionary from Phase 1a and 1b results."""
    drugs = {}

    for _, row in binary_df.iterrows():
        name = row["drug_name"]
        targets = set(str(row["targets"]).split(";"))
        covered_pws = set()
        for pw_id in CONVERGENT_PATHWAY_IDS:
            if pw_id in row.index and row[pw_id] == 1:
                covered_pws.add(pw_id)

        drugs[name] = {
            "targets": targets,
            "binary_pathways": covered_pws,
            "safety_tier": str(row.get("safety_tier", "")),
            "bbb_penetrant": str(row.get("bbb_penetrant", "")),
            "repurposing_rank": row.get("repurposing_rank", ""),
        }

    # Add proximity scores
    for _, row in weighted_df.iterrows():
        name = row["drug_name"]
        if name in drugs:
            drugs[name]["total_proximity"] = float(row.get("total_proximity_score", 0))
            drugs[name]["n_sig_pathways"] = int(row.get("n_pathways_significant", 0))
            # Per-pathway proximity scores
            for pw_id in CONVERGENT_PATHWAY_IDS:
                col = f"score_{pw_id}"
                if col in row.index:
                    drugs[name].setdefault("pw_scores", {})[pw_id] = float(row.get(col, 0))

    return drugs


def screen_combinations(
    drugs: dict[str, dict],
    max_candidates: int = 50,
) -> list[dict]:
    """Screen all pairs from top candidates for complementary coverage."""

    # Select candidate drugs for pairing
    # Prioritize known CNS drugs, then top-scoring agents
    candidates = []
    for name in drugs:
        is_priority = name.upper() in KNOWN_CNS_DRUGS
        score = len(drugs[name]["binary_pathways"])
        candidates.append((name, is_priority, score))

    # Sort: priority first, then by coverage
    candidates.sort(key=lambda x: (-x[1], -x[2]))
    candidate_names = [c[0] for c in candidates[:max_candidates]]

    print(f"   Screening {len(candidate_names)} candidate drugs "
          f"({len(list(combinations(candidate_names, 2)))} pairs)")

    results = []
    for drug_a, drug_b in combinations(candidate_names, 2):
        da = drugs[drug_a]
        db = drugs[drug_b]

        # Union pathway coverage
        union_pws = da["binary_pathways"] | db["binary_pathways"]
        union_targets = da["targets"] | db["targets"]

        # Combined target classes
        classes = set()
        for t in union_targets:
            classes.add(TARGET_CLASS.get(t, "other"))

        # Combined proximity score
        combined_prox = da.get("total_proximity", 0) + db.get("total_proximity", 0)

        # Complementarity: how many NEW pathways does adding drug_b give?
        complement_a = db["binary_pathways"] - da["binary_pathways"]
        complement_b = da["binary_pathways"] - db["binary_pathways"]
        complementarity = len(complement_a) + len(complement_b)

        # Combination score
        # Weight: coverage > complementarity > proximity > diversity
        n_union = len(union_pws)
        coverage_score = min(n_union / 4.0, 1.0)
        complement_score = min(complementarity / 4.0, 1.0)
        prox_score = min(combined_prox / 8.0, 1.0)
        diversity_score = min(len(classes) / 5.0, 1.0)

        combo_score = (
            0.30 * coverage_score
            + 0.30 * complement_score
            + 0.20 * prox_score
            + 0.20 * diversity_score
        )

        # Clinical context flags
        a_priority = drug_a.upper() in PRIORITY_DRUGS
        b_priority = drug_b.upper() in PRIORITY_DRUGS
        both_priority = a_priority and b_priority

        covered_names = [PATHWAY_NAMES.get(p, p) for p in sorted(union_pws)]

        results.append({
            "drug_a": drug_a,
            "drug_b": drug_b,
            "targets_a": ";".join(sorted(da["targets"])),
            "targets_b": ";".join(sorted(db["targets"])),
            "union_targets": ";".join(sorted(union_targets)),
            "n_union_targets": len(union_targets),
            "n_target_classes": len(classes),
            "target_classes": ";".join(sorted(classes)),
            "n_pathways_union": n_union,
            "covered_pathways": ";".join(covered_names),
            "complementarity": complementarity,
            "combined_proximity": round(combined_prox, 4),
            "combo_score": round(combo_score, 4),
            "both_priority": both_priority,
            "safety_a": da.get("safety_tier", ""),
            "safety_b": db.get("safety_tier", ""),
        })

    results.sort(key=lambda x: (-x["combo_score"], -x["n_pathways_union"]))
    return results


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 2a: Combination Screening")
    print("=" * 60)

    # Load data
    print("\n1. Loading Phase 1 results...")
    binary_df = load_binary_coverage()
    weighted_df = load_weighted_coverage()
    drugs = get_drug_data(binary_df, weighted_df)
    print(f"   {len(drugs)} drugs with coverage data")

    # Screen combinations
    print("\n2. Screening drug combinations...")
    combos = screen_combinations(drugs, max_candidates=50)
    print(f"   {len(combos)} combinations scored")

    # Save all combinations
    combo_path = OUTPUT_DIR / "drug_combinations.csv"
    fieldnames = [
        "drug_a", "drug_b", "targets_a", "targets_b", "union_targets",
        "n_union_targets", "n_target_classes", "target_classes",
        "n_pathways_union", "covered_pathways", "complementarity",
        "combined_proximity", "combo_score", "both_priority",
        "safety_a", "safety_b",
    ]
    with open(combo_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(combos)
    print(f"   Saved all to {combo_path}")

    # Top combinations
    top = combos[:50]
    top_path = OUTPUT_DIR / "top_combinations.csv"
    with open(top_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(top)
    print(f"   Top 50 saved to {top_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nTop 15 drug combinations:")
    print(f"{'Rank':<5} {'Drug A':<20} {'Drug B':<20} {'Score':<7} "
          f"{'PW':<4} {'Comp':<5} {'Classes':<8}")
    print("-" * 75)
    for i, c in enumerate(combos[:15], 1):
        print(f"{i:<5} {c['drug_a']:<20} {c['drug_b']:<20} "
              f"{c['combo_score']:.3f}  {c['n_pathways_union']:<4} "
              f"{c['complementarity']:<5} {c['target_classes']}")

    # Clinical combinations (both drugs are priority/known TS drugs)
    clinical = [c for c in combos if c["both_priority"]]
    print(f"\nClinical combinations (both drugs used in TS): {len(clinical)}")
    for c in clinical[:10]:
        print(f"  {c['drug_a']} + {c['drug_b']}: "
              f"score={c['combo_score']:.3f}, {c['n_pathways_union']} pathways, "
              f"classes={c['target_classes']}")

    # Best complementary pairs
    complementary = sorted(combos, key=lambda x: -x["complementarity"])
    print(f"\nMost complementary pairs:")
    for c in complementary[:5]:
        print(f"  {c['drug_a']} + {c['drug_b']}: "
              f"complementarity={c['complementarity']}, "
              f"union={c['n_pathways_union']} pathways")


if __name__ == "__main__":
    main()
