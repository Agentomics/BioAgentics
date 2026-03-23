"""Phase 3: Clinical validation of polypharmacology findings.

Cross-reference top-scoring combinations with existing clinical practice:
1. Are commonly co-prescribed TS combinations explained by complementary
   pathway coverage?
2. Do clinical combinations score higher than random drug pairs?
   (permutation test, p < 0.01)
3. Test dual-circuit hypothesis: 5-HT2A + D3 antagonist combination

Inputs:
  - output/tourettes/ts-convergent-polypharmacology/safe_combinations.csv
  - output/tourettes/ts-convergent-polypharmacology/polypharmacology_agents.csv
  - output/tourettes/ts-convergent-polypharmacology/drug_pathway_coverage.csv
  - data/tourettes/ts-drug-repurposing-network/ts_clinical_trials.csv

Outputs:
  - output/tourettes/ts-convergent-polypharmacology/clinical_validation.csv
  - output/tourettes/ts-convergent-polypharmacology/permutation_test_results.csv

Usage:
    uv run python -m bioagentics.tourettes.convergent_polypharmacology.06_clinical_validation
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-convergent-polypharmacology"
TRIALS_PATH = (
    REPO_ROOT / "data" / "tourettes" / "ts-drug-repurposing-network" / "ts_clinical_trials.csv"
)

# Known co-prescribed TS drug combinations from clinical literature
# These are established combinations used in clinical practice
CLINICAL_COMBINATIONS: list[dict] = [
    {
        "drug_a": "ARIPIPRAZOLE",
        "drug_b": "GUANFACINE",
        "usage": "First-line combination: D2 partial agonist + alpha-2 agonist",
        "evidence": "Both FDA-indicated for TS; complementary mechanisms",
    },
    {
        "drug_a": "RISPERIDONE",
        "drug_b": "CLONIDINE",
        "usage": "Common combination: atypical antipsychotic + alpha-2 agonist",
        "evidence": "Widely used in pediatric TS management",
    },
    {
        "drug_a": "HALOPERIDOL",
        "drug_b": "CLONIDINE",
        "usage": "Traditional combination: typical antipsychotic + alpha-2 agonist",
        "evidence": "Historical first-line before atypicals",
    },
    {
        "drug_a": "ARIPIPRAZOLE",
        "drug_b": "CLONIDINE",
        "usage": "D2 partial agonist + alpha-2 agonist",
        "evidence": "Alternative first-line combination",
    },
    {
        "drug_a": "ECOPIPAM",
        "drug_b": "GUANFACINE",
        "usage": "Novel D1 antagonist + alpha-2 agonist",
        "evidence": "Emerging combination as ecopipam gains approval",
    },
    {
        "drug_a": "RISPERIDONE",
        "drug_b": "GUANFACINE",
        "usage": "Atypical antipsychotic + alpha-2 agonist",
        "evidence": "Common in treatment-resistant TS",
    },
    {
        "drug_a": "PIMOZIDE",
        "drug_b": "CLONIDINE",
        "usage": "Classical D2 blocker + alpha-2 agonist",
        "evidence": "Historical combination; pimozide now less used",
    },
]

# Dual-circuit hypothesis combinations
DUAL_CIRCUIT_COMBOS: list[dict] = [
    {
        "drug_a": "PIMAVANSERIN",  # 5-HT2A inverse agonist
        "drug_b": "ECOPIPAM",     # D1 antagonist (striosome-selective)
        "hypothesis": "5-HT2A/PFC + D1/striosome dual-circuit targeting",
    },
    {
        "drug_a": "KETANSERIN",    # 5-HT2A antagonist
        "drug_b": "ECOPIPAM",     # D1 antagonist
        "hypothesis": "5-HT2A + D1 antagonism covers PFC and striosomal circuits",
    },
]


def load_safe_combinations(path: Path) -> list[dict]:
    """Load safety-filtered combinations."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_agents(path: Path) -> dict[str, dict]:
    """Load polypharmacological agent data."""
    agents = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            agents[row["drug_name"]] = row
    return agents


def load_drug_coverage(path: Path) -> dict[str, float]:
    """Load per-drug pathway coverage counts."""
    coverage = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            coverage[row["drug_name"]] = int(row["n_pathways_covered"])
    return coverage


def find_combination_score(
    combos: list[dict],
    drug_a: str,
    drug_b: str,
) -> dict | None:
    """Find a specific combination in the scored list."""
    for c in combos:
        pair = {c["drug_a"].upper(), c["drug_b"].upper()}
        if pair == {drug_a.upper(), drug_b.upper()}:
            return c
    return None


def permutation_test(
    clinical_scores: list[float],
    all_scores: list[float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Test whether clinical combination scores are significantly higher than random.

    Returns: (mean_clinical, mean_random, p_value)
    """
    rng = random.Random(seed)
    mean_clinical = float(np.mean(clinical_scores))

    n_clinical = len(clinical_scores)
    count_higher = 0

    for _ in range(n_permutations):
        sample = rng.sample(all_scores, min(n_clinical, len(all_scores)))
        if np.mean(sample) >= mean_clinical:
            count_higher += 1

    p_value = (count_higher + 1) / (n_permutations + 1)
    mean_random = float(np.mean(all_scores))

    return mean_clinical, mean_random, p_value


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 3: Clinical Validation")
    print("=" * 60)

    # Load data
    print("\n1. Loading scored combinations...")
    safe_combos = load_safe_combinations(OUTPUT_DIR / "safe_combinations.csv")
    print(f"   {len(safe_combos)} safe combinations")

    agents = load_agents(OUTPUT_DIR / "polypharmacology_agents.csv")
    coverage = load_drug_coverage(OUTPUT_DIR / "drug_pathway_coverage.csv")

    # Validate clinical combinations
    print("\n2. Validating clinical combinations against pathway coverage...")
    clinical_results = []

    for cc in CLINICAL_COMBINATIONS:
        combo = find_combination_score(safe_combos, cc["drug_a"], cc["drug_b"])

        cov_a = coverage.get(cc["drug_a"], coverage.get(cc["drug_a"].upper(), 0))
        cov_b = coverage.get(cc["drug_b"], coverage.get(cc["drug_b"].upper(), 0))
        agent_a = agents.get(cc["drug_a"], agents.get(cc["drug_a"].upper(), {}))
        agent_b = agents.get(cc["drug_b"], agents.get(cc["drug_b"].upper(), {}))

        result = {
            "drug_a": cc["drug_a"],
            "drug_b": cc["drug_b"],
            "clinical_usage": cc["usage"],
            "evidence": cc["evidence"],
            "pathways_a": cov_a,
            "pathways_b": cov_b,
            "targets_a": agent_a.get("targets", ""),
            "targets_b": agent_b.get("targets", ""),
            "classes_a": agent_a.get("target_classes", ""),
            "classes_b": agent_b.get("target_classes", ""),
        }

        if combo:
            result["n_pathways_union"] = combo["n_pathways_union"]
            result["adjusted_score"] = float(combo["adjusted_score"])
            result["combo_score"] = float(combo["combo_score"])
            result["ddi_severity"] = combo["ddi_severity"]
            result["covered_pathways"] = combo["covered_pathways"]
            result["found_in_screening"] = True
        else:
            result["n_pathways_union"] = ""
            result["adjusted_score"] = 0.0
            result["combo_score"] = 0.0
            result["ddi_severity"] = ""
            result["covered_pathways"] = ""
            result["found_in_screening"] = False

        clinical_results.append(result)
        found = "FOUND" if result["found_in_screening"] else "NOT FOUND"
        score = f"adj={result['adjusted_score']:.3f}" if result["found_in_screening"] else "n/a"
        print(f"   {cc['drug_a']} + {cc['drug_b']}: {found}, {score}")

    # Permutation test
    print("\n3. Permutation test: clinical vs random combinations...")
    clinical_scores = [
        r["adjusted_score"]
        for r in clinical_results
        if r["found_in_screening"] and r["adjusted_score"] > 0
    ]
    all_scores = [
        float(c["adjusted_score"])
        for c in safe_combos
        if float(c["adjusted_score"]) > 0
    ]

    if clinical_scores and all_scores:
        mean_clin, mean_rand, p_val = permutation_test(
            clinical_scores, all_scores, n_permutations=10000
        )
        print(f"   Clinical combination mean score: {mean_clin:.4f}")
        print(f"   Random combination mean score: {mean_rand:.4f}")
        if mean_rand > 0:
            print(f"   Fold enrichment: {mean_clin / mean_rand:.2f}x")
        else:
            print(f"   Fold enrichment: N/A (random mean is zero)")
        print(f"   Permutation p-value: {p_val:.4f}")
        significant = p_val < 0.01
        print(f"   Significant (p < 0.01): {'YES' if significant else 'NO'}")
    else:
        mean_clin, mean_rand, p_val = 0.0, 0.0, 1.0
        significant = False
        print("   Insufficient data for permutation test")

    # Dual-circuit hypothesis test
    print("\n4. Testing dual-circuit hypothesis...")
    for dc in DUAL_CIRCUIT_COMBOS:
        combo = find_combination_score(safe_combos, dc["drug_a"], dc["drug_b"])
        if combo:
            print(f"   {dc['drug_a']} + {dc['drug_b']}: "
                  f"adj_score={float(combo['adjusted_score']):.3f}, "
                  f"pathways={combo['n_pathways_union']}")
        else:
            # Check if individual drugs are in our dataset
            in_a = dc["drug_a"] in agents or dc["drug_a"].upper() in agents
            in_b = dc["drug_b"] in agents or dc["drug_b"].upper() in agents
            print(f"   {dc['drug_a']} + {dc['drug_b']}: not in screening set "
                  f"(drug_a={'found' if in_a else 'missing'}, "
                  f"drug_b={'found' if in_b else 'missing'})")
        print(f"     Hypothesis: {dc['hypothesis']}")

    # Save clinical validation results
    val_path = OUTPUT_DIR / "clinical_validation.csv"
    val_fields = [
        "drug_a", "drug_b", "clinical_usage", "evidence",
        "pathways_a", "pathways_b", "targets_a", "targets_b",
        "classes_a", "classes_b",
        "n_pathways_union", "adjusted_score", "combo_score",
        "ddi_severity", "covered_pathways", "found_in_screening",
    ]
    with open(val_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=val_fields)
        writer.writeheader()
        writer.writerows(clinical_results)
    print(f"\n   Saved clinical validation to {val_path}")

    # Save permutation test results
    perm_path = OUTPUT_DIR / "permutation_test_results.csv"
    with open(perm_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "test", "n_clinical", "n_all",
            "mean_clinical", "mean_random", "fold_enrichment",
            "p_value", "significant",
        ])
        writer.writeheader()
        writer.writerow({
            "test": "clinical_vs_random_combinations",
            "n_clinical": len(clinical_scores),
            "n_all": len(all_scores),
            "mean_clinical": round(mean_clin, 4),
            "mean_random": round(mean_rand, 4),
            "fold_enrichment": round(mean_clin / mean_rand, 4) if mean_rand > 0 else "",
            "p_value": round(p_val, 6),
            "significant": significant,
        })
    print(f"   Saved permutation test to {perm_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY - ALL PHASES")
    print("=" * 60)

    print("\n[Phase 1] Drug-Pathway Mapping:")
    print(f"  - 230 drugs mapped to 11 convergent pathways")
    print(f"  - 4/11 pathways directly druggable, 7 are druggability gaps")
    print(f"  - Top agents: clozapine (5 target classes), olanzapine/risperidone (4)")

    print("\n[Phase 2] Combination Screening:")
    print(f"  - 1,225 pairs screened, 1,224 passed safety filter")
    print(f"  - Top clinical pair: ecopipam+clonidine (adj=0.472)")
    print(f"  - Coverage ceiling: 4/11 pathways (limited by current drug targets)")

    print("\n[Phase 3] Clinical Validation:")
    found_count = sum(1 for r in clinical_results if r["found_in_screening"])
    print(f"  - {found_count}/{len(CLINICAL_COMBINATIONS)} clinical combinations found in screening")
    if clinical_scores:
        print(f"  - Mean clinical score: {mean_clin:.4f} vs random: {mean_rand:.4f}")
        print(f"  - Permutation test p={p_val:.4f} ({'significant' if significant else 'not significant'})")

    print("\n[Key Finding] Druggability Gap:")
    print("  7/11 convergent pathways (nervous system development, axonogenesis,")
    print("  cortex development, neuron differentiation/migration/generation,")
    print("  Notch signaling) lack pharmacological targets in current drug databases.")
    print("  This represents a major therapeutic gap that may require novel drug")
    print("  development beyond existing repurposing candidates.")


if __name__ == "__main__":
    main()
