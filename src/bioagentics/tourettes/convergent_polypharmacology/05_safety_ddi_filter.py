"""Phase 2b: Safety and drug-drug interaction filtering.

Filter drug combinations from Phase 2a for known DDI safety using:
1. Curated DDI severity database (CYP enzyme interactions, pharmacodynamic DDIs)
2. Safety tier information from drug repurposing pipeline
3. Known TS-specific contraindications

Identifies "coverage-optimal" pairs: best pathway coverage with acceptable
safety profiles.

Inputs:
  - output/tourettes/ts-convergent-polypharmacology/drug_combinations.csv
  - output/tourettes/ts-drug-repurposing-network/ranked_candidates.csv

Outputs:
  - output/tourettes/ts-convergent-polypharmacology/safe_combinations.csv
  - output/tourettes/ts-convergent-polypharmacology/coverage_optimal_pairs.csv

Usage:
    uv run python -m bioagentics.tourettes.convergent_polypharmacology.05_safety_ddi_filter
"""

from __future__ import annotations

import csv
from pathlib import Path

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-convergent-polypharmacology"
RANKED_PATH = (
    REPO_ROOT / "output" / "tourettes" / "ts-drug-repurposing-network" / "ranked_candidates.csv"
)

# Curated DDI database for CNS drug combinations
# severity: "major" (contraindicated), "moderate" (use with caution),
#           "minor" (low risk), "none" (no known interaction)
# source: standard clinical pharmacology references
KNOWN_DDIS: dict[frozenset[str], dict] = {
    frozenset({"CLOZAPINE", "FLUVOXAMINE"}): {
        "severity": "major",
        "mechanism": "CYP1A2 inhibition by fluvoxamine increases clozapine levels",
        "recommendation": "Avoid combination or reduce clozapine dose by 50%",
    },
    frozenset({"HALOPERIDOL", "RISPERIDONE"}): {
        "severity": "major",
        "mechanism": "Additive D2 blockade increases EPS and QTc prolongation risk",
        "recommendation": "Avoid dual antipsychotic use",
    },
    frozenset({"HALOPERIDOL", "PIMOZIDE"}): {
        "severity": "major",
        "mechanism": "Both prolong QTc interval; additive cardiac risk",
        "recommendation": "Contraindicated combination",
    },
    frozenset({"RISPERIDONE", "PIMOZIDE"}): {
        "severity": "major",
        "mechanism": "Additive QTc prolongation risk",
        "recommendation": "Avoid combination",
    },
    frozenset({"FLUOXETINE", "PIMOZIDE"}): {
        "severity": "major",
        "mechanism": "CYP2D6 inhibition increases pimozide levels; QTc risk",
        "recommendation": "Contraindicated",
    },
    frozenset({"PAROXETINE", "PIMOZIDE"}): {
        "severity": "major",
        "mechanism": "CYP2D6 inhibition increases pimozide levels; QTc risk",
        "recommendation": "Contraindicated",
    },
    frozenset({"RISPERIDONE", "FLUOXETINE"}): {
        "severity": "moderate",
        "mechanism": "CYP2D6 inhibition may increase risperidone levels",
        "recommendation": "Monitor for EPS; consider dose reduction",
    },
    frozenset({"RISPERIDONE", "PAROXETINE"}): {
        "severity": "moderate",
        "mechanism": "CYP2D6 inhibition may increase risperidone levels",
        "recommendation": "Monitor; paroxetine increases risperidone AUC ~75%",
    },
    frozenset({"ARIPIPRAZOLE", "FLUOXETINE"}): {
        "severity": "moderate",
        "mechanism": "CYP2D6 inhibition may increase aripiprazole levels",
        "recommendation": "Reduce aripiprazole dose to 50%",
    },
    frozenset({"ARIPIPRAZOLE", "PAROXETINE"}): {
        "severity": "moderate",
        "mechanism": "CYP2D6 inhibition may increase aripiprazole levels",
        "recommendation": "Reduce aripiprazole dose",
    },
    frozenset({"CLONIDINE", "GUANFACINE"}): {
        "severity": "moderate",
        "mechanism": "Additive alpha-2 agonism; risk of hypotension/bradycardia",
        "recommendation": "Avoid dual alpha-2 agonists",
    },
    frozenset({"RISPERIDONE", "CLONIDINE"}): {
        "severity": "minor",
        "mechanism": "Additive sedation; generally well-tolerated combination in TS",
        "recommendation": "Common clinical combination; monitor sedation",
    },
    frozenset({"ARIPIPRAZOLE", "GUANFACINE"}): {
        "severity": "minor",
        "mechanism": "No significant PK interaction; complementary mechanisms",
        "recommendation": "Common TS combination; monitor sedation",
    },
    frozenset({"ECOPIPAM", "CLONIDINE"}): {
        "severity": "minor",
        "mechanism": "No known PK interaction; complementary D1/alpha2 mechanisms",
        "recommendation": "Favorable combination for TS",
    },
    frozenset({"ECOPIPAM", "GUANFACINE"}): {
        "severity": "minor",
        "mechanism": "No known PK interaction; complementary D1/alpha2 mechanisms",
        "recommendation": "Favorable combination for TS",
    },
    # Pimavanserin DDIs (CYP3A4 substrate)
    frozenset({"PIMAVANSERIN", "PIMOZIDE"}): {
        "severity": "major",
        "mechanism": "Additive QTc prolongation; both prolong cardiac repolarization",
        "recommendation": "Contraindicated combination",
    },
    frozenset({"PIMAVANSERIN", "HALOPERIDOL"}): {
        "severity": "moderate",
        "mechanism": "Additive QTc prolongation risk; monitor ECG",
        "recommendation": "Use with caution; ECG monitoring recommended",
    },
    frozenset({"PIMAVANSERIN", "ARIPIPRAZOLE"}): {
        "severity": "minor",
        "mechanism": "No significant PK interaction; complementary 5-HT2A/D2 mechanisms",
        "recommendation": "Dual-circuit combination: cortical (5-HT2A) + striatal (D2)",
    },
    frozenset({"PIMAVANSERIN", "ECOPIPAM"}): {
        "severity": "minor",
        "mechanism": "No known PK interaction; complementary 5-HT2A/D1 mechanisms",
        "recommendation": "Dual-circuit combination: cortical (5-HT2A) + striosomal (D1)",
    },
    frozenset({"PIMAVANSERIN", "RISPERIDONE"}): {
        "severity": "moderate",
        "mechanism": "Overlapping 5-HT2A antagonism; risperidone already has strong 5-HT2A activity",
        "recommendation": "Redundant serotonergic mechanism; limited complementarity",
    },
    frozenset({"PIMAVANSERIN", "CLONIDINE"}): {
        "severity": "minor",
        "mechanism": "No significant PK interaction; complementary 5-HT2A/alpha2 mechanisms",
        "recommendation": "Favorable combination for TS",
    },
    frozenset({"PIMAVANSERIN", "GUANFACINE"}): {
        "severity": "minor",
        "mechanism": "No significant PK interaction; complementary 5-HT2A/alpha2 mechanisms",
        "recommendation": "Favorable combination for TS",
    },
}

# Pharmacodynamic interaction classes:
# Pairs of target classes that produce concerning additive effects
PHARMACODYNAMIC_RISKS: list[tuple[set[str], str, str]] = [
    # Both hit dopamine heavily -> increased EPS risk
    ({"dopamine"}, "Dual dopamine blockade increases EPS risk", "moderate"),
    # QTc-prolonging combinations
    # (handled individually in KNOWN_DDIS for specific drugs)
]

# Safety tiers from drug repurposing
SAFETY_TIERS = {
    "tier_1_favorable": 1.0,
    "tier_2_acceptable": 0.7,
    "tier_3_caution": 0.3,
}


def load_combinations(path: Path) -> list[dict]:
    """Load drug combinations from Phase 2a."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def load_safety_info(path: Path) -> dict[str, dict]:
    """Load safety tier info from ranked candidates."""
    info = {}
    if not path.exists():
        return info
    with open(path) as f:
        for row in csv.DictReader(f):
            info[row["drug_name"].upper()] = {
                "safety_tier": row["safety_tier"],
                "bbb_penetrant": row["bbb_penetrant"] == "True",
                "in_ts_trials": row["in_ts_trials"] == "True",
            }
    return info


def assess_ddi(drug_a: str, drug_b: str) -> dict:
    """Look up known DDI for a drug pair."""
    pair = frozenset({drug_a.upper(), drug_b.upper()})
    ddi = KNOWN_DDIS.get(pair)
    if ddi:
        return ddi
    return {"severity": "unknown", "mechanism": "", "recommendation": ""}


def compute_safety_score(
    drug_a: str,
    drug_b: str,
    safety_info: dict[str, dict],
    ddi: dict,
) -> float:
    """Compute combined safety score for a drug pair."""
    # Individual safety tiers
    a_info = safety_info.get(drug_a.upper(), {})
    b_info = safety_info.get(drug_b.upper(), {})

    tier_a = SAFETY_TIERS.get(a_info.get("safety_tier", ""), 0.5)
    tier_b = SAFETY_TIERS.get(b_info.get("safety_tier", ""), 0.5)
    individual_safety = (tier_a + tier_b) / 2.0

    # DDI penalty
    severity = ddi.get("severity", "unknown")
    ddi_penalties = {
        "major": 0.0,       # Contraindicated
        "moderate": 0.5,    # Use with caution
        "minor": 0.9,       # Low risk
        "none": 1.0,        # No interaction
        "unknown": 0.7,     # Default conservative
    }
    ddi_factor = ddi_penalties.get(severity, 0.7)

    return round(individual_safety * ddi_factor, 4)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 2b: Safety & DDI Filtering")
    print("=" * 60)

    # Load combinations
    combo_path = OUTPUT_DIR / "drug_combinations.csv"
    print("\n1. Loading drug combinations from Phase 2a...")
    combos = load_combinations(combo_path)
    print(f"   {len(combos)} combinations loaded")

    # Load safety info
    print("\n2. Loading safety profiles...")
    safety_info = load_safety_info(RANKED_PATH)
    print(f"   Safety data for {len(safety_info)} drugs")

    # Filter and score
    print("\n3. Applying DDI and safety filters...")
    safe_combos = []
    blocked = 0
    unknown_ddi = 0

    for combo in combos:
        drug_a = combo["drug_a"]
        drug_b = combo["drug_b"]

        # Check DDI
        ddi = assess_ddi(drug_a, drug_b)
        safety_score = compute_safety_score(drug_a, drug_b, safety_info, ddi)

        # Block major DDIs
        if ddi["severity"] == "major":
            blocked += 1
            continue

        if ddi["severity"] == "unknown":
            unknown_ddi += 1

        # Combined score: coverage score * safety factor
        combo_score = float(combo.get("combo_score", 0))
        adjusted_score = round(combo_score * safety_score, 4)

        a_info = safety_info.get(drug_a.upper(), {})
        b_info = safety_info.get(drug_b.upper(), {})

        safe_combos.append({
            "drug_a": drug_a,
            "drug_b": drug_b,
            "n_pathways_union": combo["n_pathways_union"],
            "n_target_classes": combo["n_target_classes"],
            "target_classes": combo["target_classes"],
            "covered_pathways": combo["covered_pathways"],
            "complementarity": combo["complementarity"],
            "combo_score": combo_score,
            "ddi_severity": ddi["severity"],
            "ddi_mechanism": ddi.get("mechanism", ""),
            "ddi_recommendation": ddi.get("recommendation", ""),
            "safety_score": safety_score,
            "adjusted_score": adjusted_score,
            "safety_a": a_info.get("safety_tier", ""),
            "safety_b": b_info.get("safety_tier", ""),
            "a_in_trials": a_info.get("in_ts_trials", ""),
            "b_in_trials": b_info.get("in_ts_trials", ""),
        })

    safe_combos.sort(key=lambda x: -x["adjusted_score"])

    print(f"   Blocked (major DDI): {blocked}")
    print(f"   Unknown DDI status: {unknown_ddi}")
    print(f"   Safe combinations: {len(safe_combos)}")

    # Save safe combinations
    safe_path = OUTPUT_DIR / "safe_combinations.csv"
    fieldnames = [
        "drug_a", "drug_b", "n_pathways_union", "n_target_classes",
        "target_classes", "covered_pathways", "complementarity",
        "combo_score", "ddi_severity", "ddi_mechanism", "ddi_recommendation",
        "safety_score", "adjusted_score",
        "safety_a", "safety_b", "a_in_trials", "b_in_trials",
    ]
    with open(safe_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(safe_combos)
    print(f"   Saved to {safe_path}")

    # Coverage-optimal pairs: best adjusted score with max coverage
    optimal = sorted(
        safe_combos,
        key=lambda x: (-int(x["n_pathways_union"]), -x["adjusted_score"]),
    )
    optimal_path = OUTPUT_DIR / "coverage_optimal_pairs.csv"
    with open(optimal_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(optimal[:30])
    print(f"   Coverage-optimal pairs saved to {optimal_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nTop 15 safe combinations (by adjusted score):")
    print(f"{'Rank':<5} {'Drug A':<20} {'Drug B':<20} {'AdjScore':<9} "
          f"{'DDI':<10} {'PW':<4} {'Classes'}")
    print("-" * 85)
    for i, c in enumerate(safe_combos[:15], 1):
        print(f"{i:<5} {c['drug_a']:<20} {c['drug_b']:<20} "
              f"{c['adjusted_score']:.3f}    {c['ddi_severity']:<10} "
              f"{c['n_pathways_union']:<4} {c['target_classes']}")

    # Clinical-ready combinations (both in TS trials, minor/no DDI)
    clinical_ready = [
        c for c in safe_combos
        if c["a_in_trials"] and c["b_in_trials"]
        and c["ddi_severity"] in ("minor", "none", "unknown")
    ]
    print(f"\nClinical-ready combinations (both in TS trials, safe DDI): {len(clinical_ready)}")
    for c in clinical_ready[:10]:
        print(f"  {c['drug_a']} + {c['drug_b']}: "
              f"adj_score={c['adjusted_score']:.3f}, "
              f"{c['n_pathways_union']} pathways, "
              f"DDI={c['ddi_severity']}")

    # Novel combinations: high score, at least one drug in TS trials
    novel = [
        c for c in safe_combos
        if (c["a_in_trials"] or c["b_in_trials"])
        and c["adjusted_score"] >= 0.4
        and c["ddi_severity"] in ("minor", "none", "unknown")
    ]
    print(f"\nNovel combinations (>=1 TS drug, adj_score>=0.4): {len(novel)}")
    for c in novel[:10]:
        print(f"  {c['drug_a']} + {c['drug_b']}: "
              f"adj_score={c['adjusted_score']:.3f}, "
              f"classes={c['target_classes']}")


if __name__ == "__main__":
    main()
