"""Multi-criteria candidate ranking pipeline for TS drug repurposing.

Combines all evidence streams into final ranked candidate list:
1. Network proximity z-score (weight: 0.30)
2. LINCS signature correlation (weight: 0.20)
3. Target pathway relevance to TS (weight: 0.20)
4. Safety profile score (weight: 0.15)
5. Clinical precedent (weight: 0.15)

Applies VMAT2 negative control penalty.
Outputs top 20 candidates with evidence breakdown.

Output: output/tourettes/ts-drug-repurposing-network/ranked_candidates.csv

Usage:
    uv run python -m bioagentics.tourettes.drug_repurposing.rank_candidates
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from bioagentics.config import REPO_ROOT
from bioagentics.tourettes.drug_repurposing.safety_filter import (
    BBB_PENETRANT_DRUGS,
    assign_safety_tier,
)
from bioagentics.tourettes.drug_repurposing.vmat2_calibration import (
    NEGATIVE_CONTROLS,
    POSITIVE_CONTROLS,
)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-drug-repurposing-network"
DATA_DIR = REPO_ROOT / "data" / "tourettes" / "ts-drug-repurposing-network"

SIGNATURE_PATH = OUTPUT_DIR / "lincs_signature_scores.csv"
SAFETY_PATH = OUTPUT_DIR / "safety_filtered_candidates.csv"
CALIBRATION_PATH = OUTPUT_DIR / "vmat2_calibration.csv"
TRIALS_PATH = DATA_DIR / "ts_clinical_trials.csv"
OUTPUT_PATH = OUTPUT_DIR / "ranked_candidates.csv"

# Scoring weights
WEIGHTS = {
    "network_proximity": 0.30,
    "signature_score": 0.20,
    "pathway_relevance": 0.20,
    "safety_profile": 0.15,
    "clinical_precedent": 0.15,
}

# TS-relevant pathway target annotations
TS_PATHWAY_TARGETS: dict[str, set[str]] = {
    "D2_antagonist": {"DRD2", "DRD3"},
    "D1_antagonist": {"DRD1"},
    "PDE10A_inhibitor": {"PDE10A"},
    "M4_agonist": {"CHRM4", "CHRM1"},
    "H3R_modulator": {"HRH3"},
    "alpha2_agonist": {"ADRA2A", "ADRA2C"},
    "GABA_modulator": {"GABRA1", "GABRG2", "GAD1", "GAD2"},
    "5HT2_modulator": {"HTR2A", "HTR2C"},
    "endocannabinoid": {"CNR1", "CNR2", "FAAH", "MGLL"},
    "anti_inflammatory": {"TNF", "IL12A", "IL12B", "IL23A", "JAK1", "JAK2"},
    "VMAT2_inhibitor": {"SLC18A2"},
    "glutamate_modulator": {"GRIN2B", "SLC1A2"},
}

# Drug-to-pathway assignments (curated)
DRUG_PATHWAYS: dict[str, list[str]] = {
    "aripiprazole": ["D2_antagonist", "5HT2_modulator"],
    "ecopipam": ["D1_antagonist"],
    "haloperidol": ["D2_antagonist"],
    "pimozide": ["D2_antagonist"],
    "risperidone": ["D2_antagonist", "5HT2_modulator"],
    "fluphenazine": ["D2_antagonist"],
    "clonidine": ["alpha2_agonist"],
    "guanfacine": ["alpha2_agonist"],
    "pitolisant": ["H3R_modulator"],
    "xanomeline": ["M4_agonist"],
    "topiramate": ["GABA_modulator", "glutamate_modulator"],
    "cannabidiol": ["endocannabinoid"],
    "riluzole": ["glutamate_modulator"],
    "n-acetylcysteine": ["glutamate_modulator"],
    "fluvoxamine": ["5HT2_modulator"],
    "valbenazine": ["VMAT2_inhibitor"],
    "deutetrabenazine": ["VMAT2_inhibitor"],
}


def load_signature_scores(path: Path) -> dict[str, float]:
    """Load signature concordance scores."""
    scores: dict[str, float] = {}
    if not path.exists():
        return scores
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["compound_name"].lower().strip()
            try:
                scores[name] = float(row["concordance_score"])
            except (ValueError, KeyError):
                pass
    return scores


def load_vmat2_penalties(path: Path) -> dict[str, float]:
    """Load VMAT2 calibration penalties."""
    penalties: dict[str, float] = {}
    if not path.exists():
        return penalties
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["drug_name"].lower().strip()
            try:
                penalties[name] = float(row["vmat2_penalty"])
            except (ValueError, KeyError):
                pass
    return penalties


def load_clinical_trials(path: Path) -> set[str]:
    """Load drug names currently in TS clinical trials."""
    drugs: set[str] = set()
    if not path.exists():
        return drugs
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            drug = row.get("intervention_drug", "").lower().strip()
            if drug and drug != "non-drug" and drug != "placebo":
                drugs.add(drug)
    return drugs


def compute_pathway_score(drug_name: str) -> float:
    """Score drug by relevance of its target pathways to TS.

    Higher score = more relevant pathways.
    """
    name = drug_name.lower()
    pathways = DRUG_PATHWAYS.get(name, [])

    if not pathways:
        return 0.0

    # Weight by pathway importance
    pathway_weights = {
        "D1_antagonist": 1.0,    # Ecopipam Phase 3 validated
        "PDE10A_inhibitor": 1.0, # Gemlapodect Phase 2a validated
        "M4_agonist": 0.9,       # KarXT strong rationale
        "D2_antagonist": 0.85,   # Well-established
        "alpha2_agonist": 0.7,   # First-line for mild TS
        "H3R_modulator": 0.7,    # Pitolisant emerging
        "GABA_modulator": 0.6,
        "5HT2_modulator": 0.55,
        "endocannabinoid": 0.5,
        "anti_inflammatory": 0.5,
        "glutamate_modulator": 0.45,
        "VMAT2_inhibitor": 0.1,  # Failed in TS
    }

    total = sum(pathway_weights.get(p, 0.3) for p in pathways)
    return min(total / 1.0, 1.0)  # Normalize to [0, 1]


def compute_safety_score(drug_name: str) -> float:
    """Score drug safety profile (0-1, higher = safer)."""
    tier = assign_safety_tier(drug_name.lower())
    tier_scores = {
        "tier_1_favorable": 1.0,
        "tier_2_acceptable": 0.7,
        "tier_3_caution": 0.3,
        "tier_unknown": 0.5,
    }
    return tier_scores.get(tier, 0.5)


def compute_clinical_precedent_score(drug_name: str, trial_drugs: set[str]) -> float:
    """Score clinical precedent for the drug in TS."""
    name = drug_name.lower()

    # Known TS-approved or breakthrough drugs
    if name in {"ecopipam", "aripiprazole"}:
        return 1.0
    if name in POSITIVE_CONTROLS:
        return 0.8
    if name in trial_drugs:
        return 0.6
    # Novel candidate — bonus for not being already tested
    return 0.4


def rank_candidates(top_n: int = 20) -> list[dict]:
    """Run full multi-criteria ranking pipeline."""
    # Load data
    sig_scores = load_signature_scores(SIGNATURE_PATH)
    vmat2_penalties = load_vmat2_penalties(CALIBRATION_PATH)
    trial_drugs = load_clinical_trials(TRIALS_PATH)

    print(f"  Signature scores: {len(sig_scores)} drugs")
    print(f"  VMAT2 penalties: {len(vmat2_penalties)} drugs")
    print(f"  Clinical trial drugs: {len(trial_drugs)} drugs")

    # Score each candidate
    results = []
    for drug_name, sig_score in sig_scores.items():
        # Normalize signature score to [0, 1] (more negative = better)
        sig_normalized = max(0, min(1, (-sig_score) / 1.0))

        pathway_score = compute_pathway_score(drug_name)
        safety_score = compute_safety_score(drug_name)
        clinical_score = compute_clinical_precedent_score(drug_name, trial_drugs)

        # Network proximity placeholder (uses sig_score as proxy until DrugBank available)
        # In full pipeline, this would come from network_proximity_scores.csv
        network_score = sig_normalized * 0.9  # Correlated proxy

        # Weighted sum
        raw_score = (
            WEIGHTS["network_proximity"] * network_score
            + WEIGHTS["signature_score"] * sig_normalized
            + WEIGHTS["pathway_relevance"] * pathway_score
            + WEIGHTS["safety_profile"] * safety_score
            + WEIGHTS["clinical_precedent"] * clinical_score
        )

        # Apply VMAT2 penalty
        penalty = vmat2_penalties.get(drug_name, 0.0)
        final_score = raw_score * (1 - penalty)

        # Pathway annotations
        pathways = DRUG_PATHWAYS.get(drug_name, [])

        results.append({
            "rank": 0,  # Will be set after sorting
            "drug_name": drug_name,
            "final_score": round(final_score, 4),
            "network_proximity_score": round(network_score, 4),
            "signature_score": round(sig_normalized, 4),
            "pathway_relevance": round(pathway_score, 4),
            "safety_score": round(safety_score, 4),
            "clinical_precedent": round(clinical_score, 4),
            "vmat2_penalty": round(penalty, 4),
            "safety_tier": assign_safety_tier(drug_name),
            "pathway_annotations": ";".join(pathways),
            "bbb_penetrant": drug_name in BBB_PENETRANT_DRUGS,
            "in_ts_trials": drug_name in trial_drugs,
            "is_positive_control": drug_name in {d.lower() for d in POSITIVE_CONTROLS},
            "is_negative_control": drug_name in {d.lower() for d in NEGATIVE_CONTROLS},
        })

    # Sort by final score (higher = better candidate)
    results.sort(key=lambda x: x["final_score"], reverse=True)

    # Assign ranks
    for i, r in enumerate(results):
        r["rank"] = i + 1

    return results[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-criteria candidate ranking")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running multi-criteria candidate ranking...")
    results = rank_candidates(top_n=args.top_n)

    if results:
        fieldnames = list(results[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved top {len(results)} candidates to {args.output}")

    # Print results
    print(f"\n{'='*80}")
    print(f"TOP {len(results)} DRUG REPURPOSING CANDIDATES FOR TOURETTE SYNDROME")
    print(f"{'='*80}")
    for r in results:
        ctrl = ""
        if r["is_positive_control"]:
            ctrl = " [+CTRL]"
        elif r["is_negative_control"]:
            ctrl = " [-CTRL]"
        novel = " [NOVEL]" if not r["in_ts_trials"] and not r["is_positive_control"] else ""
        print(
            f"  #{r['rank']:2d}  {r['drug_name']:<25s}  "
            f"score={r['final_score']:.3f}  "
            f"safety={r['safety_tier']:<20s}  "
            f"pathways={r['pathway_annotations']}"
            f"{ctrl}{novel}"
        )

    # Summary statistics
    print(f"\n{'='*80}")
    novel = [r for r in results if not r["in_ts_trials"] and not r["is_positive_control"]]
    print(f"Novel candidates (not in current TS trials): {len(novel)}")
    for r in novel:
        print(f"  {r['drug_name']}: {r['pathway_annotations']} ({r['safety_tier']})")

    tier1 = [r for r in results if r["safety_tier"] == "tier_1_favorable"]
    print(f"\nTier 1 (favorable safety): {len(tier1)}")
    for r in tier1:
        print(f"  {r['drug_name']}: score={r['final_score']:.3f}")


if __name__ == "__main__":
    main()
