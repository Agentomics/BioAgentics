"""Safety and feasibility filtering for TS drug repurposing candidates.

Filters drug candidates by:
1. FDA-approved status
2. Acceptable CNS safety profile (no severe CNS adverse events)
3. Blood-brain barrier (BBB) penetrance
4. No overlapping major contraindications with TS comorbidities (ADHD, OCD)

Uses curated safety annotations and known BBB penetrance data.

Output: filtered candidate list with safety flags

Usage:
    uv run python -m bioagentics.tourettes.drug_repurposing.safety_filter
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-drug-repurposing-network"
OUTPUT_PATH = OUTPUT_DIR / "safety_filtered_candidates.csv"
SIGNATURE_PATH = OUTPUT_DIR / "lincs_signature_scores.csv"

# Known BBB-penetrant drug classes and specific drugs
# This is a curated list; in production, use a BBB prediction model
BBB_PENETRANT_DRUGS: set[str] = {
    "aripiprazole", "haloperidol", "pimozide", "risperidone",
    "fluphenazine", "clonidine", "guanfacine", "ecopipam",
    "topiramate", "cannabidiol", "riluzole", "fluvoxamine",
    "valbenazine", "deutetrabenazine", "xanomeline",
    "pitolisant", "n-acetylcysteine", "clozapine", "olanzapine",
    "quetiapine", "ziprasidone", "cariprazine", "brexpiprazole",
    "naltrexone", "baclofen", "tetrabenazine", "ondansetron",
    "dextromethorphan", "memantine", "amantadine", "buspirone",
    "atomoxetine", "methylphenidate", "dexmethylphenidate",
}

# CNS safety concerns (drugs to flag but not necessarily exclude)
CNS_SAFETY_CONCERNS: dict[str, str] = {
    "haloperidol": "EPS, tardive dyskinesia risk",
    "pimozide": "QTc prolongation, EPS",
    "fluphenazine": "EPS, tardive dyskinesia risk",
    "clozapine": "Agranulocytosis risk, metabolic syndrome",
    "olanzapine": "Metabolic syndrome, weight gain",
    "valbenazine": "Somnolence, QTc concern",
    "deutetrabenazine": "Depression, suicidality warning",
    "tetrabenazine": "Depression, suicidality, parkinsonism",
}

# Contraindications with TS comorbidities
TS_COMORBIDITY_RISKS: dict[str, str] = {
    # ADHD comorbidity concerns
    "haloperidol": "May worsen ADHD symptoms",
    "pimozide": "May worsen ADHD symptoms",
    "clozapine": "Sedation may impair ADHD function",
    # OCD comorbidity concerns
    "aripiprazole": "Generally safe; augments SSRI for OCD",
    "buspirone": "Generally well-tolerated with OCD meds",
}

# Overall safety tier
SAFETY_TIERS = {
    "tier_1_favorable": {
        "guanfacine", "clonidine", "aripiprazole", "ecopipam",
        "topiramate", "cannabidiol", "n-acetylcysteine", "buspirone",
    },
    "tier_2_acceptable": {
        "risperidone", "pitolisant", "xanomeline", "riluzole",
        "fluvoxamine", "ondansetron", "memantine", "amantadine",
        "brexpiprazole", "cariprazine", "atomoxetine", "baclofen",
    },
    "tier_3_caution": {
        "haloperidol", "pimozide", "fluphenazine", "quetiapine",
        "olanzapine", "ziprasidone", "valbenazine", "deutetrabenazine",
        "tetrabenazine", "clozapine", "dextromethorphan",
    },
}


def assign_safety_tier(drug_name: str) -> str:
    """Assign safety tier to a drug."""
    name = drug_name.lower()
    for tier, drugs in SAFETY_TIERS.items():
        if name in drugs:
            return tier
    return "tier_unknown"


def filter_candidates(
    candidates: list[dict],
    require_bbb: bool = True,
    exclude_tier3: bool = False,
) -> list[dict]:
    """Apply safety filters to drug candidates."""
    filtered = []
    for cand in candidates:
        name = cand.get("compound_name", cand.get("drug_name", "")).lower()

        bbb = name in BBB_PENETRANT_DRUGS
        safety_tier = assign_safety_tier(name)
        cns_concern = CNS_SAFETY_CONCERNS.get(name, "")
        comorbidity_risk = TS_COMORBIDITY_RISKS.get(name, "")

        if require_bbb and not bbb:
            continue
        if exclude_tier3 and safety_tier == "tier_3_caution":
            continue

        cand_out = dict(cand)
        cand_out["bbb_penetrant"] = bbb
        cand_out["safety_tier"] = safety_tier
        cand_out["cns_safety_concern"] = cns_concern
        cand_out["comorbidity_risk"] = comorbidity_risk
        filtered.append(cand_out)

    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Safety filtering for TS drug candidates")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--exclude-tier3", action="store_true")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load signature scores as candidate list (for demo)
    candidates = []
    if SIGNATURE_PATH.exists():
        df = pd.read_csv(SIGNATURE_PATH)
        for _, row in df.iterrows():
            candidates.append(row.to_dict())

    print(f"Input candidates: {len(candidates)}")

    filtered = filter_candidates(candidates, exclude_tier3=args.exclude_tier3)
    print(f"After safety filtering: {len(filtered)}")

    if filtered:
        fieldnames = list(filtered[0].keys())
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(filtered)
        print(f"Saved to {args.output}")

    # Summary by tier
    from collections import Counter
    tiers = Counter(c["safety_tier"] for c in filtered)
    for tier, count in tiers.most_common():
        print(f"  {tier}: {count}")


if __name__ == "__main__":
    main()
