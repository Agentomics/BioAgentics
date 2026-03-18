"""Phase 5: Clinical concordance analysis — DepMap SL ranking vs clinical data.

Cross-references DepMap PRMT5 SL rankings with available clinical trial data
for MTA-cooperative PRMT5 inhibitors. Identifies underexplored cancer types
with strong preclinical SL but no clinical data yet.

Usage:
    uv run python -m pancancer_mtap_prmt5_atlas.06_clinical_concordance
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "pancancer-mtap-prmt5-atlas"

# Strong SL threshold (Cohen's d)
STRONG_SL_THRESHOLD = -0.5

# Available clinical response data for MTA-cooperative PRMT5 inhibitors
# Source: Published trial results and conference presentations
CLINICAL_DATA = {
    "Lung": {
        "drugs": ["vopimetostat", "BMS-986504", "AMG 193"],
        "orr_pct": 29.0,  # BMS-986504 NSCLC ORR
        "notes": "BMS-986504: 29% ORR in NSCLC (Phase 1). Vopimetostat: included in histology-selective cohort (49% ORR mixed). AMG 193: included in pan-cancer Phase 1.",
    },
    "Pancreas": {
        "drugs": ["vopimetostat", "AMG 193"],
        "orr_pct": 25.0,  # Vopimetostat PDAC
        "notes": "Vopimetostat: 25% ORR in PDAC (Phase 1/2). Pivotal trial targets PDAC. AMG 193: PDAC included in pan-cancer Phase 1.",
    },
    "CNS/Brain": {
        "drugs": ["TNG456"],
        "orr_pct": None,  # Enrolling, no response data yet
        "notes": "TNG456: brain-penetrant PRMT5i, GBM enrolling (NCT06810544). No response data yet.",
    },
    "Bladder/Urinary Tract": {
        "drugs": ["AMG 193"],
        "orr_pct": None,
        "notes": "AMG 193: bladder cancer included in pan-cancer Phase 1. No tumor-specific ORR reported.",
    },
    "Esophagus/Stomach": {
        "drugs": ["AMG 193"],
        "orr_pct": None,
        "notes": "AMG 193: esophageal cancer included in pan-cancer Phase 1. No tumor-specific ORR reported.",
    },
    "Head and Neck": {
        "drugs": ["AMG 193"],
        "orr_pct": None,
        "notes": "AMG 193: head and neck included in pan-cancer Phase 1. No tumor-specific ORR reported.",
    },
}

# Pan-cancer reference ORRs
PAN_CANCER_ORRS = {
    "vopimetostat_pan": 27.0,
    "vopimetostat_selective": 49.0,
    "AMG_193_pan": 21.4,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading PRMT5 effect sizes...")
    prmt5 = pd.read_csv(OUTPUT_DIR / "prmt5_effect_sizes.csv")

    # Load population data if available
    pop_path = OUTPUT_DIR / "patient_population_estimates.csv"
    pop_df = pd.read_csv(pop_path) if pop_path.exists() else None

    # Classify each cancer type
    classifications = []
    for _, row in prmt5.iterrows():
        cancer_type = row["cancer_type"]
        d = row["cohens_d"]
        fdr = row["fdr"]
        strong_sl = d < STRONG_SL_THRESHOLD

        clinical = CLINICAL_DATA.get(cancer_type)
        has_clinical = clinical is not None

        if strong_sl and has_clinical:
            category = "clinical_validated"
        elif strong_sl and not has_clinical:
            category = "underexplored"
        else:
            category = "weak_sl"

        entry = {
            "cancer_type": cancer_type,
            "cohens_d": float(d),
            "fdr": float(fdr),
            "sl_rank": int(row["rank"]),
            "strong_sl": strong_sl,
            "has_clinical_data": has_clinical,
            "classification": category,
        }

        if has_clinical:
            entry["drugs"] = clinical["drugs"]
            entry["orr_pct"] = clinical["orr_pct"]
            entry["clinical_notes"] = clinical["notes"]

        # Add population data
        if pop_df is not None:
            pop_row = pop_df[pop_df["cancer_type"] == cancer_type]
            if not pop_row.empty:
                entry["eligible_patients_year"] = int(pop_row.iloc[0]["eligible_patients_year"])

        classifications.append(entry)

    # Save concordance JSON
    concordance = {
        "pan_cancer_reference_orrs": PAN_CANCER_ORRS,
        "strong_sl_threshold": STRONG_SL_THRESHOLD,
        "classifications": classifications,
        "summary": {
            "clinical_validated": sum(1 for c in classifications if c["classification"] == "clinical_validated"),
            "underexplored": sum(1 for c in classifications if c["classification"] == "underexplored"),
            "weak_sl": sum(1 for c in classifications if c["classification"] == "weak_sl"),
        },
    }
    with open(OUTPUT_DIR / "clinical_concordance.json", "w") as f:
        json.dump(concordance, f, indent=2)

    print("\nClassification summary:")
    for cat in ["clinical_validated", "underexplored", "weak_sl"]:
        entries = [c for c in classifications if c["classification"] == cat]
        print(f"\n  {cat} ({len(entries)}):")
        for e in entries:
            extra = ""
            if e.get("orr_pct"):
                extra = f", ORR={e['orr_pct']}%"
            elif e.get("drugs"):
                extra = f", drugs={e['drugs']}"
            pop = f", {e['eligible_patients_year']:,}/yr" if "eligible_patients_year" in e else ""
            print(f"    {e['cancer_type']}: d={e['cohens_d']:.2f}{extra}{pop}")

    # Underexplored cancer types CSV
    underexplored = [c for c in classifications if c["classification"] == "underexplored"]
    if underexplored:
        ue_df = pd.DataFrame(underexplored)
        cols = ["cancer_type", "cohens_d", "fdr", "sl_rank"]
        if "eligible_patients_year" in ue_df.columns:
            cols.append("eligible_patients_year")
        ue_df[cols].to_csv(OUTPUT_DIR / "underexplored_cancer_types.csv", index=False)
        print(f"\nSaved {len(ue_df)} underexplored cancer types")

    # Trial expansion recommendations
    recommendations = []
    for c in sorted(classifications,
                    key=lambda x: x.get("eligible_patients_year", 0), reverse=True):
        if c["classification"] == "underexplored":
            recommendations.append({
                "cancer_type": c["cancer_type"],
                "rationale": f"Strong PRMT5 SL (d={c['cohens_d']:.2f}, FDR={c['fdr']:.3f}) "
                             f"with no PRMT5i clinical data. "
                             f"Eligible patients: {c.get('eligible_patients_year', 'unknown')}/year.",
                "priority": "high" if c["fdr"] < 0.05 else "medium",
                "cohens_d": c["cohens_d"],
                "eligible_patients_year": c.get("eligible_patients_year"),
            })

    with open(OUTPUT_DIR / "trial_expansion_recommendations.json", "w") as f:
        json.dump({"recommendations": recommendations}, f, indent=2)
    print(f"Saved {len(recommendations)} trial expansion recommendations")

    # === Clinical concordance plot ===
    # Plot DepMap |d| vs clinical ORR for cancer types with both
    print("\nGenerating clinical concordance plot...")
    has_orr = [c for c in classifications if c.get("orr_pct") is not None]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot all cancer types by category
    for c in classifications:
        if c["classification"] == "clinical_validated":
            color, marker = "#D95319", "o"
        elif c["classification"] == "underexplored":
            color, marker = "#0072BD", "s"
        else:
            color, marker = "#999999", "^"

        # x = |Cohen's d|, y = ORR (or 0 if no ORR)
        x = abs(c["cohens_d"])
        y = c.get("orr_pct", 0) or 0
        size = c.get("eligible_patients_year", 500) / 200 + 20

        ax.scatter(x, y, c=color, marker=marker, s=size, alpha=0.7,
                   edgecolors="black", linewidths=0.5)
        offset = (5, 5) if y > 0 else (5, -10)
        ax.annotate(c["cancer_type"], (x, y), fontsize=7,
                    xytext=offset, textcoords="offset points")

    # Reference lines
    ax.axhline(y=PAN_CANCER_ORRS["vopimetostat_pan"], color="gray", linestyle="--",
               alpha=0.4, label=f"Vopimetostat pan-cancer ORR ({PAN_CANCER_ORRS['vopimetostat_pan']}%)")

    ax.set_xlabel("|Cohen's d| (DepMap PRMT5 SL strength)")
    ax.set_ylabel("Clinical ORR (%)")
    ax.set_title("DepMap PRMT5 SL vs Clinical Response\n"
                  "(0% ORR = no clinical data available)")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D95319',
               markersize=8, label='Clinical validated'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#0072BD',
               markersize=8, label='Underexplored (strong SL, no trials)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#999999',
               markersize=8, label='Weak SL'),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=-2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "clinical_concordance_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: clinical_concordance_plot.png")


if __name__ == "__main__":
    main()
