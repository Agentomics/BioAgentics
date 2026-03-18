"""Phase 4: TCGA allele frequency integration and patient population estimates.

Cross-validates DepMap cell line allele distributions against TCGA patient
populations. Estimates annual US patients per PIK3CA allele per cancer type
to prioritize allele-specific therapeutic strategies.

Usage:
    uv run python -m pik3ca_allele_dependencies.05_tcga_integration
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

OUTPUT_DIR = REPO_ROOT / "output" / "pik3ca_allele_dependencies"
FIG_DIR = OUTPUT_DIR / "figures"
TCGA_PATH = REPO_ROOT / "data" / "tcga" / "pancancer_pik3ca" / "pik3ca_mutation_frequencies.csv"

# Approximate annual US cancer incidence by TCGA cancer type (SEER 2023 estimates)
# Sources: cancer.org, SEER, NCI
US_ANNUAL_INCIDENCE = {
    "BRCA": 310_720,
    "UCEC": 67_880,
    "COADREAD": 152_810,
    "CESC": 13_820,
    "BLCA": 83_190,
    "HNSC": 66_920,
    "LUSC": 70_000,  # ~30% of NSCLC
    "LUAD": 117_000,  # ~50% of NSCLC
    "STAD": 26_500,
    "OV": 19_710,
    "LIHC": 41_210,
    "PRAD": 288_300,
    "GBM": 13_410,  # GBM subset
    "LGG": 5_000,
    "SKCM": 97_610,
    "ESCA": 22_370,
    "KIRC": 51_020,
    "THCA": 43_720,
    "PAAD": 64_050,
}

# Map TCGA cancer types to DepMap OncotreePrimaryDisease for cross-validation
TCGA_TO_DEPMAP = {
    "BRCA": "Invasive Breast Carcinoma",
    "UCEC": "Endometrial Carcinoma",
    "COADREAD": "Colorectal Adenocarcinoma",
    "BLCA": "Bladder Urothelial Carcinoma",
    "OV": "Ovarian Epithelial Tumor",
    "HNSC": "Head and Neck Squamous Cell Carcinoma",
    "LUAD": "Non-Small Cell Lung Cancer",
    "LUSC": "Non-Small Cell Lung Cancer",
    "GBM": "Diffuse Glioma",
    "LGG": "Diffuse Glioma",
    "STAD": "Esophagogastric Adenocarcinoma",
    "ESCA": "Esophagogastric Adenocarcinoma",
}

ALLELE_COLS = ["H1047R_L", "E545K", "E542K", "C420R", "N345K"]


def load_tcga_frequencies(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_depmap_allele_distribution(output_dir: Path) -> pd.DataFrame:
    """Load DepMap classified lines and compute allele distributions per cancer type."""
    classified = pd.read_csv(output_dir / "pik3ca_classified_lines.csv", index_col=0)
    classified["PIK3CA_mutated"] = classified["PIK3CA_mutated"].astype(bool)
    mutant = classified[classified["PIK3CA_mutated"]]

    rows = []
    for ct, group in mutant.groupby("OncotreePrimaryDisease"):
        allele_counts = group["PIK3CA_allele"].value_counts()
        n_total = len(group)
        row = {
            "depmap_cancer_type": ct,
            "depmap_n_mutant": n_total,
            "depmap_H1047R_L": int(allele_counts.get("H1047R", 0) + allele_counts.get("H1047L", 0)),
            "depmap_E545K": int(allele_counts.get("E545K", 0)),
            "depmap_E542K": int(allele_counts.get("E542K", 0)),
            "depmap_C420R": int(allele_counts.get("C420R", 0)),
            "depmap_N345K": int(allele_counts.get("N345K", 0)),
        }
        # Fractions
        for col in ["H1047R_L", "E545K", "E542K", "C420R", "N345K"]:
            row[f"depmap_{col}_frac"] = row[f"depmap_{col}"] / n_total if n_total > 0 else 0
        rows.append(row)

    return pd.DataFrame(rows)


def cross_validate(tcga: pd.DataFrame, depmap: pd.DataFrame) -> pd.DataFrame:
    """Compare allele distributions between TCGA and DepMap per cancer type."""
    comparison_rows = []

    for _, tcga_row in tcga.iterrows():
        ct = tcga_row["cancer_type"]
        depmap_ct = TCGA_TO_DEPMAP.get(ct)
        if depmap_ct is None:
            continue

        depmap_match = depmap[depmap["depmap_cancer_type"] == depmap_ct]
        if depmap_match.empty:
            continue

        dm = depmap_match.iloc[0]
        n_tcga = tcga_row["PIK3CA_mutated"]
        n_depmap = dm["depmap_n_mutant"]

        row = {
            "tcga_type": ct,
            "depmap_type": depmap_ct,
            "tcga_n_mutant": int(n_tcga),
            "depmap_n_mutant": int(n_depmap),
        }

        for allele in ALLELE_COLS:
            tcga_frac = tcga_row[allele] / n_tcga if n_tcga > 0 else 0
            depmap_frac = dm[f"depmap_{allele}_frac"]
            row[f"tcga_{allele}_frac"] = round(tcga_frac, 3)
            row[f"depmap_{allele}_frac"] = round(depmap_frac, 3)
            row[f"diff_{allele}"] = round(depmap_frac - tcga_frac, 3)

        comparison_rows.append(row)

    return pd.DataFrame(comparison_rows)


def estimate_patient_populations(tcga: pd.DataFrame) -> pd.DataFrame:
    """Estimate annual US patients per allele per cancer type."""
    rows = []
    for _, tcga_row in tcga.iterrows():
        ct = tcga_row["cancer_type"]
        incidence = US_ANNUAL_INCIDENCE.get(ct, 0)
        if incidence == 0:
            continue

        pik3ca_rate = tcga_row["PIK3CA_pct"] / 100.0
        n_mutated = tcga_row["PIK3CA_mutated"]

        for allele in ALLELE_COLS:
            allele_count = tcga_row[allele]
            allele_frac = allele_count / n_mutated if n_mutated > 0 else 0
            est_patients = int(incidence * pik3ca_rate * allele_frac)

            rows.append({
                "cancer_type": ct,
                "allele": allele,
                "us_annual_incidence": incidence,
                "pik3ca_mutation_rate": round(pik3ca_rate, 3),
                "allele_fraction": round(allele_frac, 3),
                "estimated_annual_patients": est_patients,
            })

    df = pd.DataFrame(rows)
    return df.sort_values("estimated_annual_patients", ascending=False)


def plot_allele_comparison(comparison: pd.DataFrame, out_path: Path) -> None:
    """Side-by-side bar chart: TCGA vs DepMap allele fractions."""
    alleles = ["H1047R_L", "E545K", "E542K"]
    cancer_types = comparison["tcga_type"].tolist()

    if len(cancer_types) == 0:
        return

    fig, axes = plt.subplots(1, len(alleles), figsize=(5 * len(alleles), max(4, len(cancer_types) * 0.5)))
    if len(alleles) == 1:
        axes = [axes]

    for ax, allele in zip(axes, alleles):
        y_pos = np.arange(len(cancer_types))
        width = 0.35

        tcga_vals = comparison[f"tcga_{allele}_frac"].values
        depmap_vals = comparison[f"depmap_{allele}_frac"].values

        ax.barh(y_pos - width / 2, tcga_vals, width, label="TCGA", color="#0072BD", alpha=0.7)
        ax.barh(y_pos + width / 2, depmap_vals, width, label="DepMap", color="#D95319", alpha=0.7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(cancer_types, fontsize=8)
        ax.set_xlabel("Fraction of PIK3CA-mutant")
        ax.set_title(allele.replace("_", "/"))
        ax.legend(fontsize=8)

    fig.suptitle("PIK3CA Allele Distribution: TCGA vs DepMap", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_patient_estimates(estimates: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar chart of estimated patients per cancer type by allele."""
    alleles = ["H1047R_L", "E545K", "E542K", "C420R", "N345K"]
    colors = {"H1047R_L": "#D95319", "E545K": "#0072BD", "E542K": "#4DBEEE",
              "C420R": "#77AC30", "N345K": "#EDB120"}

    # Pivot to cancer_type x allele
    pivot = estimates.pivot_table(
        index="cancer_type", columns="allele", values="estimated_annual_patients", fill_value=0
    )
    # Keep only alleles present and sort by total
    pivot = pivot[[a for a in alleles if a in pivot.columns]]
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.sort_values("total", ascending=True).drop(columns="total")
    pivot = pivot.tail(12)  # top 12 cancer types

    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(pivot))
    for allele in pivot.columns:
        vals = pivot[allele].values
        ax.barh(range(len(pivot)), vals, left=bottom, label=allele.replace("_", "/"),
                color=colors.get(allele, "gray"), alpha=0.8)
        bottom += vals

    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Estimated Annual US Patients")
    ax.set_title("PIK3CA-Mutant Patient Population by Allele")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TCGA PIK3CA frequencies...")
    tcga = load_tcga_frequencies(TCGA_PATH)
    print(f"  {len(tcga)} cancer types, {tcga['n_sequenced'].sum()} patients")

    print("Loading DepMap allele distributions...")
    depmap = load_depmap_allele_distribution(OUTPUT_DIR)
    print(f"  {len(depmap)} cancer types with PIK3CA-mutant lines")

    # Cross-validation
    print("\nCross-validating TCGA vs DepMap allele distributions...")
    comparison = cross_validate(tcga, depmap)
    out_comp = OUTPUT_DIR / "tcga_depmap_comparison.csv"
    comparison.to_csv(out_comp, index=False)
    print(f"  {len(comparison)} cancer types compared")

    # Show major discrepancies
    for _, row in comparison.iterrows():
        flags = []
        for allele in ALLELE_COLS[:3]:
            diff = abs(row[f"diff_{allele}"])
            if diff > 0.15:
                direction = "over" if row[f"diff_{allele}"] > 0 else "under"
                flags.append(f"{allele} {direction}-represented ({diff:.0%})")
        if flags:
            print(f"  {row['tcga_type']}: {', '.join(flags)}")

    # Patient population estimates
    print("\nEstimating patient populations...")
    estimates = estimate_patient_populations(tcga)
    out_est = OUTPUT_DIR / "patient_population_estimates.json"

    # Summary by allele
    allele_totals = estimates.groupby("allele")["estimated_annual_patients"].sum()
    total_pik3ca = allele_totals.sum()
    print(f"\n  Estimated annual US PIK3CA-mutant patients: ~{total_pik3ca:,}")
    for allele, count in allele_totals.sort_values(ascending=False).items():
        print(f"    {allele}: ~{count:,} ({count / total_pik3ca:.1%})")

    # Top cancer types by patient volume
    ct_totals = estimates.groupby("cancer_type")["estimated_annual_patients"].sum()
    print(f"\n  Top cancer types by PIK3CA-mutant patient volume:")
    for ct, count in ct_totals.sort_values(ascending=False).head(8).items():
        print(f"    {ct}: ~{count:,}")

    # Save estimates
    with open(out_est, "w") as f:
        json.dump({
            "total_estimated_pik3ca_mutant_patients": int(total_pik3ca),
            "allele_totals": {str(k): int(v) for k, v in allele_totals.items()},
            "cancer_type_totals": {str(k): int(v) for k, v in ct_totals.sort_values(ascending=False).items()},
            "per_cancer_per_allele": estimates.to_dict(orient="records"),
        }, f, indent=2)
    print(f"\nSaved estimates to {out_est.name}")

    # Plots
    print("\nGenerating plots...")
    plot_allele_comparison(comparison, FIG_DIR / "tcga_vs_depmap_alleles.png")
    plot_patient_estimates(estimates, FIG_DIR / "patient_population_by_allele.png")

    print("\nSaved comparison to", out_comp.name)


if __name__ == "__main__":
    main()
