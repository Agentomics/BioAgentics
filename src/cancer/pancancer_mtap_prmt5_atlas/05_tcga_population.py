"""Phase 4: TCGA patient population estimates with C-CAT cross-validation.

Loads pre-computed TCGA pan-cancer MTAP deletion frequencies, maps to DepMap
cancer types, estimates PRMT5i-eligible patient populations, and creates a
priority matrix combining SL strength with population size.

Usage:
    uv run python -m pancancer_mtap_prmt5_atlas.05_tcga_population
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "pancancer-mtap-prmt5-atlas"
TCGA_DATA = REPO_ROOT / "data" / "tcga" / "pancancer_mtap_cn" / "mtap_cn_deletion_frequencies.csv"

# TCGA abbreviation -> DepMap OncotreeLineage mapping
# Only cancer types that map to qualifying DepMap lineages
TCGA_TO_DEPMAP = {
    "GBM": "CNS/Brain",
    "LGG": "CNS/Brain",
    "MESO": "Pleura",
    "PAAD": "Pancreas",
    "SKCM": "Skin",
    "BLCA": "Bladder/Urinary Tract",
    "LAML": "Myeloid",
    "LUAD": "Lung",
    "LUSC": "Lung",
    "ESCA": "Esophagus/Stomach",
    "STAD": "Esophagus/Stomach",
    "SARC": "Soft Tissue",
    "HNSC": "Head and Neck",
    "BRCA": "Breast",
    "OV": "Ovary/Fallopian Tube",
    "LIHC": "Liver",
    "KIRC": "Kidney",
    "KIRP": "Kidney",
    "KICH": "Kidney",
    "DLBC": "Lymphoid",
    "COADREAD": "Bowel",
    "CHOL": "Biliary Tract",
}

# US annual cancer incidence (SEER/ACS 2024 estimates)
US_INCIDENCE = {
    "CNS/Brain": 25400,
    "Pleura": 2500,     # Mesothelioma
    "Pancreas": 66440,
    "Skin": 100640,     # Melanoma
    "Bladder/Urinary Tract": 83190,
    "Myeloid": 23810,   # AML
    "Lung": 238340,
    "Esophagus/Stomach": 47770,  # Combined esophageal + stomach
    "Soft Tissue": 13590,
    "Head and Neck": 55970,
    "Breast": 310720,
    "Ovary/Fallopian Tube": 19680,
    "Liver": 41630,
    "Kidney": 82290,
    "Lymphoid": 20160,  # DLBCL
    "Bowel": 153020,
    "Biliary Tract": 12260,
    "Bone": 3970,
}

# C-CAT cross-validation (Suzuki et al., ESMO Open 2025, 51,828 patients)
CCAT_FREQUENCIES = {
    "Pancreas": 18.4,
    "Biliary Tract": 15.6,
    "Lung": 14.3,
    "Esophagus/Stomach": 9.0,  # Approximate from C-CAT esophageal/gastric
    "Bladder/Urinary Tract": 11.0,
    "Head and Neck": 10.0,
    "Breast": 3.0,
    "Ovary/Fallopian Tube": 3.0,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading TCGA pan-cancer MTAP deletion frequencies...")
    tcga = pd.read_csv(TCGA_DATA)
    print(f"  {len(tcga)} TCGA cancer types")

    # Save raw TCGA frequencies
    tcga.to_csv(OUTPUT_DIR / "tcga_mtap_frequencies.csv", index=False)

    # Map to DepMap lineage and aggregate
    tcga["depmap_lineage"] = tcga["cancer_type"].map(TCGA_TO_DEPMAP)
    mapped = tcga[tcga["depmap_lineage"].notna()].copy()

    # Aggregate by DepMap lineage (weighted average for multi-TCGA-type lineages)
    lineage_stats = []
    for lineage, group in mapped.groupby("depmap_lineage"):
        n_total = group["n_profiled"].sum()
        n_homdel = group["MTAP_homdel"].sum()
        homdel_pct = n_homdel / n_total * 100 if n_total > 0 else 0

        n_hetloss = group["MTAP_hetloss"].sum()
        any_loss = (n_homdel + n_hetloss) / n_total * 100 if n_total > 0 else 0

        incidence = US_INCIDENCE.get(lineage, 0)

        # Use C-CAT frequency for population estimates when TCGA is discrepant
        # (>5pp difference). C-CAT (N=51,828) is more reliable for prevalence.
        pop_freq = homdel_pct
        pop_freq_source = "TCGA"
        if lineage in CCAT_FREQUENCIES:
            ccat_pct = CCAT_FREQUENCIES[lineage]
            if abs(homdel_pct - ccat_pct) > 5:
                pop_freq = ccat_pct
                pop_freq_source = "C-CAT"

        eligible = int(incidence * pop_freq / 100)

        lineage_stats.append({
            "cancer_type": lineage,
            "tcga_types": ", ".join(group["cancer_type"].tolist()),
            "n_profiled": int(n_total),
            "n_homdel": int(n_homdel),
            "homdel_pct": round(homdel_pct, 1),
            "pop_freq_pct": round(pop_freq, 1),
            "pop_freq_source": pop_freq_source,
            "any_loss_pct": round(any_loss, 1),
            "us_incidence": incidence,
            "eligible_patients_year": eligible,
        })

    pop_df = pd.DataFrame(lineage_stats)
    pop_df = pop_df.sort_values("eligible_patients_year", ascending=False).reset_index(drop=True)
    pop_df.to_csv(OUTPUT_DIR / "patient_population_estimates.csv", index=False)
    print(f"\nPatient population estimates ({len(pop_df)} cancer types):")
    for _, row in pop_df.iterrows():
        src = f" [via {row['pop_freq_source']}]" if row.get("pop_freq_source") == "C-CAT" else ""
        print(f"  {row['cancer_type']}: {row['homdel_pct']:.1f}% homdel, "
              f"{row['pop_freq_pct']:.1f}% pop freq{src}, "
              f"{row['eligible_patients_year']:,} eligible/year")

    # === C-CAT cross-validation ===
    print("\nC-CAT cross-validation (Suzuki et al., ESMO Open 2025):")
    ccat_rows = []
    for lineage, ccat_pct in CCAT_FREQUENCIES.items():
        tcga_row = pop_df[pop_df["cancer_type"] == lineage]
        if tcga_row.empty:
            continue
        tcga_pct = tcga_row.iloc[0]["homdel_pct"]
        diff = abs(tcga_pct - ccat_pct)
        ccat_rows.append({
            "cancer_type": lineage,
            "tcga_homdel_pct": tcga_pct,
            "ccat_homdel_pct": ccat_pct,
            "difference_pct": round(diff, 1),
            "discrepant": diff > 5,
        })
        flag = " ***DISCREPANT***" if diff > 5 else ""
        print(f"  {lineage}: TCGA={tcga_pct:.1f}%, C-CAT={ccat_pct:.1f}%, diff={diff:.1f}%{flag}")

    ccat_df = pd.DataFrame(ccat_rows)
    # Spearman correlation
    if len(ccat_df) >= 3:
        r, p = stats.spearmanr(ccat_df["tcga_homdel_pct"], ccat_df["ccat_homdel_pct"])
        print(f"  Spearman correlation: r={r:.2f}, p={p:.3f}")
    else:
        r, p = float("nan"), float("nan")

    ccat_result = {
        "spearman_r": float(r),
        "spearman_p": float(p),
        "comparisons": ccat_df.to_dict(orient="records"),
    }
    with open(OUTPUT_DIR / "tcga_vs_ccat_validation.json", "w") as f:
        json.dump(ccat_result, f, indent=2)
    print(f"Saved tcga_vs_ccat_validation.json")

    # === Combined priority score with PRMT5 SL data ===
    prmt5_path = OUTPUT_DIR / "prmt5_effect_sizes.csv"
    if prmt5_path.exists():
        prmt5 = pd.read_csv(prmt5_path)
        merged = pop_df.merge(
            prmt5[["cancer_type", "cohens_d", "fdr", "rank"]],
            on="cancer_type", how="inner",
        )
        # Population rank (higher eligible = lower rank number)
        merged["pop_rank"] = merged["eligible_patients_year"].rank(ascending=False).astype(int)
        # Combined score: product of ranks (lower = better)
        merged["combined_rank"] = (merged["rank"] * merged["pop_rank"])
        merged = merged.sort_values("combined_rank")
        merged.to_csv(OUTPUT_DIR / "priority_ranking.csv", index=False)

        print(f"\nPriority ranking (SL strength x patient population):")
        for _, row in merged.head(10).iterrows():
            print(f"  {row['cancer_type']}: SL rank={row['rank']}, "
                  f"pop rank={row['pop_rank']}, combined={row['combined_rank']:.0f}, "
                  f"eligible={row['eligible_patients_year']:,}/year")

        # === Visualizations ===
        print("\nGenerating visualizations...")

        # Population bar chart
        fig, ax = plt.subplots(figsize=(9, max(5, len(pop_df) * 0.35)))
        y_pos = np.arange(len(pop_df))
        ax.barh(y_pos, pop_df["eligible_patients_year"], color="#4DBEEE", alpha=0.8)
        labels = [f"{row['cancer_type']} ({row['homdel_pct']:.1f}%)"
                  for _, row in pop_df.iterrows()]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Estimated MTAP-Deleted Patients/Year (US)")
        ax.set_title("PRMT5i-Eligible Patient Population by Cancer Type")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "population_bar_chart.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: population_bar_chart.png")

        # Priority bubble chart
        fig, ax = plt.subplots(figsize=(9, 7))
        for _, row in merged.iterrows():
            size = max(30, row["eligible_patients_year"] / 50)
            color = "#D95319" if row["fdr"] < 0.05 else "#999999"
            ax.scatter(abs(row["cohens_d"]), row["eligible_patients_year"],
                       s=size, c=color, alpha=0.7, edgecolors="black", linewidths=0.5)
            ax.annotate(row["cancer_type"],
                        (abs(row["cohens_d"]), row["eligible_patients_year"]),
                        fontsize=7, xytext=(5, 5), textcoords="offset points")

        ax.set_xlabel("|Cohen's d| (PRMT5 SL strength)")
        ax.set_ylabel("Eligible Patients/Year (US)")
        ax.set_title("PRMT5i Priority Matrix: SL Strength vs Patient Population\n"
                      "(red = FDR < 0.05)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "priority_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: priority_matrix.png")

    # Validation
    lung_row = pop_df[pop_df["cancer_type"] == "Lung"]
    if not lung_row.empty:
        pct = lung_row.iloc[0]["homdel_pct"]
        print(f"\nValidation: Lung homdel = {pct:.1f}% "
              f"(expect ~15.6%, LUAD 12.0% + LUSC 19.5%)")


if __name__ == "__main__":
    main()
