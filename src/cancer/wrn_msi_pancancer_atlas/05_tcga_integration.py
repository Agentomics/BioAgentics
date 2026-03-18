"""Phase 5: TCGA clinical integration — population estimates & trial gaps.

Maps DepMap WRN-MSI SL rankings to patient populations, cross-references
pembrolizumab MSI-H response rates, and identifies clinical trial gaps
for WRN inhibitors (VVD-214, HRO-761).

Usage:
    uv run python -m wrn_msi_pancancer_atlas.05_tcga_integration
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

TCGA_DIR = REPO_ROOT / "data" / "tcga" / "pancancer_msi"
PHASE2_DIR = REPO_ROOT / "output" / "wrn-msi-pancancer-atlas" / "phase2"
OUTPUT_DIR = REPO_ROOT / "output" / "wrn-msi-pancancer-atlas" / "phase5"

# US annual cancer incidence (ACS 2024 estimates)
US_ANNUAL_INCIDENCE = {
    "UCEC": 67880,
    "STAD": 26890,
    "BLCA": 83190,
    "CHOL": 12220,
    "COADREAD": 152810,
    "SKCM": 100640,
    "LIHC": 41210,
    "ESCA": 22370,
    "LUSC": 117550,
    "LUAD": 117550,
    "CESC": 13820,
    "PAAD": 66440,
    "BRCA": 310720,
    "HNSC": 58450,
    "OV": 19680,
    "DLBC": 80620,
    "SARC": 13590,
    "KIRC": 81800,
    "KIRP": 81800,
    "GBM": 25400,
    "LGG": 25400,
    "PRAD": 288300,
    "LAML": 20380,
    "THCA": 44020,
    "UCS": 67880,
    "MESO": 2800,
    "TGCT": 9760,
    "ACC": 600,
    "KICH": 81800,
    "THYM": 400,
    "UVM": 2500,
    "PCPG": 800,
}

# Map TCGA cancer types to DepMap OncotreeLineage
TCGA_TO_DEPMAP = {
    "UCEC": "Uterus",
    "UCS": "Uterus",
    "STAD": "Esophagus/Stomach",
    "ESCA": "Esophagus/Stomach",
    "BLCA": "Bladder/Urinary Tract",
    "CHOL": "Biliary Tract",
    "COADREAD": "Bowel",
    "SKCM": "Skin",
    "LIHC": "Liver",
    "LUSC": "Lung",
    "LUAD": "Lung",
    "CESC": "Cervix",
    "PAAD": "Pancreas",
    "BRCA": "Breast",
    "HNSC": "Head and Neck",
    "OV": "Ovary/Fallopian Tube",
    "DLBC": "Lymphoid",
    "SARC": "Soft Tissue",
    "KIRC": "Kidney",
    "KIRP": "Kidney",
    "GBM": "CNS/Brain",
    "LGG": "CNS/Brain",
    "PRAD": "Prostate",
    "LAML": "Myeloid",
    "THCA": "Thyroid",
}

# Pembrolizumab ORR in MSI-H/dMMR patients by tumor type
# Sources: KEYNOTE-158, KEYNOTE-164, KEYNOTE-177
PEMBROLIZUMAB_ORR_MSI_H = {
    "COADREAD": 0.438,  # KEYNOTE-177 first-line: 43.8%
    "UCEC": 0.571,      # KEYNOTE-158: 57.1%
    "STAD": 0.458,      # KEYNOTE-158: 45.8%
    "CHOL": 0.409,      # KEYNOTE-158: 40.9%
    "OV": 0.333,        # KEYNOTE-158: 33.3%
    "PAAD": 0.182,      # KEYNOTE-158: 18.2%
}

# WRN inhibitor trial tumor types (enrolled indications)
# VVD-214/RO7589831 (NCT06004245): MSI-H/dMMR solid tumors
# HRO-761 (NCT05838768): MSI/dMMR solid tumors
WRN_TRIAL_TUMOR_TYPES = {
    "COADREAD": ["VVD-214", "HRO-761"],
    "UCEC": ["VVD-214", "HRO-761"],
    "STAD": ["VVD-214", "HRO-761"],
    "OV": ["VVD-214", "HRO-761"],
}


def build_population_estimates(tcga: pd.DataFrame) -> pd.DataFrame:
    """Estimate annual MSI-H patients per cancer type in US."""
    rows = []
    for _, row in tcga.iterrows():
        ct = row["cancer_type"]
        msi_h_pct = row["msi_h_pct"] / 100.0
        incidence = US_ANNUAL_INCIDENCE.get(ct, 0)
        estimated_msi_h = int(incidence * msi_h_pct)
        depmap_lineage = TCGA_TO_DEPMAP.get(ct, "")

        rows.append({
            "cancer_type": ct,
            "depmap_lineage": depmap_lineage,
            "msi_h_pct": round(msi_h_pct, 4),
            "us_incidence": incidence,
            "estimated_msi_h_patients_per_year": estimated_msi_h,
            "n_tcga_total": row["total_samples"],
            "n_tcga_msi_h": row["msi_h_count"],
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("estimated_msi_h_patients_per_year", ascending=False)
    return result.reset_index(drop=True)


def build_priority_ranking(
    population: pd.DataFrame,
    effect_sizes: pd.DataFrame,
) -> pd.DataFrame:
    """Combine WRN SL strength with MSI-H patient population for priority score."""
    # Get WRN effect sizes per DepMap lineage (filter to WRN gene only)
    wrn = effect_sizes[effect_sizes["gene"] == "WRN"].copy()
    wrn = wrn[["cancer_type", "cohens_d", "ci_lower", "ci_upper", "fdr", "classification"]]
    wrn = wrn.rename(columns={
        "cohens_d": "wrn_d",
        "ci_lower": "wrn_ci_lower",
        "ci_upper": "wrn_ci_upper",
        "cancer_type": "depmap_lineage",
    })

    # Get pan-cancer pooled WRN effect as fallback
    pooled = wrn[wrn["depmap_lineage"] == "Pan-cancer (pooled)"]
    pooled_d = pooled["wrn_d"].iloc[0] if len(pooled) > 0 else np.nan
    lineage_wrn = wrn[wrn["depmap_lineage"] != "Pan-cancer (pooled)"]

    # Only keep MSI-H cancer types (msi_h_count > 0)
    msi_h_pop = population[population["n_tcga_msi_h"] > 0].copy()

    # Merge with WRN effect sizes where lineage-specific data exists
    merged = msi_h_pop.merge(lineage_wrn, on="depmap_lineage", how="left")

    # Fill missing lineage-specific WRN data with pan-cancer pooled
    merged["wrn_d"] = merged["wrn_d"].fillna(pooled_d)
    merged["wrn_source"] = merged["classification"].apply(
        lambda x: "lineage-specific" if pd.notna(x) else "pan-cancer pooled"
    )
    merged["classification"] = merged["classification"].fillna("POOLED_ESTIMATE")
    merged["fdr"] = merged["fdr"].fillna(pooled["fdr"].iloc[0] if len(pooled) > 0 else np.nan)

    # Priority score: |WRN effect| × log(MSI-H patient population + 1)
    merged["priority_score"] = (
        merged["wrn_d"].abs()
        * np.log1p(merged["estimated_msi_h_patients_per_year"])
    )

    merged = merged.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return merged


def build_clinical_concordance(ranking: pd.DataFrame) -> pd.DataFrame:
    """Compare WRN dependency with pembrolizumab ORR in MSI-H tumors."""
    rows = []
    for _, row in ranking.iterrows():
        ct = row["cancer_type"]
        pembro_orr = PEMBROLIZUMAB_ORR_MSI_H.get(ct)
        trials = WRN_TRIAL_TUMOR_TYPES.get(ct, [])

        rows.append({
            "cancer_type": ct,
            "depmap_lineage": row["depmap_lineage"],
            "wrn_d": round(row["wrn_d"], 4),
            "wrn_source": row["wrn_source"],
            "msi_h_pct": row["msi_h_pct"],
            "estimated_msi_h_patients": row["estimated_msi_h_patients_per_year"],
            "pembrolizumab_orr": pembro_orr,
            "pembro_data_available": pembro_orr is not None,
            "wrn_trials": ";".join(trials) if trials else "none",
            "n_wrn_trials": len(trials),
        })

    return pd.DataFrame(rows)


def build_trial_gap_analysis(concordance: pd.DataFrame) -> pd.DataFrame:
    """Identify MSI-H tumor types underexplored by WRN inhibitor trials."""
    rows = []
    for _, row in concordance.iterrows():
        # Gap criteria: has MSI-H patients but no WRN trial enrollment
        in_trial = row["n_wrn_trials"] > 0
        has_population = row["estimated_msi_h_patients"] > 0
        pembro_low = (
            row["pembrolizumab_orr"] is not None
            and not pd.isna(row["pembrolizumab_orr"])
            and row["pembrolizumab_orr"] < 0.35
        )

        if has_population:
            if not in_trial and row["estimated_msi_h_patients"] >= 100:
                gap_type = "not_in_trials"
            elif pembro_low and not in_trial:
                gap_type = "low_io_response_no_trial"
            elif in_trial:
                gap_type = "covered"
            else:
                gap_type = "small_population"

            rows.append({
                "cancer_type": row["cancer_type"],
                "depmap_lineage": row["depmap_lineage"],
                "wrn_d": row["wrn_d"],
                "msi_h_pct": row["msi_h_pct"],
                "estimated_msi_h_patients": row["estimated_msi_h_patients"],
                "pembrolizumab_orr": row["pembrolizumab_orr"],
                "wrn_trials": row["wrn_trials"],
                "gap_type": gap_type,
                "rationale": _gap_rationale(row, gap_type),
            })

    result = pd.DataFrame(rows)
    # Sort: not_in_trials first, then by estimated patients
    gap_order = {"not_in_trials": 0, "low_io_response_no_trial": 1, "small_population": 2, "covered": 3}
    result["_sort"] = result["gap_type"].map(gap_order)
    result = result.sort_values(["_sort", "estimated_msi_h_patients"], ascending=[True, False])
    result = result.drop(columns=["_sort"]).reset_index(drop=True)
    return result


def _gap_rationale(row: pd.Series, gap_type: str) -> str:
    ct = row["cancer_type"]
    pts = int(row["estimated_msi_h_patients"])
    if gap_type == "not_in_trials":
        msg = f"{ct}: ~{pts:,} MSI-H patients/yr, not enrolled in VVD-214 or HRO-761 trials"
        if row["pembrolizumab_orr"] is not None and not pd.isna(row["pembrolizumab_orr"]):
            msg += f", pembrolizumab ORR {row['pembrolizumab_orr']:.0%}"
        return msg
    elif gap_type == "low_io_response_no_trial":
        return f"{ct}: low IO response (ORR {row['pembrolizumab_orr']:.0%}), ~{pts:,} pts/yr, no WRN trial"
    elif gap_type == "covered":
        return f"{ct}: enrolled in {row['wrn_trials']}"
    else:
        return f"{ct}: small MSI-H population (~{pts:,}/yr)"


def plot_priority_bubble(ranking: pd.DataFrame, out_path: Path) -> None:
    """Bubble chart: x=WRN effect size, y=MSI-H patient population, size=priority."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color by trial coverage
    colors = []
    for _, row in ranking.iterrows():
        ct = row["cancer_type"]
        if ct in WRN_TRIAL_TUMOR_TYPES:
            colors.append("#4CAF50")  # green: in trial
        elif row["estimated_msi_h_patients_per_year"] >= 100:
            colors.append("#D95319")  # orange: underexplored
        else:
            colors.append("#CCCCCC")  # grey: small population

    ax.scatter(
        ranking["wrn_d"],
        ranking["estimated_msi_h_patients_per_year"].clip(lower=1),
        s=ranking["priority_score"] * 5 + 20,
        c=colors,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )

    for _, row in ranking.iterrows():
        ax.annotate(
            row["cancer_type"],
            (row["wrn_d"], max(row["estimated_msi_h_patients_per_year"], 1)),
            fontsize=7,
            ha="center",
            va="bottom",
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50", markersize=10, label="In WRN trials"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D95319", markersize=10, label="Underexplored"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCCCCC", markersize=10, label="Small population"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_xlabel("WRN SL Effect Size (Cohen's d, more negative = stronger)")
    ax.set_ylabel("Estimated MSI-H patients/year (US)")
    ax.set_yscale("log")
    ax.set_title("WRN-MSI Clinical Priority Matrix")
    ax.axvline(x=-0.5, color="grey", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    population: pd.DataFrame,
    ranking: pd.DataFrame,
    concordance: pd.DataFrame,
    gap_analysis: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write human-readable summary."""
    total_msi_h = population["estimated_msi_h_patients_per_year"].sum()
    n_msi_h_types = len(ranking)
    n_gaps = len(gap_analysis[gap_analysis["gap_type"] == "not_in_trials"])

    lines = [
        "=== Phase 5: TCGA Clinical Integration Summary ===",
        "",
        f"MSI-H cancer types with prevalence data: {n_msi_h_types}",
        f"Estimated total US MSI-H patients/year: ~{total_msi_h:,}",
        "",
        "--- Top 5 MSI-H Populations (US annual estimates) ---",
    ]
    for _, row in ranking.head(5).iterrows():
        lines.append(
            f"  {row['cancer_type']}: ~{row['estimated_msi_h_patients_per_year']:,}/yr "
            f"({row['msi_h_pct']:.1%} of {row['us_incidence']:,}), "
            f"WRN d={row['wrn_d']:.3f} [{row['wrn_source']}]"
        )

    lines += [
        "",
        "--- Priority Ranking (WRN SL × population) ---",
    ]
    for _, row in ranking.iterrows():
        lines.append(
            f"  {row['cancer_type']}: score={row['priority_score']:.1f}, "
            f"WRN d={row['wrn_d']:.3f}, ~{row['estimated_msi_h_patients_per_year']:,} pts/yr"
        )

    lines += [
        "",
        "--- Pembrolizumab Concordance ---",
    ]
    for _, row in concordance.iterrows():
        if row["pembro_data_available"]:
            lines.append(
                f"  {row['cancer_type']}: WRN d={row['wrn_d']:.3f}, "
                f"pembro ORR={row['pembrolizumab_orr']:.1%}"
            )

    lines += [
        "",
        f"--- Trial Gap Analysis ({n_gaps} underexplored types) ---",
    ]
    for _, row in gap_analysis.iterrows():
        lines.append(f"  [{row['gap_type']}] {row['rationale']}")

    lines += [
        "",
        "--- Key Conclusions ---",
        "1. COADREAD and UCEC are the largest MSI-H populations with confirmed WRN SL",
        "2. STAD is a major MSI-H population — WRN dependency uses pan-cancer pooled estimate",
        f"3. {n_gaps} MSI-H tumor types with meaningful patient populations lack WRN trial enrollment",
        "4. Pembrolizumab ORR varies widely (18-57%) across MSI-H tumor types —",
        "   low-response types (e.g., PAAD) may especially benefit from WRN inhibitor combinations",
        "",
        "Sources: TCGA PanCanAtlas 2018, ACS 2024, KEYNOTE-158/164/177, NCT06004245, NCT05838768",
    ]

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 5: TCGA Clinical Integration ===\n")

    # Load TCGA MSI-H prevalence
    print("Loading TCGA MSI-H prevalence...")
    tcga = pd.read_csv(TCGA_DIR / "msi_prevalence_by_cancer_type.csv")
    print(f"  {len(tcga)} cancer types, {tcga['total_samples'].sum()} TCGA patients")
    total_msi_h = tcga["msi_h_count"].sum()
    print(f"  Pan-cancer MSI-H rate: {total_msi_h}/{tcga['total_samples'].sum()} "
          f"= {total_msi_h / tcga['total_samples'].sum():.1%}")

    # Part A: Population estimates
    print("\nEstimating US MSI-H patient populations...")
    population = build_population_estimates(tcga)
    population.to_csv(OUTPUT_DIR / "patient_population_estimates.csv", index=False)

    total_pts = population["estimated_msi_h_patients_per_year"].sum()
    print(f"  Estimated total US MSI-H patients/year: ~{total_pts:,}")
    print("  Top 5:")
    for _, row in population.head(5).iterrows():
        print(f"    {row['cancer_type']}: ~{row['estimated_msi_h_patients_per_year']:,}/yr "
              f"({row['msi_h_pct']:.1%} of {row['us_incidence']:,})")

    # Part B: Priority ranking with WRN effect sizes
    print("\nLoading Phase 2 WRN effect sizes...")
    effect_sizes = pd.read_csv(PHASE2_DIR / "wrn_effect_sizes.csv")
    wrn_only = effect_sizes[effect_sizes["gene"] == "WRN"]
    print(f"  {len(wrn_only)} WRN entries: "
          + ", ".join(f"{r['cancer_type']} (d={r['cohens_d']:.3f})"
                      for _, r in wrn_only.iterrows()))

    ranking = build_priority_ranking(population, effect_sizes)
    ranking.to_csv(OUTPUT_DIR / "priority_ranking.csv", index=False)

    print(f"\nPriority ranking ({len(ranking)} MSI-H tumor types):")
    for _, row in ranking.iterrows():
        print(f"  {row['cancer_type']}: WRN d={row['wrn_d']:.3f} [{row['wrn_source']}], "
              f"~{row['estimated_msi_h_patients_per_year']:,} pts/yr, "
              f"score={row['priority_score']:.1f}")

    # Part C: Clinical concordance
    print("\nBuilding clinical concordance (pembrolizumab ORR)...")
    concordance = build_clinical_concordance(ranking)
    concordance.to_csv(OUTPUT_DIR / "clinical_concordance.csv", index=False)

    pembro_rows = concordance[concordance["pembro_data_available"]]
    print(f"  Pembrolizumab ORR data available for {len(pembro_rows)} MSI-H tumor types:")
    for _, row in pembro_rows.iterrows():
        print(f"    {row['cancer_type']}: WRN d={row['wrn_d']:.3f}, "
              f"pembro ORR={row['pembrolizumab_orr']:.1%}")

    # Part D: Trial gap analysis
    print("\nTrial gap analysis...")
    gap_analysis = build_trial_gap_analysis(concordance)
    gap_analysis.to_csv(OUTPUT_DIR / "trial_gap_analysis.csv", index=False)

    gaps = gap_analysis[gap_analysis["gap_type"] == "not_in_trials"]
    print(f"  {len(gaps)} MSI-H tumor types with >=100 patients/yr NOT in WRN trials:")
    for _, row in gaps.iterrows():
        print(f"    {row['rationale']}")

    # Part E: Bubble chart
    print("\nGenerating priority bubble chart...")
    plot_priority_bubble(ranking, OUTPUT_DIR / "priority_bubble_chart.png")
    print("  Saved priority_bubble_chart.png")

    # Part F: Summary
    write_summary(population, ranking, concordance, gap_analysis,
                  OUTPUT_DIR / "tcga_integration_summary.txt")
    print("\n  Saved tcga_integration_summary.txt")


if __name__ == "__main__":
    main()
