"""Phase 5: TCGA clinical integration — population estimates & trial mapping.

Estimates CDKN2A deletion frequency per cancer type, addressable patient
populations, and priority ranking combining effect size × population × druggability.
Maps to active CDK4/6i clinical trials. Includes mesothelioma deep-dive.

Usage:
    uv run python -m cdkn2a_pancancer_dependency_atlas.05_tcga_integration
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

PHASE1_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase1"
PHASE2_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase2"
OUTPUT_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase5"

# TCGA CDKN2A homozygous deletion frequency (%) by cancer type.
# Source: TCGA PanCanAtlas 2018, cBioPortal TCGA PanCancer, published literature.
TCGA_CDKN2A_DEL_PCT = {
    "MESO": 72.0,    # Mesothelioma — highest rate
    "GBM": 58.0,     # Glioblastoma
    "LGG": 12.0,     # Lower-grade glioma
    "PAAD": 30.0,    # Pancreatic adenocarcinoma
    "SKCM": 20.0,    # Melanoma
    "BLCA": 20.0,    # Bladder cancer
    "HNSC": 18.0,    # Head and neck squamous
    "LUSC": 15.0,    # Lung squamous
    "LUAD": 8.0,     # Lung adenocarcinoma
    "ESCA": 12.0,    # Esophageal
    "STAD": 5.0,     # Stomach
    "OV": 5.0,       # Ovarian
    "SARC": 12.0,    # Sarcoma
    "DLBC": 10.0,    # DLBCL
    "LAML": 8.0,     # AML
    "BRCA": 2.0,     # Breast
    "COADREAD": 2.0, # Colorectal
    "UCEC": 1.0,     # Endometrial
    "PRAD": 1.0,     # Prostate
    "ACC": 15.0,     # Adrenocortical
    "CHOL": 10.0,    # Cholangiocarcinoma
    "LIHC": 5.0,     # Liver
    "KIRC": 1.0,     # Kidney clear cell
    "THCA": 0.5,     # Thyroid
}

# TCGA RB1 co-loss rate in CDKN2A-deleted tumors (approximate %)
TCGA_RB1_COLOSS_PCT = {
    "MESO": 2.0,
    "GBM": 10.0,
    "PAAD": 3.0,
    "SKCM": 3.0,
    "BLCA": 15.0,
    "HNSC": 5.0,
    "LUSC": 8.0,
    "LUAD": 5.0,
    "SARC": 10.0,
}

# US annual cancer incidence (ACS 2024 estimates)
US_ANNUAL_INCIDENCE = {
    "MESO": 2800,
    "GBM": 14000,
    "LGG": 11400,
    "PAAD": 66440,
    "SKCM": 100640,
    "BLCA": 83190,
    "HNSC": 58450,
    "LUSC": 58775,
    "LUAD": 58775,
    "ESCA": 22370,
    "STAD": 26890,
    "OV": 19680,
    "SARC": 13590,
    "DLBC": 80620,
    "LAML": 20380,
    "BRCA": 310720,
    "COADREAD": 152810,
    "UCEC": 67880,
    "PRAD": 288300,
    "ACC": 600,
    "CHOL": 12220,
    "LIHC": 41210,
    "KIRC": 81800,
    "THCA": 44020,
}

# Map TCGA cancer types to DepMap OncotreeLineage
TCGA_TO_DEPMAP = {
    "GBM": "CNS/Brain",
    "LGG": "CNS/Brain",
    "PAAD": "Pancreas",
    "SKCM": "Skin",
    "BLCA": "Bladder/Urinary Tract",
    "HNSC": "Head and Neck",
    "LUSC": "Lung",
    "LUAD": "Lung",
    "ESCA": "Esophagus/Stomach",
    "STAD": "Esophagus/Stomach",
    "OV": "Ovary/Fallopian Tube",
    "SARC": "Soft Tissue",
    "DLBC": "Lymphoid",
    "LAML": "Myeloid",
    "BRCA": "Breast",
    "MESO": "Pleura",
    "COADREAD": "Bowel",
    "UCEC": "Uterus",
    "PRAD": "Prostate",
    "ACC": "Adrenal Gland",
    "CHOL": "Biliary Tract",
    "LIHC": "Liver",
    "KIRC": "Kidney",
    "THCA": "Thyroid",
}

# Active CDK4/6 inhibitor clinical trials by cancer type (selected)
CDK46I_TRIALS = {
    "GBM": ["ribociclib (NCT02933736)", "abemaciclib (NCT02981940)"],
    "PAAD": ["palbociclib + gedatolisib (NCT03065062)"],
    "SKCM": ["ribociclib + binimetinib (NCT01781572)"],
    "LUSC": ["abemaciclib (NCT02246621)"],
    "LUAD": ["palbociclib (NCT02896335)"],
    "HNSC": ["palbociclib (NCT02101034)"],
    "MESO": [],  # No active CDK4/6i trials — key gap
    "BRCA": ["palbociclib (PALOMA-2/3)", "ribociclib (MONALEESA)", "abemaciclib (monarchE)"],
    "BLCA": [],
    "OV": ["ribociclib (NCT03056833)"],
    "SARC": ["palbociclib (NCT02784795)"],
}


def build_prevalence_table() -> pd.DataFrame:
    """Build CDKN2A deletion prevalence table from TCGA data."""
    rows = []
    for ct, del_pct in sorted(TCGA_CDKN2A_DEL_PCT.items(), key=lambda x: -x[1]):
        incidence = US_ANNUAL_INCIDENCE.get(ct, 0)
        del_frac = del_pct / 100.0
        estimated_del_patients = int(incidence * del_frac)
        rb1_coloss = TCGA_RB1_COLOSS_PCT.get(ct, 5.0)  # default 5%
        # Eligible = CDKN2A-del AND RB1-intact
        rb1_intact_frac = 1.0 - rb1_coloss / 100.0
        eligible_patients = int(estimated_del_patients * rb1_intact_frac)
        depmap_lineage = TCGA_TO_DEPMAP.get(ct, "")

        rows.append({
            "cancer_type": ct,
            "depmap_lineage": depmap_lineage,
            "cdkn2a_del_pct": del_pct,
            "rb1_coloss_pct": rb1_coloss,
            "us_incidence": incidence,
            "estimated_cdkn2a_del_patients_per_year": estimated_del_patients,
            "estimated_eligible_patients_per_year": eligible_patients,
        })

    return pd.DataFrame(rows)


def build_priority_ranking(
    prevalence: pd.DataFrame,
    cdk6_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Combine CDK6 SL strength with patient population for priority score.

    Druggability = 1.0 for all (CDK4/6 inhibitors are approved drugs).
    """
    # Get CDK6 effect sizes per DepMap lineage (rb1_intact stratum)
    cdk6 = cdk6_effects[
        (cdk6_effects["gene"] == "CDK6") & (cdk6_effects["rb1_stratum"] == "rb1_intact")
    ].copy()
    cdk6_map = cdk6.set_index("cancer_type")[["cohens_d", "ci_lower", "ci_upper", "fdr", "classification"]]
    cdk6_map = cdk6_map.rename(columns={
        "cohens_d": "cdk6_d",
        "ci_lower": "cdk6_ci_lower",
        "ci_upper": "cdk6_ci_upper",
    })

    # Pan-cancer pooled fallback
    pooled = cdk6_map.loc["Pan-cancer (pooled)"] if "Pan-cancer (pooled)" in cdk6_map.index else None
    pooled_d = pooled["cdk6_d"] if pooled is not None else np.nan

    merged = prevalence.merge(
        cdk6_map.reset_index().rename(columns={"cancer_type": "depmap_lineage"}),
        on="depmap_lineage",
        how="left",
    )

    # Fill missing with pooled
    merged["cdk6_d"] = merged["cdk6_d"].fillna(pooled_d)
    merged["cdk6_source"] = merged["classification"].apply(
        lambda x: "lineage-specific" if pd.notna(x) else "pan-cancer pooled"
    )
    merged["classification"] = merged["classification"].fillna("POOLED_ESTIMATE")

    # Priority = |CDK6 effect| × log(eligible patients + 1) × druggability(=1)
    merged["priority_score"] = (
        merged["cdk6_d"].abs()
        * np.log1p(merged["estimated_eligible_patients_per_year"])
    )

    merged = merged.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return merged


def build_comutation_landscape(prevalence: pd.DataFrame) -> pd.DataFrame:
    """Summarize co-alteration landscape per cancer type."""
    rows = []
    for _, row in prevalence.iterrows():
        ct = row["cancer_type"]
        n_del = row["estimated_cdkn2a_del_patients_per_year"]
        rb1 = TCGA_RB1_COLOSS_PCT.get(ct, 5.0)

        rows.append({
            "cancer_type": ct,
            "cdkn2a_del_patients": n_del,
            "rb1_coloss_pct": rb1,
            "rb1_intact_eligible": int(n_del * (1 - rb1 / 100)),
            "note": "RB1 co-loss abolishes CDK4/6 dependency" if rb1 > 10 else "",
        })

    return pd.DataFrame(rows)


def build_trial_mapping(ranking: pd.DataFrame) -> pd.DataFrame:
    """Map cancer types to active CDK4/6i trials and identify gaps."""
    rows = []
    for _, row in ranking.iterrows():
        ct = row["cancer_type"]
        trials = CDK46I_TRIALS.get(ct, [])
        has_trials = len(trials) > 0
        eligible = row["estimated_eligible_patients_per_year"]

        if has_trials:
            gap_type = "covered"
        elif eligible >= 1000:
            gap_type = "major_gap"
        elif eligible >= 100:
            gap_type = "gap"
        else:
            gap_type = "small_population"

        rows.append({
            "cancer_type": ct,
            "depmap_lineage": row["depmap_lineage"],
            "cdk6_d": row.get("cdk6_d", np.nan),
            "eligible_patients": eligible,
            "cdkn2a_del_pct": row["cdkn2a_del_pct"],
            "active_trials": "; ".join(trials) if trials else "none",
            "n_trials": len(trials),
            "gap_type": gap_type,
        })

    result = pd.DataFrame(rows)
    gap_order = {"major_gap": 0, "gap": 1, "small_population": 2, "covered": 3}
    result["_sort"] = result["gap_type"].map(gap_order)
    result = result.sort_values(["_sort", "eligible_patients"], ascending=[True, False])
    result = result.drop(columns=["_sort"]).reset_index(drop=True)
    return result


def plot_priority_ranking(ranking: pd.DataFrame, out_path: Path) -> None:
    """Bubble chart: x=CDK6 effect, y=eligible patients, color=trial status."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = []
    for _, row in ranking.iterrows():
        ct = row["cancer_type"]
        trials = CDK46I_TRIALS.get(ct, [])
        if len(trials) > 0:
            colors.append("#4CAF50")
        elif row["estimated_eligible_patients_per_year"] >= 500:
            colors.append("#D95319")
        else:
            colors.append("#CCCCCC")

    ax.scatter(
        ranking["cdk6_d"].fillna(0),
        ranking["estimated_eligible_patients_per_year"].clip(lower=1),
        s=ranking["priority_score"].fillna(0) * 3 + 20,
        c=colors,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
    )

    for _, row in ranking.iterrows():
        ax.annotate(
            row["cancer_type"],
            (row.get("cdk6_d", 0) or 0, max(row["estimated_eligible_patients_per_year"], 1)),
            fontsize=7, ha="center", va="bottom",
        )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4CAF50", markersize=10, label="In CDK4/6i trials"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#D95319", markersize=10, label="Underexplored"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#CCCCCC", markersize=10, label="Small population"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_xlabel("CDK6 SL Effect Size (Cohen's d, more negative = stronger)")
    ax.set_ylabel("Estimated CDKN2A-del/RB1-intact patients/year (US)")
    ax.set_yscale("log")
    ax.set_title("CDKN2A-CDK6 Clinical Priority Matrix")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_prevalence_heatmap(prevalence: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of CDKN2A deletion prevalence across cancer types."""
    top = prevalence.sort_values("cdkn2a_del_pct", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(top))
    ax.barh(y_pos, top["cdkn2a_del_pct"], color="#D95319", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top["cancer_type"], fontsize=9)
    ax.set_xlabel("CDKN2A Homozygous Deletion (%)")
    ax.set_title("CDKN2A Deletion Frequency Across Cancer Types (TCGA)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    prevalence: pd.DataFrame,
    ranking: pd.DataFrame,
    comutation: pd.DataFrame,
    trial_mapping: pd.DataFrame,
    out_path: Path,
) -> None:
    """Write human-readable summary."""
    total_del = prevalence["estimated_cdkn2a_del_patients_per_year"].sum()
    total_elig = prevalence["estimated_eligible_patients_per_year"].sum()
    n_gaps = len(trial_mapping[trial_mapping["gap_type"].isin(["major_gap", "gap"])])

    lines = [
        "=" * 60,
        "CDKN2A Pan-Cancer Dependency Atlas - Phase 5: TCGA Clinical Integration",
        "=" * 60,
        "",
        f"Estimated total US CDKN2A-deleted patients/year: ~{total_del:,}",
        f"Estimated CDK4/6i-eligible (CDKN2A-del/RB1-intact): ~{total_elig:,}",
        "",
        "CDKN2A DELETION PREVALENCE",
        "-" * 50,
    ]
    for _, row in prevalence.iterrows():
        lines.append(
            f"  {row['cancer_type']}: {row['cdkn2a_del_pct']:.0f}% del, "
            f"~{row['estimated_cdkn2a_del_patients_per_year']:,} pts/yr, "
            f"eligible: ~{row['estimated_eligible_patients_per_year']:,}"
        )

    lines += ["", "PRIORITY RANKING (CDK6 SL x population)", "-" * 50]
    for _, row in ranking.head(15).iterrows():
        d = row.get("cdk6_d", float("nan"))
        d_str = f"d={d:.3f}" if pd.notna(d) else "d=N/A"
        lines.append(
            f"  {row['cancer_type']}: score={row['priority_score']:.1f}, "
            f"CDK6 {d_str} [{row.get('cdk6_source', 'N/A')}], "
            f"~{row['estimated_eligible_patients_per_year']:,} pts/yr"
        )

    lines += ["", "MESOTHELIOMA DEEP-DIVE", "-" * 50]
    meso = prevalence[prevalence["cancer_type"] == "MESO"]
    if len(meso) > 0:
        m = meso.iloc[0]
        lines += [
            f"  Incidence: {m['us_incidence']:,}/yr",
            f"  CDKN2A deletion: {m['cdkn2a_del_pct']:.0f}%",
            f"  Estimated CDKN2A-del: ~{m['estimated_cdkn2a_del_patients_per_year']:,}/yr",
            f"  RB1 co-loss: ~{m['rb1_coloss_pct']:.0f}%",
            f"  Eligible for CDK4/6i: ~{m['estimated_eligible_patients_per_year']:,}/yr",
            "  Active CDK4/6i trials: NONE (major clinical gap)",
            "  NOTE: Mesothelioma has highest CDKN2A del rate (72%),",
            "  limited treatment options, and poor prognosis — strong rationale",
            "  for CDK4/6 inhibitor clinical trials.",
        ]

    lines += ["", f"TRIAL GAP ANALYSIS ({n_gaps} underexplored types)", "-" * 50]
    for _, row in trial_mapping.iterrows():
        if row["gap_type"] in ("major_gap", "gap"):
            lines.append(
                f"  [{row['gap_type']}] {row['cancer_type']}: "
                f"~{row['eligible_patients']:,} eligible pts/yr, "
                f"{row['cdkn2a_del_pct']:.0f}% del rate, no CDK4/6i trials"
            )

    lines += [
        "",
        "CO-MUTATION LANDSCAPE",
        "-" * 50,
        "  RB1 co-loss rate varies by cancer type (2-15%).",
        "  RB1-intact tumors are the primary CDK4/6i target population.",
        "  CDKN2B co-deletion is common (~85% of CDKN2A-del in DepMap).",
        "",
        "Sources: TCGA PanCanAtlas 2018, ACS 2024 estimates, ClinicalTrials.gov",
    ]

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 5: TCGA Clinical Integration ===\n")

    # Part A: Prevalence table
    print("Building CDKN2A deletion prevalence table...")
    prevalence = build_prevalence_table()
    prevalence.to_csv(OUTPUT_DIR / "tcga_cdkn2a_prevalence.csv", index=False)

    total_del = prevalence["estimated_cdkn2a_del_patients_per_year"].sum()
    total_elig = prevalence["estimated_eligible_patients_per_year"].sum()
    print(f"  {len(prevalence)} cancer types")
    print(f"  Estimated CDKN2A-del patients/yr: ~{total_del:,}")
    print(f"  CDK4/6i-eligible (RB1-intact): ~{total_elig:,}")

    print("  Top 5 by eligible patients:")
    for _, row in prevalence.head(5).iterrows():
        print(f"    {row['cancer_type']}: {row['cdkn2a_del_pct']:.0f}% del, "
              f"~{row['estimated_eligible_patients_per_year']:,} eligible/yr")

    # Part B: Priority ranking with CDK6 effect sizes
    print("\nLoading Phase 2 CDK6 effect sizes...")
    cdk6_effects = pd.read_csv(PHASE2_DIR / "cdk6_effect_sizes.csv")
    print(f"  {len(cdk6_effects)} CDK6 entries")

    ranking = build_priority_ranking(prevalence, cdk6_effects)
    ranking.to_csv(OUTPUT_DIR / "priority_ranking.csv", index=False)

    print(f"\nPriority ranking (top 10):")
    for _, row in ranking.head(10).iterrows():
        d = row.get("cdk6_d", float("nan"))
        d_str = f"d={d:.3f}" if pd.notna(d) else "d=N/A"
        print(f"  {row['cancer_type']}: CDK6 {d_str}, "
              f"~{row['estimated_eligible_patients_per_year']:,} pts/yr, "
              f"score={row['priority_score']:.1f}")

    # Part C: Co-mutation landscape
    print("\nBuilding co-mutation landscape...")
    comutation = build_comutation_landscape(prevalence)
    comutation.to_csv(OUTPUT_DIR / "comutation_landscape.csv", index=False)

    # Part D: Trial mapping
    print("\nMapping to CDK4/6i clinical trials...")
    trial_mapping = build_trial_mapping(ranking)
    trial_mapping.to_csv(OUTPUT_DIR / "clinical_trial_mapping.csv", index=False)

    gaps = trial_mapping[trial_mapping["gap_type"].isin(["major_gap", "gap"])]
    print(f"  {len(gaps)} cancer types with clinical trial gaps:")
    for _, row in gaps.iterrows():
        print(f"    [{row['gap_type']}] {row['cancer_type']}: "
              f"~{row['eligible_patients']:,} eligible/yr")

    # Part E: Population estimates
    pop = prevalence[["cancer_type", "us_incidence", "cdkn2a_del_pct",
                       "estimated_cdkn2a_del_patients_per_year",
                       "estimated_eligible_patients_per_year"]].copy()
    pop.to_csv(OUTPUT_DIR / "patient_population_estimates.csv", index=False)

    # Part F: Plots
    print("\nGenerating plots...")
    plot_priority_ranking(ranking, OUTPUT_DIR / "priority_ranking_plot.png")
    plot_prevalence_heatmap(prevalence, OUTPUT_DIR / "prevalence_heatmap.png")
    print("  priority_ranking_plot.png, prevalence_heatmap.png")

    # Part G: Summary
    write_summary(prevalence, ranking, comutation, trial_mapping,
                  OUTPUT_DIR / "tcga_integration_summary.txt")
    print("  tcga_integration_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
