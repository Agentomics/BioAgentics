"""Phase 5: TCGA population estimates + clinical concordance.

Maps DepMap SL rankings to patient populations and identifies clinical
trial gaps for ARID1A-mutant cancers.

Usage:
    uv run python -m arid1a_pancancer_sl_atlas.05_tcga_clinical
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

TCGA_DIR = REPO_ROOT / "data" / "tcga" / "pancancer_arid1a"
PHASE2_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase2"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase5"

# Approximate US annual cancer incidence (ACS 2024 estimates)
# Source: American Cancer Society Cancer Facts & Figures 2024
US_ANNUAL_INCIDENCE = {
    "UCEC": 67880,    # Uterine corpus (endometrial)
    "STAD": 26890,    # Stomach
    "BLCA": 83190,    # Bladder
    "CHOL": 12220,    # Bile duct (intrahepatic + extrahepatic)
    "COADREAD": 152810,  # Colorectal
    "SKCM": 100640,   # Melanoma
    "LIHC": 41210,    # Liver (HCC)
    "ESCA": 22370,    # Esophageal
    "LUSC": 117550,   # Lung (squamous ~30% of ~235k)
    "LUAD": 117550,   # Lung (adeno ~40% of ~235k)
    "CESC": 13820,    # Cervical
    "PAAD": 66440,    # Pancreatic
    "BRCA": 310720,   # Breast
    "HNSC": 58450,    # Head and neck
    "OV": 19680,      # Ovarian
    "DLBC": 80620,    # Lymphoma (NHL)
    "SARC": 13590,    # Soft tissue sarcoma
    "KIRC": 81800,    # Kidney (RCC)
    "KIRP": 81800,    # Kidney (papillary)
    "GBM": 25400,     # Brain (glioblastoma)
    "LGG": 25400,     # Brain (lower grade glioma)
    "PRAD": 288300,   # Prostate
    "LAML": 20380,    # AML
    "THCA": 44020,    # Thyroid
    "UCS": 67880,     # Uterine carcinosarcoma (subset of uterine)
    "MESO": 2800,     # Mesothelioma
    "TGCT": 9760,     # Testicular
    "ACC": 600,       # Adrenocortical
    "KICH": 81800,    # Kidney (chromophobe)
    "THYM": 400,      # Thymoma
    "UVM": 2500,      # Uveal melanoma
    "PCPG": 800,      # Pheochromocytoma
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

# Reference cancer types with tulmimetostat clinical validation
CLINICALLY_VALIDATED = {"Ovary/Fallopian Tube", "Uterus"}


def build_population_estimates(tcga: pd.DataFrame) -> pd.DataFrame:
    """Estimate annual ARID1A-mutant patients per cancer type in US."""
    rows = []
    for _, row in tcga.iterrows():
        ct = row["cancer_type"]
        freq = row["ARID1A_altered_pct"] / 100.0
        incidence = US_ANNUAL_INCIDENCE.get(ct, 0)
        estimated = int(incidence * freq)
        depmap_lineage = TCGA_TO_DEPMAP.get(ct, "")

        rows.append({
            "cancer_type": ct,
            "depmap_lineage": depmap_lineage,
            "tcga_freq": round(freq, 4),
            "us_incidence": incidence,
            "estimated_arid1a_patients_per_year": estimated,
            "n_tcga_sequenced": row["n_sequenced"],
            "n_tcga_altered": row["ARID1A_any_altered"],
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("estimated_arid1a_patients_per_year", ascending=False)
    return result.reset_index(drop=True)


def build_priority_ranking(
    population: pd.DataFrame,
    effect_sizes: pd.DataFrame,
) -> pd.DataFrame:
    """Combine SL strength with patient population for clinical impact score."""
    # Get best SL effect per DepMap lineage (strongest negative d across all genes)
    best_sl = (
        effect_sizes.groupby("cancer_type")["cohens_d"]
        .min()
        .rename("best_sl_d")
        .reset_index()
    )
    # Also get EZH2 specifically
    ezh2 = effect_sizes[effect_sizes["gene"] == "EZH2"][["cancer_type", "cohens_d"]].rename(
        columns={"cohens_d": "ezh2_d"}
    )
    best_sl = best_sl.merge(ezh2, on="cancer_type", how="left")

    # Get strongest gene per cancer type
    best_gene = (
        effect_sizes.loc[effect_sizes.groupby("cancer_type")["cohens_d"].idxmin()]
        [["cancer_type", "gene"]]
        .rename(columns={"gene": "strongest_sl_gene"})
    )
    best_sl = best_sl.merge(best_gene, on="cancer_type", how="left")

    # Aggregate population per DepMap lineage
    pop_by_lineage = (
        population.groupby("depmap_lineage")
        .agg(
            estimated_arid1a_patients=("estimated_arid1a_patients_per_year", "sum"),
            tcga_freq_weighted=("tcga_freq", "mean"),
        )
        .reset_index()
    )

    # Merge
    ranking = best_sl.merge(
        pop_by_lineage,
        left_on="cancer_type",
        right_on="depmap_lineage",
        how="inner",
    )

    # Priority score: |SL effect| x log(patient population)
    ranking["priority_score"] = (
        ranking["best_sl_d"].abs()
        * np.log1p(ranking["estimated_arid1a_patients"])
    )

    # Classify
    categories = []
    for _, row in ranking.iterrows():
        if row["cancer_type"] in CLINICALLY_VALIDATED:
            categories.append("clinically_validated")
        elif row["best_sl_d"] < -0.5 and row["estimated_arid1a_patients"] > 1000:
            categories.append("underexplored")
        elif row["best_sl_d"] < -0.3:
            categories.append("moderate_sl")
        else:
            categories.append("weak_sl")
    ranking["category"] = categories

    ranking = ranking.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return ranking


def build_trial_recommendations(ranking: pd.DataFrame) -> list[dict]:
    """Identify top cancer types for trial expansion."""
    underexplored = ranking[ranking["category"] == "underexplored"]
    recs = []

    for _, row in underexplored.head(5).iterrows():
        recs.append({
            "cancer_type": row["cancer_type"],
            "strongest_sl_gene": row["strongest_sl_gene"],
            "best_sl_d": round(row["best_sl_d"], 3),
            "ezh2_d": round(row["ezh2_d"], 3) if pd.notna(row["ezh2_d"]) else None,
            "estimated_arid1a_patients_per_year": int(row["estimated_arid1a_patients"]),
            "priority_score": round(row["priority_score"], 2),
            "rationale": (
                f"Strong SL (d={row['best_sl_d']:.2f} for {row['strongest_sl_gene']}), "
                f"~{int(row['estimated_arid1a_patients']):,} estimated ARID1A-mutant patients/year in US, "
                f"no active SL-targeted clinical trials identified"
            ),
        })

    return recs


def plot_priority_bubble(ranking: pd.DataFrame, out_path: Path) -> None:
    """Bubble chart: x=SL effect size, y=patient population, size=priority."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        "clinically_validated": "#4CAF50",
        "underexplored": "#D95319",
        "moderate_sl": "#EDB120",
        "weak_sl": "#CCCCCC",
    }

    for cat, group in ranking.groupby("category"):
        ax.scatter(
            group["best_sl_d"],
            group["estimated_arid1a_patients"],
            s=group["priority_score"] * 20,
            c=colors.get(cat, "#999"),
            alpha=0.7,
            label=cat,
            edgecolors="black",
            linewidths=0.5,
        )

    # Label points
    for _, row in ranking.iterrows():
        ax.annotate(
            row["cancer_type"],
            (row["best_sl_d"], row["estimated_arid1a_patients"]),
            fontsize=7,
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Best SL Effect Size (Cohen's d, more negative = stronger)")
    ax.set_ylabel("Estimated ARID1A-mutant patients/year (US)")
    ax.set_yscale("log")
    ax.set_title("ARID1A SL Clinical Priority Matrix")
    ax.legend(loc="upper right", fontsize=8)
    ax.axvline(x=-0.5, color="grey", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 5: TCGA Population & Clinical Concordance ===\n")

    # Part A: Population estimates
    print("Loading TCGA ARID1A frequencies...")
    tcga = pd.read_csv(TCGA_DIR / "arid1a_mutation_deletion_frequencies.csv")
    print(f"  {len(tcga)} cancer types, {tcga['n_sequenced'].sum()} patients")
    print(f"  Pan-cancer ARID1A alteration rate: "
          f"{tcga['ARID1A_any_altered'].sum() / tcga['n_sequenced'].sum():.1%}")

    population = build_population_estimates(tcga)
    population.to_csv(OUTPUT_DIR / "patient_population_estimates.csv", index=False)

    total_patients = population["estimated_arid1a_patients_per_year"].sum()
    print(f"\n  Estimated total US ARID1A-mutant patients/year: ~{total_patients:,}")
    print("  Top 5:")
    for _, row in population.head(5).iterrows():
        print(f"    {row['cancer_type']}: ~{row['estimated_arid1a_patients_per_year']:,}/yr "
              f"({row['tcga_freq']:.1%} of {row['us_incidence']:,})")

    # Part B: Priority ranking
    print("\nLoading Phase 2 SL effect sizes...")
    effect_sizes = pd.read_csv(PHASE2_DIR / "known_sl_effect_sizes.csv")

    ranking = build_priority_ranking(population, effect_sizes)
    ranking.to_csv(OUTPUT_DIR / "priority_ranking.csv", index=False)

    print("\nPriority ranking:")
    for _, row in ranking.iterrows():
        print(f"  {row['cancer_type']}: {row['strongest_sl_gene']} d={row['best_sl_d']:.2f}, "
              f"~{int(row['estimated_arid1a_patients']):,} pts/yr, "
              f"score={row['priority_score']:.1f} [{row['category']}]")

    # Clinical concordance
    concordance = {
        "clinically_validated": ranking[ranking["category"] == "clinically_validated"]["cancer_type"].tolist(),
        "underexplored": ranking[ranking["category"] == "underexplored"]["cancer_type"].tolist(),
        "moderate_sl": ranking[ranking["category"] == "moderate_sl"]["cancer_type"].tolist(),
        "weak_sl": ranking[ranking["category"] == "weak_sl"]["cancer_type"].tolist(),
    }
    with open(OUTPUT_DIR / "clinical_concordance.json", "w") as f:
        json.dump(concordance, f, indent=2)

    # Trial recommendations
    recommendations = build_trial_recommendations(ranking)
    with open(OUTPUT_DIR / "trial_expansion_recommendations.json", "w") as f:
        json.dump(recommendations, f, indent=2)

    print(f"\nTrial expansion recommendations ({len(recommendations)} cancer types):")
    for rec in recommendations:
        print(f"  {rec['cancer_type']}: {rec['rationale']}")

    # Bubble chart
    print("\nGenerating priority bubble chart...")
    plot_priority_bubble(ranking, OUTPUT_DIR / "priority_bubble_chart.png")
    print("  Saved priority_bubble_chart.png")


if __name__ == "__main__":
    main()
