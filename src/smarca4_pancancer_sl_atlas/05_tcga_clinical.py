"""Phase 5: TCGA population estimates + clinical concordance for SMARCA4.

Maps DepMap SL rankings to patient populations and identifies clinical
trial opportunities for SMARCA4-mutant cancers. Incorporates data_curator
findings (task #221): LUAD dominates LOF, SKCM inflated by UV-passenger
missense.

Usage:
    uv run python -m smarca4_pancancer_sl_atlas.05_tcga_clinical
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

TCGA_DIR = REPO_ROOT / "data" / "tcga" / "pancancer_smarca4"
PHASE2_DIR = REPO_ROOT / "data" / "results" / "smarca4-pancancer-sl-atlas" / "phase2"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "smarca4-pancancer-sl-atlas" / "phase5"

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

# Cancer types with active SMARCA2 degrader clinical trials
CLINICALLY_TARGETED = {"Lung"}  # PRT3789/PRT7732 trials focus on NSCLC

# SKCM UV-passenger caution (from data_curator task #221)
UV_PASSENGER_TYPES = {"SKCM"}


def build_population_estimates(tcga: pd.DataFrame) -> pd.DataFrame:
    """Estimate annual SMARCA4-mutant patients per cancer type in US."""
    rows = []
    for _, row in tcga.iterrows():
        ct = row["cancer_type"]
        freq = row["SMARCA4_mut_pct"] / 100.0
        lof_count = row["SMARCA4_LOF"]
        missense_count = row["SMARCA4_missense"]
        homdel_count = row["SMARCA4_homdel"]
        incidence = US_ANNUAL_INCIDENCE.get(ct, 0)
        estimated_all = int(incidence * freq)

        # LOF-specific estimate (Class 1 — clinically actionable for SMARCA2 degraders)
        total_mut = row["SMARCA4_mutated"]
        lof_frac = lof_count / total_mut if total_mut > 0 else 0
        estimated_lof = int(estimated_all * lof_frac)

        depmap_lineage = TCGA_TO_DEPMAP.get(ct, "")

        notes = ""
        if ct in UV_PASSENGER_TYPES:
            notes = f"UV-passenger caution: only {lof_count}/{total_mut} are LOF"

        rows.append({
            "cancer_type": ct,
            "depmap_lineage": depmap_lineage,
            "tcga_freq": round(freq, 4),
            "n_mutated": int(total_mut),
            "n_lof": int(lof_count),
            "n_missense": int(missense_count),
            "n_homdel": int(homdel_count),
            "us_incidence": incidence,
            "estimated_smarca4_patients_per_year": estimated_all,
            "estimated_lof_patients_per_year": estimated_lof,
            "notes": notes,
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("estimated_smarca4_patients_per_year", ascending=False)
    return result.reset_index(drop=True)


def build_priority_ranking(
    population: pd.DataFrame,
    effect_sizes: pd.DataFrame,
) -> pd.DataFrame:
    """Combine SL strength with patient population for clinical impact score."""
    # Only use "all" cohort from Phase 2
    all_es = effect_sizes[effect_sizes["cohort"] == "all"] if "cohort" in effect_sizes.columns else effect_sizes

    best_sl = (
        all_es.groupby("cancer_type")["cohens_d"]
        .min()
        .rename("best_sl_d")
        .reset_index()
    )

    # SMARCA2 specifically
    smarca2 = all_es[all_es["gene"] == "SMARCA2"][["cancer_type", "cohens_d"]].rename(
        columns={"cohens_d": "smarca2_d"}
    )
    best_sl = best_sl.merge(smarca2, on="cancer_type", how="left")

    # Strongest gene per cancer type
    best_gene = (
        all_es.loc[all_es.groupby("cancer_type")["cohens_d"].idxmin()]
        [["cancer_type", "gene"]]
        .rename(columns={"gene": "strongest_sl_gene"})
    )
    best_sl = best_sl.merge(best_gene, on="cancer_type", how="left")

    # Aggregate population per DepMap lineage (using LOF patients for clinical relevance)
    pop_by_lineage = (
        population.groupby("depmap_lineage")
        .agg(
            estimated_smarca4_patients=("estimated_smarca4_patients_per_year", "sum"),
            estimated_lof_patients=("estimated_lof_patients_per_year", "sum"),
            tcga_freq_weighted=("tcga_freq", "mean"),
        )
        .reset_index()
    )

    ranking = best_sl.merge(
        pop_by_lineage,
        left_on="cancer_type",
        right_on="depmap_lineage",
        how="inner",
    )

    # Priority score: |SL effect| x log(LOF patient population + 1)
    ranking["priority_score"] = (
        ranking["best_sl_d"].abs()
        * np.log1p(ranking["estimated_lof_patients"])
    )

    categories = []
    for _, row in ranking.iterrows():
        if row["cancer_type"] in CLINICALLY_TARGETED:
            categories.append("clinically_targeted")
        elif row["best_sl_d"] < -0.5 and row["estimated_lof_patients"] > 500:
            categories.append("underexplored")
        elif row["best_sl_d"] < -0.3:
            categories.append("moderate_sl")
        else:
            categories.append("weak_sl")
    ranking["category"] = categories

    ranking = ranking.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return ranking


def build_trial_recommendations(ranking: pd.DataFrame) -> list[dict]:
    """Identify top cancer types for trial expansion beyond NSCLC."""
    underexplored = ranking[ranking["category"] == "underexplored"]
    recs = []

    for _, row in underexplored.head(5).iterrows():
        recs.append({
            "cancer_type": row["cancer_type"],
            "strongest_sl_gene": row["strongest_sl_gene"],
            "best_sl_d": round(row["best_sl_d"], 3),
            "smarca2_d": round(row["smarca2_d"], 3) if pd.notna(row["smarca2_d"]) else None,
            "estimated_lof_patients_per_year": int(row["estimated_lof_patients"]),
            "priority_score": round(row["priority_score"], 2),
            "rationale": (
                f"Strong SL (d={row['best_sl_d']:.2f} for {row['strongest_sl_gene']}), "
                f"~{int(row['estimated_lof_patients']):,} estimated SMARCA4-LOF patients/year in US, "
                f"not currently targeted by SMARCA2 degrader trials"
            ),
        })

    return recs


def plot_priority_bubble(ranking: pd.DataFrame, out_path: Path) -> None:
    """Bubble chart: x=SL effect size, y=LOF patient population, size=priority."""
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = {
        "clinically_targeted": "#4CAF50",
        "underexplored": "#D95319",
        "moderate_sl": "#EDB120",
        "weak_sl": "#CCCCCC",
    }

    for cat, group in ranking.groupby("category"):
        pop = group["estimated_lof_patients"].clip(lower=1)
        ax.scatter(
            group["best_sl_d"],
            pop,
            s=group["priority_score"].clip(lower=1) * 20,
            c=colors.get(cat, "#999"),
            alpha=0.7,
            label=cat,
            edgecolors="black",
            linewidths=0.5,
        )

    for _, row in ranking.iterrows():
        pop = max(row["estimated_lof_patients"], 1)
        ax.annotate(
            row["cancer_type"],
            (row["best_sl_d"], pop),
            fontsize=7,
            ha="center",
            va="bottom",
        )

    ax.set_xlabel("Best SL Effect Size (Cohen's d, more negative = stronger)")
    ax.set_ylabel("Estimated SMARCA4-LOF patients/year (US)")
    ax.set_yscale("log")
    ax.set_title("SMARCA4 SL Clinical Priority Matrix\n(LOF patients = Class 1 actionable population)")
    ax.legend(loc="upper right", fontsize=8)
    ax.axvline(x=-0.5, color="grey", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 5: TCGA Population & Clinical Integration ===\n")

    # Load TCGA SMARCA4 frequencies
    print("Loading TCGA SMARCA4 frequencies...")
    tcga = pd.read_csv(TCGA_DIR / "smarca4_mutation_deletion_frequencies.csv")
    total_seq = tcga["n_sequenced"].sum()
    total_mut = tcga["SMARCA4_mutated"].sum()
    total_lof = tcga["SMARCA4_LOF"].sum()
    total_homdel = tcga["SMARCA4_homdel"].sum()
    print(f"  {len(tcga)} cancer types, {total_seq} patients")
    print(f"  Pan-cancer: {total_mut} mutated ({total_mut/total_seq:.1%}), "
          f"{total_lof} LOF, {total_homdel} homdel")

    # Population estimates
    population = build_population_estimates(tcga)
    population.to_csv(OUTPUT_DIR / "patient_population_estimates.csv", index=False)

    total_patients = population["estimated_smarca4_patients_per_year"].sum()
    total_lof_patients = population["estimated_lof_patients_per_year"].sum()
    print(f"\n  Estimated US SMARCA4-mutant patients/year: ~{total_patients:,}")
    print(f"  Estimated US SMARCA4-LOF (Class 1) patients/year: ~{total_lof_patients:,}")
    print("  Top 5 by LOF patients:")
    top_lof = population.sort_values("estimated_lof_patients_per_year", ascending=False)
    for _, row in top_lof.head(5).iterrows():
        note = f"  ** {row['notes']}" if row["notes"] else ""
        print(f"    {row['cancer_type']}: ~{row['estimated_lof_patients_per_year']:,} LOF/yr "
              f"({row['n_lof']} LOF in TCGA){note}")

    # Priority ranking
    print("\nLoading Phase 2 SL effect sizes...")
    effect_sizes = pd.read_csv(PHASE2_DIR / "known_sl_effect_sizes.csv")

    ranking = build_priority_ranking(population, effect_sizes)
    ranking.to_csv(OUTPUT_DIR / "priority_ranking.csv", index=False)

    print("\nPriority ranking:")
    for _, row in ranking.iterrows():
        print(f"  {row['cancer_type']}: {row['strongest_sl_gene']} d={row['best_sl_d']:.2f}, "
              f"~{int(row['estimated_lof_patients']):,} LOF pts/yr, "
              f"score={row['priority_score']:.1f} [{row['category']}]")

    # Clinical concordance
    concordance = {
        "clinically_targeted": ranking[ranking["category"] == "clinically_targeted"]["cancer_type"].tolist(),
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

    print(f"\nTrial expansion recommendations ({len(recommendations)}):")
    for rec in recommendations:
        print(f"  {rec['cancer_type']}: {rec['rationale']}")

    # Bubble chart
    print("\nGenerating priority bubble chart...")
    plot_priority_bubble(ranking, OUTPUT_DIR / "priority_bubble_chart.png")
    print("  Saved priority_bubble_chart.png")

    print("\n=== Phase 5 Complete ===")


if __name__ == "__main__":
    main()
