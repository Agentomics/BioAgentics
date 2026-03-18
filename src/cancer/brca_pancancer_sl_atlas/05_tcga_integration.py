"""Phase 5: TCGA clinical integration and cross-atlas references.

Pan-cancer BRCA1/2 mutation frequencies from TCGA, co-mutation landscape,
addressable patient populations, cross-references with TP53/PTEN/WRN-MSI
atlases, and clinical trial landscape for PARPi combinations.

Usage:
    uv run python -m brca_pancancer_sl_atlas.05_tcga_integration
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

PHASE1_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase1"
PHASE3_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase3"
PHASE4_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase4"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase5"

# --- TCGA BRCA1/2 mutation frequencies (%) ---
# Source: TCGA PanCanAtlas 2018, cBioPortal. Includes germline + somatic LOF.
TCGA_BRCA_FREQ = {
    # cancer_type: (brca1_pct, brca2_pct, brca_any_pct, n_samples)
    "OV": (6.0, 5.5, 11.0, 585),      # High-grade serous ovarian
    "BRCA": (2.5, 3.0, 5.0, 1084),    # Breast (triple-negative enriched)
    "PRAD": (0.8, 3.0, 3.5, 500),     # Prostate (BRCA2 > BRCA1)
    "PAAD": (1.5, 3.5, 5.0, 185),     # Pancreatic
    "UCEC": (1.0, 2.0, 3.0, 530),     # Endometrial
    "COADREAD": (0.8, 1.5, 2.0, 594), # Colorectal
    "STAD": (1.0, 2.0, 2.5, 440),     # Stomach
    "BLCA": (1.0, 2.0, 2.5, 412),     # Bladder
    "LUAD": (0.5, 1.5, 2.0, 567),     # Lung adenocarcinoma
    "HNSC": (0.5, 1.0, 1.5, 510),     # Head and neck
    "SKCM": (0.5, 1.5, 2.0, 469),     # Melanoma
    "KIRC": (0.3, 1.0, 1.2, 512),     # Kidney clear cell
    "LIHC": (0.5, 1.0, 1.5, 377),     # Liver
    "GBM": (0.3, 0.5, 0.8, 396),      # Glioblastoma
    "CHOL": (1.0, 3.0, 4.0, 36),      # Cholangiocarcinoma
}

# Co-mutation rates in BRCA1/2-mutant tumors (% of BRCA-mut cases)
BRCA_COMUTATION = {
    # cancer_type: {gene: pct}
    "OV": {"TP53": 96, "PTEN": 3, "PIK3CA": 2, "CDH1": 0, "KRAS": 1, "NF1": 8},
    "BRCA": {"TP53": 80, "PTEN": 4, "PIK3CA": 30, "CDH1": 12, "KRAS": 0, "MYC_amp": 15},
    "PRAD": {"TP53": 25, "PTEN": 30, "PIK3CA": 3, "SPOP": 8, "KRAS": 0, "AR": 5},
    "PAAD": {"TP53": 70, "PTEN": 2, "PIK3CA": 2, "KRAS": 90, "CDKN2A": 25, "SMAD4": 30},
    "UCEC": {"TP53": 30, "PTEN": 60, "PIK3CA": 40, "KRAS": 15, "ARID1A": 35},
    "COADREAD": {"TP53": 55, "PTEN": 5, "PIK3CA": 15, "KRAS": 40, "APC": 70},
}

# US annual cancer incidence (ACS 2024 estimates)
US_ANNUAL_INCIDENCE = {
    "OV": 19680,
    "BRCA": 310720,
    "PRAD": 288300,
    "PAAD": 66440,
    "UCEC": 67880,
    "COADREAD": 152810,
    "STAD": 26890,
    "BLCA": 83190,
    "LUAD": 58775,
    "HNSC": 58450,
    "SKCM": 100640,
    "KIRC": 81800,
    "LIHC": 41210,
    "GBM": 14000,
    "CHOL": 12360,
}

# Map TCGA cancer types to DepMap OncotreeLineage
TCGA_TO_DEPMAP = {
    "OV": "Ovary/Fallopian Tube",
    "BRCA": "Breast",
    "PRAD": "Prostate",
    "PAAD": "Pancreas",
    "UCEC": "Uterus",
    "COADREAD": "Bowel",
    "STAD": "Esophagus/Stomach",
    "BLCA": "Bladder/Urinary Tract",
    "LUAD": "Lung",
    "HNSC": "Head and Neck",
    "SKCM": "Skin",
    "KIRC": "Kidney",
    "LIHC": "Liver",
    "GBM": "CNS/Brain",
    "CHOL": "Biliary Tract",
}

# Active/recent PARPi and DDR combination clinical trials (as of early 2026)
BRCA_TRIALS = {
    "OV": [
        "olaparib maintenance (SOLO-1/2, FDA-approved)",
        "niraparib maintenance (PRIMA, FDA-approved)",
        "rucaparib maintenance (ARIEL3, FDA-approved)",
        "olaparib + bevacizumab (PAOLA-1, FDA-approved)",
        "ceralasertib + olaparib (CAPRI, Phase 2)",
    ],
    "BRCA": [
        "olaparib adjuvant (OlympiA, FDA-approved HER2-neg gBRCA)",
        "talazoparib (EMBRACA, FDA-approved gBRCA)",
        "olaparib (OlympiAD, FDA-approved gBRCA)",
    ],
    "PRAD": [
        "olaparib (PROfound, FDA-approved HRRm mCRPC)",
        "rucaparib (TRITON2, FDA-approved BRCA mCRPC)",
        "niraparib + abiraterone (MAGNITUDE, FDA-approved HRRm)",
        "talazoparib + enzalutamide (TALAPRO-2, FDA-approved HRRm)",
    ],
    "PAAD": [
        "olaparib maintenance (POLO, FDA-approved gBRCA)",
        "rucaparib (Phase 2, gBRCA)",
    ],
    "UCEC": [],
    "COADREAD": [],
    "STAD": [],
    "BLCA": [],
    "LUAD": [],
}

# POLθi trials (emerging)
POLQ_TRIALS = [
    "ART4215 (POLθi, Phase 1/2, HRD solid tumors)",
    "novobiocin (POLθ ATPase, Phase 1, BRCA-mutant)",
]


def build_frequency_table() -> pd.DataFrame:
    """Build BRCA1/2 frequency table from TCGA data."""
    rows = []
    for ct, (b1, b2, b_any, n) in sorted(
        TCGA_BRCA_FREQ.items(), key=lambda x: -x[1][2]
    ):
        incidence = US_ANNUAL_INCIDENCE.get(ct, 0)
        est_patients = int(incidence * b_any / 100)
        depmap = TCGA_TO_DEPMAP.get(ct, "")

        rows.append({
            "cancer_type": ct,
            "depmap_lineage": depmap,
            "brca1_freq_pct": b1,
            "brca2_freq_pct": b2,
            "brca_any_freq_pct": b_any,
            "n_tcga_samples": n,
            "us_annual_incidence": incidence,
            "estimated_brca_patients_per_year": est_patients,
        })

    return pd.DataFrame(rows)


def build_comutation_table() -> pd.DataFrame:
    """Build co-mutation landscape table."""
    rows = []
    for ct, comuts in BRCA_COMUTATION.items():
        for gene, pct in comuts.items():
            rows.append({
                "cancer_type": ct,
                "co_mutated_gene": gene,
                "comutation_pct_in_brca_mut": pct,
            })
    return pd.DataFrame(rows)


def build_patient_estimates(freq_table: pd.DataFrame) -> pd.DataFrame:
    """Build patient population estimates for addressable markets."""
    rows = []
    total_patients = 0

    for _, row in freq_table.iterrows():
        ct = row["cancer_type"]
        patients = row["estimated_brca_patients_per_year"]
        total_patients += patients

        trials = BRCA_TRIALS.get(ct, [])
        has_approval = any("FDA-approved" in t for t in trials)
        has_trial = len(trials) > 0

        if has_approval:
            trial_status = "FDA-approved PARPi"
        elif has_trial:
            trial_status = "Active trials"
        elif patients >= 500:
            trial_status = "Unmet need (>500 patients)"
        elif patients >= 100:
            trial_status = "Small population"
        else:
            trial_status = "Rare"

        rows.append({
            "cancer_type": ct,
            "brca_freq_pct": row["brca_any_freq_pct"],
            "us_incidence": row["us_annual_incidence"],
            "estimated_brca_patients": patients,
            "trial_status": trial_status,
            "active_trials": "; ".join(trials) if trials else "none",
            "n_trials": len(trials),
        })

    result = pd.DataFrame(rows)
    result = result.sort_values("estimated_brca_patients", ascending=False)
    return result


def build_cross_atlas_refs() -> dict:
    """Cross-reference with TP53, PTEN, and WRN-MSI atlases."""
    refs = {
        "tp53_atlas": {
            "relevance": "BRCA1/TP53 co-mutation is near-universal in HGSOC (96%). "
                         "TP53 status may modify BRCA synthetic lethalities.",
            "key_question": "Do TP53-mutant BRCA-deficient tumors show different "
                            "dependencies vs TP53-wt BRCA-deficient?",
            "overlap_cancer_types": ["OV", "BRCA", "PAAD"],
            "comutation_rates": {
                "OV": "96% TP53 co-mutation in BRCA-mut",
                "BRCA": "80% TP53 co-mutation in BRCA-mut",
                "PAAD": "70% TP53 co-mutation in BRCA-mut",
            },
        },
        "pten_atlas": {
            "relevance": "PI3K pathway status in BRCA-deficient tumors provides "
                         "rationale for PI3Ki + PARPi combinations.",
            "key_question": "Do PTEN-lost BRCA-deficient tumors show enhanced "
                            "sensitivity to PI3K/AKT pathway inhibitors?",
            "combination_rationale": "PI3K signaling supports HR through RAD51. "
                                     "PI3Ki may synergize with PARPi by further "
                                     "impairing HR in BRCA-deficient context.",
            "overlap_cancer_types": ["PRAD", "UCEC", "BRCA"],
            "comutation_rates": {
                "PRAD": "30% PTEN co-loss in BRCA-mut prostate",
                "UCEC": "60% PTEN co-loss in BRCA-mut endometrial",
            },
        },
        "wrn_msi_atlas": {
            "relevance": "DDR drug overlap — ATRi, DNA-PKi shared between BRCA and "
                         "MSI therapeutic contexts.",
            "key_question": "Are DDR combination strategies (ATRi, DNA-PKi) effective "
                            "across both BRCA-deficient and MSI-H contexts?",
            "shared_drug_targets": ["ATR", "DNA-PKcs", "CHK1", "WEE1"],
            "note": "Different mechanisms but convergent on replication stress.",
        },
    }

    # Check if atlas results exist
    for atlas_name, atlas_dir in [
        ("tp53_atlas", REPO_ROOT / "data" / "results" / "tp53-hotspot-atlas"),
        ("pten_atlas", REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas"),
        ("wrn_msi_atlas", REPO_ROOT / "data" / "results" / "wrn-msi-pancancer-atlas"),
    ]:
        refs[atlas_name]["data_available"] = atlas_dir.exists()

    return refs


def plot_frequency_barplot(freq_table: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot of BRCA1/2 mutation frequencies by cancer type."""
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(freq_table))
    width = 0.35

    ax.bar(x - width / 2, freq_table["brca1_freq_pct"], width,
           label="BRCA1", color="#E53935", alpha=0.8)
    ax.bar(x + width / 2, freq_table["brca2_freq_pct"], width,
           label="BRCA2", color="#1E88E5", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(freq_table["cancer_type"], fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("Mutation Frequency (%)")
    ax.set_title("TCGA BRCA1/2 Mutation Frequencies by Cancer Type")
    ax.legend()

    # Add combined frequency labels
    for i, (_, row) in enumerate(freq_table.iterrows()):
        ax.text(i, max(row["brca1_freq_pct"], row["brca2_freq_pct"]) + 0.3,
                f"{row['brca_any_freq_pct']:.1f}%", ha="center", fontsize=7, fontweight="bold")

    fig.tight_layout()
    fig.savefig(output_dir / "tcga_brca_frequency_barplot.png", dpi=150)
    plt.close(fig)


def plot_comutation_heatmap(comut_table: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of co-mutation landscape in BRCA-mutant tumors."""
    if len(comut_table) == 0:
        return

    pivot = comut_table.pivot_table(
        index="co_mutated_gene", columns="cancer_type",
        values="comutation_pct_in_brca_mut", aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.5)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, fontsize=9, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Add text values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = "white" if val > 50 else "black"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Co-mutation rate (%)", shrink=0.7)
    ax.set_title("Co-mutation Landscape in BRCA1/2-Mutant Tumors")
    fig.tight_layout()
    fig.savefig(output_dir / "comutation_oncoplot.png", dpi=150)
    plt.close(fig)


def plot_patient_treemap(estimates: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of estimated BRCA-mutant patients per year (treemap alternative)."""
    if len(estimates) == 0:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_est = estimates.sort_values("estimated_brca_patients", ascending=True)

    colors = []
    for _, row in sorted_est.iterrows():
        if "FDA-approved" in row["trial_status"]:
            colors.append("#4CAF50")
        elif "Active" in row["trial_status"]:
            colors.append("#FFC107")
        elif "Unmet" in row["trial_status"]:
            colors.append("#E53935")
        else:
            colors.append("#9E9E9E")

    y_pos = range(len(sorted_est))
    ax.barh(y_pos, sorted_est["estimated_brca_patients"], color=colors, alpha=0.85)

    labels = [
        f"{row['cancer_type']} ({row['brca_freq_pct']:.1f}%)"
        for _, row in sorted_est.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Estimated BRCA-Mutant Patients/Year (US)")
    ax.set_title("Addressable Patient Populations by Cancer Type\n"
                 "Green=FDA-approved PARPi, Yellow=Active trials, Red=Unmet need")

    # Add value labels
    for i, (_, row) in enumerate(sorted_est.iterrows()):
        ax.text(row["estimated_brca_patients"] + 100, i,
                f"{row['estimated_brca_patients']:,}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_dir / "patient_population_treemap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 5: TCGA Clinical Integration ===\n")

    # --- BRCA1/2 frequency table ---
    print("Building TCGA BRCA1/2 frequency table...")
    freq_table = build_frequency_table()
    freq_table.to_csv(OUTPUT_DIR / "tcga_brca_frequencies.csv", index=False)
    print(f"  {len(freq_table)} cancer types")

    total_patients = freq_table["estimated_brca_patients_per_year"].sum()
    print(f"  Estimated total US BRCA-mutant patients/year: {total_patients:,}")

    for _, row in freq_table.head(5).iterrows():
        print(f"    {row['cancer_type']}: BRCA1={row['brca1_freq_pct']}% "
              f"BRCA2={row['brca2_freq_pct']}% → {row['estimated_brca_patients_per_year']:,} patients/yr")

    # --- Co-mutation landscape ---
    print("\nBuilding co-mutation landscape...")
    comut_table = build_comutation_table()
    comut_table.to_csv(OUTPUT_DIR / "comutation_landscape.csv", index=False)
    print(f"  {len(comut_table)} gene-cancer type pairs")

    # Key finding: TP53 co-mutation
    for ct in ["OV", "BRCA", "PAAD"]:
        tp53_pct = BRCA_COMUTATION.get(ct, {}).get("TP53", 0)
        print(f"    {ct}: TP53 co-mutation = {tp53_pct}% of BRCA-mut cases")

    # --- Patient population estimates ---
    print("\nBuilding patient population estimates...")
    estimates = build_patient_estimates(freq_table)
    estimates.to_csv(OUTPUT_DIR / "patient_population_estimates.csv", index=False)

    for _, row in estimates.head(8).iterrows():
        print(f"    {row['cancer_type']}: {row['estimated_brca_patients']:,} patients — "
              f"{row['trial_status']}")

    # Identify unmet needs
    unmet = estimates[estimates["trial_status"].str.contains("Unmet")]
    if len(unmet) > 0:
        unmet_total = unmet["estimated_brca_patients"].sum()
        print(f"\n  Unmet need: {len(unmet)} cancer types, {unmet_total:,} patients/year")

    # --- Cross-atlas references ---
    print("\nBuilding cross-atlas references...")
    cross_refs = build_cross_atlas_refs()
    with open(OUTPUT_DIR / "cross_atlas_references.json", "w") as f:
        json.dump(cross_refs, f, indent=2, default=str)

    for atlas, info in cross_refs.items():
        available = "data available" if info.get("data_available") else "no data yet"
        print(f"  {atlas}: {info['relevance'][:60]}... ({available})")

    # --- Clinical trial landscape ---
    print("\nClinical trial landscape:")
    trial_summary = {}
    for ct, trials in BRCA_TRIALS.items():
        if trials:
            trial_summary[ct] = trials
            print(f"  {ct}: {len(trials)} trials")
            for t in trials[:2]:
                print(f"    - {t}")
    print(f"  POLθi trials: {len(POLQ_TRIALS)}")
    for t in POLQ_TRIALS:
        print(f"    - {t}")

    # --- Plots ---
    print("\nGenerating plots...")
    plot_frequency_barplot(freq_table, OUTPUT_DIR)
    print("  tcga_brca_frequency_barplot.png")

    plot_comutation_heatmap(comut_table, OUTPUT_DIR)
    print("  comutation_oncoplot.png")

    plot_patient_treemap(estimates, OUTPUT_DIR)
    print("  patient_population_treemap.png")

    # --- Summary ---
    top_3 = freq_table.nlargest(3, "estimated_brca_patients_per_year")
    top_3_strs = []
    for _, r in top_3.iterrows():
        ct = r["cancer_type"]
        pts = r["estimated_brca_patients_per_year"]
        top_3_strs.append(f"{ct}({pts:,})")
    print("\n=== Phase 5 Complete ===")
    print(f"  {len(freq_table)} cancer types analyzed")
    print(f"  Total estimated US BRCA-mutant patients/year: {total_patients:,}")
    print(f"  Top 3 by patient volume: {', '.join(top_3_strs)}")


if __name__ == "__main__":
    main()
