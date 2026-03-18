"""Phase 4: PRISM drug validation for SMARCA4 SL targets.

Tests whether SMARCA4-deficient cell lines show differential sensitivity to
SL-targeting drugs in PRISM 24Q2. Priority drug classes: EZH2 inhibitors,
BET inhibitors, HDAC inhibitors. Correlates PRISM drug sensitivity with
CRISPR SL effect sizes from Phase 2.

Usage:
    uv run python -m smarca4_pancancer_sl_atlas.04_prism_drug_validation
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "data" / "results" / "smarca4-pancancer-sl-atlas" / "phase1"
PHASE2_DIR = REPO_ROOT / "data" / "results" / "smarca4-pancancer-sl-atlas" / "phase2"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "smarca4-pancancer-sl-atlas" / "phase4"

# Drug targets to search for in PRISM (SWI/SNF-relevant drug classes)
DRUG_TARGETS = {
    # EZH2 inhibitors
    "CPI-1205": ("EZH2 inhibitor", "EZH2"),
    "EPZ020411": ("EZH2 inhibitor", "EZH2"),
    "tazemetostat": ("EZH2 inhibitor", "EZH2"),
    # BET inhibitors
    "JQ1": ("BET inhibitor", "BRD4"),
    "OTX015": ("BET inhibitor", "BRD4"),
    "I-BET-762": ("BET inhibitor", "BRD4"),
    # HDAC inhibitors
    "vorinostat": ("HDAC inhibitor", "HDAC1"),
    "panobinostat": ("HDAC inhibitor", "HDAC1"),
    "entinostat": ("HDAC inhibitor", "HDAC1"),
    # ATR inhibitors
    "berzosertib": ("ATR inhibitor", "ATR"),
    "ceralasertib": ("ATR inhibitor", "ATR"),
    # CDK inhibitors (relevant in NSCLC)
    "palbociclib": ("CDK4/6 inhibitor", "CDK4"),
    "ribociclib": ("CDK4/6 inhibitor", "CDK4"),
}


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    fdr = np.empty(n)
    for i in range(n):
        fdr[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
    fdr_sorted = fdr[sorted_idx]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[sorted_idx] = fdr_sorted
    return np.minimum(fdr, 1.0)


def find_available_drugs(meta: pd.DataFrame) -> dict[str, dict]:
    """Search PRISM metadata for target drugs."""
    available = {}
    all_names = set(meta["name"].dropna().unique())

    for drug_name, (mechanism, crispr_gene) in DRUG_TARGETS.items():
        in_prism = drug_name in all_names
        if in_prism:
            drug_rows = meta[meta["name"] == drug_name]
            broad_id = drug_rows["broad_id"].iloc[0]
            available[drug_name] = {
                "mechanism": mechanism,
                "crispr_gene": crispr_gene,
                "broad_id": broad_id,
                "matrix_key": f"BRD:{broad_id}",
            }

    return available


def compute_drug_sensitivity(
    prism_matrix: pd.DataFrame,
    classified: pd.DataFrame,
    available_drugs: dict,
    qualifying_types: list[str],
) -> pd.DataFrame:
    """Compute differential drug sensitivity (SMARCA4-deficient vs intact) per cancer type."""
    rows = []

    for drug_name, drug_info in available_drugs.items():
        matrix_key = drug_info["matrix_key"]
        if matrix_key not in prism_matrix.index:
            print(f"    {drug_name}: matrix key {matrix_key} not in data matrix")
            continue

        drug_sens = prism_matrix.loc[matrix_key]

        for cancer_type in qualifying_types:
            ct_lines = classified[classified["OncotreeLineage"] == cancer_type]
            def_lines = ct_lines[ct_lines["smarca4_status"] == "deficient"].index
            intact_lines = ct_lines[ct_lines["smarca4_status"] == "intact"].index

            def_vals = drug_sens.reindex(def_lines).dropna().values
            intact_vals = drug_sens.reindex(intact_lines).dropna().values

            if len(def_vals) < 3 or len(intact_vals) < 3:
                continue

            _, pval = stats.mannwhitneyu(def_vals, intact_vals, alternative="two-sided")
            d = cohens_d(def_vals, intact_vals)

            rows.append({
                "cancer_type": cancer_type,
                "drug": drug_name,
                "mechanism": drug_info["mechanism"],
                "cohens_d": d,
                "p_value": float(pval),
                "n_def": len(def_vals),
                "n_intact": len(intact_vals),
                "median_sens_def": float(np.median(def_vals)),
                "median_sens_intact": float(np.median(intact_vals)),
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["p_value"].values)
    return result


def compute_crispr_prism_concordance(
    prism_matrix: pd.DataFrame,
    classified: pd.DataFrame,
    available_drugs: dict,
    qualifying_types: list[str],
) -> pd.DataFrame:
    """Correlate PRISM drug sensitivity with CRISPR dependency per cell line."""
    from bioagentics.data.gene_ids import load_depmap_matrix

    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")

    rows = []
    for drug_name, drug_info in available_drugs.items():
        matrix_key = drug_info["matrix_key"]
        gene = drug_info["crispr_gene"]

        if matrix_key not in prism_matrix.index or gene not in crispr.columns:
            continue

        drug_sens = prism_matrix.loc[matrix_key]
        gene_dep = crispr[gene]

        common = drug_sens.dropna().index.intersection(gene_dep.dropna().index)
        common = common.intersection(classified.index)

        if len(common) < 10:
            continue

        r, p = stats.spearmanr(drug_sens[common].values, gene_dep[common].values)

        rows.append({
            "gene": gene,
            "drug": drug_name,
            "mechanism": drug_info["mechanism"],
            "spearman_r": float(r),
            "p_value": float(p),
            "n_lines": len(common),
        })

        for cancer_type in qualifying_types:
            ct_lines = classified.loc[
                classified["OncotreeLineage"] == cancer_type
            ].index.intersection(common)
            if len(ct_lines) < 5:
                continue

            r_ct, p_ct = stats.spearmanr(
                drug_sens[ct_lines].values, gene_dep[ct_lines].values
            )
            rows.append({
                "gene": gene,
                "drug": drug_name,
                "mechanism": drug_info["mechanism"],
                "cancer_type": cancer_type,
                "spearman_r": float(r_ct),
                "p_value": float(p_ct),
                "n_lines": len(ct_lines),
            })

    return pd.DataFrame(rows)


def plot_drug_heatmap(sensitivity: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of drug sensitivity effect sizes."""
    if sensitivity.empty:
        return

    pivot = sensitivity.pivot_table(
        index="cancer_type", columns="drug", values="cohens_d", aggfunc="first"
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.5)))

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 1.0)
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > vmax * 0.6 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d (negative = SMARCA4-deficient more sensitive)")
    ax.set_title("PRISM Drug Sensitivity: SMARCA4-deficient vs Intact")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4: PRISM Drug Validation ===\n")

    # Load Phase 1 outputs
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "smarca4_classified_lines.csv", index_col=0)
    summary = pd.read_csv(PHASE1_DIR / "cancer_type_summary.csv")
    qualifying = summary[summary["qualifies"]]["cancer_type"].tolist()

    # Load PRISM metadata
    print("Loading PRISM 24Q2 treatment metadata...")
    meta = pd.read_csv(DEPMAP_DIR / "Repurposing_Public_24Q2_Treatment_Meta_Data.csv")

    # Find available drugs
    available = find_available_drugs(meta)

    # Drug availability report
    availability_rows = []
    for drug_name, (mechanism, _) in DRUG_TARGETS.items():
        in_prism = drug_name in available
        availability_rows.append({
            "drug_name": drug_name,
            "mechanism": mechanism,
            "in_prism": in_prism,
        })
    avail_df = pd.DataFrame(availability_rows)
    avail_df.to_csv(OUTPUT_DIR / "prism_drug_availability.csv", index=False)

    print(f"\nDrug availability:")
    for _, row in avail_df.iterrows():
        status = "AVAILABLE" if row["in_prism"] else "not found"
        print(f"  {row['drug_name']} ({row['mechanism']}): {status}")

    if not available:
        print("\nNo target drugs found in PRISM. Exiting.")
        return

    # Load PRISM data matrix
    print("\nLoading PRISM data matrix...")
    prism = pd.read_csv(
        DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        index_col=0,
    )
    print(f"  {prism.shape[0]} treatments x {prism.shape[1]} cell lines")

    # Compute drug sensitivity
    print("\nComputing differential drug sensitivity...")
    sensitivity = compute_drug_sensitivity(prism, classified, available, qualifying)

    if len(sensitivity) > 0:
        sensitivity.to_csv(OUTPUT_DIR / "prism_drug_sensitivity.csv", index=False)
        print(f"\n  {len(sensitivity)} drug-cancer type tests:")
        for _, row in sensitivity.sort_values("cohens_d").iterrows():
            sig = " *" if row.get("fdr", 1) < 0.05 else ""
            print(f"    {row['cancer_type']} / {row['drug']}: d={row['cohens_d']:.3f} "
                  f"p={row['p_value']:.3e}{sig}")
    else:
        print("  No drug sensitivity tests possible (insufficient overlapping lines)")
        pd.DataFrame().to_csv(OUTPUT_DIR / "prism_drug_sensitivity.csv", index=False)

    # CRISPR-PRISM concordance
    print("\nComputing CRISPR-PRISM concordance...")
    concordance = compute_crispr_prism_concordance(
        prism, classified, available, qualifying
    )
    concordance.to_csv(OUTPUT_DIR / "crispr_prism_concordance.csv", index=False)

    if len(concordance) > 0:
        print("  Concordance results:")
        for _, row in concordance.head(15).iterrows():
            ct = row.get("cancer_type") if "cancer_type" in concordance.columns else None
            ct_str = f" [{ct}]" if pd.notna(ct) else " [pan-cancer]"
            print(f"    {row['drug']} vs {row['gene']}{ct_str}: "
                  f"r={row['spearman_r']:.3f} p={row['p_value']:.3e}")

    # Heatmap
    print("\nGenerating drug sensitivity heatmap...")
    plot_drug_heatmap(sensitivity, OUTPUT_DIR / "drug_sensitivity_heatmap.png")
    print("  Saved drug_sensitivity_heatmap.png")

    print("\n=== Phase 4 Complete ===")


if __name__ == "__main__":
    main()
