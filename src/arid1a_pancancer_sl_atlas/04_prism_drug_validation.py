"""Phase 4: PRISM drug validation for ARID1A SL targets.

Tests whether ARID1A-mutant cell lines show differential sensitivity to
SL-targeting drugs in PRISM 24Q2. Correlates PRISM drug sensitivity with
CRISPR SL effect sizes from Phase 2.

Usage:
    uv run python -m arid1a_pancancer_sl_atlas.04_prism_drug_validation
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
PHASE1_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase1"
PHASE2_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase2"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase4"

# Drug targets to search for in PRISM
DRUG_TARGETS = {
    # drug_name: (mechanism, CRISPR_gene_to_correlate)
    "CPI-1205": ("EZH2 inhibitor", "EZH2"),
    "EPZ020411": ("EZH2 inhibitor", "EZH2"),
    "tazemetostat": ("EZH2 inhibitor", "EZH2"),
    "berzosertib": ("ATR inhibitor", "ATR"),
    "ceralasertib": ("ATR inhibitor", "ATR"),
    "alpelisib": ("PI3K inhibitor", "PIK3CA"),
    "inavolisib": ("PI3K inhibitor", "PIK3CA"),
    "vorinostat": ("HDAC inhibitor", "HDAC1"),
    "panobinostat": ("HDAC inhibitor", "HDAC1"),
    "tucidinostat": ("HDAC inhibitor", "HDAC1"),
    "JQ1": ("BET inhibitor", "BRD4"),
    "OTX015": ("BET inhibitor", "BRD4"),
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
    """Search PRISM metadata for target drugs. Return available ones with info."""
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
    """Compute differential drug sensitivity (ARID1A-mut vs WT) per cancer type."""
    rows = []

    for drug_name, drug_info in available_drugs.items():
        matrix_key = drug_info["matrix_key"]
        if matrix_key not in prism_matrix.index:
            print(f"    {drug_name}: matrix key {matrix_key} not in data matrix")
            continue

        # Get sensitivity values (columns are cell line IDs)
        drug_sens = prism_matrix.loc[matrix_key]

        for cancer_type in qualifying_types:
            ct_lines = classified[classified["OncotreeLineage"] == cancer_type]
            mut_lines = ct_lines[ct_lines["ARID1A_status"] == "mutant"].index
            wt_lines = ct_lines[ct_lines["ARID1A_status"] == "WT"].index

            # Get sensitivity for mut and WT lines (lower = more sensitive in PRISM)
            mut_vals = drug_sens.reindex(mut_lines).dropna().values
            wt_vals = drug_sens.reindex(wt_lines).dropna().values

            if len(mut_vals) < 3 or len(wt_vals) < 3:
                continue

            _, pval = stats.mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")
            d = cohens_d(mut_vals, wt_vals)

            rows.append({
                "cancer_type": cancer_type,
                "drug": drug_name,
                "mechanism": drug_info["mechanism"],
                "cohens_d": d,
                "p_value": float(pval),
                "n_mut": len(mut_vals),
                "n_wt": len(wt_vals),
                "median_sens_mut": float(np.median(mut_vals)),
                "median_sens_wt": float(np.median(wt_vals)),
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["p_value"].values)
    return result


def compute_crispr_prism_concordance(
    prism_matrix: pd.DataFrame,
    crispr_effect_sizes: pd.DataFrame,
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

        # Find common cell lines
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

        # Per cancer type
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
    """Heatmap of drug sensitivity effect sizes (cancer types x drugs)."""
    if sensitivity.empty:
        return

    pivot = sensitivity.pivot_table(
        index="cancer_type", columns="drug", values="cohens_d", aggfunc="first"
    )
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.5), max(4, len(pivot) * 0.45)))

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
    cbar.set_label("Cohen's d (negative = ARID1A-mut more sensitive)")
    ax.set_title("PRISM Drug Sensitivity: ARID1A-mutant vs WT")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4: PRISM Drug Validation ===\n")

    # Load Phase 1 outputs
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "all_cell_lines_classified.csv", index_col=0)
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
            sig = "*" if row.get("fdr", 1) < 0.05 else ""
            print(f"    {row['cancer_type']} / {row['drug']}: d={row['cohens_d']:.3f} "
                  f"p={row['p_value']:.3e}{sig}")
    else:
        print("  No drug sensitivity tests possible (insufficient overlapping lines)")
        pd.DataFrame().to_csv(OUTPUT_DIR / "prism_drug_sensitivity.csv", index=False)

    # CRISPR-PRISM concordance
    print("\nComputing CRISPR-PRISM concordance...")
    phase2_results = pd.read_csv(PHASE2_DIR / "known_sl_effect_sizes.csv")
    concordance = compute_crispr_prism_concordance(
        prism, phase2_results, classified, available, qualifying
    )
    concordance.to_csv(OUTPUT_DIR / "crispr_prism_concordance.csv", index=False)

    if len(concordance) > 0:
        for _, row in concordance.head(10).iterrows():
            ct = row.get("cancer_type") if "cancer_type" in concordance.columns else None
            ct_str = f" [{ct}]" if pd.notna(ct) else " [pan-cancer]"
            print(f"    {row['drug']} vs {row['gene']}{ct_str}: "
                  f"r={row['spearman_r']:.3f} p={row['p_value']:.3e}")

    # Heatmap
    print("\nGenerating drug sensitivity heatmap...")
    plot_drug_heatmap(sensitivity, OUTPUT_DIR / "drug_sensitivity_heatmap.png")
    print("  Saved drug_sensitivity_heatmap.png")


if __name__ == "__main__":
    main()
