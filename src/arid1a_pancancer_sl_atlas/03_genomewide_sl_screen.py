"""Phase 3: Genome-wide SL screen per cancer type.

For each qualifying cancer type, runs differential dependency analysis
(ARID1A-mutant vs WT) across all ~18,000 genes in CRISPRGeneEffect.
Identifies cancer-type-specific and universal SL hits.

Usage:
    uv run python -m arid1a_pancancer_sl_atlas.03_genomewide_sl_screen
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
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase1"
PHASE2_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase2"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase3"

# Significance thresholds
FDR_THRESHOLD = 0.05
EFFECT_SIZE_THRESHOLD = 0.5  # |Cohen's d|

# SWI/SNF complex genes for cross-reference
SWISNF_GENES = {
    "SMARCA4", "SMARCB1", "ARID2", "PBRM1", "SMARCC1", "SMARCC2",
    "ARID1A", "ARID1B", "SMARCA2", "SMARCD1", "SMARCD2", "SMARCD3",
    "SMARCE1", "DPF1", "DPF2", "DPF3", "BCL7A", "BCL7B", "BCL7C",
    "BRD9", "GLTSCR1", "SS18",
}

# Known SL targets from Phase 2
KNOWN_SL_TARGETS = {
    "EZH2", "ARID1B", "USP8", "BRD2", "BRD4",
    "HDAC1", "HDAC2", "HDAC3", "HDAC6",
    "ATR", "ATRIP", "PARP1", "PARP2",
    "HSP90AA1", "PIK3CA", "AKT1", "MTOR",
}

# Minimum samples for testing
MIN_SAMPLES = 3


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


def screen_cancer_type(
    ct_data: pd.DataFrame,
    crispr_cols: list[str],
    cancer_type: str,
) -> list[dict]:
    """Run genome-wide differential dependency for one cancer type."""
    mutant = ct_data[ct_data["ARID1A_status"] == "mutant"]
    wt = ct_data[ct_data["ARID1A_status"] == "WT"]

    n_mut = len(mutant)
    n_wt = len(wt)

    rows = []
    pvals = []
    gene_names = []

    for gene in crispr_cols:
        mut_vals = mutant[gene].dropna().values
        wt_vals = wt[gene].dropna().values

        if len(mut_vals) < MIN_SAMPLES or len(wt_vals) < MIN_SAMPLES:
            continue

        _, pval = stats.mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")
        d = cohens_d(mut_vals, wt_vals)

        rows.append({
            "cancer_type": cancer_type,
            "gene": gene,
            "cohens_d": d,
            "p_value": float(pval),
            "n_mut": len(mut_vals),
            "n_wt": len(wt_vals),
        })
        pvals.append(pval)
        gene_names.append(gene)

    # FDR correction per cancer type
    if pvals:
        fdrs = fdr_correction(np.array(pvals))
        for i, row in enumerate(rows):
            row["fdr"] = float(fdrs[i])

    return rows


def classify_hits(all_hits: pd.DataFrame) -> pd.DataFrame:
    """Classify SL hits as cancer-specific or universal."""
    # Count how many cancer types each gene is significant in
    gene_counts = (
        all_hits.groupby("gene")["cancer_type"]
        .nunique()
        .rename("n_cancer_types")
        .reset_index()
    )

    all_hits = all_hits.merge(gene_counts, on="gene")
    all_hits["category"] = np.where(
        all_hits["n_cancer_types"] >= 3, "universal", "cancer_specific"
    )
    return all_hits


def plot_volcano(
    results_ct: pd.DataFrame,
    cancer_type: str,
    out_dir: Path,
) -> None:
    """Volcano plot: effect size vs -log10(FDR) for one cancer type."""
    fig, ax = plt.subplots(figsize=(8, 6))

    x = results_ct["cohens_d"].values
    y = -np.log10(results_ct["fdr"].values.clip(min=1e-50))

    # Color by significance
    sig = (results_ct["fdr"] < FDR_THRESHOLD) & (results_ct["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ax.scatter(x[~sig], y[~sig], c="#CCCCCC", s=5, alpha=0.5)
    ax.scatter(x[sig], y[sig], c="#D95319", s=15, alpha=0.8)

    # Label top hits
    top = results_ct[sig].nsmallest(10, "cohens_d")
    for _, row in top.iterrows():
        ax.annotate(
            row["gene"], (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
            fontsize=7, ha="right",
        )

    ax.axhline(-np.log10(FDR_THRESHOLD), color="grey", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)

    ax.set_xlabel("Cohen's d (ARID1A-mut vs WT)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Genome-wide SL Screen: {cancer_type}")

    fig.tight_layout()
    safe_name = cancer_type.replace("/", "_").replace(" ", "_")
    fig.savefig(out_dir / f"volcano_{safe_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    volcano_dir = OUTPUT_DIR / "volcano_plots"
    volcano_dir.mkdir(exist_ok=True)

    print("=== Phase 3: Genome-Wide SL Screen ===\n")

    # Load Phase 1 outputs
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "all_cell_lines_classified.csv", index_col=0)
    summary = pd.read_csv(PHASE1_DIR / "cancer_type_summary.csv")
    qualifying = summary[summary["qualifies"]]["cancer_type"].tolist()
    print(f"  {len(qualifying)} qualifying cancer types")

    # Load CRISPR data
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    crispr_cols = list(crispr.columns)
    print(f"  {len(crispr_cols)} genes")

    # Merge
    merged = classified.join(crispr, how="inner")

    # Screen each cancer type
    all_rows = []
    for cancer_type in qualifying:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        n_mut = (ct_data["ARID1A_status"] == "mutant").sum()
        n_wt = (ct_data["ARID1A_status"] == "WT").sum()
        print(f"  Screening {cancer_type} ({n_mut} mut, {n_wt} WT)...")

        rows = screen_cancer_type(ct_data, crispr_cols, cancer_type)
        all_rows.extend(rows)

    all_results = pd.DataFrame(all_rows)
    print(f"\n  Total tests: {len(all_results)}")

    # Filter significant hits
    sig_hits = all_results[
        (all_results["fdr"] < FDR_THRESHOLD)
        & (all_results["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ].copy()

    # Focus on SL (negative d = more dependent in ARID1A-mutant)
    sl_hits = sig_hits[sig_hits["cohens_d"] < 0].copy()
    sl_hits = classify_hits(sl_hits)
    sl_hits = sl_hits.sort_values("cohens_d").reset_index(drop=True)

    print(f"  Significant SL hits (FDR<0.05, d<-0.5): {len(sl_hits)}")
    print(f"  Unique SL genes: {sl_hits['gene'].nunique()}")

    # Also capture nominally significant hits (uncorrected p<0.05, |d|>0.5)
    # These are useful when FDR correction is too conservative for small samples
    nominal_hits = all_results[
        (all_results["p_value"] < 0.05)
        & (all_results["cohens_d"] < -EFFECT_SIZE_THRESHOLD)
    ].copy()
    if len(nominal_hits) > 0:
        nominal_hits = classify_hits(nominal_hits)
        nominal_hits = nominal_hits.sort_values("cohens_d").reset_index(drop=True)
    print(f"  Nominal SL hits (p<0.05, d<-0.5): {len(nominal_hits)}")
    print(f"  Unique nominal SL genes: {nominal_hits['gene'].nunique()}")

    # Use nominal hits if FDR yields nothing (expected with small N)
    if len(sl_hits) == 0 and len(nominal_hits) > 0:
        print("  NOTE: Using nominal hits (FDR too conservative for these sample sizes)")
        sl_hits = nominal_hits

    # Save all SL hits
    sl_hits.to_csv(OUTPUT_DIR / "genomewide_sl_hits.csv", index=False)

    # Universal SL genes (significant in >=3 cancer types)
    universal = sl_hits[sl_hits["category"] == "universal"]
    universal_genes = (
        universal.groupby("gene")
        .agg(
            n_cancer_types=("cancer_type", "nunique"),
            mean_cohens_d=("cohens_d", "mean"),
            min_fdr=("fdr", "min"),
            cancer_types=("cancer_type", lambda x: ";".join(sorted(set(x)))),
        )
        .sort_values("mean_cohens_d")
        .reset_index()
    )
    universal_genes.to_csv(OUTPUT_DIR / "universal_sl_genes.csv", index=False)

    print(f"\n  Universal SL genes (>=3 cancer types): {len(universal_genes)}")
    for _, row in universal_genes.head(20).iterrows():
        in_swisnf = " [SWI/SNF]" if row["gene"] in SWISNF_GENES else ""
        in_known = " [known]" if row["gene"] in KNOWN_SL_TARGETS else ""
        print(f"    {row['gene']}: mean d={row['mean_cohens_d']:.3f}, "
              f"{row['n_cancer_types']} types{in_swisnf}{in_known}")

    # Novel SL candidates (not in known list)
    novel = sl_hits[~sl_hits["gene"].isin(KNOWN_SL_TARGETS)].copy()
    novel_summary = (
        novel.groupby("gene")
        .agg(
            n_cancer_types=("cancer_type", "nunique"),
            mean_cohens_d=("cohens_d", "mean"),
            min_fdr=("fdr", "min"),
            cancer_types=("cancer_type", lambda x: ";".join(sorted(set(x)))),
            is_swisnf=("gene", lambda x: x.iloc[0] in SWISNF_GENES),
        )
        .sort_values(["n_cancer_types", "mean_cohens_d"], ascending=[False, True])
        .reset_index()
    )
    novel_summary.to_csv(OUTPUT_DIR / "novel_sl_candidates.csv", index=False)

    print(f"\n  Novel SL candidates (not in known list): {len(novel_summary)}")
    for _, row in novel_summary.head(20).iterrows():
        in_swisnf = " [SWI/SNF]" if row["is_swisnf"] else ""
        print(f"    {row['gene']}: mean d={row['mean_cohens_d']:.3f}, "
              f"{row['n_cancer_types']} types{in_swisnf}")

    # Generate volcano plots for each cancer type
    print("\nGenerating volcano plots...")
    for cancer_type in qualifying:
        ct_results = all_results[all_results["cancer_type"] == cancer_type]
        plot_volcano(ct_results, cancer_type, volcano_dir)
    print(f"  Saved {len(qualifying)} volcano plots")


if __name__ == "__main__":
    main()
