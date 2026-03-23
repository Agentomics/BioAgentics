"""Phase 2: Pan-cancer genome-wide KEAP1/NRF2-specific dependency screen.

For each gene in DepMap CRISPR (~18K genes), computes Cohen's d effect size
comparing KEAP1/NRF2-altered vs wild-type lines. Runs both pan-cancer pooled
and per qualifying lineage. Applies Benjamini-Hochberg FDR correction (q < 0.1).

Output: ranked synthetic lethal gene list with effect sizes and p-values.

Usage:
    uv run python -m keap1_nrf2_pancancer_dependency_atlas.02_pancancer_effect_sizes
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
PHASE1_DIR = (
    REPO_ROOT / "output" / "cancer" / "keap1-nrf2-pancancer-dependency-atlas" / "phase1"
)
OUTPUT_DIR = (
    REPO_ROOT / "output" / "cancer" / "keap1-nrf2-pancancer-dependency-atlas" / "phase2"
)

# Significance thresholds
FDR_THRESHOLD = 0.1
STRICT_FDR = 0.05
EFFECT_SIZE_THRESHOLD = 0.5  # |d| > 0.5 per plan
HIGH_CONFIDENCE_ES = 0.8  # |d| > 0.8 for high-confidence hits

MIN_SAMPLES = 3

# NRF2/KEAP1 pathway genes (to distinguish direct pathway members from SL hits)
NRF2_PATHWAY_GENES = {
    "KEAP1", "NFE2L2", "CUL3", "RBX1",
    "NQO1", "GCLM", "GCLC", "HMOX1", "TXNRD1", "AKR1C1", "AKR1C2", "AKR1C3",
    "GPX2", "GSR", "SLC7A11", "ABCC1", "ABCC2", "ABCG2",
    "ME1", "IDH1", "G6PD", "PGD", "TKT", "TALDO1",
    "FTH1", "FTL", "SQSTM1", "MAFG", "MAFK", "MAFF",
}

# Genes of interest from journal evidence
PRIORITY_GENES = {"UXS1", "ATR"}


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
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
    # Enforce monotonicity
    fdr_sorted = fdr[sorted_idx]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[sorted_idx] = fdr_sorted
    return np.minimum(fdr, 1.0)


def screen_one_context(
    altered_data: pd.DataFrame,
    wt_data: pd.DataFrame,
    crispr_cols: list[str],
    context_name: str,
) -> list[dict]:
    """Run genome-wide differential dependency for one context."""
    rows = []
    pvals = []

    for gene in crispr_cols:
        alt_vals = altered_data[gene].dropna().values
        wt_vals = wt_data[gene].dropna().values

        if len(alt_vals) < MIN_SAMPLES or len(wt_vals) < MIN_SAMPLES:
            continue

        _, pval = stats.mannwhitneyu(alt_vals, wt_vals, alternative="two-sided")
        d = cohens_d(alt_vals, wt_vals)

        rows.append({
            "cancer_type": context_name,
            "gene": gene,
            "cohens_d": round(d, 4),
            "p_value": float(pval),
            "n_altered": len(alt_vals),
            "n_wt": len(wt_vals),
            "median_dep_altered": round(float(np.median(alt_vals)), 4),
            "median_dep_wt": round(float(np.median(wt_vals)), 4),
        })
        pvals.append(pval)

    if pvals:
        fdrs = fdr_correction(np.array(pvals))
        for i, row in enumerate(rows):
            row["fdr"] = float(fdrs[i])

    return rows


def plot_volcano(results_ct: pd.DataFrame, context_name: str, out_dir: Path) -> None:
    if "fdr" not in results_ct.columns or len(results_ct) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    x = results_ct["cohens_d"].values
    y = -np.log10(results_ct["fdr"].values.clip(min=1e-50))

    sig = (results_ct["fdr"] < FDR_THRESHOLD) & (results_ct["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ax.scatter(x[~sig], y[~sig], c="#CCCCCC", s=5, alpha=0.5)

    gained = sig & (results_ct["cohens_d"] < 0)
    lost_dep = sig & (results_ct["cohens_d"] > 0)
    ax.scatter(x[gained], y[gained], c="#D95319", s=15, alpha=0.8, label="Gained dep. in altered")
    ax.scatter(x[lost_dep], y[lost_dep], c="#4DBEEE", s=15, alpha=0.8, label="Lost dep. in altered")

    # Label top gained hits
    top = results_ct[gained].nsmallest(10, "cohens_d")
    for _, row in top.iterrows():
        ax.annotate(row["gene"], (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                     fontsize=7, ha="right")

    # Highlight priority genes (UXS1, ATR)
    priority_mask = results_ct["gene"].isin(PRIORITY_GENES)
    if priority_mask.any():
        ax.scatter(x[priority_mask], y[priority_mask], c="none", edgecolors="red",
                   s=50, linewidths=1.5, label="Priority (UXS1/ATR)")
        for _, row in results_ct[priority_mask].iterrows():
            ax.annotate(row["gene"], (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                         fontsize=8, ha="left", fontweight="bold", color="red")

    # Highlight NRF2 pathway genes
    pathway_mask = results_ct["gene"].isin(NRF2_PATHWAY_GENES) & sig
    if pathway_mask.any():
        ax.scatter(x[pathway_mask], y[pathway_mask], c="none", edgecolors="green",
                   s=40, linewidths=1.0, label="NRF2 pathway")

    ax.axhline(-np.log10(FDR_THRESHOLD), color="grey", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)

    ax.set_xlabel("Cohen's d (KEAP1/NRF2-altered vs WT)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"KEAP1/NRF2 Dependency Screen: {context_name}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    safe = context_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(out_dir / f"volcano_{safe}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: Pan-Cancer KEAP1/NRF2 Genome-Wide Screen ===\n")

    # Load Phase 1
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "keap1_nrf2_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()
    print(f"  {len(classified)} lines, {len(qualifying_types)} qualifying cancer types")

    # Load CRISPR
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    crispr_cols = list(crispr.columns)
    print(f"  {len(crispr_cols)} genes, {len(crispr)} cell lines")

    merged = classified.join(crispr, how="inner")
    print(f"  {len(merged)} lines with both classification and CRISPR data")

    # Screen each qualifying cancer type + pan-cancer
    all_rows = []
    contexts = qualifying_types + ["Pan-cancer (pooled)"]

    for context in contexts:
        if context == "Pan-cancer (pooled)":
            ct_data = merged
        else:
            ct_data = merged[merged["OncotreeLineage"] == context]

        altered_lines = ct_data[ct_data["KEAP1_NRF2_altered"] == True]  # noqa: E712
        wt_lines = ct_data[ct_data["pathway_status"] == "WT"]
        print(f"  Screening {context} ({len(altered_lines)} altered, {len(wt_lines)} WT)...")

        rows = screen_one_context(altered_lines, wt_lines, crispr_cols, context)
        all_rows.extend(rows)

    all_results = pd.DataFrame(all_rows)
    print(f"\n  Total tests: {len(all_results)}")

    # Save full results
    all_results.to_csv(OUTPUT_DIR / "genomewide_all_results.csv", index=False)

    # Significant hits at primary threshold (FDR < 0.1, |d| > 0.5)
    sig_hits = all_results[
        (all_results["fdr"] < FDR_THRESHOLD) & (all_results["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ].copy()

    # Negative d = more essential in altered lines (gained dependency = synthetic lethal)
    gained = sig_hits[sig_hits["cohens_d"] < 0].sort_values("cohens_d")
    lost = sig_hits[sig_hits["cohens_d"] > 0].sort_values("cohens_d", ascending=False)

    print(f"  SL gained dependencies (FDR<{FDR_THRESHOLD}, |d|>{EFFECT_SIZE_THRESHOLD}): {len(gained)}")
    print(f"  Lost dependencies: {len(lost)}")

    # High-confidence hits (FDR < 0.05, |d| > 0.8)
    hc_hits = all_results[
        (all_results["fdr"] < STRICT_FDR) & (all_results["cohens_d"].abs() > HIGH_CONFIDENCE_ES)
    ]
    hc_gained = hc_hits[hc_hits["cohens_d"] < 0]
    print(f"  High-confidence SL hits (FDR<{STRICT_FDR}, |d|>{HIGH_CONFIDENCE_ES}): {len(hc_gained)}")

    # Check priority genes
    print(f"\nPriority gene results:")
    for gene in sorted(PRIORITY_GENES):
        gene_results = all_results[all_results["gene"] == gene]
        if len(gene_results) > 0:
            for _, row in gene_results.iterrows():
                sig_flag = " ***" if row.get("fdr", 1) < FDR_THRESHOLD and row["cohens_d"] < -EFFECT_SIZE_THRESHOLD else ""
                print(f"  {gene} [{row['cancer_type']}]: d={row['cohens_d']:.3f}, "
                      f"FDR={row.get('fdr', 'N/A'):.3e}{sig_flag}")
        else:
            print(f"  {gene}: not tested (insufficient samples)")

    # NRF2 pathway genes in hits (these are direct pathway, not novel SL)
    pathway_in_hits = set(gained["gene"]) & NRF2_PATHWAY_GENES
    novel_sl = gained[~gained["gene"].isin(NRF2_PATHWAY_GENES)]
    print(f"\n  NRF2 pathway genes in gained dependencies: {len(pathway_in_hits)}")
    if pathway_in_hits:
        print(f"    {', '.join(sorted(pathway_in_hits))}")
    print(f"  Novel SL candidates (excluding pathway): {len(novel_sl)}")

    # Top hits table
    print(f"\nTop SL hits (pan-cancer):")
    pancancer_gained = gained[gained["cancer_type"] == "Pan-cancer (pooled)"]
    for _, row in pancancer_gained.head(20).iterrows():
        label = ""
        if row["gene"] in PRIORITY_GENES:
            label = " [PRIORITY]"
        elif row["gene"] in NRF2_PATHWAY_GENES:
            label = " [NRF2 pathway]"
        print(f"  {row['gene']}: d={row['cohens_d']:.3f}, FDR={row['fdr']:.3e}, "
              f"n_alt={row['n_altered']}, n_wt={row['n_wt']}{label}")

    # Save significant hits
    gained.to_csv(OUTPUT_DIR / "sl_gained_dependencies.csv", index=False)
    lost.to_csv(OUTPUT_DIR / "lost_dependencies.csv", index=False)

    # Build ranked SL gene list (pan-cancer, excluding NRF2 pathway)
    pancancer_all = all_results[all_results["cancer_type"] == "Pan-cancer (pooled)"].copy()
    pancancer_all = pancancer_all.sort_values("cohens_d").reset_index(drop=True)
    pancancer_all["is_nrf2_pathway"] = pancancer_all["gene"].isin(NRF2_PATHWAY_GENES)
    pancancer_all["is_priority"] = pancancer_all["gene"].isin(PRIORITY_GENES)
    pancancer_all.to_csv(OUTPUT_DIR / "pancancer_ranked_genes.csv", index=False)

    # Volcano plots
    print("\nGenerating volcano plots...")
    for context in contexts:
        ct_res = all_results[all_results["cancer_type"] == context]
        plot_volcano(ct_res, context, OUTPUT_DIR)
    print(f"  Saved {len(contexts)} volcano plots")

    # Summary text
    summary_lines = [
        "=" * 70,
        "KEAP1/NRF2 Pan-Cancer Dependency Atlas - Phase 2: Effect Sizes",
        "=" * 70,
        "",
        f"Total tests: {len(all_results)}",
        f"Contexts screened: {', '.join(contexts)}",
        "",
        f"SL gained dependencies (FDR<{FDR_THRESHOLD}, |d|>{EFFECT_SIZE_THRESHOLD}): {len(gained)}",
        f"High-confidence SL (FDR<{STRICT_FDR}, |d|>{HIGH_CONFIDENCE_ES}): {len(hc_gained)}",
        f"Lost dependencies: {len(lost)}",
        f"NRF2 pathway genes in hits: {len(pathway_in_hits)}",
        f"Novel SL candidates: {len(novel_sl)}",
        "",
        "TOP PAN-CANCER SL HITS",
        "-" * 60,
    ]
    for _, row in pancancer_gained.head(30).iterrows():
        label = ""
        if row["gene"] in PRIORITY_GENES:
            label = " [PRIORITY]"
        elif row["gene"] in NRF2_PATHWAY_GENES:
            label = " [NRF2 pathway]"
        summary_lines.append(
            f"  {row['gene']}: d={row['cohens_d']:.3f}, FDR={row['fdr']:.3e}{label}"
        )

    summary_lines += [
        "",
        "PRIORITY GENE STATUS",
        "-" * 60,
    ]
    for gene in sorted(PRIORITY_GENES):
        gene_results = all_results[all_results["gene"] == gene]
        for _, row in gene_results.iterrows():
            summary_lines.append(
                f"  {gene} [{row['cancer_type']}]: d={row['cohens_d']:.3f}, FDR={row.get('fdr', 'N/A'):.3e}"
            )

    summary_lines.append("")

    with open(OUTPUT_DIR / "effect_sizes_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("  effect_sizes_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
