"""Phase 2a: PIK3CA mutant vs WT genome-wide dependency screen per cancer type.

For each powered cancer type and pan-cancer, runs Mann-Whitney U + Cohen's d
on every gene in CRISPRGeneEffect to identify PIK3CA-mutant-specific
dependencies. PIK3CA itself serves as a positive control.

Usage:
    uv run python -m pik3ca_allele_dependencies.02_mutant_vs_wt_dependencies
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "pik3ca_allele_dependencies"
FIG_DIR = OUTPUT_DIR / "figures"

MIN_N = 5  # minimum samples per group
FDR_THRESHOLD = 0.05
EFFECT_THRESHOLD = 0.3  # |Cohen's d|

# Genes to highlight on plots
HIGHLIGHT_GENES = {
    "PIK3CA", "PIK3R1", "AKT1", "AKT2", "MTOR", "PTEN",
    "KRAS", "BRAF", "MAP2K1", "MAP2K2",
    "CDK4", "CDK6", "RB1", "CCND1",
}


def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """Cohen's d effect size (g1 - g2)."""
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled)


def cohens_d_ci(d: float, n1: int, n2: int, alpha: float = 0.05) -> tuple[float, float]:
    """Approximate CI for Cohen's d using normal approximation."""
    se = np.sqrt(1 / n1 + 1 / n2 + d**2 / (2 * (n1 + n2)))
    z = stats.norm.ppf(1 - alpha / 2)
    return float(d - z * se), float(d + z * se)


def genome_wide_screen(
    crispr: pd.DataFrame, mutant_ids: list[str], wt_ids: list[str], label: str,
) -> pd.DataFrame:
    """Run Mann-Whitney U for every gene: mutant vs WT."""
    print(f"  Screening {label}: {len(mutant_ids)} mutant vs {len(wt_ids)} WT "
          f"({len(crispr.columns)} genes)...")

    results = []
    n_genes = len(crispr.columns)

    for i, gene in enumerate(crispr.columns):
        if (i + 1) % 3000 == 0:
            print(f"    {i + 1}/{n_genes} genes...", file=sys.stderr)

        mut_vals = crispr.loc[mutant_ids, gene].dropna()
        wt_vals = crispr.loc[wt_ids, gene].dropna()

        if len(mut_vals) < MIN_N or len(wt_vals) < MIN_N:
            continue

        mut_arr = mut_vals.values.astype(float)
        wt_arr = wt_vals.values.astype(float)
        try:
            stat, pval = stats.mannwhitneyu(mut_arr, wt_arr, alternative="two-sided")
        except ValueError:
            continue

        d = cohens_d(mut_arr, wt_arr)
        ci_lo, ci_hi = cohens_d_ci(d, len(mut_arr), len(wt_arr))

        results.append({
            "gene": gene,
            "n_mutant": len(mut_arr),
            "n_wt": len(wt_arr),
            "median_mutant": float(np.median(mut_arr)),
            "median_wt": float(np.median(wt_arr)),
            "cohens_d": d,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "mannwhitney_U": float(stat),
            "mannwhitney_p": pval,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    reject, fdr, _, _ = multipletests(df["mannwhitney_p"], method="fdr_bh")
    df["fdr"] = fdr
    df["significant"] = reject & (df["cohens_d"].abs() > EFFECT_THRESHOLD)

    return df.sort_values("cohens_d")


def plot_volcano(screen: pd.DataFrame, cancer_type: str, out_path: Path) -> None:
    """Volcano plot: Cohen's d vs -log10(FDR)."""
    if len(screen) == 0:
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    neglog_fdr = -np.log10(screen["fdr"].clip(lower=1e-50))
    d = screen["cohens_d"]

    sig = screen["fdr"] < FDR_THRESHOLD
    strong = sig & (d.abs() > EFFECT_THRESHOLD)

    ax.scatter(d[~sig], neglog_fdr[~sig], alpha=0.15, s=8, color="gray", label="NS")
    ax.scatter(d[sig & ~strong], neglog_fdr[sig & ~strong], alpha=0.4, s=12,
               color="#4DBEEE", label=f"FDR<{FDR_THRESHOLD}")
    ax.scatter(d[strong], neglog_fdr[strong], alpha=0.6, s=20,
               color="#D95319", label=f"FDR<{FDR_THRESHOLD} & |d|>{EFFECT_THRESHOLD}")

    for _, row in screen.iterrows():
        if row["gene"] in HIGHLIGHT_GENES:
            x = row["cohens_d"]
            y = -np.log10(max(row["fdr"], 1e-50))
            ax.annotate(row["gene"], (x, y), fontsize=7, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")
            ax.scatter([x], [y], s=50, edgecolors="black", facecolors="none",
                       linewidths=1.5, zorder=5)

    ax.axhline(-np.log10(FDR_THRESHOLD), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_THRESHOLD, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(EFFECT_THRESHOLD, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Cohen's d (mutant - WT)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"PIK3CA Mutant vs WT: {cancer_type}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_analysis_for_group(
    crispr: pd.DataFrame, classified: pd.DataFrame,
    cancer_type: str, cancer_mask: pd.Series | None,
) -> dict | None:
    """Run mutant-vs-WT screen for one cancer type (or pan-cancer if mask is None)."""
    if cancer_mask is not None:
        subset = classified[cancer_mask]
    else:
        subset = classified

    mutant_ids = list(set(subset[subset["PIK3CA_mutated"]].index) & set(crispr.index))
    wt_ids = list(set(subset[~subset["PIK3CA_mutated"]].index) & set(crispr.index))

    if len(mutant_ids) < MIN_N or len(wt_ids) < MIN_N:
        return None

    screen = genome_wide_screen(crispr, mutant_ids, wt_ids, cancer_type)
    if len(screen) == 0:
        return None

    # Save full results
    safe_name = cancer_type.lower().replace(" ", "_").replace("/", "_")
    out_csv = OUTPUT_DIR / f"mutant_vs_wt_{safe_name}.csv"
    screen.to_csv(out_csv, index=False)

    # Volcano plot
    plot_volcano(screen, cancer_type, FIG_DIR / f"volcano_mvw_{safe_name}.png")

    # Positive control: PIK3CA itself
    pik3ca_row = screen[screen["gene"] == "PIK3CA"]
    pik3ca_result = None
    if not pik3ca_row.empty:
        r = pik3ca_row.iloc[0]
        pik3ca_result = {
            "cohens_d": round(r["cohens_d"], 4),
            "fdr": float(r["fdr"]),
            "passes_positive_control": bool(r["cohens_d"] < -0.5),  # mutant more dependent
        }

    n_sig = int(screen["significant"].sum())
    top_hits = []
    sig_genes = screen[screen["significant"]].head(20)
    for _, row in sig_genes.iterrows():
        top_hits.append({
            "gene": row["gene"],
            "cohens_d": round(row["cohens_d"], 4),
            "fdr": float(row["fdr"]),
        })

    return {
        "cancer_type": cancer_type,
        "n_mutant": len(mutant_ids),
        "n_wt": len(wt_ids),
        "n_genes_tested": len(screen),
        "n_significant": n_sig,
        "positive_control_PIK3CA": pik3ca_result,
        "top_hits": top_hits,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading classified cell lines...")
    classified = pd.read_csv(OUTPUT_DIR / "pik3ca_classified_lines.csv", index_col=0)
    classified["PIK3CA_mutated"] = classified["PIK3CA_mutated"].astype(bool)

    print("Loading CRISPR dependency data...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    print(f"  {len(crispr)} cell lines, {len(crispr.columns)} genes")

    # Keep only lines present in both
    common = list(set(classified.index) & set(crispr.index))
    classified = classified.loc[classified.index.isin(common)]
    print(f"  {len(classified)} lines with both classification and CRISPR data")

    # Load cancer type summary to find powered types
    with open(OUTPUT_DIR / "cancer_type_summary.json") as f:
        summary = json.load(f)

    powered_types = [
        ct["cancer_type"]
        for ct in summary["cancer_types"]
        if ct["powered_mutant_vs_wt"]
    ]
    print(f"\n{len(powered_types)} powered cancer types + pan-cancer")

    all_results = []

    # Pan-cancer first (most statistical power)
    print("\n--- Pan-cancer ---")
    result = run_analysis_for_group(crispr, classified, "Pan-cancer", None)
    if result:
        all_results.append(result)
        pc = result.get("positive_control_PIK3CA")
        if pc:
            status = "PASS" if pc["passes_positive_control"] else "WARN"
            print(f"  Positive control PIK3CA: d={pc['cohens_d']:.3f}, "
                  f"FDR={pc['fdr']:.2e} [{status}]")
        print(f"  {result['n_significant']} significant genes (FDR<{FDR_THRESHOLD}, "
              f"|d|>{EFFECT_THRESHOLD})")

    # Per cancer type
    for ct in powered_types:
        print(f"\n--- {ct} ---")
        mask = classified["OncotreePrimaryDisease"] == ct
        result = run_analysis_for_group(crispr, classified, ct, mask)
        if result:
            all_results.append(result)
            pc = result.get("positive_control_PIK3CA")
            if pc:
                status = "PASS" if pc["passes_positive_control"] else "WARN"
                print(f"  Positive control PIK3CA: d={pc['cohens_d']:.3f}, "
                      f"FDR={pc['fdr']:.2e} [{status}]")
            print(f"  {result['n_significant']} significant genes")

    # Save summary
    out_summary = OUTPUT_DIR / "mutant_vs_wt_summary.json"
    with open(out_summary, "w") as f:
        json.dump({"analyses": all_results}, f, indent=2)
    print(f"\nSaved summary to {out_summary.name}")


if __name__ == "__main__":
    main()
