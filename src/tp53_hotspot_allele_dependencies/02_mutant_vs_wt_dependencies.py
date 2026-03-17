"""Phase 2a: TP53 mutant vs WT genome-wide dependency screen.

For each powered cancer type and pan-cancer, runs Mann-Whitney U + Cohen's d
on every gene in CRISPRGeneEffect to identify TP53-mutant-specific dependencies.

CRITICAL POSITIVE CONTROL: MDM2 must be more essential in TP53-WT than mutant
(Cohen's d > 0, positive = mutant less dependent). This is the most well-established
TP53-related dependency in cancer biology.

Usage:
    uv run python -m tp53_hotspot_allele_dependencies.02_mutant_vs_wt_dependencies
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
OUTPUT_DIR = REPO_ROOT / "output" / "tp53_hotspot_allele_dependencies"
FIG_DIR = OUTPUT_DIR / "figures"

MIN_N = 10  # minimum samples per group
FDR_THRESHOLD = 0.05
EFFECT_THRESHOLD = 0.3  # |Cohen's d|

# Genes to highlight on plots
HIGHLIGHT_GENES = {
    "MDM2", "MDM4", "USP7",
    "CHEK1", "WEE1", "ATR",
    "CDKN1A", "CDKN2A",
    "TP53", "TP53BP1",
    "HSP90AA1", "HSP90AB1",
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
            "median_dep_mutant": float(np.median(mut_arr)),
            "median_dep_wt": float(np.median(wt_arr)),
            "cohens_d": d,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "mann_whitney_U": float(stat),
            "p_value": pval,
        })

    df = pd.DataFrame(results)
    if len(df) == 0:
        return df

    reject, fdr, _, _ = multipletests(df["p_value"], method="fdr_bh")
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
    ax.set_xlabel("Cohen's d (mutant − WT)\n← mutant more essential | WT more essential →")
    ax.set_ylabel("−log10(FDR)")
    ax.set_title(f"TP53 Mutant vs WT: {cancer_type}")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def check_mdm2_control(screen: pd.DataFrame, cancer_type: str) -> dict | None:
    """Check MDM2 positive control: must be more essential in TP53-WT."""
    mdm2_row = screen[screen["gene"] == "MDM2"]
    if mdm2_row.empty:
        print(f"  WARNING: MDM2 not found in {cancer_type} results")
        return None

    r = mdm2_row.iloc[0]
    # MDM2 d > 0 means mutant is LESS dependent (higher score) than WT
    passes = bool(r["cohens_d"] > 0.3 and r["fdr"] < FDR_THRESHOLD)
    result = {
        "cohens_d": round(r["cohens_d"], 4),
        "fdr": float(r["fdr"]),
        "median_dep_mutant": round(r["median_dep_mutant"], 4),
        "median_dep_wt": round(r["median_dep_wt"], 4),
        "passes": passes,
    }

    status = "PASS" if passes else "WARN"
    print(f"  MDM2 control: d={r['cohens_d']:.3f}, FDR={r['fdr']:.2e} [{status}]")
    return result


def run_analysis_for_group(
    crispr: pd.DataFrame, classified: pd.DataFrame,
    cancer_type: str, cancer_mask: pd.Series | None,
) -> dict | None:
    """Run mutant-vs-WT screen for one cancer type (or pan-cancer)."""
    if cancer_mask is not None:
        subset = classified[cancer_mask]
    else:
        subset = classified

    mutant_ids = list(set(subset[subset["TP53_mutated"]].index) & set(crispr.index))
    wt_ids = list(set(subset[~subset["TP53_mutated"]].index) & set(crispr.index))

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

    # MDM2 positive control
    mdm2_result = check_mdm2_control(screen, cancer_type)

    # Also check MDM4
    mdm4_row = screen[screen["gene"] == "MDM4"]
    mdm4_result = None
    if not mdm4_row.empty:
        r = mdm4_row.iloc[0]
        mdm4_result = {
            "cohens_d": round(r["cohens_d"], 4),
            "fdr": float(r["fdr"]),
        }

    # Check highlighted genes
    highlighted = {}
    for gene in HIGHLIGHT_GENES:
        row = screen[screen["gene"] == gene]
        if not row.empty:
            r = row.iloc[0]
            highlighted[gene] = {
                "cohens_d": round(r["cohens_d"], 4),
                "fdr": float(r["fdr"]),
            }

    n_sig = int(screen["significant"].sum())

    # Top hits (most significant by effect size)
    top_neg = screen[screen["significant"]].head(10)  # sorted ascending by d
    top_pos = screen[screen["significant"]].tail(10)
    top_hits = []
    for _, row in pd.concat([top_neg, top_pos]).drop_duplicates("gene").iterrows():
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
        "positive_control_MDM2": mdm2_result,
        "MDM4": mdm4_result,
        "highlighted_genes": highlighted,
        "top_hits": top_hits,
    }


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading classified cell lines...")
    classified = pd.read_csv(OUTPUT_DIR / "tp53_classified_lines.csv", index_col=0)
    classified["TP53_mutated"] = classified["TP53_mutated"].astype(bool)

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
        print(f"  {result['n_significant']} significant genes "
              f"(FDR<{FDR_THRESHOLD}, |d|>{EFFECT_THRESHOLD})")

    # Per cancer type
    for ct in powered_types:
        print(f"\n--- {ct} ---")
        mask = classified["OncotreePrimaryDisease"] == ct
        result = run_analysis_for_group(crispr, classified, ct, mask)
        if result:
            all_results.append(result)
            print(f"  {result['n_significant']} significant genes")

    # Save summary
    out_summary = OUTPUT_DIR / "mutant_vs_wt_summary.json"
    with open(out_summary, "w") as f:
        json.dump({"analyses": all_results}, f, indent=2)
    print(f"\nSaved summary to {out_summary.name}")


if __name__ == "__main__":
    main()
