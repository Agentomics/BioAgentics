"""CRC KRAS allele-specific differential dependency analysis.

For each KRAS allele group (>=5 lines) vs KRAS-WT, computes differential
CRISPR dependency using Mann-Whitney U, Cohen's d, and BH-FDR correction.

Usage:
    uv run python -m bioagentics.models.crc_dependency
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.crc_depmap import annotate_crc_lines, DEFAULT_DEPMAP_DIR
from bioagentics.data.gene_ids import load_depmap_matrix

DEFAULT_DEST = REPO_ROOT / "output" / "crc-kras-dependencies" / "allele_dependency_results.json"

# Allele groups with enough lines for statistical testing (>=5)
TESTABLE_ALLELES = ["G12D", "G13D", "G12V", "G12C"]
# Allele groups for descriptive statistics only
DESCRIPTIVE_ALLELES = ["G12A", "Q61H", "A146T"]


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def _fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    ranked = np.argsort(pvalues)
    fdr = np.empty(n)
    for i, rank_idx in enumerate(reversed(ranked)):
        rank = n - i
        if i == 0:
            fdr[rank_idx] = pvalues[rank_idx]
        else:
            fdr[rank_idx] = min(pvalues[rank_idx] * n / rank, fdr[ranked[n - i]])
    return np.minimum(fdr, 1.0)


def _differential_test(
    crispr: pd.DataFrame,
    allele_ids: list[str],
    wt_ids: list[str],
) -> pd.DataFrame:
    """Run Mann-Whitney U for each gene: allele group vs WT."""
    results = []
    for gene in crispr.columns:
        allele_scores = crispr.loc[allele_ids, gene].dropna().values
        wt_scores = crispr.loc[wt_ids, gene].dropna().values
        if len(allele_scores) < 3 or len(wt_scores) < 3:
            continue
        stat, pval = stats.mannwhitneyu(
            allele_scores, wt_scores, alternative="two-sided"
        )
        d = _cohens_d(allele_scores, wt_scores)
        results.append({
            "gene": gene,
            "pvalue": pval,
            "cohens_d": d,
            "mean_allele": float(allele_scores.mean()),
            "mean_wt": float(wt_scores.mean()),
            "n_allele": len(allele_scores),
            "n_wt": len(wt_scores),
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["fdr"] = _fdr_correction(df["pvalue"].values)
    return df


def _descriptive_stats(
    crispr: pd.DataFrame,
    allele_ids: list[str],
) -> list[dict]:
    """Compute descriptive stats for small allele groups (no testing)."""
    results = []
    for gene in crispr.columns:
        scores = crispr.loc[allele_ids, gene].dropna().values
        if len(scores) < 1:
            continue
        results.append({
            "gene": gene,
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "n": len(scores),
        })
    # Sort by mean (most essential first, more negative = more essential)
    results.sort(key=lambda x: x["mean"])
    return results[:50]


def compute_allele_dependencies(
    depmap_dir: str | Path,
) -> dict:
    """Run allele-specific dependency analysis.

    Returns dict with per-allele results, all-KRAS-mut vs WT, and summary.
    """
    depmap_dir = Path(depmap_dir)

    # Load classified CRC lines
    classified = annotate_crc_lines(depmap_dir)

    # Load CRISPR dependency data, filter to CRC lines
    crispr = load_depmap_matrix(depmap_dir / "CRISPRGeneEffect.csv")
    crc_ids_with_crispr = list(set(classified.index) & set(crispr.index))
    crispr = crispr.loc[crc_ids_with_crispr]
    classified = classified.loc[crc_ids_with_crispr]

    print(f"CRC lines with CRISPR data: {len(classified)}")
    print(f"Genes in CRISPR matrix: {crispr.shape[1]}")

    # KRAS-WT reference group (includes BRAF V600E lines — they are KRAS-WT)
    wt_ids = list(classified[classified["KRAS_allele"] == "WT"].index)
    # All KRAS-mutant (any allele)
    all_mut_ids = list(classified[classified["KRAS_allele"] != "WT"].index)

    print(f"KRAS-WT reference: {len(wt_ids)} lines")
    print(f"KRAS-mutant total: {len(all_mut_ids)} lines")

    results = {"allele_comparisons": {}, "descriptive_only": {}, "summary": {}}

    # Per-allele testing (>=5 lines)
    for allele in TESTABLE_ALLELES:
        allele_ids = list(classified[classified["KRAS_allele"] == allele].index)
        n = len(allele_ids)
        print(f"\n{allele}: {n} lines")
        if n < 5:
            print(f"  Skipping {allele} — fewer than 5 lines")
            results["descriptive_only"][allele] = _descriptive_stats(crispr, allele_ids)
            continue

        df = _differential_test(crispr, allele_ids, wt_ids)
        sig = df[(df["fdr"] < 0.05) & (df["cohens_d"].abs() > 0.5)]
        print(f"  Genes tested: {len(df)}")
        print(f"  Significant (FDR<0.05, |d|>0.5): {len(sig)}")

        # Top 50 by FDR
        top = df.nsmallest(50, "fdr")
        results["allele_comparisons"][allele] = {
            "n_lines": n,
            "n_genes_tested": len(df),
            "n_significant": len(sig),
            "top_dependencies": top.to_dict(orient="records"),
        }

    # All KRAS-mutant vs WT
    print(f"\nAll KRAS-mutant vs WT: {len(all_mut_ids)} vs {len(wt_ids)}")
    df_all = _differential_test(crispr, all_mut_ids, wt_ids)
    sig_all = df_all[(df_all["fdr"] < 0.05) & (df_all["cohens_d"].abs() > 0.5)]
    print(f"  Genes tested: {len(df_all)}")
    print(f"  Significant (FDR<0.05, |d|>0.5): {len(sig_all)}")

    top_all = df_all.nsmallest(50, "fdr")
    results["allele_comparisons"]["all_KRAS_mut"] = {
        "n_lines": len(all_mut_ids),
        "n_genes_tested": len(df_all),
        "n_significant": len(sig_all),
        "top_dependencies": top_all.to_dict(orient="records"),
    }

    # Descriptive stats for small groups
    for allele in DESCRIPTIVE_ALLELES:
        allele_ids = list(classified[classified["KRAS_allele"] == allele].index)
        n = len(allele_ids)
        print(f"\n{allele} (descriptive only): {n} lines")
        if n > 0:
            results["descriptive_only"][allele] = _descriptive_stats(crispr, allele_ids)

    # Summary
    results["summary"] = {
        "total_crc_lines": len(classified),
        "total_kras_wt": len(wt_ids),
        "total_kras_mutant": len(all_mut_ids),
        "allele_counts": classified["KRAS_allele"].value_counts().to_dict(),
        "total_genes_in_crispr": int(crispr.shape[1]),
    }

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CRC KRAS allele-specific differential dependency analysis",
    )
    parser.add_argument(
        "--depmap-dir", type=Path, default=DEFAULT_DEPMAP_DIR,
        help=f"DepMap data directory (default: {DEFAULT_DEPMAP_DIR})",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
        help="Output JSON path",
    )
    args = parser.parse_args(argv)

    results = compute_allele_dependencies(args.depmap_dir)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dest, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.dest}")


if __name__ == "__main__":
    main()
