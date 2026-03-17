"""CRC KRAS allele-vs-allele and co-mutation interaction analyses.

Extends Phase 2a with head-to-head comparisons:
1. G12D vs G13D direct comparison
2. G12C vs non-G12C KRAS-mutant
3. PIK3CA co-mutation x KRAS interaction
4. MSI-H vs MSS within KRAS-mutant

Usage:
    uv run python -m bioagentics.models.crc_interactions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.crc_depmap import annotate_crc_lines, DEFAULT_DEPMAP_DIR
from bioagentics.data.gene_ids import load_depmap_matrix
from bioagentics.models.crc_dependency import _cohens_d, _fdr_correction

DEFAULT_DEST = REPO_ROOT / "output" / "crc-kras-dependencies" / "allele_comparison_results.json"

MIN_GROUP_SIZE = 3  # Minimum lines per group for testing


def _run_comparison(
    crispr: pd.DataFrame,
    group1_ids: list[str],
    group2_ids: list[str],
    label: str,
) -> dict:
    """Run Mann-Whitney comparison between two groups, return results dict."""
    print(f"\n{label}: {len(group1_ids)} vs {len(group2_ids)} lines")

    if len(group1_ids) < MIN_GROUP_SIZE or len(group2_ids) < MIN_GROUP_SIZE:
        print(f"  Skipped — insufficient sample size")
        return {"label": label, "skipped": True, "reason": "insufficient sample size",
                "n_group1": len(group1_ids), "n_group2": len(group2_ids)}

    results = []
    for gene in crispr.columns:
        g1 = crispr.loc[group1_ids, gene].dropna().values
        g2 = crispr.loc[group2_ids, gene].dropna().values
        if len(g1) < MIN_GROUP_SIZE or len(g2) < MIN_GROUP_SIZE:
            continue
        _, pval = stats.mannwhitneyu(g1, g2, alternative="two-sided")
        d = _cohens_d(g1, g2)
        results.append({
            "gene": gene,
            "pvalue": pval,
            "cohens_d": d,
            "mean_group1": float(g1.mean()),
            "mean_group2": float(g2.mean()),
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["fdr"] = _fdr_correction(df["pvalue"].values)
        sig = df[(df["fdr"] < 0.05) & (df["cohens_d"].abs() > 0.5)]
        top = df.nsmallest(50, "fdr")
        print(f"  Genes tested: {len(df)}")
        print(f"  Significant (FDR<0.05, |d|>0.5): {len(sig)}")
    else:
        sig = pd.DataFrame()
        top = pd.DataFrame()

    return {
        "label": label,
        "skipped": False,
        "n_group1": len(group1_ids),
        "n_group2": len(group2_ids),
        "n_genes_tested": len(df),
        "n_significant": len(sig),
        "top_results": top.to_dict(orient="records") if len(top) > 0 else [],
    }


def compute_interactions(depmap_dir: str | Path) -> dict:
    """Run all interaction analyses."""
    depmap_dir = Path(depmap_dir)

    classified = annotate_crc_lines(depmap_dir)
    crispr = load_depmap_matrix(depmap_dir / "CRISPRGeneEffect.csv")
    crc_ids = list(set(classified.index) & set(crispr.index))
    crispr = crispr.loc[crc_ids]
    classified = classified.loc[crc_ids]

    kras_mut = classified[classified["KRAS_allele"] != "WT"]

    results = {}

    # Analysis 1: G12D vs G13D
    g12d_ids = list(classified[classified["KRAS_allele"] == "G12D"].index)
    g13d_ids = list(classified[classified["KRAS_allele"] == "G13D"].index)
    results["g12d_vs_g13d"] = _run_comparison(
        crispr, g12d_ids, g13d_ids, "G12D vs G13D"
    )

    # Analysis 2: G12C vs non-G12C KRAS-mutant
    g12c_ids = list(classified[classified["KRAS_allele"] == "G12C"].index)
    non_g12c_mut_ids = list(
        kras_mut[kras_mut["KRAS_allele"] != "G12C"].index
    )
    results["g12c_vs_non_g12c"] = _run_comparison(
        crispr, g12c_ids, non_g12c_mut_ids, "G12C vs non-G12C KRAS-mutant"
    )

    # Analysis 3: PIK3CA co-mutation within KRAS-mutant
    pik3ca_co = list(kras_mut[kras_mut["PIK3CA_mutated"]].index)
    pik3ca_wt = list(kras_mut[~kras_mut["PIK3CA_mutated"]].index)
    results["pik3ca_interaction"] = _run_comparison(
        crispr, pik3ca_co, pik3ca_wt, "PIK3CA-co-mutant vs PIK3CA-WT (within KRAS-mut)"
    )

    # Analysis 4: MSI-H vs MSS within KRAS-mutant
    msi_h_ids = list(kras_mut[kras_mut["MSI_status"] == "MSI-H"].index)
    mss_ids = list(kras_mut[kras_mut["MSI_status"] == "MSS"].index)
    results["msi_interaction"] = _run_comparison(
        crispr, msi_h_ids, mss_ids, "MSI-H vs MSS (within KRAS-mut)"
    )

    # Summary
    results["summary"] = {
        "total_kras_mutant_with_crispr": len(kras_mut),
        "allele_counts": kras_mut["KRAS_allele"].value_counts().to_dict(),
        "pik3ca_co_mutation_rate": f"{kras_mut['PIK3CA_mutated'].mean():.0%}",
        "msi_h_in_kras_mut": len(msi_h_ids),
    }

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CRC KRAS allele interaction analyses",
    )
    parser.add_argument(
        "--depmap-dir", type=Path, default=DEFAULT_DEPMAP_DIR,
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_DEST,
    )
    args = parser.parse_args(argv)

    results = compute_interactions(args.depmap_dir)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.dest, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.dest}")


if __name__ == "__main__":
    main()
