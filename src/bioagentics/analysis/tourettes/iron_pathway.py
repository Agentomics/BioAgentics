"""Iron homeostasis pathway spatial profiling across CSTC nodes.

Maps iron metabolism gene expression across the CSTC circuit and tests
concordance with the 7T MRI iron depletion pattern described in
Brain Communications 2025 (circuit-wide iron depletion in caudate,
pallidum, STN, thalamus, red nucleus, substantia nigra).

Pipeline:
1. Extract iron pathway gene expression from AHBA across extended CSTC nodes
2. Compare expression pattern with 7T MRI iron depletion ranks
3. Spearman correlation concordance test
4. Generate iron pathway heatmap and concordance scatter plot

Usage:
    uv run python -m bioagentics.analysis.tourettes.iron_pathway
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.analysis.tourettes.ahba_spatial import (
    DONOR_IDS,
    fetch_gene_expression,
)
from bioagentics.config import REPO_ROOT
from bioagentics.data.tourettes.gene_sets import get_gene_set

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "cstc-circuit-expression-atlas"

# Extended CSTC structures for iron analysis (includes brainstem nuclei
# showing iron depletion in the 7T MRI study)
IRON_CSTC_STRUCTURES: dict[str, int] = {
    "caudate": 4229,
    "putamen": 4278,
    "GPe": 4249,
    "GPi": 4250,
    "STN": 10657,
    "thalamus": 4394,
    "substantia_nigra": 4290,
    "red_nucleus": 9512,
}

# 7T MRI iron depletion severity ranking (Brain Communications 2025)
# Higher rank = more severe iron depletion in TS vs controls
# Based on quantitative susceptibility mapping (QSM) differences.
MRI_IRON_DEPLETION_RANK: dict[str, int] = {
    "caudate": 6,           # Severe depletion, D1R correlation with tic severity
    "GPi": 5,               # Severe depletion (internal pallidum)
    "GPe": 4,               # Moderate-severe depletion
    "STN": 3,               # Moderate depletion
    "substantia_nigra": 7,  # Most severe depletion
    "thalamus": 2,          # Mild-moderate depletion
    "red_nucleus": 8,       # Most severe depletion
    "putamen": 1,           # Mild depletion
}


def fetch_iron_expression(cache: bool = True) -> pd.DataFrame:
    """Fetch AHBA expression for iron homeostasis genes.

    Uses the ahba_spatial module's fetch_gene_expression but targeted
    at the iron gene set.
    """
    iron_genes = get_gene_set("iron_homeostasis")
    gene_list = sorted(iron_genes.keys())
    logger.info("Fetching expression for %d iron pathway genes: %s",
                len(gene_list), ", ".join(gene_list))

    expr_df = fetch_gene_expression(gene_list, donors=DONOR_IDS, cache=cache)
    return expr_df


def compute_iron_enrichment(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute regional expression enrichment for iron pathway genes.

    Returns a gene x region matrix of z-scored expression values.
    """
    if expr_df.empty:
        return pd.DataFrame()

    # Filter to iron-relevant CSTC regions
    iron_regions = set(IRON_CSTC_STRUCTURES.keys())
    filtered = expr_df[expr_df["cstc_region"].isin(iron_regions)]

    if filtered.empty:
        logger.warning("No samples found in iron-relevant CSTC regions")
        return pd.DataFrame()

    # Average across donors
    avg = (
        filtered.groupby(["gene_symbol", "cstc_region"])["mean_zscore"]
        .mean()
        .reset_index()
    )

    # Pivot to gene x region
    pivot = avg.pivot(index="gene_symbol", columns="cstc_region", values="mean_zscore")

    # Reorder columns by circuit position
    region_order = list(IRON_CSTC_STRUCTURES.keys())
    available = [r for r in region_order if r in pivot.columns]
    return pivot[available]


def concordance_test(enrichment: pd.DataFrame) -> dict:
    """Test concordance between iron gene expression and MRI iron depletion.

    For each iron gene, compute Spearman correlation between its regional
    expression rank and the MRI iron depletion rank. Also compute the
    mean expression profile correlation.
    """
    from scipy import stats

    if enrichment.empty:
        return {"error": "No enrichment data"}

    # Get regions present in both expression and MRI data
    shared_regions = [r for r in enrichment.columns if r in MRI_IRON_DEPLETION_RANK]
    if len(shared_regions) < 3:
        return {"error": f"Need >= 3 shared regions, found {len(shared_regions)}"}

    mri_ranks = [MRI_IRON_DEPLETION_RANK[r] for r in shared_regions]

    # Per-gene correlations
    gene_results: list[dict] = []
    for gene, row in enrichment.iterrows():
        expr_values = [row[r] for r in shared_regions if pd.notna(row.get(r))]
        valid_mri = [mri_ranks[i] for i, r in enumerate(shared_regions) if pd.notna(row.get(r))]

        if len(expr_values) >= 3:
            rho, pval = stats.spearmanr(expr_values, valid_mri)
            gene_results.append({
                "gene_symbol": gene,
                "spearman_rho": float(rho),
                "p_value": float(pval),
                "n_regions": len(expr_values),
            })

    # Mean expression profile across all iron genes
    mean_expr = enrichment[shared_regions].mean(axis=0)
    mean_values = [mean_expr[r] for r in shared_regions]
    overall_rho, overall_p = stats.spearmanr(mean_values, mri_ranks)

    return {
        "overall_spearman_rho": float(overall_rho),
        "overall_p_value": float(overall_p),
        "concordant": bool(overall_p < 0.05),
        "n_regions": len(shared_regions),
        "regions_tested": shared_regions,
        "per_gene_results": gene_results,
    }


def generate_iron_heatmap(
    enrichment: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Generate heatmap of iron pathway gene expression across CSTC nodes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        enrichment,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        xticklabels=[r.replace("_", " ").title() for r in enrichment.columns],
    )
    ax.set_title("Iron Homeostasis Gene Expression Across CSTC Nodes", fontsize=13)
    ax.set_ylabel("Gene")
    ax.set_xlabel("CSTC Region")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_concordance_plot(
    enrichment: pd.DataFrame,
    concordance: dict,
    output_path: Path,
) -> Path:
    """Generate scatter plot of mean iron gene expression vs MRI iron depletion."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    regions = concordance.get("regions_tested", [])
    if not regions:
        return output_path

    mean_expr = enrichment[regions].mean(axis=0)
    mri_ranks = [MRI_IRON_DEPLETION_RANK[r] for r in regions]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(mri_ranks, [mean_expr[r] for r in regions], s=80, c="steelblue", edgecolors="black")

    for r, mri_r in zip(regions, mri_ranks):
        ax.annotate(
            r.replace("_", " ").title(),
            (mri_r, mean_expr[r]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=9,
        )

    rho = concordance.get("overall_spearman_rho", 0)
    p = concordance.get("overall_p_value", 1)
    ax.set_title(f"Iron Gene Expression vs MRI Iron Depletion\n"
                 f"Spearman rho={rho:.3f}, p={p:.4f}", fontsize=12)
    ax.set_xlabel("MRI Iron Depletion Severity Rank")
    ax.set_ylabel("Mean Iron Gene Expression (z-score)")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_analysis(
    output_dir: Path = OUTPUT_DIR,
    cache: bool = True,
) -> dict:
    """Run full iron homeostasis pathway profiling analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fetch expression
    expr_df = fetch_iron_expression(cache=cache)
    if expr_df.empty:
        return {"error": "No expression data retrieved"}

    # Compute enrichment
    enrichment = compute_iron_enrichment(expr_df)
    if enrichment.empty:
        return {"error": "No enrichment data computed"}

    # Concordance test
    concordance = concordance_test(enrichment)

    # Save outputs
    enrichment_path = output_dir / "iron_enrichment.csv"
    enrichment.to_csv(enrichment_path)

    concordance_path = output_dir / "iron_concordance.json"
    with open(concordance_path, "w") as f:
        json.dump(concordance, f, indent=2)

    heatmap_path = output_dir / "iron_heatmap.png"
    generate_iron_heatmap(enrichment, heatmap_path)

    scatter_path = output_dir / "iron_concordance_plot.png"
    generate_concordance_plot(enrichment, concordance, scatter_path)

    logger.info("Iron pathway analysis complete")
    return {
        "n_genes": enrichment.shape[0],
        "n_regions": enrichment.shape[1],
        "concordance": concordance,
        "outputs": {
            "enrichment_csv": str(enrichment_path),
            "concordance_json": str(concordance_path),
            "heatmap_png": str(heatmap_path),
            "concordance_plot_png": str(scatter_path),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Iron homeostasis pathway spatial profiling across CSTC"
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = run_analysis(output_dir=args.output, cache=not args.no_cache)

    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    conc = results["concordance"]
    print(f"\nIron pathway analysis complete")
    print(f"  Genes profiled: {results['n_genes']}")
    print(f"  Regions: {results['n_regions']}")
    print(f"  MRI concordance: rho={conc.get('overall_spearman_rho', 0):.3f}, "
          f"p={conc.get('overall_p_value', 1):.4f}")
    if conc.get("concordant"):
        print("  ** Significant concordance with 7T MRI iron depletion pattern **")


if __name__ == "__main__":
    main()
