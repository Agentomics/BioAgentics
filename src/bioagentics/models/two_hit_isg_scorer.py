"""Interferon Signature Gene (ISG) scorer using the AGS 6-gene panel.

Computes an interferon signature score for PANS/PANDAS samples using the
Aicardi-Goutieres syndrome (AGS) canonical 6-gene panel. Compares PANS ISG
scores to control groups (healthy, primary OCD, autoimmune encephalitis).

AGS 6-gene panel (Rice et al. 2013, Crow & Manel 2015):
  IFI27, IFI44L, IFIT1, ISG15, RSAD2, SIGLEC1

Scoring method: median z-score of the 6 ISG genes per sample, a standard
approach for interferon signature quantification in clinical genomics.

Usage:
    uv run python -m bioagentics.models.two_hit_isg_scorer [--expression FILE] [--dest DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/pandas_pans/two-hit-interferonopathy-model")

# AGS canonical 6-gene ISG panel
ISG_PANEL = ["IFI27", "IFI44L", "IFIT1", "ISG15", "RSAD2", "SIGLEC1"]

# Extended ISG panel for robustness (if some genes missing from dataset)
ISG_PANEL_EXTENDED = [
    *ISG_PANEL,
    "MX1", "OAS1", "IFI44", "IFIT3", "IFI6", "HERC5", "LY6E", "USP18",
]

# Minimum ISG genes required for a valid score
MIN_ISG_GENES = 3


def compute_isg_scores(
    expression_df: pd.DataFrame,
    gene_col: str | None = None,
    sample_cols: list[str] | None = None,
    panel: list[str] | None = None,
) -> pd.DataFrame:
    """Compute ISG scores from an expression matrix.

    Args:
        expression_df: Expression matrix. Genes in rows, samples in columns
            (or genes as a column specified by gene_col).
        gene_col: Column name containing gene symbols. If None, uses the index.
        sample_cols: Which columns are samples. If None, all numeric columns.
        panel: ISG gene panel to use. Defaults to AGS 6-gene panel.

    Returns:
        DataFrame with columns: sample, isg_score, n_isg_genes, isg_genes_found.
    """
    if panel is None:
        panel = ISG_PANEL

    df = expression_df.copy()

    # Normalize gene identifiers to index
    if gene_col is not None:
        df = df.set_index(gene_col)
    df.index = df.index.astype(str).str.upper().str.strip()

    # Identify sample columns (numeric)
    if sample_cols is None:
        sample_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not sample_cols:
        logger.warning("No numeric sample columns found")
        return pd.DataFrame(columns=["sample", "isg_score", "n_isg_genes", "isg_genes_found"])

    # Find which ISG genes are present
    panel_upper = [g.upper() for g in panel]
    found_genes = [g for g in panel_upper if g in df.index]
    missing_genes = [g for g in panel_upper if g not in df.index]

    if missing_genes:
        logger.info("ISG genes not in dataset: %s", ", ".join(missing_genes))

    if len(found_genes) < MIN_ISG_GENES:
        logger.warning("Only %d/%d ISG genes found (need >= %d). "
                       "Trying extended panel.", len(found_genes), len(panel), MIN_ISG_GENES)
        ext_upper = [g.upper() for g in ISG_PANEL_EXTENDED]
        found_genes = [g for g in ext_upper if g in df.index]
        if len(found_genes) < MIN_ISG_GENES:
            logger.error("Insufficient ISG genes (%d found). Cannot compute scores.",
                         len(found_genes))
            return pd.DataFrame(columns=["sample", "isg_score", "n_isg_genes", "isg_genes_found"])

    logger.info("Computing ISG scores using %d genes: %s",
                len(found_genes), ", ".join(found_genes))

    # Extract ISG expression sub-matrix
    isg_expr = df.loc[found_genes, sample_cols].astype(float)

    # Z-score normalize each gene across samples
    gene_means = isg_expr.mean(axis=1)
    gene_stds = isg_expr.std(axis=1).replace(0, 1)  # avoid div by zero
    isg_z = isg_expr.sub(gene_means, axis=0).div(gene_stds, axis=0)

    # ISG score per sample = median z-score across ISG genes
    scores = isg_z.median(axis=0)

    results = pd.DataFrame({
        "sample": scores.index,
        "isg_score": scores.values,
        "n_isg_genes": len(found_genes),
        "isg_genes_found": ", ".join(found_genes),
    })

    return results


def compare_groups(
    scores_df: pd.DataFrame,
    group_col: str = "group",
    score_col: str = "isg_score",
) -> dict:
    """Compare ISG scores between groups using Wilcoxon rank-sum test.

    Args:
        scores_df: DataFrame with isg_score and group columns.
        group_col: Column identifying sample groups.
        score_col: Column with ISG scores.

    Returns:
        Dict with pairwise comparison statistics.
    """
    groups = scores_df[group_col].unique()
    comparisons = {}
    summary = {}

    for g in groups:
        g_scores = scores_df.loc[scores_df[group_col] == g, score_col].dropna()
        summary[str(g)] = {
            "n": int(len(g_scores)),
            "mean": float(g_scores.mean()) if len(g_scores) > 0 else None,
            "median": float(g_scores.median()) if len(g_scores) > 0 else None,
            "std": float(g_scores.std()) if len(g_scores) > 1 else None,
        }

    # Pairwise comparisons
    for i, g1 in enumerate(groups):
        for g2 in groups[i + 1:]:
            s1 = scores_df.loc[scores_df[group_col] == g1, score_col].dropna()
            s2 = scores_df.loc[scores_df[group_col] == g2, score_col].dropna()

            if len(s1) < 2 or len(s2) < 2:
                continue

            stat, pval = stats.mannwhitneyu(s1, s2, alternative="two-sided")
            effect_size = abs(s1.median() - s2.median())

            comparisons[f"{g1}_vs_{g2}"] = {
                "group1": str(g1),
                "group2": str(g2),
                "n1": int(len(s1)),
                "n2": int(len(s2)),
                "median1": float(s1.median()),
                "median2": float(s2.median()),
                "u_statistic": float(stat),
                "p_value": float(pval),
                "effect_size_median_diff": float(effect_size),
                "significant_005": bool(pval < 0.05),
            }

    return {"group_summary": summary, "pairwise_comparisons": comparisons}


def plot_isg_boxplot(
    scores_df: pd.DataFrame,
    dest: Path,
    group_col: str = "group",
    score_col: str = "isg_score",
    title: str = "ISG Score by Group (AGS 6-Gene Panel)",
) -> None:
    """Generate box plot of ISG scores by group."""
    groups = scores_df[group_col].unique()

    fig, ax = plt.subplots(figsize=(8, 6))

    group_data = [scores_df.loc[scores_df[group_col] == g, score_col].dropna().values
                  for g in groups]

    bp = ax.boxplot(group_data, tick_labels=list(groups), patch_artist=True, widths=0.6)

    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12", "#9B59B6"]
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, data in enumerate(group_data):
        x = np.random.normal(i + 1, 0.05, size=len(data))
        ax.scatter(x, data, alpha=0.5, color="black", s=20, zorder=3)

    ax.set_ylabel("ISG Score (median z-score)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ISG score box plot: %s", dest)


def generate_synthetic_demo(n_per_group: int = 20, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic expression data for demonstration and testing.

    Creates samples from 4 groups with different ISG expression levels:
    - healthy: baseline ISG expression
    - primary_ocd: similar to healthy (no IFN signature)
    - pans: elevated ISG expression (interferonopathy hypothesis)
    - autoimmune_encephalitis: moderately elevated ISG expression

    Returns:
        Expression DataFrame with ISG genes in rows, samples in columns,
        plus a metadata DataFrame.
    """
    rng = np.random.default_rng(seed)

    groups = {
        "healthy": {"mean_shift": 0.0, "n": n_per_group},
        "primary_ocd": {"mean_shift": 0.1, "n": n_per_group},
        "pans": {"mean_shift": 1.5, "n": n_per_group},
        "autoimmune_encephalitis": {"mean_shift": 0.8, "n": n_per_group},
    }

    all_samples = {}
    metadata_rows = []

    for group_name, params in groups.items():
        for i in range(params["n"]):
            sample_id = f"{group_name}_{i + 1:02d}"
            # Simulate log2 expression values with group-specific ISG elevation
            base_expr = rng.normal(loc=8.0, scale=1.5, size=len(ISG_PANEL))
            isg_shift = rng.normal(loc=params["mean_shift"], scale=0.5, size=len(ISG_PANEL))
            all_samples[sample_id] = base_expr + isg_shift
            metadata_rows.append({"sample": sample_id, "group": group_name})

    expr_df = pd.DataFrame(all_samples, index=ISG_PANEL)
    meta_df = pd.DataFrame(metadata_rows)
    return expr_df, meta_df


def run_isg_scoring(
    expression_path: Path | None = None,
    metadata_path: Path | None = None,
    dest_dir: Path | None = None,
    use_demo: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Run the full ISG scoring pipeline.

    If no expression data is provided, generates synthetic demo data.

    Args:
        expression_path: Path to expression CSV (genes in rows, samples in columns).
        metadata_path: Path to metadata CSV with 'sample' and 'group' columns.
        dest_dir: Output directory.
        use_demo: Force use of synthetic demo data.

    Returns:
        Tuple of (scores DataFrame, comparison statistics dict).
    """
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    if expression_path is not None and expression_path.exists() and not use_demo:
        logger.info("Loading expression data from %s", expression_path)
        expr_df = pd.read_csv(expression_path, index_col=0)
        if metadata_path is not None and metadata_path.exists():
            meta_df = pd.read_csv(metadata_path)
        else:
            # No metadata — all samples in one group
            meta_df = pd.DataFrame({
                "sample": expr_df.columns,
                "group": "unknown",
            })
    else:
        if not use_demo:
            logger.info("No expression data provided. Generating synthetic demo data. "
                        "When real PANS transcriptomic data is available, re-run with "
                        "--expression flag.")
        expr_df, meta_df = generate_synthetic_demo()

    # Compute ISG scores
    scores_df = compute_isg_scores(expr_df)

    if scores_df.empty:
        logger.error("No ISG scores computed. Aborting.")
        return scores_df, {}

    # Merge with metadata
    scores_df = scores_df.merge(meta_df, on="sample", how="left")
    scores_df["group"] = scores_df["group"].fillna("unknown")

    # Save scores
    scores_path = dest_dir / "isg_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    logger.info("Saved ISG scores: %s (%d samples)", scores_path, len(scores_df))

    # Compare groups if multiple exist
    comparison_stats = {}
    unique_groups = scores_df["group"].nunique()
    if unique_groups > 1:
        comparison_stats = compare_groups(scores_df)

        # Save stats
        stats_path = dest_dir / "isg_comparison_stats.json"
        with open(stats_path, "w") as f:
            json.dump(comparison_stats, f, indent=2)
        logger.info("Saved comparison stats: %s", stats_path)

        # Generate box plot
        plot_isg_boxplot(scores_df, dest_dir / "isg_scores_boxplot.png")

        # Log key results
        logger.info("=== ISG Comparison Results ===")
        for comp_name, comp_data in comparison_stats.get("pairwise_comparisons", {}).items():
            sig = "*" if comp_data["significant_005"] else ""
            logger.info("  %s: p=%.4f%s (median diff=%.3f)",
                        comp_name, comp_data["p_value"], sig,
                        comp_data["effect_size_median_diff"])
    else:
        logger.info("Single group detected — skipping group comparison")

    return scores_df, comparison_stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ISG scorer using AGS 6-gene panel for PANS interferonopathy"
    )
    parser.add_argument("--expression", type=Path, default=None,
                        help="Expression CSV (genes in rows, samples in columns)")
    parser.add_argument("--metadata", type=Path, default=None,
                        help="Metadata CSV with 'sample' and 'group' columns")
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--demo", action="store_true",
                        help="Force use of synthetic demo data")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scores_df, comparison_stats = run_isg_scoring(
        expression_path=args.expression,
        metadata_path=args.metadata,
        dest_dir=args.dest,
        use_demo=args.demo,
    )

    if scores_df.empty:
        print("No scores computed.")
        return

    print(f"\nISG scores computed for {len(scores_df)} samples")
    print(f"ISG genes used: {scores_df.iloc[0]['isg_genes_found']}")

    # Print group summaries
    if "group" in scores_df.columns:
        print("\nGroup summary:")
        for group, grp_df in scores_df.groupby("group"):
            print(f"  {group}: n={len(grp_df)}, "
                  f"median ISG={grp_df['isg_score'].median():.3f}, "
                  f"mean ISG={grp_df['isg_score'].mean():.3f}")

    # Print significant comparisons
    for comp_name, comp_data in comparison_stats.get("pairwise_comparisons", {}).items():
        if comp_data["significant_005"]:
            print(f"\n  * {comp_name}: p={comp_data['p_value']:.4f} "
                  f"(median diff={comp_data['effect_size_median_diff']:.3f})")


if __name__ == "__main__":
    main()
