"""Step 03 — Extract and visualize TS risk gene developmental trajectories.

Uses BrainSpan RNA-seq data to extract expression trajectories for all TS
developmental gene sets across CSTC brain regions and developmental stages.
Generates per-set and combined trajectory plots and peak-window summaries.

Task: #780 (expression trajectories)
Project: ts-developmental-trajectory-modeling

Usage:
    uv run python -m bioagentics.tourettes.developmental_trajectory.03_expression_trajectories
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.analysis.tourettes.brainspan_trajectories import (
    CACHE_DIR,
    CSTC_REGIONS,
    DEV_STAGES,
    extract_trajectories,
    identify_peak_windows,
    generate_trajectory_plot,
)
from bioagentics.config import REPO_ROOT
from bioagentics.data.tourettes.gene_sets import get_gene_set, list_gene_sets

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-developmental-trajectory-modeling"

# Stage order for consistent plotting and reporting
STAGE_ORDER = list(DEV_STAGES.keys())


def extract_all_trajectories(
    gene_lists: dict[str, dict[str, str]],
    cache_dir: Path = CACHE_DIR,
) -> dict[str, pd.DataFrame]:
    """Extract developmental trajectories for each gene set.

    Returns dict of {set_name: trajectories_df}.
    """
    results: dict[str, pd.DataFrame] = {}

    for set_name, genes in gene_lists.items():
        if set_name == "ts_developmental_all":
            continue  # handle combined separately
        gene_symbols = sorted(genes.keys())
        if not gene_symbols:
            continue

        logger.info("Extracting trajectories for %s (%d genes)...", set_name, len(gene_symbols))
        traj = extract_trajectories(gene_symbols, cache_dir)
        if not traj.empty:
            traj["gene_set"] = set_name
            results[set_name] = traj
            logger.info("  -> %d trajectory records for %d genes",
                        len(traj), traj["gene_symbol"].nunique())
        else:
            logger.warning("  -> No data for %s", set_name)

    return results


def compute_stage_means(trajectories: pd.DataFrame) -> pd.DataFrame:
    """Compute mean expression per developmental stage and CSTC region.

    Returns DataFrame with columns: dev_stage, cstc_region, mean_rpkm,
    mean_log2_rpkm, n_genes, stage_order.
    """
    if trajectories.empty:
        return pd.DataFrame()

    agg = (
        trajectories
        .groupby(["dev_stage", "cstc_region"])
        .agg(
            mean_rpkm=("mean_rpkm", "mean"),
            mean_log2_rpkm=("mean_log2_rpkm", "mean"),
            n_genes=("gene_symbol", "nunique"),
        )
        .reset_index()
    )

    # Add stage ordering
    stage_rank = {s: i for i, s in enumerate(STAGE_ORDER)}
    agg["stage_order"] = agg["dev_stage"].map(stage_rank)
    agg = agg.sort_values(["cstc_region", "stage_order"])

    return agg


def generate_heatmap(
    trajectories: pd.DataFrame,
    output_path: Path,
    title: str = "TS Risk Gene Expression Heatmap",
) -> Path:
    """Generate a heatmap of gene expression across stages and regions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if trajectories.empty:
        logger.warning("No data for heatmap")
        return output_path

    # Pivot: gene x stage (averaged across regions)
    avg = (
        trajectories
        .groupby(["gene_symbol", "dev_stage"])["mean_log2_rpkm"]
        .mean()
        .reset_index()
    )
    pivot = avg.pivot(index="gene_symbol", columns="dev_stage", values="mean_log2_rpkm")

    # Reorder columns by stage
    available_stages = [s for s in STAGE_ORDER if s in pivot.columns]
    pivot = pivot[available_stages]
    pivot = pivot.dropna(thresh=len(available_stages) // 2)

    if pivot.empty:
        logger.warning("No genes with sufficient stage coverage for heatmap")
        return output_path

    # Z-score per gene for visualization
    zscore = pivot.apply(
        lambda row: (row - row.mean()) / row.std() if row.std() > 0 else row * 0,
        axis=1,
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(12, max(6, len(zscore) * 0.3)))
    im = ax.imshow(zscore.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)

    ax.set_yticks(range(len(zscore)))
    ax.set_yticklabels(zscore.index, fontsize=7)
    ax.set_xticks(range(len(available_stages)))
    ax.set_xticklabels([s.replace("_", "\n") for s in available_stages],
                       fontsize=8, rotation=45, ha="right")
    ax.set_title(title, fontsize=12)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="Z-score")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap to %s", output_path)
    return output_path


def run(
    output_dir: Path = OUTPUT_DIR,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Extract trajectories for all TS gene sets, generate plots and summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assemble gene lists using gene_sets registry
    gene_lists: dict[str, dict[str, str]] = {}
    for name in list_gene_sets():
        if name == "ts_combined":
            continue
        gene_lists[name] = get_gene_set(name)
    # Combined set
    gene_lists["ts_developmental_all"] = get_gene_set("ts_combined")

    # Extract trajectories per set
    set_trajectories = extract_all_trajectories(gene_lists, cache_dir)

    if not set_trajectories:
        return {"error": "No trajectory data extracted for any gene set"}

    # Combine all trajectories
    combined = pd.concat(set_trajectories.values(), ignore_index=True)
    combined.to_csv(output_dir / "trajectories_combined.csv", index=False)

    # Per-set outputs
    set_summaries: dict[str, dict] = {}
    for set_name, traj in set_trajectories.items():
        # Save trajectory data
        traj.to_csv(output_dir / f"trajectories_{set_name}.csv", index=False)

        # Peak windows
        peaks = identify_peak_windows(traj)
        peaks.to_csv(output_dir / f"peak_windows_{set_name}.csv", index=False)

        # Stage means
        stage_means = compute_stage_means(traj)
        stage_means.to_csv(output_dir / f"stage_means_{set_name}.csv", index=False)

        # Trajectory plot
        generate_trajectory_plot(
            traj,
            output_dir / f"trajectories_{set_name}.png",
            title=f"Developmental Trajectories: {set_name}",
        )

        set_summaries[set_name] = {
            "n_genes_found": int(traj["gene_symbol"].nunique()),
            "peak_stage_distribution": peaks["peak_stage"].value_counts().to_dict()
            if not peaks.empty else {},
            "regions_covered": sorted(traj["cstc_region"].unique().tolist()),
        }

    # Combined heatmap
    generate_heatmap(
        combined,
        output_dir / "heatmap_all_ts_genes.png",
        title="TS Developmental Gene Expression (Z-scored)",
    )

    # Combined trajectory plot
    generate_trajectory_plot(
        combined,
        output_dir / "trajectories_all_ts_genes.png",
        title="All TS Risk Genes — Developmental Trajectories",
    )

    # Combined peak windows
    all_peaks = identify_peak_windows(combined)
    all_peaks.to_csv(output_dir / "peak_windows_all.csv", index=False)

    # Summary report
    summary = {
        "total_genes_found": int(combined["gene_symbol"].nunique()),
        "total_trajectory_records": len(combined),
        "gene_sets": set_summaries,
        "overall_peak_distribution": all_peaks["peak_stage"].value_counts().to_dict()
        if not all_peaks.empty else {},
        "stages_covered": sorted(combined["dev_stage"].unique().tolist()),
        "regions_covered": sorted(combined["cstc_region"].unique().tolist()),
    }

    summary_path = output_dir / "expression_trajectories_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Summary saved to %s", summary_path)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract and visualize TS risk gene developmental trajectories"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = run(output_dir=args.output, cache_dir=args.cache)

    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print(f"\nExpression Trajectory Extraction")
    print(f"  Total genes found: {summary['total_genes_found']}")
    print(f"  Trajectory records: {summary['total_trajectory_records']}")
    print(f"\n  Per-set results:")
    for name, info in summary.get("gene_sets", {}).items():
        print(f"    {name}: {info['n_genes_found']} genes")
    print(f"\n  Overall peak stage distribution:")
    for stage, count in summary.get("overall_peak_distribution", {}).items():
        print(f"    {stage}: {count} genes")


if __name__ == "__main__":
    main()
