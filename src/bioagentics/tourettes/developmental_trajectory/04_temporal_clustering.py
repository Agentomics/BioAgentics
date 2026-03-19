"""Step 04 — Temporal clustering of TS risk gene expression patterns.

Clusters TS risk genes by their developmental expression trajectories in
CSTC brain regions using k-means on z-scored expression profiles. Categorizes
clusters into biologically meaningful temporal groups:
  - Early-peak: high fetal/infant expression, declining with age
  - Onset-window: expression peak at 4-8 years (tic onset)
  - Adolescent-peak: expression peak at 12-18 years (remission window)
  - Flat/constitutive: stable expression across development

Task: #781 (temporal clustering)
Project: ts-developmental-trajectory-modeling

Usage:
    uv run python -m bioagentics.tourettes.developmental_trajectory.04_temporal_clustering
    uv run python -m bioagentics.tourettes.developmental_trajectory.04_temporal_clustering --clusters 4
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from bioagentics.analysis.tourettes.brainspan_trajectories import (
    CACHE_DIR,
    DEV_STAGES,
    extract_trajectories,
)
from bioagentics.config import REPO_ROOT
from bioagentics.data.tourettes.gene_sets import get_gene_set, list_gene_sets

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-developmental-trajectory-modeling"

STAGE_ORDER = list(DEV_STAGES.keys())

# Map stages to biological windows for cluster characterization
BIOLOGICAL_WINDOWS = {
    "early_prenatal": "prenatal",
    "mid_prenatal": "prenatal",
    "late_prenatal": "prenatal",
    "infancy": "infancy",
    "early_childhood": "onset_window",      # 1-6y, approaching tic onset
    "late_childhood": "onset_window",       # 6-12y, tic onset/peak severity
    "adolescence": "remission_window",      # 12-20y, spontaneous remission
    "adulthood": "adult",
}


def build_expression_matrix(
    trajectories: pd.DataFrame,
) -> pd.DataFrame:
    """Build gene x stage expression matrix from trajectories.

    Averages across CSTC regions per gene per stage.
    Returns pivot table (genes as rows, stages as columns).
    """
    avg = (
        trajectories
        .groupby(["gene_symbol", "dev_stage"])["mean_log2_rpkm"]
        .mean()
        .reset_index()
    )
    pivot = avg.pivot(index="gene_symbol", columns="dev_stage", values="mean_log2_rpkm")

    # Reorder columns by developmental stage
    available = [s for s in STAGE_ORDER if s in pivot.columns]
    pivot = pivot[available]

    # Drop genes with >50% missing stages
    pivot = pivot.dropna(thresh=len(available) // 2)

    return pivot


def zscore_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    """Z-score each gene (row) across developmental stages."""
    return matrix.apply(
        lambda row: (row - row.mean()) / row.std() if row.std() > 0 else row * 0,
        axis=1,
    ).fillna(0)


def cluster_genes(
    zscore: pd.DataFrame,
    n_clusters: int = 4,
) -> tuple[pd.DataFrame, np.ndarray, KMeans]:
    """Cluster genes by z-scored expression patterns.

    Returns (assignments_df, centroids, kmeans_model).
    """
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km.fit_predict(zscore.values)

    assignments = pd.DataFrame({
        "gene_symbol": zscore.index,
        "cluster": labels,
    })

    return assignments, km.cluster_centers_, km


def characterize_clusters(
    centroids: np.ndarray,
    stage_columns: list[str],
) -> list[dict]:
    """Characterize each cluster by its peak developmental window.

    Returns list of cluster descriptions with peak stage, biological
    window, and temporal pattern classification.
    """
    descriptions = []
    for i, centroid in enumerate(centroids):
        peak_idx = int(np.argmax(centroid))
        trough_idx = int(np.argmin(centroid))
        peak_stage = stage_columns[peak_idx]
        trough_stage = stage_columns[trough_idx]
        peak_window = BIOLOGICAL_WINDOWS.get(peak_stage, "unknown")

        # Classify temporal pattern
        peak_half = peak_idx / max(len(stage_columns) - 1, 1)
        amplitude = float(centroid[peak_idx] - centroid[trough_idx])

        if amplitude < 0.5:
            pattern = "flat_constitutive"
        elif peak_half < 0.3:
            pattern = "early_peak"
        elif peak_half < 0.6:
            pattern = "onset_window_peak"
        elif peak_half < 0.8:
            pattern = "adolescent_peak"
        else:
            pattern = "adult_peak"

        descriptions.append({
            "cluster": i,
            "peak_stage": peak_stage,
            "trough_stage": trough_stage,
            "peak_biological_window": peak_window,
            "temporal_pattern": pattern,
            "amplitude": round(amplitude, 3),
            "centroid": [round(float(v), 4) for v in centroid],
        })

    return descriptions


def generate_cluster_plot(
    zscore: pd.DataFrame,
    assignments: pd.DataFrame,
    centroids: np.ndarray,
    cluster_info: list[dict],
    output_path: Path,
) -> Path:
    """Generate cluster visualization: centroid plots + heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = list(zscore.columns)
    n_clusters = len(cluster_info)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                             gridspec_kw={"width_ratios": [1.2, 1]})

    # Left panel: centroid line plots
    ax = axes[0]
    colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
    for info in cluster_info:
        i = info["cluster"]
        n_genes = (assignments["cluster"] == i).sum()
        label = f"C{i}: {info['temporal_pattern']} (n={n_genes})"
        ax.plot(range(len(stages)), centroids[i], color=colors[i],
                linewidth=2.5, marker="o", markersize=5, label=label)

    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([s.replace("_", "\n") for s in stages],
                       fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Z-score")
    ax.set_title("Cluster Centroids")
    ax.legend(fontsize=8, loc="best")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)

    # Right panel: sorted heatmap
    ax2 = axes[1]
    merged = assignments.set_index("gene_symbol").join(zscore)
    merged = merged.sort_values("cluster")
    data = merged.drop(columns=["cluster"]).values

    im = ax2.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax2.set_xticks(range(len(stages)))
    ax2.set_xticklabels([s.replace("_", "\n") for s in stages],
                        fontsize=7, rotation=45, ha="right")
    ax2.set_ylabel("Genes (sorted by cluster)")
    ax2.set_title("Gene Expression Heatmap")

    # Cluster boundaries
    boundaries = merged["cluster"].values
    for idx in range(1, len(boundaries)):
        if boundaries[idx] != boundaries[idx - 1]:
            ax2.axhline(idx - 0.5, color="black", linewidth=1)

    fig.colorbar(im, ax=ax2, shrink=0.6, label="Z-score")
    fig.suptitle("Temporal Clustering of TS Risk Genes", fontsize=13, y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cluster plot to %s", output_path)
    return output_path


def run(
    n_clusters: int = 4,
    output_dir: Path = OUTPUT_DIR,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Run temporal clustering pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all TS genes
    all_genes = get_gene_set("ts_combined")
    gene_symbols = sorted(all_genes.keys())
    logger.info("Clustering %d TS genes...", len(gene_symbols))

    # Extract trajectories
    trajectories = extract_trajectories(gene_symbols, cache_dir)
    if trajectories.empty:
        return {"error": "No trajectory data extracted"}

    # Build expression matrix
    expr_matrix = build_expression_matrix(trajectories)
    logger.info("Expression matrix: %d genes x %d stages", *expr_matrix.shape)

    if expr_matrix.shape[0] < n_clusters:
        return {"error": f"Too few genes ({expr_matrix.shape[0]}) for {n_clusters} clusters"}

    # Z-score and cluster
    zscore = zscore_matrix(expr_matrix)
    assignments, centroids, km = cluster_genes(zscore, n_clusters)

    # Characterize clusters
    cluster_info = characterize_clusters(centroids, list(zscore.columns))

    # Add gene set membership to assignments
    gene_set_map: dict[str, list[str]] = {}
    for name in list_gene_sets():
        if name == "ts_combined":
            continue
        gs = get_gene_set(name)
        for sym in gs:
            gene_set_map.setdefault(sym, []).append(name)

    assignments["gene_sets"] = assignments["gene_symbol"].map(
        lambda s: ",".join(gene_set_map.get(s, []))
    )

    # Save outputs
    assignments.to_csv(output_dir / "temporal_clusters.csv", index=False)
    zscore.to_csv(output_dir / "zscore_matrix.csv")

    cluster_info_path = output_dir / "cluster_characterization.json"
    with open(cluster_info_path, "w") as f:
        json.dump(cluster_info, f, indent=2)

    # Generate plot
    generate_cluster_plot(
        zscore, assignments, centroids, cluster_info,
        output_dir / "temporal_clusters_plot.png",
    )

    # Per-cluster gene lists
    cluster_gene_lists: dict[int, list[str]] = {}
    for info in cluster_info:
        c = info["cluster"]
        genes = sorted(assignments[assignments["cluster"] == c]["gene_symbol"].tolist())
        cluster_gene_lists[c] = genes
        logger.info("  Cluster %d (%s): %d genes — %s",
                     c, info["temporal_pattern"], len(genes),
                     ", ".join(genes[:5]) + ("..." if len(genes) > 5 else ""))

    # Summary
    summary = {
        "n_genes_clustered": len(assignments),
        "n_clusters": n_clusters,
        "stages_used": list(zscore.columns),
        "clusters": [],
    }
    for info in cluster_info:
        c = info["cluster"]
        summary["clusters"].append({
            **info,
            "n_genes": len(cluster_gene_lists[c]),
            "genes": cluster_gene_lists[c],
        })

    summary_path = output_dir / "temporal_clustering_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Cluster TS risk genes by developmental expression pattern"
    )
    parser.add_argument("--clusters", type=int, default=4,
                        help="Number of temporal clusters (default: 4)")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = run(n_clusters=args.clusters, output_dir=args.output, cache_dir=args.cache)

    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print(f"\nTemporal Clustering Results")
    print(f"  Genes clustered: {summary['n_genes_clustered']}")
    print(f"  Clusters: {summary['n_clusters']}")
    for c in summary.get("clusters", []):
        print(f"\n  Cluster {c['cluster']} — {c['temporal_pattern']}:")
        print(f"    Peak: {c['peak_stage']}, Amplitude: {c['amplitude']}")
        print(f"    Genes ({c['n_genes']}): {', '.join(c['genes'][:8])}"
              + ("..." if c['n_genes'] > 8 else ""))


if __name__ == "__main__":
    main()
