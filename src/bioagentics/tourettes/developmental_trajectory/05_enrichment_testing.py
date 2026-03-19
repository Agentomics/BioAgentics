"""Step 05 — Test TS risk gene enrichment in developmental temporal clusters.

Tests whether TS GWAS/rare variant genes are significantly enriched in
specific developmental temporal clusters relative to a genome-wide background.
Uses both hypergeometric testing and permutation-based p-values.

Key question: Are TS risk genes disproportionately represented in the
onset-window, peak-severity, or remission-window temporal clusters?

Task: #782 (enrichment testing)
Project: ts-developmental-trajectory-modeling

Usage:
    uv run python -m bioagentics.tourettes.developmental_trajectory.05_enrichment_testing
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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


def load_or_build_clusters(
    output_dir: Path,
    cache_dir: Path = CACHE_DIR,
    n_clusters: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load existing clusters or build them from BrainSpan data.

    Returns (cluster_assignments, zscore_matrix).
    """
    cluster_path = output_dir / "temporal_clusters.csv"
    zscore_path = output_dir / "zscore_matrix.csv"

    if cluster_path.exists() and zscore_path.exists():
        logger.info("Loading existing clusters from %s", cluster_path)
        assignments = pd.read_csv(cluster_path)
        zscore = pd.read_csv(zscore_path, index_col=0)
        return assignments, zscore

    # Build from scratch using step 04 logic
    from sklearn.cluster import KMeans

    all_genes = get_gene_set("ts_combined")
    gene_symbols = sorted(all_genes.keys())
    trajectories = extract_trajectories(gene_symbols, cache_dir)

    if trajectories.empty:
        return pd.DataFrame(), pd.DataFrame()

    avg = (
        trajectories
        .groupby(["gene_symbol", "dev_stage"])["mean_log2_rpkm"]
        .mean()
        .reset_index()
    )
    pivot = avg.pivot(index="gene_symbol", columns="dev_stage", values="mean_log2_rpkm")
    available = [s for s in STAGE_ORDER if s in pivot.columns]
    pivot = pivot[available].dropna(thresh=len(available) // 2)

    zscore = pivot.apply(
        lambda row: (row - row.mean()) / row.std() if row.std() > 0 else row * 0,
        axis=1,
    ).fillna(0)

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km.fit_predict(zscore.values)
    assignments = pd.DataFrame({
        "gene_symbol": zscore.index,
        "cluster": labels,
    })

    return assignments, zscore


def build_background_clusters(
    ts_cluster_assignments: pd.DataFrame,
    cache_dir: Path = CACHE_DIR,
    n_background: int = 500,
    n_clusters: int = 4,
) -> pd.DataFrame:
    """Build background gene clusters for enrichment comparison.

    Randomly samples non-TS genes from BrainSpan, extracts their trajectories,
    and clusters them using the same k-means approach.
    """
    from bioagentics.analysis.tourettes.brainspan_trajectories import download_brainspan
    from sklearn.cluster import KMeans

    expression, rows_meta, cols_meta = download_brainspan(cache_dir)

    # Find gene symbol column
    gene_col = next(
        (c for c in rows_meta.columns if "gene_symbol" in c.lower() or "symbol" in c.lower()),
        None,
    )
    if gene_col is None:
        logger.warning("Cannot find gene symbol column for background")
        return pd.DataFrame()

    all_brainspan_genes = set(rows_meta[gene_col].dropna().unique())
    ts_genes = set(ts_cluster_assignments["gene_symbol"])

    # Sample background genes (exclude TS genes)
    non_ts = sorted(all_brainspan_genes - ts_genes)
    rng = np.random.default_rng(42)
    n_sample = min(n_background, len(non_ts))
    bg_genes = list(rng.choice(non_ts, size=n_sample, replace=False))

    logger.info("Extracting trajectories for %d background genes...", n_sample)
    bg_traj = extract_trajectories(bg_genes, cache_dir)

    if bg_traj.empty:
        return pd.DataFrame()

    avg = (
        bg_traj
        .groupby(["gene_symbol", "dev_stage"])["mean_log2_rpkm"]
        .mean()
        .reset_index()
    )
    pivot = avg.pivot(index="gene_symbol", columns="dev_stage", values="mean_log2_rpkm")
    available = [s for s in STAGE_ORDER if s in pivot.columns]
    pivot = pivot[available].dropna(thresh=len(available) // 2)

    if pivot.shape[0] < n_clusters:
        return pd.DataFrame()

    zscore = pivot.apply(
        lambda row: (row - row.mean()) / row.std() if row.std() > 0 else row * 0,
        axis=1,
    ).fillna(0)

    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels = km.fit_predict(zscore.values)

    return pd.DataFrame({
        "gene_symbol": zscore.index,
        "cluster": labels,
    })


def test_cluster_enrichment(
    ts_assignments: pd.DataFrame,
    ts_gene_set: set[str],
    cluster_info_path: Path | None = None,
    n_permutations: int = 5000,
) -> list[dict]:
    """Test enrichment of specific TS gene subsets in each temporal cluster.

    Tests whether a particular gene subset (e.g., GWAS hits, rare variants)
    is overrepresented in a specific temporal cluster compared to random
    expectation among all clustered genes.
    """
    all_clustered = set(ts_assignments["gene_symbol"])
    test_genes = ts_gene_set & all_clustered

    N = len(all_clustered)
    K = len(test_genes)

    if K == 0:
        logger.warning("No test genes found among clustered genes")
        return []

    # Load cluster characterization if available
    cluster_names = {}
    if cluster_info_path and cluster_info_path.exists():
        with open(cluster_info_path) as f:
            for info in json.load(f):
                cluster_names[info["cluster"]] = info.get("temporal_pattern", f"cluster_{info['cluster']}")

    rng = np.random.default_rng(42)
    all_genes_list = list(all_clustered)
    results = []

    for cluster_id in sorted(ts_assignments["cluster"].unique()):
        cluster_genes = set(
            ts_assignments[ts_assignments["cluster"] == cluster_id]["gene_symbol"]
        )
        n = len(cluster_genes)
        k = len(test_genes & cluster_genes)

        # Hypergeometric p-value (one-sided: enrichment)
        hyper_p = float(1 - stats.hypergeom.cdf(k - 1, N, K, n))

        # Permutation p-value
        perm_count = 0
        for _ in range(n_permutations):
            random_set = set(rng.choice(all_genes_list, size=K, replace=False))
            if len(random_set & cluster_genes) >= k:
                perm_count += 1
        perm_p = (perm_count + 1) / (n_permutations + 1)

        expected = K * n / N if N > 0 else 0
        fold = k / max(expected, 0.001)

        results.append({
            "cluster": int(cluster_id),
            "cluster_name": cluster_names.get(cluster_id, f"cluster_{cluster_id}"),
            "cluster_size": n,
            "test_genes_in_cluster": k,
            "test_genes_expected": round(expected, 2),
            "fold_enrichment": round(fold, 2),
            "hypergeometric_p": hyper_p,
            "permutation_p": perm_p,
            "significant_005": perm_p < 0.05,
            "significant_001": perm_p < 0.01,
            "genes_found": sorted(test_genes & cluster_genes),
        })

    return results


def test_stage_enrichment(
    ts_assignments: pd.DataFrame,
    zscore: pd.DataFrame,
    ts_gene_set: set[str],
    n_permutations: int = 5000,
) -> list[dict]:
    """Test enrichment of TS genes among top-expressed genes at each stage.

    For each developmental stage, identifies the top 25% of genes by
    expression and tests whether TS risk genes are overrepresented.
    """
    all_clustered = set(ts_assignments["gene_symbol"])
    test_genes = ts_gene_set & all_clustered

    N = len(all_clustered)
    K = len(test_genes)

    if K == 0 or zscore.empty:
        return []

    rng = np.random.default_rng(42)
    all_genes_list = list(all_clustered)
    results = []

    for stage in zscore.columns:
        # Top quartile by expression at this stage
        stage_vals = zscore[stage].dropna()
        threshold = stage_vals.quantile(0.75)
        top_genes = set(stage_vals[stage_vals >= threshold].index)

        n = len(top_genes)
        k = len(test_genes & top_genes)

        hyper_p = float(1 - stats.hypergeom.cdf(k - 1, N, K, n))

        perm_count = 0
        for _ in range(n_permutations):
            random_set = set(rng.choice(all_genes_list, size=K, replace=False))
            if len(random_set & top_genes) >= k:
                perm_count += 1
        perm_p = (perm_count + 1) / (n_permutations + 1)

        expected = K * n / N if N > 0 else 0
        fold = k / max(expected, 0.001)

        results.append({
            "stage": stage,
            "top_quartile_size": n,
            "test_genes_in_top": k,
            "expected": round(expected, 2),
            "fold_enrichment": round(fold, 2),
            "hypergeometric_p": hyper_p,
            "permutation_p": perm_p,
            "significant_005": perm_p < 0.05,
            "genes_found": sorted(test_genes & top_genes),
        })

    return results


def generate_enrichment_plot(
    cluster_results: dict[str, list[dict]],
    output_path: Path,
) -> Path:
    """Generate enrichment bar plot for all tested gene subsets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gene_set_names = list(cluster_results.keys())
    n_sets = len(gene_set_names)

    if n_sets == 0:
        return output_path

    # Get cluster IDs from first result
    cluster_ids = [r["cluster"] for r in cluster_results[gene_set_names[0]]]
    n_clusters = len(cluster_ids)

    fig, ax = plt.subplots(figsize=(max(10, n_clusters * 2), 6))

    bar_width = 0.8 / n_sets
    colors = plt.cm.Set2(np.linspace(0, 1, n_sets))

    for i, set_name in enumerate(gene_set_names):
        enrichments = cluster_results[set_name]
        x = np.arange(n_clusters) + i * bar_width
        fold_vals = [r["fold_enrichment"] for r in enrichments]
        sig = [r["significant_005"] for r in enrichments]

        bars = ax.bar(x, fold_vals, bar_width, label=set_name.replace("_", " "),
                      color=colors[i], edgecolor="black", linewidth=0.5)

        # Mark significant bars
        for j, (bar, is_sig) in enumerate(zip(bars, sig)):
            if is_sig:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        "*", ha="center", fontsize=14, fontweight="bold")

    ax.set_xticks(np.arange(n_clusters) + bar_width * (n_sets - 1) / 2)
    cluster_labels = [
        cluster_results[gene_set_names[0]][j].get("cluster_name", f"C{j}")
        for j in range(n_clusters)
    ]
    ax.set_xticklabels(cluster_labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Fold Enrichment")
    ax.set_title("TS Gene Enrichment in Temporal Clusters\n(* = permutation p < 0.05)")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=8)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved enrichment plot to %s", output_path)
    return output_path


def run(
    output_dir: Path = OUTPUT_DIR,
    cache_dir: Path = CACHE_DIR,
    n_clusters: int = 4,
    n_permutations: int = 5000,
) -> dict:
    """Run enrichment testing pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or build clusters
    ts_assignments, zscore = load_or_build_clusters(output_dir, cache_dir, n_clusters)
    if ts_assignments.empty:
        return {"error": "No cluster data available"}

    logger.info("Testing enrichment for %d clustered genes", len(ts_assignments))

    cluster_info_path = output_dir / "cluster_characterization.json"

    # Test enrichment for each TS gene subset
    test_sets = {
        "tsaicg_gwas": set(get_gene_set("tsaicg_gwas").keys()),
        "rare_variant": set(get_gene_set("rare_variant").keys()),
        "de_novo_variant": set(get_gene_set("de_novo_variant").keys()),
        "ts_combined": set(get_gene_set("ts_combined").keys()),
    }

    cluster_results: dict[str, list[dict]] = {}
    stage_results: dict[str, list[dict]] = {}

    for set_name, gene_set in test_sets.items():
        logger.info("Testing cluster enrichment for %s (%d genes)...", set_name, len(gene_set))
        cluster_results[set_name] = test_cluster_enrichment(
            ts_assignments, gene_set, cluster_info_path, n_permutations,
        )

        logger.info("Testing stage enrichment for %s...", set_name)
        stage_results[set_name] = test_stage_enrichment(
            ts_assignments, zscore, gene_set, n_permutations,
        )

    # Save results
    with open(output_dir / "cluster_enrichment.json", "w") as f:
        json.dump(cluster_results, f, indent=2)

    with open(output_dir / "stage_enrichment.json", "w") as f:
        json.dump(stage_results, f, indent=2)

    # Generate plot
    generate_enrichment_plot(
        cluster_results,
        output_dir / "enrichment_plot.png",
    )

    # Summary
    significant_findings = []
    for set_name, results in cluster_results.items():
        for r in results:
            if r["significant_005"]:
                significant_findings.append({
                    "gene_set": set_name,
                    "cluster": r["cluster"],
                    "cluster_name": r["cluster_name"],
                    "fold_enrichment": r["fold_enrichment"],
                    "permutation_p": r["permutation_p"],
                    "genes": r["genes_found"],
                })

    for set_name, results in stage_results.items():
        for r in results:
            if r["significant_005"]:
                significant_findings.append({
                    "gene_set": set_name,
                    "stage": r["stage"],
                    "fold_enrichment": r["fold_enrichment"],
                    "permutation_p": r["permutation_p"],
                    "genes": r["genes_found"],
                })

    summary = {
        "n_clustered_genes": len(ts_assignments),
        "n_clusters": n_clusters,
        "n_permutations": n_permutations,
        "gene_sets_tested": list(test_sets.keys()),
        "n_significant_findings": len(significant_findings),
        "significant_findings": significant_findings,
    }

    with open(output_dir / "enrichment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Test TS risk gene enrichment in developmental temporal clusters"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = run(
        output_dir=args.output,
        cache_dir=args.cache,
        n_clusters=args.clusters,
        n_permutations=args.permutations,
    )

    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print(f"\nEnrichment Testing Results")
    print(f"  Clustered genes: {summary['n_clustered_genes']}")
    print(f"  Gene sets tested: {', '.join(summary['gene_sets_tested'])}")
    print(f"  Significant findings: {summary['n_significant_findings']}")

    for finding in summary.get("significant_findings", []):
        loc = finding.get("cluster_name", finding.get("stage", "?"))
        print(f"\n  {finding['gene_set']} in {loc}:")
        print(f"    Fold enrichment: {finding['fold_enrichment']}")
        print(f"    p = {finding['permutation_p']:.4f}")
        print(f"    Genes: {', '.join(finding['genes'])}")


if __name__ == "__main__":
    main()
