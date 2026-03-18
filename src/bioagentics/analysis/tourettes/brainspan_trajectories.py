"""BrainSpan developmental trajectory analysis for TS risk genes.

Uses BrainSpan Atlas of the Developing Human Brain (RNA-seq) to identify
critical developmental periods when TS risk genes are most highly expressed
in CSTC-relevant brain regions.

Pipeline:
1. Download/cache BrainSpan RNA-seq expression data
2. Extract trajectories for TS risk genes in striatal and cortical regions
3. Identify peak expression windows (prenatal/postnatal/childhood/adolescent)
4. Cluster genes by temporal expression pattern (k-means on z-scored trajectories)
5. Generate trajectory plots

Data source: BrainSpan — developmental transcriptome, prenatal through adult.
Download: https://www.brainspan.org/api/v2/well_known_file_download/267666525

Usage:
    uv run python -m bioagentics.analysis.tourettes.brainspan_trajectories
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bioagentics.config import REPO_ROOT
from bioagentics.data.tourettes.gene_sets import get_gene_set, list_gene_sets

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "cstc-circuit-expression-atlas"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "brainspan"

# BrainSpan RNA-seq dataset download URL
BRAINSPAN_URL = "https://www.brainspan.org/api/v2/well_known_file_download/267666525"

# Developmental stage bins for trajectory analysis
DEV_STAGES: dict[str, tuple[float, float]] = {
    "early_prenatal": (8, 16),       # 8-16 pcw (post-conception weeks)
    "mid_prenatal": (16, 24),        # 16-24 pcw
    "late_prenatal": (24, 38),       # 24-38 pcw
    "infancy": (0.0, 1.0),          # 0-1 years postnatal
    "early_childhood": (1.0, 6.0),   # 1-6 years
    "late_childhood": (6.0, 12.0),   # 6-12 years (typical TS onset window)
    "adolescence": (12.0, 20.0),     # 12-20 years
    "adulthood": (20.0, 100.0),      # 20+ years
}

# CSTC-relevant BrainSpan region keywords
CSTC_REGIONS = {
    "striatum": ["striatum", "caudate", "putamen", "basal ganglia"],
    "cortex": ["frontal cortex", "prefrontal", "motor cortex", "primary motor",
               "dorsolateral prefrontal", "orbital frontal"],
    "thalamus": ["thalamus", "mediodorsal nucleus", "thalamic"],
}


def download_brainspan(cache_dir: Path = CACHE_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download and cache BrainSpan RNA-seq dataset.

    Returns:
        expression: DataFrame (genes x samples) of RPKM values
        rows_metadata: gene annotations (gene_symbol, ensembl_gene_id, etc.)
        columns_metadata: sample annotations (age, structure, donor, etc.)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    expr_path = cache_dir / "expression_matrix.parquet"
    rows_path = cache_dir / "rows_metadata.parquet"
    cols_path = cache_dir / "columns_metadata.parquet"

    if expr_path.exists() and rows_path.exists() and cols_path.exists():
        logger.info("Loading cached BrainSpan data from %s", cache_dir)
        return (
            pd.read_parquet(expr_path),
            pd.read_parquet(rows_path),
            pd.read_parquet(cols_path),
        )

    logger.info("Downloading BrainSpan RNA-seq data...")
    resp = requests.get(BRAINSPAN_URL, timeout=120)
    resp.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        file_list = zf.namelist()
        logger.info("  ZIP contents: %s", file_list)

        # Find the expression matrix and metadata files
        expr_file = next(f for f in file_list if "expression" in f.lower() and f.endswith(".csv"))
        rows_file = next(f for f in file_list if "rows" in f.lower() and f.endswith(".csv"))
        cols_file = next(f for f in file_list if "columns" in f.lower() and f.endswith(".csv"))

        expression = pd.read_csv(zf.open(expr_file), header=None)
        rows_metadata = pd.read_csv(zf.open(rows_file))
        columns_metadata = pd.read_csv(zf.open(cols_file))

    # Cache for next time
    expression.to_parquet(expr_path, index=False)
    rows_metadata.to_parquet(rows_path, index=False)
    columns_metadata.to_parquet(cols_path, index=False)

    logger.info("  Cached: %d genes x %d samples", expression.shape[0], expression.shape[1])
    return expression, rows_metadata, columns_metadata


def parse_age(age_str: str) -> tuple[str, float]:
    """Parse BrainSpan age string into (period, numeric_value).

    Examples: '8 pcw' -> ('prenatal', 8.0), '4 mos' -> ('postnatal', 0.33),
              '8 yrs' -> ('postnatal', 8.0)
    """
    parts = age_str.strip().split()
    value = float(parts[0])
    unit = parts[1].lower() if len(parts) > 1 else ""

    if "pcw" in unit:
        return ("prenatal", value)
    elif "mos" in unit:
        return ("postnatal", value / 12.0)
    elif "yrs" in unit or "yr" in unit:
        return ("postnatal", value)
    else:
        return ("unknown", value)


def classify_dev_stage(period: str, value: float) -> str:
    """Classify a sample into a developmental stage bin."""
    if period == "prenatal":
        for stage, (lo, hi) in DEV_STAGES.items():
            if "prenatal" in stage and lo <= value < hi:
                return stage
        return "late_prenatal"  # fallback for >38 pcw
    else:
        for stage, (lo, hi) in DEV_STAGES.items():
            if "prenatal" not in stage and lo <= value < hi:
                return stage
        return "adulthood"


def match_cstc_region(structure_name: str) -> str | None:
    """Match a BrainSpan structure name to a CSTC region category."""
    name_lower = structure_name.lower()
    for region, keywords in CSTC_REGIONS.items():
        if any(kw in name_lower for kw in keywords):
            return region
    return None


def extract_trajectories(
    gene_symbols: list[str],
    cache_dir: Path = CACHE_DIR,
) -> pd.DataFrame:
    """Extract developmental expression trajectories for genes in CSTC regions.

    Returns DataFrame with columns: gene_symbol, dev_stage, cstc_region,
    mean_rpkm, log2_rpkm, n_samples.
    """
    expression, rows_meta, cols_meta = download_brainspan(cache_dir)

    # Find gene symbol column
    gene_col = None
    for col in rows_meta.columns:
        if "gene_symbol" in col.lower() or "symbol" in col.lower():
            gene_col = col
            break
    if gene_col is None:
        raise ValueError(f"Cannot find gene symbol column in rows_metadata: {rows_meta.columns.tolist()}")

    # Map gene symbols to row indices
    gene_to_idx: dict[str, list[int]] = {}
    for idx, row in rows_meta.iterrows():
        sym = str(row[gene_col]).upper()
        if sym in [g.upper() for g in gene_symbols]:
            gene_to_idx.setdefault(sym, []).append(idx)

    found = set(gene_to_idx.keys())
    missing = set(g.upper() for g in gene_symbols) - found
    if missing:
        logger.warning("Genes not found in BrainSpan: %s", ", ".join(sorted(missing)))
    logger.info("Found %d/%d genes in BrainSpan", len(found), len(gene_symbols))

    # Parse sample metadata
    age_col = next(c for c in cols_meta.columns if "age" in c.lower())
    struct_col = next(
        (c for c in cols_meta.columns if "structure" in c.lower()),
        cols_meta.columns[-1],
    )

    sample_info: list[dict] = []
    for idx, row in cols_meta.iterrows():
        age_str = str(row[age_col])
        period, value = parse_age(age_str)
        stage = classify_dev_stage(period, value)
        cstc = match_cstc_region(str(row[struct_col]))
        sample_info.append({
            "sample_idx": idx,
            "age": age_str,
            "period": period,
            "age_value": value,
            "dev_stage": stage,
            "structure": str(row[struct_col]),
            "cstc_region": cstc,
        })

    sample_df = pd.DataFrame(sample_info)
    cstc_samples = sample_df[sample_df["cstc_region"].notna()]

    if cstc_samples.empty:
        logger.warning("No CSTC-relevant samples found in BrainSpan metadata")
        return pd.DataFrame()

    # Extract expression for each gene across CSTC samples and stages
    records: list[dict] = []
    for gene_upper, row_indices in gene_to_idx.items():
        # Average across probes/rows for same gene
        gene_expr = expression.iloc[row_indices].mean(axis=0)

        for _, sample in cstc_samples.iterrows():
            sidx = sample["sample_idx"]
            if sidx < len(gene_expr):
                rpkm = gene_expr.iloc[sidx]
                records.append({
                    "gene_symbol": gene_upper,
                    "dev_stage": sample["dev_stage"],
                    "cstc_region": sample["cstc_region"],
                    "rpkm": float(rpkm),
                    "log2_rpkm": float(np.log2(rpkm + 1)),
                    "age_value": sample["age_value"],
                    "period": sample["period"],
                })

    result = pd.DataFrame(records)

    if result.empty:
        return result

    # Aggregate: mean per gene x stage x region
    agg = (
        result.groupby(["gene_symbol", "dev_stage", "cstc_region"])
        .agg(
            mean_rpkm=("rpkm", "mean"),
            mean_log2_rpkm=("log2_rpkm", "mean"),
            n_samples=("rpkm", "count"),
        )
        .reset_index()
    )

    return agg


def identify_peak_windows(trajectories: pd.DataFrame) -> pd.DataFrame:
    """For each gene, identify the developmental stage with peak expression.

    Returns DataFrame with gene_symbol, peak_stage, peak_region, peak_expression.
    """
    if trajectories.empty:
        return pd.DataFrame()

    peaks = (
        trajectories
        .sort_values("mean_log2_rpkm", ascending=False)
        .groupby("gene_symbol")
        .first()
        .reset_index()
        [["gene_symbol", "dev_stage", "cstc_region", "mean_log2_rpkm"]]
        .rename(columns={
            "dev_stage": "peak_stage",
            "cstc_region": "peak_region",
            "mean_log2_rpkm": "peak_expression",
        })
    )
    return peaks


def cluster_temporal_patterns(
    trajectories: pd.DataFrame,
    n_clusters: int = 3,
) -> pd.DataFrame:
    """Cluster genes by their temporal expression patterns using k-means.

    Z-scores each gene's trajectory across stages, then clusters.
    Returns DataFrame with gene_symbol and cluster assignment.
    """
    from sklearn.cluster import KMeans

    if trajectories.empty:
        return pd.DataFrame()

    # Pivot: gene x stage (averaged across regions)
    avg = (
        trajectories
        .groupby(["gene_symbol", "dev_stage"])["mean_log2_rpkm"]
        .mean()
        .reset_index()
    )
    pivot = avg.pivot(index="gene_symbol", columns="dev_stage", values="mean_log2_rpkm")
    pivot = pivot.dropna(thresh=pivot.shape[1] // 2)  # drop genes with >50% missing

    if pivot.shape[0] < n_clusters:
        logger.warning("Too few genes (%d) for %d clusters", pivot.shape[0], n_clusters)
        return pd.DataFrame()

    # Z-score per gene (row)
    zscore = pivot.apply(
        lambda row: (row - row.mean()) / row.std() if row.std() > 0 else row * 0,
        axis=1,
    ).fillna(0)

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(zscore)

    result = pd.DataFrame({
        "gene_symbol": pivot.index,
        "cluster": labels,
    })

    # Add cluster characterization (which stage has highest centroid)
    for i in range(n_clusters):
        centroid = km.cluster_centers_[i]
        peak_idx = np.argmax(centroid)
        peak_stage = zscore.columns[peak_idx]
        count = (labels == i).sum()
        logger.info("  Cluster %d (%d genes): peak at %s", i, count, peak_stage)

    return result


def generate_trajectory_plot(
    trajectories: pd.DataFrame,
    output_path: Path,
    title: str = "TS Risk Gene Developmental Trajectories",
) -> Path:
    """Generate developmental trajectory line plots for TS risk genes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stage_order = list(DEV_STAGES.keys())

    fig, axes = plt.subplots(1, len(CSTC_REGIONS), figsize=(18, 6), sharey=True)
    if not hasattr(axes, "__iter__"):
        axes = [axes]

    for ax, region in zip(axes, CSTC_REGIONS.keys()):
        region_data = trajectories[trajectories["cstc_region"] == region]
        if region_data.empty:
            ax.set_title(f"{region.title()}\n(no data)")
            continue

        pivot = region_data.pivot(
            index="dev_stage", columns="gene_symbol", values="mean_log2_rpkm"
        )
        # Reorder stages
        available = [s for s in stage_order if s in pivot.index]
        pivot = pivot.loc[available]

        for gene in pivot.columns:
            ax.plot(range(len(available)), pivot[gene], alpha=0.5, linewidth=1)

        # Plot mean trajectory
        mean_traj = pivot.mean(axis=1)
        ax.plot(range(len(available)), mean_traj, color="red", linewidth=2.5,
                label="Mean", zorder=10)

        ax.set_xticks(range(len(available)))
        ax.set_xticklabels([s.replace("_", "\n") for s in available],
                           fontsize=7, rotation=45, ha="right")
        ax.set_title(f"{region.title()}", fontsize=11)
        if region == list(CSTC_REGIONS.keys())[0]:
            ax.set_ylabel("log2(RPKM + 1)")
        ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved trajectory plot to %s", output_path)
    return output_path


def run_analysis(
    gene_set_name: str = "ts_combined",
    output_dir: Path = OUTPUT_DIR,
    n_clusters: int = 3,
) -> dict:
    """Run the full BrainSpan developmental trajectory analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)

    genes = get_gene_set(gene_set_name)
    gene_list = sorted(genes.keys())
    logger.info("Analyzing %d genes from set '%s'", len(gene_list), gene_set_name)

    # Extract trajectories
    trajectories = extract_trajectories(gene_list)
    if trajectories.empty:
        return {"error": "No trajectory data extracted"}

    # Peak windows
    peaks = identify_peak_windows(trajectories)

    # Clustering
    clusters = cluster_temporal_patterns(trajectories, n_clusters=n_clusters)

    # Save outputs
    traj_path = output_dir / f"dev_trajectories_{gene_set_name}.csv"
    trajectories.to_csv(traj_path, index=False)

    peaks_path = output_dir / f"dev_peak_windows_{gene_set_name}.csv"
    peaks.to_csv(peaks_path, index=False)

    if not clusters.empty:
        clusters_path = output_dir / f"dev_clusters_{gene_set_name}.csv"
        clusters.to_csv(clusters_path, index=False)

    plot_path = output_dir / f"dev_trajectories_{gene_set_name}.png"
    generate_trajectory_plot(trajectories, plot_path)

    # Summary stats
    stage_counts = peaks["peak_stage"].value_counts().to_dict() if not peaks.empty else {}

    summary = {
        "gene_set": gene_set_name,
        "n_genes_queried": len(gene_list),
        "n_genes_found": trajectories["gene_symbol"].nunique(),
        "peak_stage_distribution": stage_counts,
        "n_clusters": n_clusters if not clusters.empty else 0,
    }

    summary_path = output_dir / f"dev_summary_{gene_set_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="BrainSpan developmental trajectory analysis for TS risk genes"
    )
    parser.add_argument("--gene-set", default="ts_combined", choices=list_gene_sets())
    parser.add_argument("--clusters", type=int, default=3, help="Number of temporal clusters")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = run_analysis(args.gene_set, args.output, args.clusters)

    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    print(f"\nBrainSpan analysis complete for: {results['gene_set']}")
    print(f"  Genes found: {results['n_genes_found']}/{results['n_genes_queried']}")
    print(f"  Peak stage distribution:")
    for stage, count in results.get("peak_stage_distribution", {}).items():
        print(f"    {stage}: {count} genes")


if __name__ == "__main__":
    main()
