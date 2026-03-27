"""Prefrontal cortex cell-type expression profiling of TS risk genes.

Maps Tourette syndrome risk gene expression onto PFC cell types using the
Wang group DLPFC snRNA-seq dataset (72,340 cells from 9 TS/CTL donors;
GEO GSE311334; doi:10.64898/2026.01.14.699521).

This fills the cortical (C) node of the CSTC circuit atlas by providing
direct single-cell resolution of TS risk gene expression in PFC cell types,
complementing the AHBA-based striatal deconvolution.

Cell types profiled (from Allen Brain reference mapping):
  Excitatory: IT (L2/3), L4 IT, L5 ET, L5/6 NP, L6 CT, L6b
  Inhibitory: PVALB, SST, VIP, LAMP5
  Non-neuronal: Astrocyte, Oligodendrocyte, OPC, Microglia

Dependencies:
  - bioagentics.data.tourettes.gene_sets (TS risk genes)
  - Wang DLPFC data in data/tourettes/cstc-circuit-expression-atlas/wang_dlpfc_snrnaseq/

Usage:
    uv run python -m bioagentics.analysis.tourettes.pfc_celltypes
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.tourettes.gene_sets import (
    get_gene_set,
    list_gene_sets,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "cstc-circuit-expression-atlas"
DATA_DIR = (
    REPO_ROOT
    / "data"
    / "tourettes"
    / "cstc-circuit-expression-atlas"
    / "wang_dlpfc_snrnaseq"
)
COUNTS_FILE = DATA_DIR / "GSE311334_counts_matrix_24149.csv.gz"
METADATA_FILE = DATA_DIR / "GSE311334_metadata_24149.csv.gz"

# PFC cell types grouped from predicted.subclass_label
PFC_CELL_TYPES: list[str] = [
    "IT",
    "L4 IT",
    "L5 ET",
    "L5/6 NP",
    "L5/6 IT Car3",
    "L6 CT",
    "L6b",
    "PVALB",
    "SST",
    "VIP",
    "LAMP5",
    "Astrocyte",
    "Oligodendrocyte",
    "OPC",
    "Microglia",
]

# Broad category mapping for summary statistics
BROAD_CATEGORIES: dict[str, str] = {
    "IT": "Excitatory",
    "L4 IT": "Excitatory",
    "L5 ET": "Excitatory",
    "L5/6 NP": "Excitatory",
    "L5/6 IT Car3": "Excitatory",
    "L6 CT": "Excitatory",
    "L6b": "Excitatory",
    "PVALB": "Inhibitory",
    "SST": "Inhibitory",
    "VIP": "Inhibitory",
    "LAMP5": "Inhibitory",
    "Astrocyte": "Non-neuronal",
    "Oligodendrocyte": "Non-neuronal",
    "OPC": "Non-neuronal",
    "Microglia": "Non-neuronal",
}

# Canonical immediate early genes for overlap analysis
IEG_GENES: list[str] = [
    "FOS",
    "FOSB",
    "JUN",
    "JUNB",
    "EGR1",
    "EGR2",
    "EGR3",
    "ARC",
    "NPAS4",
    "NR4A1",
    "NR4A2",
    "NR4A3",
    "IER2",
    "BTG2",
    "DUSP1",
    "ATF3",
]


def load_metadata(path: Path = METADATA_FILE) -> pd.DataFrame:
    """Load cell metadata and filter to annotated PFC cell types.

    Returns DataFrame indexed by cell barcode with columns:
    group, Cell_type, predicted.subclass_label, IEG_Module, etc.
    """
    logger.info("Loading metadata from %s", path)
    meta = pd.read_csv(path, index_col=0)
    logger.info("  %d total cells loaded", len(meta))
    logger.info(
        "  Groups: %s",
        dict(meta["group"].value_counts()),
    )

    # Filter to cells with valid PFC subclass labels
    valid = meta[meta["predicted.subclass_label"].isin(PFC_CELL_TYPES)]
    logger.info(
        "  %d cells with valid PFC cell-type labels (of %d)",
        len(valid),
        len(meta),
    )
    return valid


def extract_gene_expression(
    gene_symbols: list[str],
    counts_path: Path = COUNTS_FILE,
) -> pd.DataFrame:
    """Stream counts matrix and extract rows for requested genes only.

    Memory-safe: reads one gene (row) at a time, keeping only matches.
    The counts matrix is genes x cells (32,338 genes x 72,340 cells).

    Returns DataFrame: rows = genes, columns = cell barcodes, values = counts.
    """
    target_set = {g.upper() for g in gene_symbols}
    logger.info(
        "Extracting %d genes from counts matrix (streaming)...",
        len(target_set),
    )

    found: dict[str, np.ndarray] = {}
    cell_ids: list[str] | None = None

    with gzip.open(counts_path, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)
        cell_ids = header  # cell barcodes (no gene name column in header row 0)

        for row in reader:
            gene_name = row[0]
            if gene_name.upper() in target_set:
                counts = np.array(row[1:], dtype=np.float32)
                found[gene_name] = counts
                if len(found) == len(target_set):
                    break  # all genes found, stop early

    logger.info("  Found %d of %d requested genes", len(found), len(target_set))

    if not found:
        return pd.DataFrame()

    df = pd.DataFrame(found, index=cell_ids).T
    df.index.name = "gene_symbol"
    return df


def compute_celltype_mean_expression(
    expr_df: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mean expression per gene per PFC cell type.

    Parameters
    ----------
    expr_df
        Genes x cells DataFrame from extract_gene_expression.
    metadata
        Cell metadata with predicted.subclass_label column.

    Returns
    -------
    pd.DataFrame
        Rows = genes, columns = PFC cell types, values = mean CPM-normalized
        expression.
    """
    # Align cells present in both expression and metadata
    common_cells = expr_df.columns.intersection(metadata.index)
    if len(common_cells) == 0:
        logger.warning("No cells in common between expression and metadata")
        return pd.DataFrame()

    expr = expr_df[common_cells]
    meta = metadata.loc[common_cells]
    logger.info("  %d cells in common for mean expression", len(common_cells))

    # Compute per-cell library size for CPM normalization
    lib_sizes = expr.sum(axis=0)
    # Avoid division by zero for cells with no counts for these genes
    lib_sizes = lib_sizes.replace(0, 1)

    records: list[dict] = []
    for gene in expr.index:
        row: dict[str, object] = {"gene_symbol": gene}
        gene_counts = expr.loc[gene]

        for ct in PFC_CELL_TYPES:
            ct_cells = meta.index[meta["predicted.subclass_label"] == ct]
            ct_cells = ct_cells.intersection(gene_counts.index)
            if len(ct_cells) == 0:
                row[ct] = 0.0
                continue
            # Mean of raw counts per cell type
            row[ct] = float(gene_counts[ct_cells].mean())

        records.append(row)

    return pd.DataFrame(records).set_index("gene_symbol")


def compute_tau_specificity(expression_vector: np.ndarray) -> float:
    """Compute tau specificity index (Yanai et al. 2005).

    0 = ubiquitous, 1 = highly cell-type-specific.
    """
    vals = expression_vector[~np.isnan(expression_vector)]
    if len(vals) < 2:
        return np.nan
    vals = vals - vals.min()
    max_val = vals.max()
    if max_val == 0:
        return 0.0
    n = len(vals)
    hat = vals / max_val
    return float(np.sum(1 - hat) / (n - 1))


def compute_specificity_scores(
    mean_expr: pd.DataFrame,
) -> pd.DataFrame:
    """Add tau specificity and top cell type to mean expression matrix.

    Returns the input DataFrame augmented with tau, top_cell_type,
    and broad_category columns.
    """
    ct_cols = [c for c in PFC_CELL_TYPES if c in mean_expr.columns]

    taus: list[float] = []
    top_cts: list[str] = []
    top_cats: list[str] = []

    for _, row in mean_expr.iterrows():
        vals = row[ct_cols].values.astype(float)
        taus.append(compute_tau_specificity(vals))
        top_ct = ct_cols[int(np.nanargmax(vals))] if not np.all(np.isnan(vals)) else "none"
        top_cts.append(top_ct)
        top_cats.append(BROAD_CATEGORIES.get(top_ct, "unknown"))

    result = mean_expr.copy()
    result["tau"] = taus
    result["top_cell_type"] = top_cts
    result["broad_category"] = top_cats
    return result


def compute_td_vs_ctl_enrichment(
    expr_df: pd.DataFrame,
    metadata: pd.DataFrame,
    gene_symbols: list[str],
) -> pd.DataFrame:
    """Test differential expression of TS risk genes between TD and CTL per cell type.

    Uses Wilcoxon rank-sum test per gene per cell type.
    Returns DataFrame with columns: gene_symbol, cell_type, mean_td, mean_ctl,
    log2fc, p_value, significant.
    """
    from scipy.stats import mannwhitneyu

    common_cells = expr_df.columns.intersection(metadata.index)
    expr = expr_df[common_cells]
    meta = metadata.loc[common_cells]

    records: list[dict] = []
    n_tests = 0

    for ct in PFC_CELL_TYPES:
        ct_mask = meta["predicted.subclass_label"] == ct
        td_cells = meta.index[ct_mask & (meta["group"] == "TD")]
        ctl_cells = meta.index[ct_mask & (meta["group"] == "CTL")]

        td_cells = td_cells.intersection(expr.columns)
        ctl_cells = ctl_cells.intersection(expr.columns)

        if len(td_cells) < 5 or len(ctl_cells) < 5:
            continue

        for gene in gene_symbols:
            if gene not in expr.index:
                continue

            td_vals = expr.loc[gene, td_cells].values.astype(float)
            ctl_vals = expr.loc[gene, ctl_cells].values.astype(float)

            mean_td = float(np.mean(td_vals))
            mean_ctl = float(np.mean(ctl_vals))

            # Log2 fold change (add pseudocount)
            pseudo = 0.01
            log2fc = float(np.log2((mean_td + pseudo) / (mean_ctl + pseudo)))

            # Wilcoxon rank-sum (Mann-Whitney U)
            try:
                _, p_val = mannwhitneyu(td_vals, ctl_vals, alternative="two-sided")
            except ValueError:
                p_val = 1.0

            n_tests += 1
            records.append({
                "gene_symbol": gene,
                "cell_type": ct,
                "n_td": len(td_cells),
                "n_ctl": len(ctl_cells),
                "mean_td": mean_td,
                "mean_ctl": mean_ctl,
                "log2fc": log2fc,
                "p_value": float(p_val),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Bonferroni correction
    df["significant"] = df["p_value"] < (0.05 / max(n_tests, 1))
    return df


def compute_ieg_overlap(
    specificity_df: pd.DataFrame,
    metadata: pd.DataFrame,
    expr_df: pd.DataFrame,
) -> dict:
    """Test whether IEG-high cell types also show elevated TS risk gene expression.

    Uses the IEG_Module score from metadata to identify IEG-activated cells,
    then tests whether TS risk genes are more highly expressed in those cells.
    """
    common_cells = expr_df.columns.intersection(metadata.index)
    meta = metadata.loc[common_cells]
    expr = expr_df[common_cells]

    # Check if IEG_Module is available and numeric
    if "IEG_Module" not in meta.columns:
        logger.warning("IEG_Module column not found in metadata")
        return {"available": False}

    ieg_scores = pd.to_numeric(meta["IEG_Module"], errors="coerce")
    if ieg_scores.isna().all():
        return {"available": False}

    # Identify IEG-high cells (top quartile per cell type)
    results_per_ct: dict[str, dict] = {}

    for ct in PFC_CELL_TYPES:
        ct_mask = meta["predicted.subclass_label"] == ct
        ct_ieg = ieg_scores[ct_mask].dropna()
        if len(ct_ieg) < 20:
            continue

        threshold = ct_ieg.quantile(0.75)
        high_cells = ct_ieg.index[ct_ieg >= threshold]
        low_cells = ct_ieg.index[ct_ieg < threshold]

        high_cells = high_cells.intersection(expr.columns)
        low_cells = low_cells.intersection(expr.columns)

        if len(high_cells) < 5 or len(low_cells) < 5:
            continue

        # Mean TS risk gene expression in IEG-high vs IEG-low cells
        ts_genes_in_data = [g for g in specificity_df.index if g in expr.index]
        if not ts_genes_in_data:
            continue

        mean_high = float(expr.loc[ts_genes_in_data, high_cells].values.mean())
        mean_low = float(expr.loc[ts_genes_in_data, low_cells].values.mean())

        results_per_ct[ct] = {
            "n_ieg_high": len(high_cells),
            "n_ieg_low": len(low_cells),
            "mean_ts_expr_ieg_high": mean_high,
            "mean_ts_expr_ieg_low": mean_low,
            "ratio": mean_high / mean_low if mean_low > 0 else float("inf"),
        }

    # Also check overlap of IEG genes with TS risk genes
    ts_genes = set(specificity_df.index)
    ieg_set = set(IEG_GENES)
    overlap = ts_genes & ieg_set

    return {
        "available": True,
        "ts_ieg_overlap_genes": sorted(overlap),
        "n_overlap": len(overlap),
        "per_cell_type": results_per_ct,
    }


def generate_specificity_heatmap(
    specificity_df: pd.DataFrame,
    output_path: Path,
    title: str = "TS Risk Gene PFC Cell-Type Expression",
) -> Path:
    """Generate heatmap of gene x PFC cell-type mean expression."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ct_cols = [c for c in PFC_CELL_TYPES if c in specificity_df.columns]
    plot_data = specificity_df[ct_cols].copy()

    # Z-score across cell types for visualization
    row_means = plot_data.mean(axis=1)
    row_stds = plot_data.std(axis=1).replace(0, 1)
    plot_data = plot_data.sub(row_means, axis=0).div(row_stds, axis=0)

    fig, ax = plt.subplots(figsize=(14, max(6, len(plot_data) * 0.35)))
    sns.heatmap(
        plot_data,
        cmap="RdBu_r",
        center=0,
        annot=False,
        linewidths=0.3,
        ax=ax,
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel("TS Risk Gene")
    ax.set_xlabel("PFC Cell Type")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved specificity heatmap to %s", output_path)
    return output_path


def generate_cortical_vs_subcortical_plot(
    pfc_specificity: pd.DataFrame,
    striatal_csv: Path | None,
    output_path: Path,
) -> Path | None:
    """Compare PFC vs striatal cell-type vulnerability for TS risk genes.

    If striatal specificity CSV exists, generates a scatter/bar comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if striatal_csv is None or not striatal_csv.exists():
        logger.info("Striatal specificity CSV not found; skipping comparison plot")
        return None

    striatal = pd.read_csv(striatal_csv, index_col=0)

    # Find genes in common
    common_genes = pfc_specificity.index.intersection(striatal.index)
    if len(common_genes) < 3:
        logger.warning("Too few common genes (%d) for comparison", len(common_genes))
        return None

    # Compute max specificity per gene in each region
    pfc_ct_cols = [c for c in PFC_CELL_TYPES if c in pfc_specificity.columns]
    striatal_ct_cols = [c for c in striatal.columns if c not in ("tau", "top_cell_type")]

    pfc_max = pfc_specificity.loc[common_genes, pfc_ct_cols].max(axis=1)
    striatal_max = striatal.loc[common_genes, striatal_ct_cols].max(axis=1)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(striatal_max, pfc_max, alpha=0.7, s=40)
    for gene in common_genes:
        ax.annotate(gene, (striatal_max[gene], pfc_max[gene]), fontsize=7, alpha=0.7)

    lim = max(pfc_max.max(), striatal_max.max()) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="Equal")
    ax.set_xlabel("Max Striatal Cell-Type Expression")
    ax.set_ylabel("Max PFC Cell-Type Expression")
    ax.set_title("Cortical vs Subcortical TS Risk Gene Expression", fontsize=13)
    ax.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved cortical vs subcortical comparison to %s", output_path)
    return output_path


def run_analysis(
    gene_set_name: str = "ts_combined",
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Run the full PFC cell-type expression profiling analysis.

    Steps:
    1. Load metadata and filter to annotated PFC cell types
    2. Extract TS risk gene expression from counts matrix (streaming)
    3. Compute mean expression per cell type
    4. Compute tau specificity scores
    5. TD vs CTL differential expression per cell type
    6. IEG overlap analysis
    7. Generate heatmap and comparison plots
    8. Save results

    Returns dict with results summary and output paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load metadata
    logger.info("Step 1: Loading metadata")
    metadata = load_metadata()
    if metadata.empty:
        return {"error": "No valid metadata loaded"}

    # 2. Get gene list and extract expression
    logger.info("Step 2: Extracting TS risk gene expression")
    risk_genes = get_gene_set(gene_set_name)
    risk_symbols = sorted(risk_genes.keys())
    logger.info("  %d TS risk genes in set '%s'", len(risk_symbols), gene_set_name)

    # Also extract IEG genes for overlap analysis
    all_genes = sorted(set(risk_symbols) | set(IEG_GENES))
    expr_df = extract_gene_expression(all_genes)

    if expr_df.empty:
        return {"error": "No TS risk genes found in counts matrix"}

    ts_expr = expr_df.loc[expr_df.index.intersection(risk_symbols)]
    logger.info("  %d TS risk genes found in data", len(ts_expr))

    # 3. Compute mean expression per cell type
    logger.info("Step 3: Computing mean expression per PFC cell type")
    mean_expr = compute_celltype_mean_expression(ts_expr, metadata)

    if mean_expr.empty:
        return {"error": "Could not compute mean expression"}

    # 4. Add specificity scores
    logger.info("Step 4: Computing tau specificity")
    specificity = compute_specificity_scores(mean_expr)

    specificity_path = output_dir / f"pfc_specificity_{gene_set_name}.csv"
    specificity.to_csv(specificity_path)
    logger.info("Saved specificity matrix: %s", specificity_path)

    # 5. TD vs CTL differential expression
    logger.info("Step 5: Testing TD vs CTL differential expression")
    de_results = compute_td_vs_ctl_enrichment(
        ts_expr, metadata, risk_symbols,
    )
    de_path = output_dir / f"pfc_td_vs_ctl_{gene_set_name}.csv"
    if not de_results.empty:
        de_results.to_csv(de_path, index=False)
        logger.info("Saved DE results: %s", de_path)
        n_sig = int(de_results["significant"].sum())
        logger.info("  %d significant gene-cell type pairs", n_sig)
    else:
        n_sig = 0
        logger.warning("No DE results computed")

    # 6. IEG overlap analysis
    logger.info("Step 6: IEG overlap analysis")
    ieg_results = compute_ieg_overlap(specificity, metadata, expr_df)

    ieg_path = output_dir / f"pfc_ieg_overlap_{gene_set_name}.json"
    with open(ieg_path, "w") as f:
        json.dump(ieg_results, f, indent=2, default=str)
    logger.info("Saved IEG overlap: %s", ieg_path)

    # 7. Generate figures
    logger.info("Step 7: Generating figures")
    heatmap_path = output_dir / f"pfc_specificity_heatmap_{gene_set_name}.png"
    generate_specificity_heatmap(
        specificity,
        heatmap_path,
        title=f"TS Risk Gene PFC Cell-Type Expression ({gene_set_name})",
    )

    # Cortical vs subcortical comparison
    striatal_csv = output_dir / f"striatal_specificity_{gene_set_name}.csv"
    comparison_path = output_dir / f"pfc_vs_striatal_{gene_set_name}.png"
    generate_cortical_vs_subcortical_plot(
        specificity, striatal_csv, comparison_path,
    )

    # 8. Summary
    ct_counts = metadata["predicted.subclass_label"].value_counts().to_dict()
    top_genes_per_ct: dict[str, list[str]] = {}
    ct_cols = [c for c in PFC_CELL_TYPES if c in specificity.columns]
    for ct in ct_cols:
        top = specificity[ct].nlargest(3).index.tolist()
        if top:
            top_genes_per_ct[ct] = top

    summary: dict = {
        "gene_set": gene_set_name,
        "n_risk_genes_scored": len(specificity),
        "n_cells_total": len(metadata),
        "n_td_cells": int((metadata["group"] == "TD").sum()),
        "n_ctl_cells": int((metadata["group"] == "CTL").sum()),
        "n_pfc_cell_types": len(ct_cols),
        "cells_per_type": {k: ct_counts.get(k, 0) for k in PFC_CELL_TYPES},
        "mean_tau": float(specificity["tau"].mean()),
        "top_genes_per_cell_type": top_genes_per_ct,
        "n_de_significant": n_sig,
        "ieg_overlap": {
            "n_ts_ieg_genes": ieg_results.get("n_overlap", 0),
            "overlapping_genes": ieg_results.get("ts_ieg_overlap_genes", []),
        },
        "outputs": {
            "specificity_csv": str(specificity_path),
            "de_csv": str(de_path),
            "ieg_json": str(ieg_path),
            "heatmap_png": str(heatmap_path),
            "comparison_png": str(comparison_path),
        },
    }

    summary_path = output_dir / f"pfc_celltypes_summary_{gene_set_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary: %s", summary_path)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PFC cell-type expression profiling of TS risk genes"
    )
    parser.add_argument(
        "--gene-set",
        default="ts_combined",
        choices=list_gene_sets(),
        help="Gene set to analyze (default: ts_combined)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = run_analysis(
        gene_set_name=args.gene_set,
        output_dir=args.output,
    )

    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    print(f"\nPFC cell-type profiling complete: {results['gene_set']}")
    print(f"  Risk genes scored: {results['n_risk_genes_scored']}")
    print(f"  Cells: {results['n_cells_total']} ({results['n_td_cells']} TD, {results['n_ctl_cells']} CTL)")
    print(f"  PFC cell types: {results['n_pfc_cell_types']}")
    print(f"  Mean tau specificity: {results['mean_tau']:.3f}")
    print(f"  Significant DE pairs: {results['n_de_significant']}")
    ieg = results["ieg_overlap"]
    if ieg["n_ts_ieg_genes"]:
        print(f"  TS-IEG overlap: {ieg['n_ts_ieg_genes']} genes ({', '.join(ieg['overlapping_genes'])})")
    print("\nOutputs:")
    for name, path in results["outputs"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
