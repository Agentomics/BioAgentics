"""Compute innate-to-adaptive immune gene expression ratio.

Accepts both scRNA-seq pseudobulk and bulk RNA-seq count matrices.
Uses gene modules from innate_immunity_modules to compute per-sample
ratios of innate vs adaptive immune gene expression.

Usage::

    uv run python -m pandas_pans.innate_immunity_deficiency_model.innate_adaptive_ratio
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.config import DATA_DIR, REPO_ROOT
from bioagentics.pandas_pans.innate_immunity_modules import (
    INNATE_MODULES,
    ADAPTIVE_MODULES,
    get_all_adaptive_genes,
    get_all_innate_genes,
)

logger = logging.getLogger(__name__)

BULK_PATH = DATA_DIR / "pandas_pans" / "ivig-mechanism-single-cell-analysis" / "bulk" / "GSE278678_pans_ivig_counts.csv.gz"
RAW_DIR = DATA_DIR / "pandas_pans" / "ivig-mechanism-single-cell-analysis" / "raw"
OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "innate-immunity-deficiency-model"


def _assign_condition(sample: str) -> str:
    """Map sample name to condition."""
    if sample.startswith("Control"):
        return "control"
    elif sample.startswith("PreIVIG"):
        return "pre"
    elif sample.startswith("PostIVIG"):
        return "post"
    elif sample.startswith("Secondpost"):
        return "second_post"
    return "unknown"


def compute_ratio_from_bulk(
    bulk_path: Path | None = None,
    innate_genes: list[str] | None = None,
    adaptive_genes: list[str] | None = None,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """Compute innate/adaptive gene expression ratio from bulk RNA-seq counts.

    For each sample, sums expression of innate module genes and adaptive
    module genes, then computes the log2 ratio.

    Args:
        bulk_path: Path to bulk counts CSV (gene × sample).
        innate_genes: List of innate gene symbols.
        adaptive_genes: List of adaptive gene symbols.
        pseudocount: Added before log2 to avoid division by zero.

    Returns:
        DataFrame with per-sample ratios: sample, condition,
        innate_sum, adaptive_sum, ratio, log2_ratio, and per-module sums.
    """
    if bulk_path is None:
        bulk_path = BULK_PATH
    if innate_genes is None:
        innate_genes = get_all_innate_genes()
    if adaptive_genes is None:
        adaptive_genes = get_all_adaptive_genes()

    df = pd.read_csv(bulk_path)
    sample_cols = [c for c in df.columns if c not in ("Unnamed: 0", "ENTREZID", "SYMBOL")]

    # Build gene-to-expression matrix
    df = df.set_index("SYMBOL")
    df = df[sample_cols]

    results: list[dict] = []
    for sample in sample_cols:
        row: dict = {
            "sample": sample,
            "condition": _assign_condition(sample),
        }

        # Per-module innate sums
        innate_total = 0.0
        for mod_name, mod_genes in INNATE_MODULES.items():
            available = [g for g in mod_genes if g in df.index]
            mod_sum = float(df.loc[available, sample].sum()) if available else 0.0
            row[f"innate_{mod_name}"] = mod_sum
            innate_total += mod_sum

        # Per-module adaptive sums
        adaptive_total = 0.0
        for mod_name, mod_genes in ADAPTIVE_MODULES.items():
            available = [g for g in mod_genes if g in df.index]
            mod_sum = float(df.loc[available, sample].sum()) if available else 0.0
            row[f"adaptive_{mod_name}"] = mod_sum
            adaptive_total += mod_sum

        row["innate_sum"] = innate_total
        row["adaptive_sum"] = adaptive_total
        row["ratio"] = innate_total / (adaptive_total + pseudocount)
        row["log2_ratio"] = np.log2((innate_total + pseudocount) / (adaptive_total + pseudocount))

        results.append(row)

    result_df = pd.DataFrame(results)
    return result_df.sort_values(by=["condition", "sample"])


def compute_ratio_from_scrna(
    raw_dir: Path | None = None,
    innate_genes: list[str] | None = None,
    adaptive_genes: list[str] | None = None,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """Compute innate/adaptive ratio from scRNA-seq TCM files per sample.

    Processes each TCM file one at a time (memory-safe). For each sample,
    sums expression of innate and adaptive genes across all cells.

    Returns:
        DataFrame with per-sample pseudobulk ratios.
    """
    if raw_dir is None:
        raw_dir = RAW_DIR
    if innate_genes is None:
        innate_genes = get_all_innate_genes()
    if adaptive_genes is None:
        adaptive_genes = get_all_adaptive_genes()

    all_target = list(set(innate_genes + adaptive_genes))

    tcm_files = sorted(raw_dir.glob("*_TCM.tsv.gz"))
    if not tcm_files:
        logger.warning("No TCM files found in %s", raw_dir)
        return pd.DataFrame()

    results: list[dict] = []

    for tcm_path in tcm_files:
        from pandas_pans.innate_immunity_deficiency_model.lectin_complement_extraction import (
            _parse_sample_info,
            _load_tcm_target_genes,
        )

        sample_info = _parse_sample_info(tcm_path.name)
        logger.info("  Processing %s", sample_info["patient"])

        expr_df, _found = _load_tcm_target_genes(tcm_path, all_target)
        if expr_df.empty:
            continue

        n_cells = expr_df.shape[1]

        row: dict = {
            "patient": sample_info["patient"],
            "condition": sample_info["condition"],
            "gsm": sample_info["gsm"],
            "n_cells": n_cells,
        }

        # Sum per module
        innate_total = 0.0
        for mod_name in INNATE_MODULES:
            mod_genes = [g for g in INNATE_MODULES[mod_name] if g in expr_df.index]
            mod_sum = float(expr_df.loc[mod_genes].values.sum()) if mod_genes else 0.0
            row[f"innate_{mod_name}"] = mod_sum
            innate_total += mod_sum

        adaptive_total = 0.0
        for mod_name in ADAPTIVE_MODULES:
            mod_genes = [g for g in ADAPTIVE_MODULES[mod_name] if g in expr_df.index]
            mod_sum = float(expr_df.loc[mod_genes].values.sum()) if mod_genes else 0.0
            row[f"adaptive_{mod_name}"] = mod_sum
            adaptive_total += mod_sum

        row["innate_sum"] = innate_total
        row["adaptive_sum"] = adaptive_total
        row["ratio"] = innate_total / (adaptive_total + pseudocount)
        row["log2_ratio"] = np.log2((innate_total + pseudocount) / (adaptive_total + pseudocount))

        # Normalize by cell count for per-cell comparison
        row["innate_per_cell"] = innate_total / n_cells
        row["adaptive_per_cell"] = adaptive_total / n_cells

        results.append(row)
        del expr_df

    return pd.DataFrame(results).sort_values(by=["condition", "patient"])


def run_ratio_pipeline() -> dict[str, Path]:
    """Run full innate/adaptive ratio computation."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # Bulk
    logger.info("=== Computing bulk innate/adaptive ratios ===")
    bulk_ratios = compute_ratio_from_bulk()
    if not bulk_ratios.empty:
        path = OUTPUT_DIR / "innate_adaptive_ratio_bulk.csv"
        bulk_ratios.to_csv(path, index=False)
        outputs["bulk_ratios"] = path
        logger.info("Saved bulk ratios: %s", path)

        # Summary per condition
        summary = bulk_ratios.groupby("condition").agg(
            mean_ratio=("ratio", "mean"),
            std_ratio=("ratio", "std"),
            mean_log2=("log2_ratio", "mean"),
            n=("sample", "count"),
        ).reset_index()
        logger.info("Bulk ratio summary:\n%s", summary.to_string(index=False))

    # scRNA-seq
    logger.info("=== Computing scRNA-seq innate/adaptive ratios ===")
    scrna_ratios = compute_ratio_from_scrna()
    if not scrna_ratios.empty:
        path = OUTPUT_DIR / "innate_adaptive_ratio_scrna.csv"
        scrna_ratios.to_csv(path, index=False)
        outputs["scrna_ratios"] = path
        logger.info("Saved scRNA ratios: %s", path)

        summary = scrna_ratios.groupby("condition").agg(
            mean_ratio=("ratio", "mean"),
            std_ratio=("ratio", "std"),
            mean_log2=("log2_ratio", "mean"),
            n=("patient", "count"),
        ).reset_index()
        logger.info("scRNA ratio summary:\n%s", summary.to_string(index=False))

    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_ratio_pipeline()
