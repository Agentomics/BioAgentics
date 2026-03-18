"""Step 01 — Download and validate BrainSpan RNA-seq data.

Downloads the BrainSpan Atlas of the Developing Human Brain RNA-seq dataset,
caches it locally, and produces a validation report for downstream trajectory
analysis.

Task: #778 (BrainSpan data download)
Project: ts-developmental-trajectory-modeling

Usage:
    uv run python -m bioagentics.tourettes.developmental_trajectory.01_download_brainspan
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
    download_brainspan,
    match_cstc_region,
    parse_age,
    classify_dev_stage,
)
from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-developmental-trajectory-modeling"


def validate_brainspan(
    expression: pd.DataFrame,
    rows_meta: pd.DataFrame,
    cols_meta: pd.DataFrame,
) -> dict:
    """Validate BrainSpan data and produce summary statistics.

    Returns a validation report dict with data quality metrics.
    """
    report: dict = {
        "n_genes": expression.shape[0],
        "n_samples": expression.shape[1],
        "rows_meta_columns": rows_meta.columns.tolist(),
        "cols_meta_columns": cols_meta.columns.tolist(),
    }

    # Gene coverage
    gene_col = next(
        (c for c in rows_meta.columns if "gene_symbol" in c.lower() or "symbol" in c.lower()),
        None,
    )
    if gene_col:
        symbols = rows_meta[gene_col].dropna().unique()
        report["n_unique_gene_symbols"] = len(symbols)
    else:
        report["n_unique_gene_symbols"] = None
        report["warning_no_gene_symbol_column"] = True

    # Sample age coverage
    age_col = next((c for c in cols_meta.columns if "age" in c.lower()), None)
    if age_col:
        ages = cols_meta[age_col].dropna().unique()
        report["n_unique_ages"] = len(ages)
        report["age_values"] = sorted(ages.tolist())

        # Parse developmental stages
        stage_counts: dict[str, int] = {}
        for age_str in ages:
            period, value = parse_age(str(age_str))
            stage = classify_dev_stage(period, value)
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        report["dev_stage_coverage"] = stage_counts

    # CSTC region coverage
    struct_col = next(
        (c for c in cols_meta.columns if "structure" in c.lower()),
        None,
    )
    if struct_col:
        structures = cols_meta[struct_col].dropna().unique()
        report["n_unique_structures"] = len(structures)

        cstc_counts: dict[str, int] = {}
        non_cstc = 0
        for s in structures:
            region = match_cstc_region(str(s))
            if region:
                cstc_counts[region] = cstc_counts.get(region, 0) + 1
            else:
                non_cstc += 1
        report["cstc_region_coverage"] = cstc_counts
        report["n_non_cstc_structures"] = non_cstc

    # Expression data quality
    expr_values = expression.values
    report["expression_stats"] = {
        "min": float(np.nanmin(expr_values)),
        "max": float(np.nanmax(expr_values)),
        "mean": float(np.nanmean(expr_values)),
        "median": float(np.nanmedian(expr_values)),
        "pct_zero": float((expr_values == 0).sum() / expr_values.size * 100),
        "pct_nan": float(np.isnan(expr_values).sum() / expr_values.size * 100),
    }

    # Validation checks
    checks = []
    if report["n_genes"] < 10000:
        checks.append("WARNING: Fewer than 10,000 genes — expected ~20,000+")
    if report["n_samples"] < 100:
        checks.append("WARNING: Fewer than 100 samples — expected ~500+")
    if report.get("dev_stage_coverage"):
        missing_stages = set(DEV_STAGES.keys()) - set(report["dev_stage_coverage"].keys())
        if missing_stages:
            checks.append(f"WARNING: Missing developmental stages: {sorted(missing_stages)}")
    if report.get("cstc_region_coverage"):
        missing_regions = set(CSTC_REGIONS.keys()) - set(report["cstc_region_coverage"].keys())
        if missing_regions:
            checks.append(f"WARNING: Missing CSTC regions: {sorted(missing_regions)}")

    report["validation_checks"] = checks
    report["valid"] = len(checks) == 0

    return report


def run(output_dir: Path = OUTPUT_DIR, cache_dir: Path = CACHE_DIR) -> dict:
    """Download BrainSpan data and produce validation report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading BrainSpan RNA-seq data...")
    expression, rows_meta, cols_meta = download_brainspan(cache_dir)

    logger.info("Validating data: %d genes x %d samples", expression.shape[0], expression.shape[1])
    report = validate_brainspan(expression, rows_meta, cols_meta)

    # Save validation report
    report_path = output_dir / "brainspan_validation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Validation report saved to %s", report_path)

    # Save sample metadata summary for reference
    age_col_name: str | None = next(
        (str(c) for c in cols_meta.columns if "age" in str(c).lower()), None
    )
    struct_col_name: str | None = next(
        (str(c) for c in cols_meta.columns if "structure" in str(c).lower()),
        None,
    )
    if age_col_name is not None and struct_col_name is not None:
        summary = cols_meta[[age_col_name, struct_col_name]].copy()
        summary["cstc_region"] = [
            match_cstc_region(str(s)) for s in summary[struct_col_name]
        ]
        periods = []
        age_values = []
        for age_str in summary[age_col_name]:
            period, value = parse_age(str(age_str))
            periods.append(period)
            age_values.append(value)
        summary["period"] = periods
        summary["age_value"] = age_values
        summary["dev_stage"] = [
            classify_dev_stage(p, v) for p, v in zip(periods, age_values)
        ]
        summary_path = output_dir / "brainspan_sample_summary.csv"
        summary.to_csv(summary_path, index=False)
        logger.info("Sample summary saved to %s", summary_path)

    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate BrainSpan RNA-seq data for TS developmental trajectory modeling"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report = run(output_dir=args.output, cache_dir=args.cache)

    print(f"\nBrainSpan Download & Validation")
    print(f"  Genes: {report['n_genes']}")
    print(f"  Samples: {report['n_samples']}")
    print(f"  Unique gene symbols: {report.get('n_unique_gene_symbols', 'N/A')}")

    if report.get("dev_stage_coverage"):
        print(f"  Developmental stages covered: {len(report['dev_stage_coverage'])}/{len(DEV_STAGES)}")
        for stage, count in sorted(report["dev_stage_coverage"].items()):
            print(f"    {stage}: {count} age points")

    if report.get("cstc_region_coverage"):
        print(f"  CSTC regions covered: {len(report['cstc_region_coverage'])}/{len(CSTC_REGIONS)}")
        for region, count in sorted(report["cstc_region_coverage"].items()):
            print(f"    {region}: {count} structures")

    if report["validation_checks"]:
        print(f"\n  Validation issues:")
        for check in report["validation_checks"]:
            print(f"    {check}")
    else:
        print(f"\n  All validation checks passed.")


if __name__ == "__main__":
    main()
