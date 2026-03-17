"""Phase 1a: Classify MTAP deletion status across ALL DepMap cell lines.

Extends the NSCLC-specific classifier to pan-cancer. Groups cell lines by
OncotreeLineage and identifies cancer types with sufficient sample sizes
for downstream PRMT5/MAT2A synthetic lethality analysis.

Usage:
    uv run python -m pancancer_mtap_prmt5_atlas.01_pancancer_mtap_classifier
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix, load_depmap_model_metadata

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "pancancer-mtap-prmt5-atlas"

# Same threshold validated in NSCLC analysis
MTAP_CN_THRESHOLD = 0.5

# Minimum samples per group for statistical power
MIN_DELETED = 5
MIN_INTACT = 5


def load_all_lines(depmap_dir: Path) -> pd.DataFrame:
    """Load DepMap Model.csv for all cell lines."""
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    # Keep lines with a cancer type annotation
    meta = meta[meta["OncotreeLineage"].notna()].copy()
    return meta


def add_mtap_copy_number(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add MTAP copy number and deletion call."""
    cn = load_depmap_matrix(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    if "MTAP" not in cn.columns:
        raise ValueError("MTAP not found in CN data columns")

    mtap_cn = cn["MTAP"].rename("MTAP_CN")
    df = df.join(mtap_cn, how="left")
    df["MTAP_deleted"] = df["MTAP_CN"] < MTAP_CN_THRESHOLD
    return df


def add_mtap_expression(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add MTAP expression (TPM log+1) for cross-validation."""
    expr = load_depmap_matrix(
        depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    if "MTAP" not in expr.columns:
        raise ValueError("MTAP not found in expression data columns")

    mtap_expr = expr["MTAP"].rename("MTAP_expression")
    df = df.join(mtap_expr, how="left")
    return df


def cross_validate_expression(df: pd.DataFrame) -> dict:
    """Validate that MTAP-deleted lines have lower expression than intact (pan-cancer)."""
    has_both = df.dropna(subset=["MTAP_deleted", "MTAP_expression"])
    deleted = has_both[has_both["MTAP_deleted"]]["MTAP_expression"]
    intact = has_both[~has_both["MTAP_deleted"]]["MTAP_expression"]

    if len(deleted) == 0 or len(intact) == 0:
        return {"valid": False, "reason": "No lines in one group"}

    stat, pval = stats.mannwhitneyu(deleted, intact, alternative="less")
    expr_q25 = has_both["MTAP_expression"].quantile(0.25)
    frac_below_q25 = (deleted < expr_q25).mean()

    return {
        "valid": True,
        "n_deleted": len(deleted),
        "n_intact": len(intact),
        "median_expr_deleted": float(deleted.median()),
        "median_expr_intact": float(intact.median()),
        "mannwhitney_p": float(pval),
        "frac_deleted_below_q25": float(frac_below_q25),
    }


def build_cancer_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize MTAP deletion per cancer type (OncotreeLineage)."""
    # Only lines with CN data
    has_cn = df.dropna(subset=["MTAP_CN"])

    summary_rows = []
    for lineage, group in has_cn.groupby("OncotreeLineage"):
        n_total = len(group)
        n_deleted = int(group["MTAP_deleted"].sum())
        n_intact = n_total - n_deleted
        freq = n_deleted / n_total if n_total > 0 else 0.0
        qualifies = n_deleted >= MIN_DELETED and n_intact >= MIN_INTACT
        summary_rows.append({
            "cancer_type": lineage,
            "N_total": n_total,
            "N_deleted": n_deleted,
            "N_intact": n_intact,
            "deletion_freq": round(freq, 4),
            "qualifies": qualifies,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("deletion_freq", ascending=False).reset_index(drop=True)
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading all DepMap cell lines...")
    df = load_all_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with OncotreeLineage annotation")

    print("Adding MTAP copy number...")
    df = add_mtap_copy_number(df, DEPMAP_DIR)
    n_with_cn = df["MTAP_CN"].notna().sum()
    n_deleted = df["MTAP_deleted"].sum()
    print(f"  {n_with_cn} lines with CN data, {n_deleted} MTAP-deleted")

    print("Adding MTAP expression...")
    df = add_mtap_expression(df, DEPMAP_DIR)
    n_with_expr = df["MTAP_expression"].notna().sum()
    print(f"  {n_with_expr} lines with expression data")

    print("Cross-validating CN vs expression (pan-cancer)...")
    validation = cross_validate_expression(df)
    if validation["valid"]:
        print(f"  Median expr deleted={validation['median_expr_deleted']:.2f} "
              f"vs intact={validation['median_expr_intact']:.2f} "
              f"(p={validation['mannwhitney_p']:.2e})")
        print(f"  {validation['frac_deleted_below_q25']:.0%} of deleted lines "
              f"below Q25 expression")

    # Build classified table — keep columns needed by downstream scripts
    output_cols = [
        "CellLineName", "OncotreeLineage", "OncotreePrimaryDisease",
        "OncotreeSubtype", "MTAP_CN", "MTAP_deleted", "MTAP_expression",
    ]
    # Only keep columns that exist
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()

    # Save all classified cell lines
    out_classified = OUTPUT_DIR / "all_cell_lines_classified.csv"
    result.to_csv(out_classified)
    print(f"\nSaved {len(result)} classified cell lines to {out_classified.name}")

    # Build and save cancer type summary
    summary = build_cancer_type_summary(df)
    out_summary = OUTPUT_DIR / "cancer_type_summary.csv"
    summary.to_csv(out_summary, index=False)

    n_qualifying = summary["qualifies"].sum()
    print(f"\nCancer type summary ({len(summary)} types, {n_qualifying} qualifying):")
    qualifying = summary[summary["qualifies"]]
    for _, row in qualifying.iterrows():
        print(f"  {row['cancer_type']}: {row['N_deleted']}/{row['N_total']} deleted "
              f"({row['deletion_freq']:.1%}), N_intact={row['N_intact']}")

    # Validate NSCLC matches prior analysis (~30.9% in cell lines)
    nsclc_row = summary[summary["cancer_type"] == "Lung"]
    if not nsclc_row.empty:
        nsclc_freq = nsclc_row.iloc[0]["deletion_freq"]
        print(f"\nValidation: Lung deletion freq = {nsclc_freq:.1%} "
              f"(expect ~30.9% from prior NSCLC analysis)")

    print(f"\nSaved summary to {out_summary.name}")


if __name__ == "__main__":
    main()
