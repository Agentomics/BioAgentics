"""Phase 1a: Classify MTAP deletion status in DepMap NSCLC cell lines.

Produces a classified cell line table with MTAP copy number, expression,
deletion status, and co-mutation annotations (KRAS allele, STK11, KEAP1,
TP53, NFE2L2) for downstream PRMT5 synthetic lethality analysis.

Usage:
    uv run python -m mtap_prmt5_nsclc_sl.01_mtap_classifier
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import (
    load_depmap_matrix,
    load_depmap_model_metadata,
    load_depmap_mutations,
)
from bioagentics.data.nsclc_common import classify_kras_allele

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "mtap-prmt5-nsclc-sl"

# Genes to annotate for mutation status
MUTATION_GENES = ["KRAS", "STK11", "KEAP1", "TP53", "NFE2L2"]

# MTAP deletion threshold: DepMap 25Q3 PortalOmicsCNGeneLog2 stores CN ratios
# (not log2 despite filename). Diploid genes cluster around 1.0, homozygous
# deletions near 0. Threshold < 0.5 captures homo+hemizygous deletions,
# equivalent to the PLAN's "log2 ratio < -1" in standard format.
# Cross-validated against MTAP expression: deleted lines show median expr 0.2
# vs 5.1 intact (p < 1e-13), with >73% below Q25 expression.
MTAP_CN_THRESHOLD = 0.5


def load_nsclc_lines(depmap_dir: Path) -> pd.DataFrame:
    """Load DepMap Model.csv and filter to NSCLC lines."""
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    nsclc = meta[
        meta["OncotreePrimaryDisease"] == "Non-Small Cell Lung Cancer"
    ].copy()
    return nsclc


def add_mtap_copy_number(nsclc: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add MTAP copy number and deletion call.

    DepMap 25Q3 PortalOmicsCNGeneLog2 values are CN ratios (diploid ~1.0,
    deletion ~0). We store the raw value as MTAP_CN_log2 for column name
    consistency with downstream scripts.
    """
    cn = load_depmap_matrix(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    if "MTAP" not in cn.columns:
        raise ValueError("MTAP not found in CN data columns")

    mtap_cn = cn["MTAP"].rename("MTAP_CN_log2")
    nsclc = nsclc.join(mtap_cn, how="left")
    nsclc["MTAP_deleted"] = nsclc["MTAP_CN_log2"] < MTAP_CN_THRESHOLD
    return nsclc


def add_mtap_expression(nsclc: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add MTAP expression (TPM log+1) for cross-validation."""
    expr = load_depmap_matrix(
        depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    if "MTAP" not in expr.columns:
        raise ValueError("MTAP not found in expression data columns")

    mtap_expr = expr["MTAP"].rename("MTAP_expression")
    nsclc = nsclc.join(mtap_expr, how="left")
    return nsclc


def add_mutation_status(nsclc: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add mutation status for KRAS (with allele), STK11, KEAP1, TP53, NFE2L2."""
    muts = load_depmap_mutations(depmap_dir / "OmicsSomaticMutations.csv")
    nsclc_ids = set(nsclc.index)

    # Filter to NSCLC lines, target genes, HIGH/MODERATE impact
    driver_muts = muts[
        (muts["ModelID"].isin(nsclc_ids))
        & (muts["HugoSymbol"].isin(MUTATION_GENES))
        & (muts["VepImpact"].isin(["HIGH", "MODERATE"]))
    ].copy()

    # Binary mutation status for each gene
    for gene in MUTATION_GENES:
        gene_muts = driver_muts[driver_muts["HugoSymbol"] == gene]
        mutated_ids = set(gene_muts["ModelID"])
        col_name = f"{gene}_mut" if gene != "KRAS" else "KRAS_status"
        nsclc[col_name] = nsclc.index.isin(mutated_ids)

    # KRAS allele classification
    kras_muts = driver_muts[driver_muts["HugoSymbol"] == "KRAS"]
    kras_alleles = kras_muts.groupby("ModelID")["ProteinChange"].apply(list)

    def get_allele(model_id: str) -> str:
        if model_id in kras_alleles.index:
            return classify_kras_allele(kras_alleles[model_id])
        return "WT"

    nsclc["KRAS_allele"] = [get_allele(mid) for mid in nsclc.index]

    return nsclc


def cross_validate_expression(nsclc: pd.DataFrame) -> dict:
    """Validate that MTAP-deleted lines have lower expression than intact."""
    has_both = nsclc.dropna(subset=["MTAP_deleted", "MTAP_expression"])
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


def build_classified_table(depmap_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Build the full classified NSCLC cell line table.

    Returns (classified_df, validation_stats).
    """
    print("Loading NSCLC cell lines...")
    nsclc = load_nsclc_lines(depmap_dir)
    print(f"  Found {len(nsclc)} NSCLC lines")

    print("Adding MTAP copy number...")
    nsclc = add_mtap_copy_number(nsclc, depmap_dir)
    n_with_cn = nsclc["MTAP_CN_log2"].notna().sum()
    n_deleted = nsclc["MTAP_deleted"].sum()
    print(f"  {n_with_cn} lines with CN data, {n_deleted} MTAP-deleted")

    print("Adding MTAP expression...")
    nsclc = add_mtap_expression(nsclc, depmap_dir)
    n_with_expr = nsclc["MTAP_expression"].notna().sum()
    print(f"  {n_with_expr} lines with expression data")

    print("Cross-validating CN vs expression...")
    validation = cross_validate_expression(nsclc)
    if validation["valid"]:
        print(f"  Median expr deleted={validation['median_expr_deleted']:.2f} "
              f"vs intact={validation['median_expr_intact']:.2f} "
              f"(p={validation['mannwhitney_p']:.2e})")
        print(f"  {validation['frac_deleted_below_q25']:.0%} of deleted lines "
              f"below Q25 expression")

    print("Adding mutation status...")
    nsclc = add_mutation_status(nsclc, depmap_dir)

    # Select output columns
    output_cols = [
        "CellLineName", "OncotreeSubtype",
        "MTAP_CN_log2", "MTAP_deleted", "MTAP_expression",
        "KRAS_status", "KRAS_allele",
        "STK11_mut", "KEAP1_mut", "TP53_mut", "NFE2L2_mut",
    ]
    result = nsclc[output_cols].copy()

    # Summary
    n_total = len(result)
    n_del = result["MTAP_deleted"].sum()
    pct_del = n_del / n_total * 100 if n_total > 0 else 0
    print(f"\nSummary: {n_total} NSCLC lines, {n_del} MTAP-deleted ({pct_del:.1f}%)")
    print(f"  KRAS alleles: {result['KRAS_allele'].value_counts().to_dict()}")
    print(f"  STK11 mut: {result['STK11_mut'].sum()}")
    print(f"  KEAP1 mut: {result['KEAP1_mut'].sum()}")
    print(f"  TP53 mut: {result['TP53_mut'].sum()}")
    print(f"  NFE2L2 mut: {result['NFE2L2_mut'].sum()}")

    return result, validation


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result, validation = build_classified_table(DEPMAP_DIR)

    out_path = OUTPUT_DIR / "nsclc_cell_lines_classified.csv"
    result.to_csv(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
