"""Phase 1: Classify ARID1A status across all DepMap 25Q3 cell lines.

Classifies cell lines as ARID1A-loss (LOF mutation OR homozygous deletion)
vs wildtype. Cross-validates with expression data and annotates co-occurring
SWI/SNF mutations.

Usage:
    uv run python -m arid1a_pancancer_sl_atlas.01_arid1a_classifier
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import (
    load_depmap_matrix,
    load_depmap_model_metadata,
)

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase1"

# CN threshold for homozygous deletion (log2 scale)
HOMDEL_CN_THRESHOLD = -1.0

# Minimum samples per group for powered analysis
MIN_MUTANT = 5
MIN_WT = 5

# SWI/SNF complex genes to check for co-mutations
SWISNF_GENES = ["SMARCA4", "SMARCB1", "ARID2", "PBRM1", "SMARCC1", "SMARCC2"]


def load_all_lines(depmap_dir: Path) -> pd.DataFrame:
    """Load DepMap Model.csv for all cell lines with cancer type annotation."""
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    meta = meta[meta["OncotreeLineage"].notna()].copy()
    return meta


def load_arid1a_lof_mutations(depmap_dir: Path) -> pd.DataFrame:
    """Identify cell lines with ARID1A loss-of-function mutations.

    LOF = nonsense (stop_gained), frameshift, splice site variants.
    Uses DepMap's LikelyLoF annotation (VepImpact == HIGH).
    """
    cols = [
        "ModelID", "HugoSymbol", "VariantType", "VariantInfo",
        "ProteinChange", "VepImpact", "LikelyLoF",
    ]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    # Filter to ARID1A LOF mutations
    arid1a = mutations[
        (mutations["HugoSymbol"] == "ARID1A") & (mutations["LikelyLoF"] == True)
    ].copy()

    # Summarize per cell line: take the most severe mutation type
    per_line = (
        arid1a.groupby("ModelID")
        .agg(
            mutation_type=("VariantInfo", lambda x: ";".join(sorted(set(x)))),
            n_lof_mutations=("VariantInfo", "count"),
            protein_changes=("ProteinChange", lambda x: ";".join(
                sorted(set(str(v) for v in x if pd.notna(v)))
            )),
        )
        .reset_index()
    )
    per_line["has_lof_mutation"] = True
    return per_line


def load_swisnf_comutations(depmap_dir: Path) -> pd.DataFrame:
    """Identify co-occurring SWI/SNF complex LOF mutations per cell line."""
    cols = ["ModelID", "HugoSymbol", "LikelyLoF"]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    swisnf = mutations[
        (mutations["HugoSymbol"].isin(SWISNF_GENES))
        & (mutations["LikelyLoF"] == True)
    ].copy()

    # Summarize per cell line
    per_line = (
        swisnf.groupby("ModelID")["HugoSymbol"]
        .apply(lambda x: ";".join(sorted(set(x))))
        .rename("co_SWI_SNF_mutations")
        .reset_index()
    )
    return per_line


def add_arid1a_copy_number(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add ARID1A copy number and homozygous deletion call."""
    cn = load_depmap_matrix(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    if "ARID1A" not in cn.columns:
        raise ValueError("ARID1A not found in CN data columns")

    arid1a_cn = cn["ARID1A"].rename("CN_log2")
    df = df.join(arid1a_cn, how="left")
    df["has_homdel"] = df["CN_log2"] < HOMDEL_CN_THRESHOLD
    return df


def add_arid1a_expression(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add ARID1A expression (TPM log+1) for cross-validation."""
    expr = load_depmap_matrix(
        depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    if "ARID1A" not in expr.columns:
        raise ValueError("ARID1A not found in expression data columns")

    arid1a_expr = expr["ARID1A"].rename("expression")
    df = df.join(arid1a_expr, how="left")
    return df


def classify_arid1a_status(
    df: pd.DataFrame,
    lof_df: pd.DataFrame,
) -> pd.DataFrame:
    """Classify ARID1A status: mutant (LOF mutation OR homdel) vs WT."""
    # Merge LOF mutation info
    df = df.merge(
        lof_df[["ModelID", "has_lof_mutation", "mutation_type", "n_lof_mutations", "protein_changes"]],
        left_index=True,
        right_on="ModelID",
        how="left",
    ).set_index("ModelID")

    df["has_lof_mutation"] = df["has_lof_mutation"].fillna(False)
    df["has_homdel"] = df["has_homdel"].fillna(False)

    # ARID1A-loss = LOF mutation OR homozygous deletion (union)
    df["ARID1A_status"] = np.where(
        df["has_lof_mutation"] | df["has_homdel"],
        "mutant",
        "WT",
    )
    return df


def cross_validate_expression(df: pd.DataFrame) -> dict:
    """Validate that ARID1A-loss lines have significantly lower expression."""
    has_data = df.dropna(subset=["expression"])
    mutant = has_data[has_data["ARID1A_status"] == "mutant"]["expression"]
    wt = has_data[has_data["ARID1A_status"] == "WT"]["expression"]

    if len(mutant) == 0 or len(wt) == 0:
        return {"valid": False, "reason": "No lines in one group"}

    stat, pval = stats.mannwhitneyu(mutant, wt, alternative="less")

    return {
        "valid": True,
        "n_mutant": len(mutant),
        "n_wt": len(wt),
        "median_expr_mutant": float(mutant.median()),
        "median_expr_wt": float(wt.median()),
        "mannwhitney_p": float(pval),
    }


def plot_expression_validation(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate boxplot of ARID1A expression by status."""
    has_data = df.dropna(subset=["expression"])
    mutant = has_data[has_data["ARID1A_status"] == "mutant"]["expression"]
    wt = has_data[has_data["ARID1A_status"] == "WT"]["expression"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [wt, mutant],
        tick_labels=["WT", "ARID1A-loss"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")

    stat, pval = stats.mannwhitneyu(mutant, wt, alternative="less")
    ax.set_title(f"ARID1A Expression by Status\n(Mann-Whitney p={pval:.2e})")
    ax.set_ylabel("ARID1A Expression (log2 TPM+1)")
    ax.set_xlabel(f"WT (n={len(wt)})    ARID1A-loss (n={len(mutant)})")

    plt.tight_layout()
    fig.savefig(output_dir / "arid1a_expression_validation.png", dpi=150)
    plt.close(fig)


def build_cancer_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize ARID1A status per cancer type."""
    has_status = df[df["ARID1A_status"].notna()]

    summary_rows = []
    for lineage, group in has_status.groupby("OncotreeLineage"):
        n_total = len(group)
        n_mutant = int((group["ARID1A_status"] == "mutant").sum())
        n_wt = n_total - n_mutant
        freq = n_mutant / n_total if n_total > 0 else 0.0
        qualifies = n_mutant >= MIN_MUTANT and n_wt >= MIN_WT

        # Flag SKCM UV-passenger concern
        notes = ""
        if lineage == "Skin":
            notes = "UV-passenger ARID1A mutations may inflate count"

        summary_rows.append({
            "cancer_type": lineage,
            "n_mutant": n_mutant,
            "n_wt": n_wt,
            "n_total": n_total,
            "mutation_freq": round(freq, 4),
            "qualifies": qualifies,
            "notes": notes,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("mutation_freq", ascending=False).reset_index(drop=True)
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: ARID1A Cell Line Classifier ===\n")

    # Step 1: Load all cell lines
    print("Loading all DepMap cell lines...")
    df = load_all_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with OncotreeLineage annotation")

    # Step 2: Load ARID1A LOF mutations
    print("Loading ARID1A LOF mutations...")
    lof_df = load_arid1a_lof_mutations(DEPMAP_DIR)
    print(f"  {len(lof_df)} cell lines with ARID1A LOF mutations")

    # Step 3: Add ARID1A copy number
    print("Adding ARID1A copy number...")
    df = add_arid1a_copy_number(df, DEPMAP_DIR)
    n_homdel = df["has_homdel"].sum()
    print(f"  {n_homdel} cell lines with ARID1A homozygous deletion (CN log2 < {HOMDEL_CN_THRESHOLD})")

    # Step 4: Add ARID1A expression
    print("Adding ARID1A expression...")
    df = add_arid1a_expression(df, DEPMAP_DIR)
    n_with_expr = df["expression"].notna().sum()
    print(f"  {n_with_expr} cell lines with expression data")

    # Step 5: Classify ARID1A status
    print("Classifying ARID1A status (LOF mutation OR homdel)...")
    df = classify_arid1a_status(df, lof_df)
    n_mutant = (df["ARID1A_status"] == "mutant").sum()
    n_wt = (df["ARID1A_status"] == "WT").sum()

    # Count contribution sources
    n_lof_only = ((df["has_lof_mutation"]) & (~df["has_homdel"])).sum()
    n_homdel_only = ((~df["has_lof_mutation"]) & (df["has_homdel"])).sum()
    n_both = ((df["has_lof_mutation"]) & (df["has_homdel"])).sum()
    print(f"  {n_mutant} ARID1A-loss ({n_lof_only} LOF-only, {n_homdel_only} homdel-only, {n_both} both)")
    print(f"  {n_wt} WT")

    # Step 6: Cross-validate with expression
    print("\nCross-validating with ARID1A expression...")
    validation = cross_validate_expression(df)
    if validation["valid"]:
        print(f"  Median expression: mutant={validation['median_expr_mutant']:.2f} "
              f"vs WT={validation['median_expr_wt']:.2f}")
        print(f"  Mann-Whitney p={validation['mannwhitney_p']:.2e}")

    # Step 7: Generate expression validation plot
    print("Generating expression validation plot...")
    plot_expression_validation(df, OUTPUT_DIR)

    # Step 8: Annotate SWI/SNF co-mutations
    print("Annotating SWI/SNF co-mutations...")
    swisnf_df = load_swisnf_comutations(DEPMAP_DIR)
    df = df.merge(swisnf_df, left_index=True, right_on="ModelID", how="left").set_index("ModelID")
    df["co_SWI_SNF_mutations"] = df["co_SWI_SNF_mutations"].fillna("")
    n_comut = (df["co_SWI_SNF_mutations"] != "").sum()
    print(f"  {n_comut} cell lines with co-occurring SWI/SNF LOF mutations")

    # Step 9: Save classified cell lines
    output_cols = [
        "OncotreeLineage", "ARID1A_status", "mutation_type",
        "CN_log2", "expression", "co_SWI_SNF_mutations",
        "has_lof_mutation", "has_homdel", "n_lof_mutations", "protein_changes",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()
    out_classified = OUTPUT_DIR / "all_cell_lines_classified.csv"
    result.to_csv(out_classified)
    print(f"\nSaved {len(result)} classified cell lines to {out_classified.name}")

    # Step 10: Cancer type summary
    summary = build_cancer_type_summary(df)
    out_summary = OUTPUT_DIR / "cancer_type_summary.csv"
    summary.to_csv(out_summary, index=False)

    n_qualifying = summary["qualifies"].sum()
    print(f"\nCancer type summary ({len(summary)} types, {n_qualifying} qualifying):")
    qualifying = summary[summary["qualifies"]]
    for _, row in qualifying.iterrows():
        note = f"  ** {row['notes']}" if row["notes"] else ""
        print(f"  {row['cancer_type']}: {row['n_mutant']}/{row['n_total']} mutant "
              f"({row['mutation_freq']:.1%}), n_wt={row['n_wt']}{note}")

    print(f"\nSaved summary to {out_summary.name}")
    print(f"Saved expression plot to arid1a_expression_validation.png")


if __name__ == "__main__":
    main()
