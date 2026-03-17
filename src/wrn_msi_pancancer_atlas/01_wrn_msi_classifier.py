"""Phase 1: Classify MSI status across all DepMap 25Q3 cell lines.

Classifies cell lines as MSI-H (microsatellite instability-high) vs MSS
using Model.csv annotations as primary source, supplemented with stringent
molecular criteria (MLH1 silencing, MSH2 LOF mutations).

Sub-classifies MSI-H lines as MLH1-methylated vs MMR-mutated.
Reports qualifying cancer types for downstream WRN dependency analysis.

Usage:
    uv run python -m wrn_msi_pancancer_atlas.01_wrn_msi_classifier
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import (
    load_depmap_matrix,
    load_depmap_model_metadata,
)

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "wrn-msi-pancancer-atlas" / "phase1"

# MMR genes
MMR_GENES = ["MLH1", "MSH2", "MSH6", "PMS2"]

# MLH1 expression threshold for silencing (log2(TPM+1)).
# True MLH1 methylation causes near-complete silencing.
# < 1.0 corresponds to TPM < 1, consistent with promoter methylation.
MLH1_SILENCING_THRESHOLD = 1.0

# Minimum samples per group for powered analysis
MIN_MSI = 5
MIN_MSS = 10


def load_all_lines(depmap_dir: Path) -> pd.DataFrame:
    """Load DepMap Model.csv for all cell lines with cancer type annotation."""
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    meta = meta[meta["OncotreeLineage"].notna()].copy()
    return meta


def load_msi_annotations(meta: pd.DataFrame) -> pd.Series:
    """Extract MSI status from ModelSubtypeFeatures column.

    Returns a Series indexed by ModelID: 'MSI-H' or empty string.
    """
    has_msi = meta["ModelSubtypeFeatures"].str.contains("MSI", na=False)
    msi_status = pd.Series("", index=meta.index, name="annotation_msi", dtype="object")
    msi_status[has_msi] = "MSI-H"
    return msi_status


def load_mlh1_expression(depmap_dir: Path) -> pd.Series:
    """Load MLH1 expression (TPM log+1) for all cell lines."""
    expr = load_depmap_matrix(
        depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    if "MLH1" not in expr.columns:
        raise ValueError("MLH1 not found in expression data columns")
    return expr["MLH1"].rename("MLH1_expression")


def load_mmr_mutations(depmap_dir: Path) -> pd.DataFrame:
    """Load LOF mutations in MMR genes per cell line.

    Returns DataFrame indexed by ModelID with columns:
    - mmr_lof_genes: semicolon-separated LOF-mutated MMR genes
    - has_mlh1_lof, has_msh2_lof, has_msh6_lof, has_pms2_lof: bool
    """
    cols = [
        "ModelID", "HugoSymbol", "VariantInfo", "ProteinChange",
        "VepImpact", "LikelyLoF",
    ]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    # Filter to MMR gene LOF mutations
    mmr_lof = mutations[
        (mutations["HugoSymbol"].isin(MMR_GENES)) & (mutations["LikelyLoF"] == True)
    ].copy()

    # Summarize per cell line
    per_line = (
        mmr_lof.groupby("ModelID")["HugoSymbol"]
        .apply(lambda x: ";".join(sorted(set(x))))
        .rename("mmr_lof_genes")
        .reset_index()
        .set_index("ModelID")
    )

    # Add per-gene flags
    for gene in MMR_GENES:
        col = f"has_{gene.lower()}_lof"
        gene_lines = set(
            mmr_lof[mmr_lof["HugoSymbol"] == gene]["ModelID"].unique()
        )
        per_line[col] = per_line.index.isin(gene_lines)

    return per_line


def load_crispr_lines(depmap_dir: Path) -> set[str]:
    """Return set of ModelIDs that have CRISPR data."""
    crispr_full = pd.read_csv(
        depmap_dir / "CRISPRGeneEffect.csv", usecols=[0]
    )
    return set(crispr_full.iloc[:, 0])


def classify_msi_status(
    meta: pd.DataFrame,
    mlh1_expr: pd.Series,
    mmr_mut: pd.DataFrame,
    crispr_lines: set[str],
) -> pd.DataFrame:
    """Classify all cell lines by MSI status using tiered criteria.

    Tier 1 (Confirmed): Model.csv annotation contains "MSI"
    Tier 2 (High-confidence molecular): MLH1 expression < 1.0 (silenced)
        AND no MLH1 LOF mutation → MLH1 promoter methylation
    Tier 3 (Probable molecular): MSH2 LOF mutation → the dominant
        mutation-driven MSI pathway (MSH2 loss destabilizes MSH6)
    Remaining lines: MSS

    MSH6-only and PMS2-only LOF are recorded but not sufficient for
    MSI-H classification (can cause MSI-L, not reliably MSI-H).
    """
    df = meta[["OncotreeLineage", "OncotreePrimaryDisease", "ModelSubtypeFeatures"]].copy()

    # Annotation-based MSI
    df["annotation_msi"] = load_msi_annotations(meta)

    # MLH1 expression
    df = df.join(mlh1_expr, how="left")
    df["mlh1_silenced"] = df["MLH1_expression"] < MLH1_SILENCING_THRESHOLD

    # MMR mutation info
    df = df.join(mmr_mut, how="left")
    df["mmr_lof_genes"] = df["mmr_lof_genes"].fillna("")
    for gene in MMR_GENES:
        col = f"has_{gene.lower()}_lof"
        if col in df.columns:
            df[col] = df[col].fillna(False)
        else:
            df[col] = False

    # CRISPR availability
    df["has_crispr"] = df.index.isin(crispr_lines)

    # --- Tiered MSI classification ---
    is_annotated = df["annotation_msi"] == "MSI-H"
    is_mlh1_meth = df["mlh1_silenced"].fillna(False) & ~df["has_mlh1_lof"]
    is_msh2_lof = df["has_msh2_lof"]

    df["msi_status"] = "MSS"
    df.loc[is_annotated | is_mlh1_meth | is_msh2_lof, "msi_status"] = "MSI-H"

    # Classification tier
    df["classification_tier"] = ""
    # Assign in reverse priority so higher tiers overwrite
    df.loc[is_msh2_lof & (df["msi_status"] == "MSI-H"), "classification_tier"] = "tier3_MSH2_LOF"
    df.loc[is_mlh1_meth & (df["msi_status"] == "MSI-H"), "classification_tier"] = "tier2_MLH1_silenced"
    df.loc[is_annotated, "classification_tier"] = "tier1_annotated"
    # Lines that are both annotated + molecular
    df.loc[
        is_annotated & (is_mlh1_meth | is_msh2_lof),
        "classification_tier",
    ] = "tier1_annotated+molecular"

    # Sub-classify MSI-H mechanism
    df["msi_source"] = ""
    msi_h = df["msi_status"] == "MSI-H"

    mlh1_meth_flag = msi_h & is_mlh1_meth
    mmr_mut_flag = msi_h & (df["mmr_lof_genes"] != "")

    df.loc[mlh1_meth_flag & ~mmr_mut_flag, "msi_source"] = "MLH1-methylated"
    df.loc[~mlh1_meth_flag & mmr_mut_flag, "msi_source"] = "MMR-mutated"
    df.loc[mlh1_meth_flag & mmr_mut_flag, "msi_source"] = "MLH1-methylated+MMR-mutated"
    df.loc[msi_h & (df["msi_source"] == ""), "msi_source"] = "annotated-only"

    return df


def build_cancer_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize MSI status per cancer type. Identify qualifying types."""
    summary_rows = []
    for lineage, group in df.groupby("OncotreeLineage"):
        n_total = len(group)
        n_msi = int((group["msi_status"] == "MSI-H").sum())
        n_mss = n_total - n_msi

        msi_crispr = int(
            ((group["msi_status"] == "MSI-H") & group["has_crispr"]).sum()
        )
        mss_crispr = int(
            ((group["msi_status"] == "MSS") & group["has_crispr"]).sum()
        )

        qualifies = msi_crispr >= MIN_MSI and mss_crispr >= MIN_MSS
        msi_freq = n_msi / n_total if n_total > 0 else 0.0

        msi_lines = group[group["msi_status"] == "MSI-H"]
        n_mlh1_meth = int(msi_lines["msi_source"].str.contains("MLH1-methylated").sum())
        n_mmr_mut = int(msi_lines["msi_source"].str.contains("MMR-mutated").sum())
        n_annotated = int((msi_lines["annotation_msi"] == "MSI-H").sum())

        summary_rows.append({
            "cancer_type": lineage,
            "n_total": n_total,
            "n_msi_h": n_msi,
            "n_mss": n_mss,
            "msi_h_freq": round(msi_freq, 4),
            "n_msi_h_crispr": msi_crispr,
            "n_mss_crispr": mss_crispr,
            "n_mlh1_methylated": n_mlh1_meth,
            "n_mmr_mutated": n_mmr_mut,
            "n_annotated_msi": n_annotated,
            "qualifies": qualifies,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("msi_h_freq", ascending=False).reset_index(drop=True)
    return summary


def write_summary_txt(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    qualifying: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write human-readable summary text file."""
    n_total = len(df)
    n_msi = int((df["msi_status"] == "MSI-H").sum())
    n_mss = n_total - n_msi
    n_crispr = int(df["has_crispr"].sum())

    lines = [
        "=" * 60,
        "WRN-MSI Pan-Cancer Atlas — Phase 1: MSI/MMR Classifier",
        "=" * 60,
        "",
        "OVERVIEW",
        f"  Total cell lines with lineage annotation: {n_total}",
        f"  Cell lines with CRISPR data: {n_crispr}",
        f"  MSI-H classified: {n_msi} ({n_msi/n_total:.1%})",
        f"  MSS classified: {n_mss}",
        "",
        "CLASSIFICATION TIERS",
    ]

    for tier, count in df[df["msi_status"] == "MSI-H"]["classification_tier"].value_counts().items():
        lines.append(f"  {tier}: {count}")

    lines += [
        "",
        "MSI-H SUBTYPES (mechanism)",
    ]
    for src, count in df[df["msi_status"] == "MSI-H"]["msi_source"].value_counts().items():
        lines.append(f"  {src}: {count}")

    lines += [
        "",
        f"MLH1 SILENCING THRESHOLD: {MLH1_SILENCING_THRESHOLD} (log2 TPM+1)",
        "  Tier 1: Model.csv annotation (ModelSubtypeFeatures = MSI)",
        "  Tier 2: MLH1 expression < threshold + no MLH1 LOF (methylation)",
        "  Tier 3: MSH2 LOF mutation (dominant mutation-driven MSI)",
        "  MSH6/PMS2-only LOF: recorded but not sufficient for MSI-H call",
        "",
        f"QUALIFYING CANCER TYPES (>={MIN_MSI} MSI-H + >={MIN_MSS} MSS with CRISPR)",
        "-" * 60,
    ]

    if len(qualifying) == 0:
        lines.append("  None qualify with current thresholds.")
    else:
        for _, row in qualifying.iterrows():
            lines.append(
                f"  {row['cancer_type']}: {row['n_msi_h_crispr']} MSI-H / "
                f"{row['n_mss_crispr']} MSS (CRISPR), "
                f"{row['n_msi_h']}/{row['n_total']} total ({row['msi_h_freq']:.1%})"
            )

    lines += [
        "",
        "ALL CANCER TYPES WITH MSI-H LINES",
        "-" * 60,
    ]
    with_msi = summary[summary["n_msi_h"] > 0]
    for _, row in with_msi.iterrows():
        q_flag = " *" if row["qualifies"] else ""
        lines.append(
            f"  {row['cancer_type']}: {row['n_msi_h']} MSI-H / {row['n_mss']} MSS "
            f"(CRISPR: {row['n_msi_h_crispr']}/{row['n_mss_crispr']}){q_flag}"
        )

    lines += [
        "",
        "POWER ANALYSIS NOTE",
        f"  Qualifying threshold: >={MIN_MSI} MSI-H + >={MIN_MSS} MSS with CRISPR data.",
        "  This ensures adequate statistical power for Cohen's d effect",
        "  size estimation with bootstrap CI in Phase 2.",
        "",
    ]

    with open(output_dir / "msi_classifier_summary.txt", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: MSI/MMR Cell Line Classifier ===\n")

    # Step 1: Load all cell lines
    print("Loading all DepMap cell lines...")
    meta = load_all_lines(DEPMAP_DIR)
    print(f"  {len(meta)} cell lines with OncotreeLineage annotation")

    # Step 2: Load MLH1 expression
    print("Loading MLH1 expression...")
    mlh1_expr = load_mlh1_expression(DEPMAP_DIR)
    print(f"  {len(mlh1_expr)} lines with expression data")

    # Step 3: Load MMR mutations
    print("Loading MMR gene mutations...")
    mmr_mut = load_mmr_mutations(DEPMAP_DIR)
    print(f"  {len(mmr_mut)} lines with MMR LOF mutations")

    # Step 4: Load CRISPR line IDs
    print("Loading CRISPR line IDs...")
    crispr_lines = load_crispr_lines(DEPMAP_DIR)
    print(f"  {len(crispr_lines)} lines with CRISPR data")

    # Step 5: Classify MSI status
    print("\nClassifying MSI status...")
    df = classify_msi_status(meta, mlh1_expr, mmr_mut, crispr_lines)

    n_msi = int((df["msi_status"] == "MSI-H").sum())
    n_mss = int((df["msi_status"] == "MSS").sum())
    print(f"  {n_msi} MSI-H, {n_mss} MSS")
    print("  Tiers:")
    for tier, count in df[df["msi_status"] == "MSI-H"]["classification_tier"].value_counts().items():
        print(f"    {tier}: {count}")

    # Step 6: Cancer type summary
    print("\nBuilding cancer type summary...")
    summary = build_cancer_type_summary(df)
    qualifying = summary[summary["qualifies"]].copy()
    n_qualifying = len(qualifying)
    print(f"  {len(summary)} cancer types, {n_qualifying} qualifying")

    print(f"\nQualifying cancer types (>={MIN_MSI} MSI-H + >={MIN_MSS} MSS with CRISPR):")
    if n_qualifying == 0:
        print("  None qualify.")
    else:
        for _, row in qualifying.iterrows():
            print(
                f"  {row['cancer_type']}: {row['n_msi_h_crispr']} MSI-H / "
                f"{row['n_mss_crispr']} MSS (CRISPR)"
            )

    # Step 7: Save outputs
    print("\nSaving outputs...")

    output_cols = [
        "OncotreeLineage", "OncotreePrimaryDisease", "msi_status",
        "msi_source", "classification_tier",
        "MLH1_expression", "mlh1_silenced", "mmr_lof_genes",
        "has_mlh1_lof", "has_msh2_lof", "has_msh6_lof", "has_pms2_lof",
        "has_crispr", "annotation_msi",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_csv(OUTPUT_DIR / "msi_classification.csv")
    print(f"  msi_classification.csv — {len(df)} lines")

    qualifying.to_csv(OUTPUT_DIR / "qualifying_cancer_types.csv", index=False)
    print(f"  qualifying_cancer_types.csv — {n_qualifying} types")

    write_summary_txt(df, summary, qualifying, OUTPUT_DIR)
    print("  msi_classifier_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
