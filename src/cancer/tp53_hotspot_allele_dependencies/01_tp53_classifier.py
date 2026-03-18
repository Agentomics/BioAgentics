"""Phase 1: Classify TP53 allele status across DepMap cell lines.

Annotates all DepMap cell lines with TP53 mutation status, allele classification
(R175H, R248W, R273H, etc.), LOH status, and TP53 expression level.
Identifies structural vs contact mutants and flags underpowered alleles.

Usage:
    uv run python -m tp53_hotspot_allele_dependencies.01_tp53_classifier
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import (
    load_depmap_matrix,
    load_depmap_model_metadata,
    load_depmap_mutations,
)
from bioagentics.data.tp53_common import (
    ALLELE_PRIORITY,
    CONTACT_ALLELES,
    HOTSPOT_ALLELES,
    STRUCTURAL_ALLELES,
    classify_tp53_allele,
    is_contact,
    is_structural,
)

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "tp53_hotspot_allele_dependencies"

MIN_PER_GROUP = 10  # minimum lines per allele for powered analysis

# TP53 LOH threshold on log2(CN/2) scale (PortalOmicsCNGeneLog2.csv).
# 0 = diploid, -1.0 = single copy. Threshold -0.5 ≈ 70% of normal copy number.
LOH_CN_THRESHOLD = -0.5


def load_cell_lines(depmap_dir: Path) -> pd.DataFrame:
    """Load DepMap metadata, keeping lines with cancer type annotation."""
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    meta = meta[meta["OncotreePrimaryDisease"].notna()].copy()
    return meta


def get_lines_with_dependency_data(depmap_dir: Path) -> set[str]:
    """Get ModelIDs that have CRISPR dependency data."""
    crispr_ids = pd.read_csv(
        depmap_dir / "CRISPRGeneEffect.csv", usecols=[0]
    ).iloc[:, 0]
    return set(crispr_ids)


def add_tp53_mutations(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add TP53 mutation status, allele classification, and structural/contact flags."""
    muts = load_depmap_mutations(depmap_dir / "OmicsSomaticMutations.csv")

    # Filter to TP53, HIGH/MODERATE impact
    tp53_muts = muts[
        (muts["HugoSymbol"] == "TP53")
        & (muts["VepImpact"].isin(["HIGH", "MODERATE"]))
    ].copy()

    # Collect protein changes and consequences per model
    pc_per_model = tp53_muts.groupby("ModelID")["ProteinChange"].apply(list)
    cons_per_model = tp53_muts.groupby("ModelID")["MolecularConsequence"].apply(list)

    # Binary mutation status
    mutated_ids = set(tp53_muts["ModelID"])
    df["TP53_mutated"] = df.index.isin(mutated_ids)

    # Allele classification
    def get_allele_info(model_id: str) -> tuple[str, str]:
        if model_id in pc_per_model.index:
            pcs = pc_per_model[model_id]
            cons = cons_per_model[model_id] if model_id in cons_per_model.index else None
            return classify_tp53_allele(pcs, cons)
        return "TP53_WT", "WT"

    allele_data = [get_allele_info(mid) for mid in df.index]
    df["TP53_allele"] = [a[0] for a in allele_data]
    df["allele_class"] = [a[1] for a in allele_data]

    # Structural / contact flags
    df["is_structural"] = df["TP53_allele"].apply(is_structural)
    df["is_contact"] = df["TP53_allele"].apply(is_contact)

    # Store raw protein changes for reference
    pc_str = tp53_muts.groupby("ModelID")["ProteinChange"].apply(
        lambda x: ";".join(x.dropna().unique())
    )
    df["TP53_protein_change"] = df.index.map(pc_str).fillna("")

    # Flag SKCM non-hotspot missense (potential UV-passenger)
    skcm_mask = df["OncotreeLineage"].str.contains("Skin", case=False, na=False)
    df["skcm_uv_flag"] = (
        skcm_mask & (df["allele_class"] == "other_missense")
    )

    print(f"  TP53 mutations found in {len(mutated_ids)} cell lines")
    allele_counts = df[df["TP53_mutated"]]["TP53_allele"].value_counts()
    for allele, count in allele_counts.items():
        print(f"    {allele}: {count}")

    return df


def add_loh_status(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Annotate TP53 LOH status from copy number data."""
    cn_path = depmap_dir / "PortalOmicsCNGeneLog2.csv"
    cn_df = pd.read_csv(cn_path, index_col=0)

    # Find exact TP53 column: "TP53 (7157)"
    tp53_col = "TP53 (7157)"
    if tp53_col not in cn_df.columns:
        print("  WARNING: TP53 (7157) column not found in CN data")
        df["tp53_cn_ratio"] = float("nan")
        df["loh_status"] = "unknown"
        return df

    tp53_cn = cn_df[tp53_col]

    df["tp53_cn_ratio"] = df.index.map(tp53_cn)
    df["loh_status"] = "intact"
    df.loc[df["tp53_cn_ratio"] < LOH_CN_THRESHOLD, "loh_status"] = "LOH"
    df.loc[df["tp53_cn_ratio"].isna(), "loh_status"] = "unknown"

    n_loh = (df["loh_status"] == "LOH").sum()
    n_intact = (df["loh_status"] == "intact").sum()
    print(f"  TP53 LOH status: {n_loh} LOH, {n_intact} intact, "
          f"{(df['loh_status'] == 'unknown').sum()} unknown")

    return df


def add_tp53_expression(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Annotate TP53 expression level from RNA-seq data."""
    expr_path = depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    expr_df = load_depmap_matrix(expr_path)

    if "TP53" not in expr_df.columns:
        print("  WARNING: TP53 not found in expression data")
        df["tp53_expression"] = float("nan")
        return df

    tp53_expr = expr_df["TP53"]
    df["tp53_expression"] = df.index.map(tp53_expr)

    # Summary stats by allele class
    for cls in ["hotspot", "other_missense", "truncating", "WT"]:
        mask = df["allele_class"] == cls
        if mask.any():
            median_expr = df.loc[mask, "tp53_expression"].median()
            print(f"  TP53 expression median ({cls}): {median_expr:.2f}")

    return df


def build_cancer_type_summary(df: pd.DataFrame) -> list[dict]:
    """Summarize TP53 mutation status per cancer type."""
    summary_rows = []

    for cancer_type, group in df.groupby("OncotreePrimaryDisease"):
        n_total = len(group)
        n_mutant = int(group["TP53_mutated"].sum())
        n_wt = n_total - n_mutant
        freq = n_mutant / n_total if n_total > 0 else 0.0

        # Allele breakdown within mutants
        allele_counts = {}
        if n_mutant > 0:
            ac = group[group["TP53_mutated"]]["TP53_allele"].value_counts()
            allele_counts = ac.to_dict()

        # Structural vs contact counts
        n_structural = int(group["is_structural"].sum())
        n_contact = int(group["is_contact"].sum())

        # Power flags
        powered_mutant_vs_wt = n_mutant >= MIN_PER_GROUP and n_wt >= MIN_PER_GROUP
        powered_structural_vs_contact = (
            n_structural >= MIN_PER_GROUP and n_contact >= MIN_PER_GROUP
        )

        summary_rows.append({
            "cancer_type": cancer_type,
            "N_total": n_total,
            "N_mutant": n_mutant,
            "N_wt": n_wt,
            "mutation_freq": round(freq, 4),
            "allele_counts": allele_counts,
            "N_structural": n_structural,
            "N_contact": n_contact,
            "powered_mutant_vs_wt": powered_mutant_vs_wt,
            "powered_structural_vs_contact": powered_structural_vs_contact,
        })

    summary_rows.sort(key=lambda x: x["mutation_freq"], reverse=True)
    return summary_rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading DepMap cell lines...")
    df = load_cell_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with cancer type annotation")

    print("Filtering to lines with CRISPR dependency data...")
    crispr_ids = get_lines_with_dependency_data(DEPMAP_DIR)
    df["has_crispr_data"] = df.index.isin(crispr_ids)
    n_crispr = df["has_crispr_data"].sum()
    print(f"  {n_crispr} lines have CRISPR dependency data")

    print("Adding TP53 mutation status...")
    df = add_tp53_mutations(df, DEPMAP_DIR)

    print("Adding TP53 LOH status...")
    df = add_loh_status(df, DEPMAP_DIR)

    print("Adding TP53 expression levels...")
    df = add_tp53_expression(df, DEPMAP_DIR)

    # Build classified output table
    output_cols = [
        "CellLineName", "StrippedCellLineName",
        "OncotreeLineage", "OncotreePrimaryDisease", "OncotreeSubtype",
        "TP53_mutated", "TP53_allele", "allele_class",
        "is_structural", "is_contact",
        "TP53_protein_change",
        "loh_status", "tp53_cn_ratio",
        "tp53_expression",
        "skcm_uv_flag",
        "has_crispr_data",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()

    out_classified = OUTPUT_DIR / "tp53_classified_lines.csv"
    result.to_csv(out_classified)
    print(f"\nSaved {len(result)} classified cell lines to {out_classified.name}")

    # Cancer type summary (restrict to lines with CRISPR data for power analysis)
    crispr_df = df[df["has_crispr_data"]].copy()
    print(f"\nCancer type summary (among {len(crispr_df)} lines with CRISPR data):")
    summary = build_cancer_type_summary(crispr_df)

    powered_mvw = [s for s in summary if s["powered_mutant_vs_wt"]]
    powered_svc = [s for s in summary if s["powered_structural_vs_contact"]]

    print(f"\n  {len(powered_mvw)} cancer types powered for mutant-vs-WT (N>={MIN_PER_GROUP}):")
    for s in powered_mvw:
        print(f"    {s['cancer_type']}: {s['N_mutant']}/{s['N_total']} mutant "
              f"({s['mutation_freq']:.1%}), N_wt={s['N_wt']}")

    print(f"\n  {len(powered_svc)} cancer types powered for structural-vs-contact:")
    for s in powered_svc:
        print(f"    {s['cancer_type']}: structural={s['N_structural']}, "
              f"contact={s['N_contact']}")

    # Pan-cancer allele distribution
    pan_mutant = crispr_df[crispr_df["TP53_mutated"]]
    print(f"\n  Pan-cancer allele distribution ({len(pan_mutant)} mutant lines "
          f"with CRISPR data):")
    allele_vc = pan_mutant["TP53_allele"].value_counts()
    for allele, count in allele_vc.items():
        flag = "" if count >= MIN_PER_GROUP else " [UNDERPOWERED]"
        print(f"    {allele}: {count}{flag}")

    # Structural vs contact distribution
    n_structural = int(pan_mutant["is_structural"].sum())
    n_contact = int(pan_mutant["is_contact"].sum())
    print(f"\n  Structural: {n_structural}, Contact: {n_contact}")

    # LOH breakdown among mutants
    loh_vc = pan_mutant["loh_status"].value_counts()
    print(f"\n  LOH status among TP53-mutant lines:")
    for status, count in loh_vc.items():
        print(f"    {status}: {count}")

    # SKCM UV flag summary
    n_skcm_flagged = int(crispr_df["skcm_uv_flag"].sum())
    if n_skcm_flagged > 0:
        print(f"\n  WARNING: {n_skcm_flagged} SKCM lines with non-hotspot TP53 missense "
              "(potential UV-passenger)")

    # Save summary JSON
    out_summary = OUTPUT_DIR / "cancer_type_summary.json"
    with open(out_summary, "w") as f:
        json.dump({
            "total_lines_with_crispr": len(crispr_df),
            "total_tp53_mutant_with_crispr": len(pan_mutant),
            "n_structural": n_structural,
            "n_contact": n_contact,
            "powered_mutant_vs_wt_count": len(powered_mvw),
            "powered_structural_vs_contact_count": len(powered_svc),
            "allele_counts": allele_vc.to_dict(),
            "cancer_types": summary,
        }, f, indent=2)
    print(f"\nSaved summary to {out_summary.name}")


if __name__ == "__main__":
    main()
