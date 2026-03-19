"""Phase 1a: Classify SWI/SNF (ARID1A + SMARCA4) status across DepMap 25Q3.

Combines ARID1A and SMARCA4 classifiers into a unified SWI/SNF status:
  - ARID1A-mutant: LOF mutation or homozygous deletion in ARID1A
  - SMARCA4-mutant: LOF mutation or homozygous deletion in SMARCA4
  - dual-mutant: Both ARID1A and SMARCA4 disrupted
  - WT: Neither gene disrupted

Reuses patterns from existing atlas classifiers. Outputs per-line classification
and per-cancer-type summary with qualification flags.

Usage:
    uv run python -m swisnf_metabolic_convergence.01_swisnf_classifier
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
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase1a"

# CN threshold for homozygous deletion (log2 scale)
HOMDEL_CN_THRESHOLD = -1.0

# Minimum samples per group for powered analysis
MIN_MUTANT = 5
MIN_WT = 5

# LOF variant types (truncating)
TRUNCATING_TYPES = {
    "nonsense", "frameshift", "splice_site",
    "stop_gained", "frameshift_variant", "splice_donor_variant",
    "splice_acceptor_variant",
}


def load_all_lines(depmap_dir: Path) -> pd.DataFrame:
    """Load DepMap Model.csv for all cell lines with cancer type annotation."""
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    meta = meta[meta["OncotreeLineage"].notna()].copy()
    return meta


def load_gene_lof_mutations(depmap_dir: Path, gene: str) -> pd.DataFrame:
    """Identify cell lines with LOF mutations for a given gene.

    LOF = LikelyLoF == True OR VepImpact == HIGH.
    Returns per-cell-line summary with mutation details.
    """
    cols = [
        "ModelID", "HugoSymbol", "VariantType", "VariantInfo",
        "ProteinChange", "VepImpact", "LikelyLoF",
    ]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    gene_muts = mutations[
        (mutations["HugoSymbol"] == gene)
        & ((mutations["LikelyLoF"] == True) | (mutations["VepImpact"] == "HIGH"))
    ].copy()

    if len(gene_muts) == 0:
        return pd.DataFrame(columns=[
            "ModelID", f"{gene}_has_lof", f"{gene}_mutation_type",
            f"{gene}_n_lof", f"{gene}_protein_changes",
        ])

    per_line = (
        gene_muts.groupby("ModelID")
        .agg(
            mutation_type=("VariantInfo", lambda x: ";".join(
                sorted(set(str(v) for v in x if pd.notna(v)))
            )),
            n_lof=("VariantInfo", "count"),
            protein_changes=("ProteinChange", lambda x: ";".join(
                sorted(set(str(v) for v in x if pd.notna(v)))
            )),
        )
        .reset_index()
    )
    per_line[f"{gene}_has_lof"] = True
    per_line = per_line.rename(columns={
        "mutation_type": f"{gene}_mutation_type",
        "n_lof": f"{gene}_n_lof",
        "protein_changes": f"{gene}_protein_changes",
    })
    return per_line


def add_gene_copy_number(
    df: pd.DataFrame, depmap_dir: Path, gene: str,
) -> pd.DataFrame:
    """Add copy number and homozygous deletion call for a gene."""
    cn = load_depmap_matrix(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    if gene not in cn.columns:
        raise ValueError(f"{gene} not found in CN data columns")

    gene_cn = cn[gene].rename(f"{gene}_CN_log2")
    df = df.join(gene_cn, how="left")
    df[f"{gene}_has_homdel"] = df[f"{gene}_CN_log2"] < HOMDEL_CN_THRESHOLD
    return df


def classify_swisnf_status(
    df: pd.DataFrame,
    arid1a_lof: pd.DataFrame,
    smarca4_lof: pd.DataFrame,
) -> pd.DataFrame:
    """Classify combined SWI/SNF status for each cell line.

    Status categories:
      - ARID1A_mutant: ARID1A disrupted only
      - SMARCA4_mutant: SMARCA4 disrupted only
      - dual_mutant: Both disrupted
      - WT: Neither disrupted
    """
    # Merge ARID1A LOF info
    df = df.merge(
        arid1a_lof[["ModelID", "ARID1A_has_lof", "ARID1A_mutation_type",
                     "ARID1A_n_lof", "ARID1A_protein_changes"]],
        left_index=True, right_on="ModelID", how="left",
    ).set_index("ModelID")

    # Merge SMARCA4 LOF info
    df = df.merge(
        smarca4_lof[["ModelID", "SMARCA4_has_lof", "SMARCA4_mutation_type",
                      "SMARCA4_n_lof", "SMARCA4_protein_changes"]],
        left_index=True, right_on="ModelID", how="left",
    ).set_index("ModelID")

    # Fill missing values
    df["ARID1A_has_lof"] = df["ARID1A_has_lof"].fillna(False)
    df["SMARCA4_has_lof"] = df["SMARCA4_has_lof"].fillna(False)
    df["ARID1A_has_homdel"] = df["ARID1A_has_homdel"].fillna(False)
    df["SMARCA4_has_homdel"] = df["SMARCA4_has_homdel"].fillna(False)

    # Individual gene status
    df["ARID1A_disrupted"] = df["ARID1A_has_lof"] | df["ARID1A_has_homdel"]
    df["SMARCA4_disrupted"] = df["SMARCA4_has_lof"] | df["SMARCA4_has_homdel"]

    # Combined SWI/SNF status
    conditions = [
        df["ARID1A_disrupted"] & df["SMARCA4_disrupted"],
        df["ARID1A_disrupted"] & ~df["SMARCA4_disrupted"],
        ~df["ARID1A_disrupted"] & df["SMARCA4_disrupted"],
    ]
    choices = ["dual_mutant", "ARID1A_mutant", "SMARCA4_mutant"]
    df["swisnf_status"] = np.select(conditions, choices, default="WT")

    # Simplified binary: any SWI/SNF mutation vs WT
    df["swisnf_any_mutant"] = df["ARID1A_disrupted"] | df["SMARCA4_disrupted"]

    return df


def build_cancer_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize SWI/SNF status per cancer type."""
    rows = []
    for lineage, group in df.groupby("OncotreeLineage"):
        n_total = len(group)
        n_arid1a = int((group["swisnf_status"] == "ARID1A_mutant").sum())
        n_smarca4 = int((group["swisnf_status"] == "SMARCA4_mutant").sum())
        n_dual = int((group["swisnf_status"] == "dual_mutant").sum())
        n_any_mut = n_arid1a + n_smarca4 + n_dual
        n_wt = n_total - n_any_mut

        # Qualification: enough mutant and WT for each comparison
        qualifies_arid1a = n_arid1a + n_dual >= MIN_MUTANT and n_wt >= MIN_WT
        qualifies_smarca4 = n_smarca4 + n_dual >= MIN_MUTANT and n_wt >= MIN_WT
        qualifies_combined = n_any_mut >= MIN_MUTANT and n_wt >= MIN_WT

        rows.append({
            "cancer_type": lineage,
            "n_total": n_total,
            "n_ARID1A_mutant": n_arid1a,
            "n_SMARCA4_mutant": n_smarca4,
            "n_dual_mutant": n_dual,
            "n_any_swisnf_mutant": n_any_mut,
            "n_WT": n_wt,
            "freq_any_swisnf": round(n_any_mut / n_total, 4) if n_total > 0 else 0.0,
            "qualifies_arid1a": qualifies_arid1a,
            "qualifies_smarca4": qualifies_smarca4,
            "qualifies_combined": qualifies_combined,
        })

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("freq_any_swisnf", ascending=False).reset_index(drop=True)
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1a: SWI/SNF Cell Line Classifier ===\n")

    # Step 1: Load all cell lines
    print("Loading all DepMap cell lines...")
    df = load_all_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with OncotreeLineage annotation")

    # Step 2: Load LOF mutations for both genes
    print("Loading ARID1A LOF mutations...")
    arid1a_lof = load_gene_lof_mutations(DEPMAP_DIR, "ARID1A")
    print(f"  {len(arid1a_lof)} cell lines with ARID1A LOF mutations")

    print("Loading SMARCA4 LOF mutations...")
    smarca4_lof = load_gene_lof_mutations(DEPMAP_DIR, "SMARCA4")
    print(f"  {len(smarca4_lof)} cell lines with SMARCA4 LOF mutations")

    # Step 3: Add copy number for both genes
    print("Adding ARID1A copy number...")
    df = add_gene_copy_number(df, DEPMAP_DIR, "ARID1A")
    n_arid1a_homdel = df["ARID1A_has_homdel"].sum()
    print(f"  {n_arid1a_homdel} cell lines with ARID1A homdel")

    print("Adding SMARCA4 copy number...")
    df = add_gene_copy_number(df, DEPMAP_DIR, "SMARCA4")
    n_smarca4_homdel = df["SMARCA4_has_homdel"].sum()
    print(f"  {n_smarca4_homdel} cell lines with SMARCA4 homdel")

    # Step 4: Classify combined SWI/SNF status
    print("\nClassifying SWI/SNF status...")
    df = classify_swisnf_status(df, arid1a_lof, smarca4_lof)

    status_counts = df["swisnf_status"].value_counts()
    print("  Status counts:")
    for status, count in status_counts.items():
        print(f"    {status}: {count}")

    # Step 5: Save classified cell lines
    output_cols = [
        "OncotreeLineage", "OncotreeSubtype",
        "swisnf_status", "swisnf_any_mutant",
        "ARID1A_disrupted", "ARID1A_has_lof", "ARID1A_has_homdel",
        "ARID1A_CN_log2", "ARID1A_mutation_type", "ARID1A_n_lof", "ARID1A_protein_changes",
        "SMARCA4_disrupted", "SMARCA4_has_lof", "SMARCA4_has_homdel",
        "SMARCA4_CN_log2", "SMARCA4_mutation_type", "SMARCA4_n_lof", "SMARCA4_protein_changes",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()
    out_path = OUTPUT_DIR / "swisnf_classified_lines.csv"
    result.to_csv(out_path)
    print(f"\nSaved {len(result)} classified cell lines to {out_path.name}")

    # Step 6: Cancer type summary
    summary = build_cancer_type_summary(df)
    out_summary = OUTPUT_DIR / "cancer_type_summary.csv"
    summary.to_csv(out_summary, index=False)

    n_qual_arid1a = summary["qualifies_arid1a"].sum()
    n_qual_smarca4 = summary["qualifies_smarca4"].sum()
    n_qual_combined = summary["qualifies_combined"].sum()
    print(f"\nCancer type summary ({len(summary)} types):")
    print(f"  Qualifying for ARID1A analysis: {n_qual_arid1a}")
    print(f"  Qualifying for SMARCA4 analysis: {n_qual_smarca4}")
    print(f"  Qualifying for combined SWI/SNF analysis: {n_qual_combined}")

    print("\nQualifying cancer types (combined):")
    for _, row in summary[summary["qualifies_combined"]].iterrows():
        print(f"  {row['cancer_type']}: {row['n_any_swisnf_mutant']}/{row['n_total']} "
              f"({row['freq_any_swisnf']:.1%}) — "
              f"ARID1A:{row['n_ARID1A_mutant']} SMARCA4:{row['n_SMARCA4_mutant']} "
              f"dual:{row['n_dual_mutant']} WT:{row['n_WT']}")

    print(f"\nSaved summary to {out_summary.name}")


if __name__ == "__main__":
    main()
