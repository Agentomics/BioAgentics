"""Phase 1: Classify SMARCA4 (BRG1) status across all DepMap 25Q3 cell lines.

Classifies cell lines as SMARCA4-deficient (LOF mutation OR homozygous deletion)
vs intact. Stratifies mutants by mutation class:
  - Class 1: truncating (nonsense, frameshift, splice) = complete ATPase loss
  - Class 2: missense (especially ATPase domain) = partial activity retained

Cross-validates with expression data and annotates co-occurring mutations
relevant to SWI/SNF biology and SMARCA4-mutant lung cancer context.

Usage:
    uv run python -m smarca4_pancancer_sl_atlas.01_smarca4_classifier
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
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "smarca4-pancancer-sl-atlas" / "phase1"

# CN threshold for homozygous deletion (log2 scale)
HOMDEL_CN_THRESHOLD = -1.0

# Minimum samples per group for powered analysis
MIN_DEFICIENT = 5
MIN_INTACT = 10

# Co-mutation genes to annotate (SWI/SNF + lung cancer context)
COMUTATION_GENES = [
    "SMARCA2", "ARID1A", "ARID1B", "PBRM1", "SMARCB1",
    "TP53", "KRAS", "STK11", "KEAP1",
]

# Truncating variant types that define Class 1
CLASS1_VARIANT_TYPES = {
    "nonsense", "frameshift", "splice_site",
    "stop_gained", "frameshift_variant", "splice_donor_variant",
    "splice_acceptor_variant",
}


def load_all_lines(depmap_dir: Path) -> pd.DataFrame:
    """Load DepMap Model.csv for all cell lines with cancer type annotation."""
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    meta = meta[meta["OncotreeLineage"].notna()].copy()
    return meta


def load_smarca4_mutations(depmap_dir: Path) -> pd.DataFrame:
    """Load all SMARCA4 mutations (LOF and missense) for classification.

    Returns per-line summary with mutation class:
      - Class 1: any truncating mutation (nonsense, frameshift, splice)
      - Class 2: only missense mutations (no truncating)
    """
    cols = [
        "ModelID", "HugoSymbol", "VariantType", "VariantInfo",
        "ProteinChange", "VepImpact", "LikelyLoF",
    ]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )

    # All SMARCA4 mutations
    smarca4 = mutations[mutations["HugoSymbol"] == "SMARCA4"].copy()

    # LOF mutations: LikelyLoF == True OR VepImpact == HIGH
    smarca4_lof = smarca4[
        (smarca4["LikelyLoF"] == True) | (smarca4["VepImpact"] == "HIGH")
    ].copy()

    if len(smarca4_lof) == 0:
        return pd.DataFrame(columns=[
            "ModelID", "has_lof", "mutation_type", "n_lof_mutations",
            "protein_changes", "mutation_class",
        ])

    # Classify each mutation as truncating or missense
    smarca4_lof["is_truncating"] = smarca4_lof["VariantInfo"].str.lower().apply(
        lambda x: any(t in str(x) for t in CLASS1_VARIANT_TYPES)
    )

    # Per-line aggregation
    per_line = (
        smarca4_lof.groupby("ModelID")
        .agg(
            mutation_type=("VariantInfo", lambda x: ";".join(sorted(set(str(v) for v in x if pd.notna(v))))),
            n_lof_mutations=("VariantInfo", "count"),
            protein_changes=("ProteinChange", lambda x: ";".join(
                sorted(set(str(v) for v in x if pd.notna(v)))
            )),
            has_any_truncating=("is_truncating", "any"),
        )
        .reset_index()
    )

    # Class 1 = any truncating mutation, Class 2 = missense only
    per_line["mutation_class"] = np.where(
        per_line["has_any_truncating"], "Class_1", "Class_2"
    )
    per_line["has_lof"] = True
    per_line = per_line.drop(columns=["has_any_truncating"])

    return per_line


def load_comutations(depmap_dir: Path) -> pd.DataFrame:
    """Identify co-occurring LOF mutations in key genes per cell line."""
    cols = ["ModelID", "HugoSymbol", "LikelyLoF"]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    comut = mutations[
        (mutations["HugoSymbol"].isin(COMUTATION_GENES))
        & (mutations["LikelyLoF"] == True)
    ].copy()

    per_line = (
        comut.groupby("ModelID")["HugoSymbol"]
        .apply(lambda x: ";".join(sorted(set(x))))
        .rename("co_mutations")
        .reset_index()
    )
    return per_line


def add_smarca4_copy_number(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add SMARCA4 copy number and homozygous deletion call."""
    cn = load_depmap_matrix(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    if "SMARCA4" not in cn.columns:
        raise ValueError("SMARCA4 not found in CN data columns")

    smarca4_cn = cn["SMARCA4"].rename("CN_log2")
    df = df.join(smarca4_cn, how="left")
    df["has_homdel"] = df["CN_log2"] < HOMDEL_CN_THRESHOLD
    return df


def add_smarca4_expression(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add SMARCA4 expression (TPM log+1) for cross-validation."""
    expr = load_depmap_matrix(
        depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    if "SMARCA4" not in expr.columns:
        raise ValueError("SMARCA4 not found in expression data columns")

    smarca4_expr = expr["SMARCA4"].rename("expression")
    df = df.join(smarca4_expr, how="left")
    return df


def classify_smarca4_status(
    df: pd.DataFrame,
    lof_df: pd.DataFrame,
) -> pd.DataFrame:
    """Classify SMARCA4 status: deficient (LOF mutation OR homdel) vs intact."""
    # Merge LOF mutation info
    merge_cols = ["ModelID", "has_lof", "mutation_type", "n_lof_mutations",
                  "protein_changes", "mutation_class"]
    merge_cols = [c for c in merge_cols if c in lof_df.columns]

    df = df.merge(
        lof_df[merge_cols],
        left_index=True,
        right_on="ModelID",
        how="left",
    ).set_index("ModelID")

    df["has_lof"] = df["has_lof"].fillna(False)
    df["has_homdel"] = df["has_homdel"].fillna(False)

    # SMARCA4-deficient = LOF mutation OR homozygous deletion
    df["smarca4_status"] = np.where(
        df["has_lof"] | df["has_homdel"],
        "deficient",
        "intact",
    )

    # For homdel-only lines (no LOF mutation), set mutation_class to Class_1
    # (complete loss via deletion = equivalent to truncating)
    homdel_only = (df["smarca4_status"] == "deficient") & (~df["has_lof"]) & (df["has_homdel"])
    df.loc[homdel_only, "mutation_class"] = "Class_1"

    # Expression confirmation: deficient lines should have low SMARCA4 expression
    df["expression_confirmed"] = np.where(
        df["smarca4_status"] == "deficient",
        np.where(df["expression"].notna(), df["expression"] < np.log2(1.0 + 1), pd.NA),
        pd.NA,
    )

    return df


def cross_validate_expression(df: pd.DataFrame) -> dict:
    """Validate that SMARCA4-deficient lines have significantly lower expression."""
    has_data = df.dropna(subset=["expression"])
    deficient = has_data[has_data["smarca4_status"] == "deficient"]["expression"]
    intact = has_data[has_data["smarca4_status"] == "intact"]["expression"]

    if len(deficient) == 0 or len(intact) == 0:
        return {"valid": False, "reason": "No lines in one group"}

    stat, pval = stats.mannwhitneyu(deficient, intact, alternative="less")

    return {
        "valid": True,
        "n_deficient": len(deficient),
        "n_intact": len(intact),
        "median_expr_deficient": float(deficient.median()),
        "median_expr_intact": float(intact.median()),
        "mannwhitney_p": float(pval),
    }


def plot_expression_validation(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate boxplot of SMARCA4 expression by status."""
    has_data = df.dropna(subset=["expression"])
    deficient = has_data[has_data["smarca4_status"] == "deficient"]["expression"]
    intact = has_data[has_data["smarca4_status"] == "intact"]["expression"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [intact, deficient],
        tick_labels=["Intact", "SMARCA4-deficient"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")

    stat, pval = stats.mannwhitneyu(deficient, intact, alternative="less")
    ax.set_title(f"SMARCA4 Expression by Status\n(Mann-Whitney p={pval:.2e})")
    ax.set_ylabel("SMARCA4 Expression (log2 TPM+1)")
    ax.set_xlabel(f"Intact (n={len(intact)})    Deficient (n={len(deficient)})")

    plt.tight_layout()
    fig.savefig(output_dir / "smarca4_expression_validation.png", dpi=150)
    plt.close(fig)


def plot_classification_summary(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    """Generate bar plots of SMARCA4 classification results."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: cancer type bar chart (qualifying types)
    qual = summary[summary["qualifies"]].sort_values("n_deficient", ascending=True)
    if len(qual) > 0:
        ax = axes[0]
        y_pos = range(len(qual))
        ax.barh(y_pos, qual["n_deficient"], color="#F44336", label="Deficient")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(qual["cancer_type"], fontsize=8)
        ax.set_xlabel("Number of cell lines")
        ax.set_title("SMARCA4-Deficient Lines by Cancer Type\n(qualifying types)")
        ax.legend()
    else:
        axes[0].text(0.5, 0.5, "No qualifying cancer types", ha="center", va="center",
                     transform=axes[0].transAxes)
        axes[0].set_title("SMARCA4-Deficient Lines by Cancer Type")

    # Right: mutation class pie chart
    deficient = df[df["smarca4_status"] == "deficient"]
    class_counts = deficient["mutation_class"].value_counts()
    if len(class_counts) > 0:
        ax = axes[1]
        colors = {"Class_1": "#E53935", "Class_2": "#FB8C00"}
        ax.pie(
            class_counts.values,
            labels=[f"{k}\n(n={v})" for k, v in class_counts.items()],
            colors=[colors.get(k, "#999") for k in class_counts.index],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title("Mutation Class Distribution\n(Class 1=truncating, Class 2=missense)")
    else:
        axes[1].text(0.5, 0.5, "No deficient lines", ha="center", va="center",
                     transform=axes[1].transAxes)

    plt.tight_layout()
    fig.savefig(output_dir / "classification_summary.png", dpi=150)
    plt.close(fig)


def build_cancer_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize SMARCA4 status per cancer type (OncotreeLineage)."""
    has_status = df[df["smarca4_status"].notna()]

    summary_rows = []
    for lineage, group in has_status.groupby("OncotreeLineage"):
        n_total = len(group)
        n_deficient = int((group["smarca4_status"] == "deficient").sum())
        n_intact = n_total - n_deficient
        freq = n_deficient / n_total if n_total > 0 else 0.0
        qualifies = n_deficient >= MIN_DEFICIENT and n_intact >= MIN_INTACT

        # Count mutation classes
        def_group = group[group["smarca4_status"] == "deficient"]
        n_class1 = int((def_group["mutation_class"] == "Class_1").sum())
        n_class2 = int((def_group["mutation_class"] == "Class_2").sum())

        # Subtypes present
        subtypes = ";".join(sorted(group["OncotreeSubtype"].dropna().unique())) if "OncotreeSubtype" in group.columns else ""

        notes = ""
        if lineage == "Skin":
            notes = "UV-passenger SMARCA4 mutations may inflate count"

        summary_rows.append({
            "cancer_type": lineage,
            "n_deficient": n_deficient,
            "n_intact": n_intact,
            "n_total": n_total,
            "mutation_freq": round(freq, 4),
            "n_class1": n_class1,
            "n_class2": n_class2,
            "qualifies": qualifies,
            "subtypes": subtypes,
            "notes": notes,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("mutation_freq", ascending=False).reset_index(drop=True)
    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: SMARCA4 (BRG1) Cell Line Classifier ===\n")

    # Step 1: Load all cell lines
    print("Loading all DepMap cell lines...")
    df = load_all_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with OncotreeLineage annotation")

    # Step 2: Load SMARCA4 mutations (LOF + missense for class annotation)
    print("Loading SMARCA4 mutations...")
    lof_df = load_smarca4_mutations(DEPMAP_DIR)
    print(f"  {len(lof_df)} cell lines with SMARCA4 LOF/HIGH-impact mutations")
    if len(lof_df) > 0:
        n_c1 = (lof_df["mutation_class"] == "Class_1").sum()
        n_c2 = (lof_df["mutation_class"] == "Class_2").sum()
        print(f"  Mutation classes: {n_c1} Class 1 (truncating), {n_c2} Class 2 (missense)")

    # Step 3: Add SMARCA4 copy number
    print("Adding SMARCA4 copy number...")
    df = add_smarca4_copy_number(df, DEPMAP_DIR)
    n_homdel = df["has_homdel"].sum()
    print(f"  {n_homdel} cell lines with SMARCA4 homozygous deletion (CN log2 < {HOMDEL_CN_THRESHOLD})")

    # Step 4: Add SMARCA4 expression
    print("Adding SMARCA4 expression...")
    df = add_smarca4_expression(df, DEPMAP_DIR)
    n_with_expr = df["expression"].notna().sum()
    print(f"  {n_with_expr} cell lines with expression data")

    # Step 5: Classify SMARCA4 status
    print("Classifying SMARCA4 status (LOF mutation OR homdel)...")
    df = classify_smarca4_status(df, lof_df)
    n_deficient = (df["smarca4_status"] == "deficient").sum()
    n_intact = (df["smarca4_status"] == "intact").sum()

    n_lof_only = ((df["has_lof"]) & (~df["has_homdel"])).sum()
    n_homdel_only = ((~df["has_lof"]) & (df["has_homdel"])).sum()
    n_both = ((df["has_lof"]) & (df["has_homdel"])).sum()
    print(f"  {n_deficient} SMARCA4-deficient ({n_lof_only} LOF-only, {n_homdel_only} homdel-only, {n_both} both)")
    print(f"  {n_intact} intact")

    # Step 6: Cross-validate with expression
    print("\nCross-validating with SMARCA4 expression...")
    validation = cross_validate_expression(df)
    if validation["valid"]:
        print(f"  Median expression: deficient={validation['median_expr_deficient']:.2f} "
              f"vs intact={validation['median_expr_intact']:.2f}")
        print(f"  Mann-Whitney p={validation['mannwhitney_p']:.2e}")

    # Step 7: Expression validation plot
    print("Generating expression validation plot...")
    plot_expression_validation(df, OUTPUT_DIR)

    # Step 8: Annotate co-mutations
    print("Annotating co-mutations (SMARCA2, ARID1A, TP53, KRAS, STK11, KEAP1, ...)...")
    comut_df = load_comutations(DEPMAP_DIR)
    df = df.merge(comut_df, left_index=True, right_on="ModelID", how="left").set_index("ModelID")
    df["co_mutations"] = df["co_mutations"].fillna("")
    n_comut = (df["co_mutations"] != "").sum()
    print(f"  {n_comut} cell lines with co-occurring LOF mutations in key genes")

    # Step 9: Save classified cell lines
    output_cols = [
        "OncotreeLineage", "OncotreeSubtype", "smarca4_status", "mutation_class",
        "has_lof", "has_homdel", "expression", "expression_confirmed",
        "CN_log2", "mutation_type", "n_lof_mutations", "protein_changes",
        "co_mutations",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()
    out_classified = OUTPUT_DIR / "smarca4_classified_lines.csv"
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
        print(f"  {row['cancer_type']}: {row['n_deficient']}/{row['n_total']} deficient "
              f"({row['mutation_freq']:.1%}), n_intact={row['n_intact']}, "
              f"Class1={row['n_class1']}, Class2={row['n_class2']}{note}")

    # Step 11: Classification summary plot
    print("\nGenerating classification summary plot...")
    plot_classification_summary(df, summary, OUTPUT_DIR)

    print(f"\nSaved summary to {out_summary.name}")
    print("Saved plots: smarca4_expression_validation.png, classification_summary.png")
    print("\n=== Phase 1 Complete ===")


if __name__ == "__main__":
    main()
