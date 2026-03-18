"""Phase 1: Classify PTEN status across all DepMap 25Q3 cell lines.

Classifies cell lines as PTEN-lost (deep deletion OR truncating mutation OR both)
vs PTEN-intact. Cross-validates with expression data, annotates PIK3CA co-mutation,
TP53 and RB1 co-alteration status, and PTEN loss mechanism.

Excludes ambiguous lines (missense-only VUS) from both groups.

Usage:
    uv run python -m pten_loss_pancancer_dependency_atlas.01_pten_classifier
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
OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas" / "phase1"

# CN threshold for deep deletion (PortalOmicsCNGeneLog2 ratio scale: 1.0 = diploid)
HOMDEL_CN_THRESHOLD = 0.3

# Minimum samples per group for powered analysis
MIN_LOST = 5
MIN_INTACT = 10

# PIK3CA activating hotspot mutations
PIK3CA_HOTSPOTS = {
    "E542K", "E545K", "E545A", "E545G", "E545Q",
    "H1047R", "H1047L", "H1047Y",
    "C420R", "N345K", "R88Q",
}


def load_all_lines(depmap_dir: Path) -> pd.DataFrame:
    """Load DepMap Model.csv for all cell lines with cancer type annotation."""
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    meta = meta[meta["OncotreeLineage"].notna()].copy()
    return meta


def load_crispr_lines(depmap_dir: Path) -> set[str]:
    """Return set of ModelIDs that have CRISPR data."""
    crispr = pd.read_csv(depmap_dir / "CRISPRGeneEffect.csv", usecols=[0])
    return set(crispr.iloc[:, 0])


def load_pten_truncating_mutations(depmap_dir: Path) -> pd.DataFrame:
    """Identify cell lines with PTEN truncating mutations (nonsense, frameshift, splice).

    Uses DepMap's LikelyLoF annotation.
    """
    cols = [
        "ModelID", "HugoSymbol", "VariantType", "VariantInfo",
        "ProteinChange", "VepImpact", "LikelyLoF",
    ]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    # Filter to PTEN LOF mutations (nonsense, frameshift, splice)
    pten_lof = mutations[
        (mutations["HugoSymbol"] == "PTEN") & (mutations["LikelyLoF"] == True)
    ].copy()

    # Summarize per cell line
    per_line = (
        pten_lof.groupby("ModelID")
        .agg(
            pten_mutation_type=("VariantInfo", lambda x: ";".join(sorted(set(x)))),
            n_lof_mutations=("VariantInfo", "count"),
            pten_protein_changes=("ProteinChange", lambda x: ";".join(
                sorted(set(str(v) for v in x if pd.notna(v)))
            )),
        )
        .reset_index()
    )
    per_line["has_truncating"] = True
    return per_line


def load_pten_missense_only(depmap_dir: Path, lof_model_ids: set[str]) -> set[str]:
    """Identify cell lines with PTEN missense mutations but NO LOF mutations.

    These are ambiguous (VUS) and should be excluded from both groups.
    """
    cols = ["ModelID", "HugoSymbol", "VariantInfo", "VepImpact", "LikelyLoF"]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    pten_muts = mutations[mutations["HugoSymbol"] == "PTEN"].copy()

    # Lines with any PTEN mutation
    all_pten_mutated = set(pten_muts["ModelID"].unique())

    # Lines with PTEN missense only (no LOF)
    missense_only = all_pten_mutated - lof_model_ids

    return missense_only


def add_pten_copy_number(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add PTEN copy number and deep deletion call."""
    cn = load_depmap_matrix(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    if "PTEN" not in cn.columns:
        raise ValueError("PTEN not found in CN data columns")

    pten_cn = cn["PTEN"].rename("pten_CN_log2")
    df = df.join(pten_cn, how="left")
    df["has_deep_deletion"] = df["pten_CN_log2"] <= HOMDEL_CN_THRESHOLD
    return df


def add_pten_expression(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add PTEN expression (TPM log+1) for cross-validation."""
    expr = load_depmap_matrix(
        depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    if "PTEN" not in expr.columns:
        raise ValueError("PTEN not found in expression data columns")

    pten_expr = expr["PTEN"].rename("pten_expression")
    df = df.join(pten_expr, how="left")
    return df


def load_comutation_status(depmap_dir: Path) -> pd.DataFrame:
    """Load PIK3CA, TP53, and RB1 mutation/alteration status per cell line."""
    cols = [
        "ModelID", "HugoSymbol", "VariantInfo", "ProteinChange",
        "VepImpact", "LikelyLoF",
    ]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )

    results = {}

    # PIK3CA activating mutations (hotspot check)
    pik3ca = mutations[mutations["HugoSymbol"] == "PIK3CA"].copy()
    pik3ca_hotspot = pik3ca[
        pik3ca["ProteinChange"].apply(
            lambda x: any(hs in str(x) for hs in PIK3CA_HOTSPOTS) if pd.notna(x) else False
        )
    ]
    pik3ca_any = set(pik3ca[pik3ca["VepImpact"].isin(["HIGH", "MODERATE"])]["ModelID"])
    pik3ca_hs = set(pik3ca_hotspot["ModelID"])

    # TP53 mutations (HIGH or MODERATE impact)
    tp53 = mutations[
        (mutations["HugoSymbol"] == "TP53")
        & (mutations["VepImpact"].isin(["HIGH", "MODERATE"]))
    ]
    tp53_mutated = set(tp53["ModelID"])

    # RB1 LOF mutations
    rb1 = mutations[
        (mutations["HugoSymbol"] == "RB1") & (mutations["LikelyLoF"] == True)
    ]
    rb1_lof = set(rb1["ModelID"])

    return pik3ca_hs, pik3ca_any, tp53_mutated, rb1_lof


def add_rb1_copy_number(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Add RB1 deep deletion status for co-alteration annotation."""
    cn = load_depmap_matrix(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    if "RB1" in cn.columns:
        rb1_cn = cn["RB1"].rename("rb1_CN_log2")
        df = df.join(rb1_cn, how="left")
        df["rb1_deep_deletion"] = df["rb1_CN_log2"] <= HOMDEL_CN_THRESHOLD
    else:
        df["rb1_deep_deletion"] = False
    return df


def classify_pten_status(
    df: pd.DataFrame,
    lof_df: pd.DataFrame,
    missense_only_ids: set[str],
    pik3ca_hotspot_ids: set[str],
    pik3ca_any_ids: set[str],
    tp53_ids: set[str],
    rb1_lof_ids: set[str],
) -> pd.DataFrame:
    """Classify PTEN status and annotate co-alterations.

    PTEN-lost = deep deletion OR truncating mutation OR both.
    Excludes missense-only VUS lines from both groups.
    """
    # Merge truncating mutation info
    df = df.merge(
        lof_df[["ModelID", "has_truncating", "pten_mutation_type",
                 "n_lof_mutations", "pten_protein_changes"]],
        left_index=True,
        right_on="ModelID",
        how="left",
    ).set_index("ModelID")

    df["has_truncating"] = df["has_truncating"].fillna(False)
    df["has_deep_deletion"] = df["has_deep_deletion"].fillna(False)

    # Determine loss mechanism
    both = df["has_truncating"] & df["has_deep_deletion"]
    del_only = (~df["has_truncating"]) & df["has_deep_deletion"]
    mut_only = df["has_truncating"] & (~df["has_deep_deletion"])

    df["loss_mechanism"] = ""
    df.loc[both, "loss_mechanism"] = "deletion+mutation"
    df.loc[del_only, "loss_mechanism"] = "deletion-only"
    df.loc[mut_only, "loss_mechanism"] = "mutation-only"

    # PTEN status: lost, intact, or excluded (missense VUS)
    is_lost = df["has_truncating"] | df["has_deep_deletion"]
    is_missense_vus = df.index.isin(missense_only_ids) & ~is_lost
    is_intact = ~is_lost & ~is_missense_vus

    df["PTEN_status"] = "intact"
    df.loc[is_lost, "PTEN_status"] = "lost"
    df.loc[is_missense_vus, "PTEN_status"] = "excluded_missense_VUS"

    # Co-alteration annotations
    df["PIK3CA_hotspot"] = df.index.isin(pik3ca_hotspot_ids)
    df["PIK3CA_mutated"] = df.index.isin(pik3ca_any_ids)
    df["TP53_mutated"] = df.index.isin(tp53_ids)
    df["RB1_altered"] = df.index.isin(rb1_lof_ids) | df["rb1_deep_deletion"].fillna(False)

    return df


def cross_validate_expression(df: pd.DataFrame) -> dict:
    """Validate that PTEN-lost lines have significantly lower expression."""
    has_data = df[df["PTEN_status"].isin(["lost", "intact"])].dropna(subset=["pten_expression"])
    lost = has_data[has_data["PTEN_status"] == "lost"]["pten_expression"]
    intact = has_data[has_data["PTEN_status"] == "intact"]["pten_expression"]

    if len(lost) == 0 or len(intact) == 0:
        return {"valid": False, "reason": "No lines in one group"}

    stat, pval = stats.mannwhitneyu(lost, intact, alternative="less")

    # Expression quartile check
    q25_intact = intact.quantile(0.25)
    n_lost_below_q25 = (lost < q25_intact).sum()
    pct_lost_below_q25 = n_lost_below_q25 / len(lost) * 100

    # Flag discordant lines (PTEN-lost but high expression)
    q75_intact = intact.quantile(0.75)
    n_discordant = (lost > q75_intact).sum()

    return {
        "valid": True,
        "n_lost": len(lost),
        "n_intact": len(intact),
        "median_expr_lost": float(lost.median()),
        "median_expr_intact": float(intact.median()),
        "mannwhitney_p": float(pval),
        "q25_intact": float(q25_intact),
        "pct_lost_below_q25": float(pct_lost_below_q25),
        "n_discordant_high_expr": int(n_discordant),
    }


def plot_expression_validation(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate boxplot of PTEN expression by status."""
    has_data = df[df["PTEN_status"].isin(["lost", "intact"])].dropna(subset=["pten_expression"])
    lost = has_data[has_data["PTEN_status"] == "lost"]["pten_expression"]
    intact = has_data[has_data["PTEN_status"] == "intact"]["pten_expression"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [intact, lost],
        tick_labels=["PTEN-intact", "PTEN-lost"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")

    stat, pval = stats.mannwhitneyu(lost, intact, alternative="less")
    ax.set_title(f"PTEN Expression by Status\n(Mann-Whitney p={pval:.2e})")
    ax.set_ylabel("PTEN Expression (log2 TPM+1)")
    ax.set_xlabel(f"Intact (n={len(intact)})    Lost (n={len(lost)})")

    plt.tight_layout()
    fig.savefig(output_dir / "pten_expression_validation.png", dpi=150)
    plt.close(fig)


def build_cancer_type_summary(
    df: pd.DataFrame,
    crispr_lines: set[str],
) -> pd.DataFrame:
    """Summarize PTEN status per cancer type with power analysis."""
    analyzable = df[df["PTEN_status"].isin(["lost", "intact"])]

    summary_rows = []
    for lineage, group in analyzable.groupby("OncotreeLineage"):
        n_total = len(group)
        n_lost = int((group["PTEN_status"] == "lost").sum())
        n_intact = n_total - n_lost
        freq = n_lost / n_total if n_total > 0 else 0.0

        # CRISPR-available counts
        has_crispr = group.index.isin(crispr_lines)
        n_lost_crispr = int((has_crispr & (group["PTEN_status"] == "lost")).sum())
        n_intact_crispr = int((has_crispr & (group["PTEN_status"] == "intact")).sum())

        qualifies = n_lost_crispr >= MIN_LOST and n_intact_crispr >= MIN_INTACT

        # Co-alteration frequencies among PTEN-lost
        lost_group = group[group["PTEN_status"] == "lost"]
        n_pik3ca_comut = int(lost_group["PIK3CA_hotspot"].sum()) if n_lost > 0 else 0
        n_tp53_comut = int(lost_group["TP53_mutated"].sum()) if n_lost > 0 else 0

        # Loss mechanism breakdown
        n_del_only = int((lost_group["loss_mechanism"] == "deletion-only").sum()) if n_lost > 0 else 0
        n_mut_only = int((lost_group["loss_mechanism"] == "mutation-only").sum()) if n_lost > 0 else 0
        n_both = int((lost_group["loss_mechanism"] == "deletion+mutation").sum()) if n_lost > 0 else 0

        summary_rows.append({
            "cancer_type": lineage,
            "n_lost": n_lost,
            "n_intact": n_intact,
            "n_total": n_total,
            "pten_loss_freq": round(freq, 4),
            "n_lost_crispr": n_lost_crispr,
            "n_intact_crispr": n_intact_crispr,
            "qualifies": qualifies,
            "n_deletion_only": n_del_only,
            "n_mutation_only": n_mut_only,
            "n_both": n_both,
            "n_pik3ca_comut": n_pik3ca_comut,
            "n_tp53_comut": n_tp53_comut,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("pten_loss_freq", ascending=False).reset_index(drop=True)
    return summary


def write_summary_text(
    output_dir: Path,
    df: pd.DataFrame,
    summary: pd.DataFrame,
    validation: dict,
    n_excluded: int,
) -> None:
    """Write human-readable summary text file."""
    lines = []
    lines.append("=" * 70)
    lines.append("PTEN Loss Pan-Cancer Dependency Atlas - Phase 1 Classifier Summary")
    lines.append("=" * 70)
    lines.append("")

    analyzable = df[df["PTEN_status"].isin(["lost", "intact"])]
    n_lost = (df["PTEN_status"] == "lost").sum()
    n_intact = (df["PTEN_status"] == "intact").sum()

    lines.append(f"Total cell lines with lineage annotation: {len(df)}")
    lines.append(f"PTEN-lost:    {n_lost}")
    lines.append(f"PTEN-intact:  {n_intact}")
    lines.append(f"Excluded (missense VUS): {n_excluded}")
    lines.append("")

    # Loss mechanism breakdown
    lost = df[df["PTEN_status"] == "lost"]
    n_del_only = (lost["loss_mechanism"] == "deletion-only").sum()
    n_mut_only = (lost["loss_mechanism"] == "mutation-only").sum()
    n_both = (lost["loss_mechanism"] == "deletion+mutation").sum()
    lines.append("Loss mechanism breakdown:")
    lines.append(f"  Deletion only:      {n_del_only}")
    lines.append(f"  Mutation only:      {n_mut_only}")
    lines.append(f"  Both (del+mut):     {n_both}")
    lines.append("")

    # Co-alteration frequencies
    lines.append("Co-alteration frequencies (among PTEN-lost):")
    lines.append(f"  PIK3CA hotspot: {lost['PIK3CA_hotspot'].sum()}/{n_lost} "
                 f"({lost['PIK3CA_hotspot'].mean()*100:.1f}%)")
    lines.append(f"  TP53 mutated:   {lost['TP53_mutated'].sum()}/{n_lost} "
                 f"({lost['TP53_mutated'].mean()*100:.1f}%)")
    lines.append(f"  RB1 altered:    {lost['RB1_altered'].sum()}/{n_lost} "
                 f"({lost['RB1_altered'].mean()*100:.1f}%)")
    lines.append("")

    # Expression cross-validation
    lines.append("Expression cross-validation:")
    if validation["valid"]:
        lines.append(f"  Median expression (lost):   {validation['median_expr_lost']:.2f}")
        lines.append(f"  Median expression (intact): {validation['median_expr_intact']:.2f}")
        lines.append(f"  Mann-Whitney p-value:       {validation['mannwhitney_p']:.2e}")
        lines.append(f"  % lost below intact Q25:    {validation['pct_lost_below_q25']:.1f}%")
        lines.append(f"  Discordant (lost w/ high expr): {validation['n_discordant_high_expr']}")
    else:
        lines.append(f"  FAILED: {validation.get('reason', 'unknown')}")
    lines.append("")

    # Qualifying cancer types
    qualifying = summary[summary["qualifies"]]
    lines.append(f"Qualifying cancer types ({len(qualifying)} of {len(summary)}):")
    lines.append(f"  (threshold: >= {MIN_LOST} PTEN-lost + >= {MIN_INTACT} PTEN-intact with CRISPR)")
    lines.append("")
    for _, row in qualifying.iterrows():
        lines.append(
            f"  {row['cancer_type']:30s} lost={row['n_lost']:3d} "
            f"intact={row['n_intact']:3d} freq={row['pten_loss_freq']:.1%} "
            f"CRISPR: {row['n_lost_crispr']}/{row['n_intact_crispr']}"
        )
    lines.append("")

    # Non-qualifying types with PTEN loss
    non_q = summary[(~summary["qualifies"]) & (summary["n_lost"] > 0)]
    if len(non_q) > 0:
        lines.append(f"Non-qualifying types with PTEN loss ({len(non_q)}):")
        for _, row in non_q.iterrows():
            lines.append(
                f"  {row['cancer_type']:30s} lost={row['n_lost']:3d} "
                f"intact={row['n_intact']:3d} CRISPR: {row['n_lost_crispr']}/{row['n_intact_crispr']}"
            )
    lines.append("")

    text = "\n".join(lines)
    (output_dir / "pten_classifier_summary.txt").write_text(text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: PTEN Loss Cell Line Classifier ===\n")

    # Step 1: Load all cell lines
    print("Loading all DepMap cell lines...")
    df = load_all_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with OncotreeLineage annotation")

    # Step 2: Load CRISPR line set
    print("Loading CRISPR line IDs...")
    crispr_lines = load_crispr_lines(DEPMAP_DIR)
    print(f"  {len(crispr_lines)} cell lines with CRISPR data")

    # Step 3: Load PTEN truncating mutations
    print("Loading PTEN truncating mutations...")
    lof_df = load_pten_truncating_mutations(DEPMAP_DIR)
    print(f"  {len(lof_df)} cell lines with PTEN truncating mutations")

    # Step 4: Identify missense-only VUS lines (to exclude)
    lof_model_ids = set(lof_df["ModelID"])
    missense_only_ids = load_pten_missense_only(DEPMAP_DIR, lof_model_ids)
    print(f"  {len(missense_only_ids)} cell lines with PTEN missense-only (excluded as VUS)")

    # Step 5: Add PTEN copy number
    print("Adding PTEN copy number...")
    df = add_pten_copy_number(df, DEPMAP_DIR)
    n_homdel = df["has_deep_deletion"].sum()
    print(f"  {n_homdel} cell lines with PTEN deep deletion (CN log2 < {HOMDEL_CN_THRESHOLD})")

    # Step 6: Add PTEN expression
    print("Adding PTEN expression...")
    df = add_pten_expression(df, DEPMAP_DIR)
    n_with_expr = df["pten_expression"].notna().sum()
    print(f"  {n_with_expr} cell lines with expression data")

    # Step 7: Add RB1 copy number for co-alteration
    print("Adding RB1 copy number...")
    df = add_rb1_copy_number(df, DEPMAP_DIR)

    # Step 8: Load co-mutation data
    print("Loading co-mutation data (PIK3CA, TP53, RB1)...")
    pik3ca_hs, pik3ca_any, tp53_ids, rb1_lof = load_comutation_status(DEPMAP_DIR)
    print(f"  PIK3CA hotspot: {len(pik3ca_hs)}, TP53 mutated: {len(tp53_ids)}, RB1 LOF: {len(rb1_lof)}")

    # Step 9: Classify PTEN status
    print("Classifying PTEN status...")
    df = classify_pten_status(
        df, lof_df, missense_only_ids,
        pik3ca_hs, pik3ca_any, tp53_ids, rb1_lof,
    )
    n_lost = (df["PTEN_status"] == "lost").sum()
    n_intact = (df["PTEN_status"] == "intact").sum()
    n_excluded = (df["PTEN_status"] == "excluded_missense_VUS").sum()

    # Loss mechanism breakdown
    lost = df[df["PTEN_status"] == "lost"]
    n_del_only = (lost["loss_mechanism"] == "deletion-only").sum()
    n_mut_only = (lost["loss_mechanism"] == "mutation-only").sum()
    n_both_mech = (lost["loss_mechanism"] == "deletion+mutation").sum()
    print(f"  {n_lost} PTEN-lost ({n_del_only} del-only, {n_mut_only} mut-only, {n_both_mech} both)")
    print(f"  {n_intact} PTEN-intact")
    print(f"  {n_excluded} excluded (missense VUS)")

    # Step 10: Cross-validate with expression
    print("\nCross-validating with PTEN expression...")
    validation = cross_validate_expression(df)
    if validation["valid"]:
        print(f"  Median expression: lost={validation['median_expr_lost']:.2f} "
              f"vs intact={validation['median_expr_intact']:.2f}")
        print(f"  Mann-Whitney p={validation['mannwhitney_p']:.2e}")
        print(f"  {validation['pct_lost_below_q25']:.1f}% of lost lines below intact Q25")
        print(f"  {validation['n_discordant_high_expr']} discordant (lost w/ high expression)")

    # Step 11: Generate expression validation plot
    print("Generating expression validation plot...")
    plot_expression_validation(df, OUTPUT_DIR)

    # Step 12: Compute expression quartile for each line
    intact_expr = df.loc[df["PTEN_status"] == "intact", "pten_expression"].dropna()
    if len(intact_expr) > 0:
        q25 = intact_expr.quantile(0.25)
        q50 = intact_expr.quantile(0.50)
        q75 = intact_expr.quantile(0.75)

        def expr_quartile(val):
            if pd.isna(val):
                return ""
            if val < q25:
                return "Q1"
            elif val < q50:
                return "Q2"
            elif val < q75:
                return "Q3"
            else:
                return "Q4"

        df["expression_quartile"] = df["pten_expression"].apply(expr_quartile)
    else:
        df["expression_quartile"] = ""

    # Step 13: Save classified cell lines
    output_cols = [
        "OncotreeLineage", "PTEN_status", "loss_mechanism",
        "pten_CN_log2", "pten_expression", "expression_quartile",
        "has_truncating", "has_deep_deletion",
        "pten_mutation_type", "n_lof_mutations", "pten_protein_changes",
        "PIK3CA_hotspot", "PIK3CA_mutated", "TP53_mutated", "RB1_altered",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()
    out_classified = OUTPUT_DIR / "pten_classification.csv"
    result.to_csv(out_classified)
    print(f"\nSaved {len(result)} classified cell lines to {out_classified.name}")

    # Step 14: Cancer type summary with power analysis
    summary = build_cancer_type_summary(df, crispr_lines)
    out_summary = OUTPUT_DIR / "qualifying_cancer_types.csv"
    summary.to_csv(out_summary, index=False)

    n_qualifying = summary["qualifies"].sum()
    print(f"\nCancer type summary ({len(summary)} types, {n_qualifying} qualifying):")
    print(f"  (threshold: >= {MIN_LOST} PTEN-lost + >= {MIN_INTACT} PTEN-intact with CRISPR)")
    qualifying = summary[summary["qualifies"]]
    for _, row in qualifying.iterrows():
        print(f"  {row['cancer_type']:30s} lost={row['n_lost']:3d} "
              f"intact={row['n_intact']:3d} freq={row['pten_loss_freq']:.1%} "
              f"CRISPR: {row['n_lost_crispr']}/{row['n_intact_crispr']}")

    # Step 15: Write summary text
    print("\nWriting summary text...")
    write_summary_text(OUTPUT_DIR, df, summary, validation, n_excluded)

    print(f"\nSaved qualifying_cancer_types.csv")
    print(f"Saved pten_expression_validation.png")
    print(f"Saved pten_classifier_summary.txt")
    print("\nDone.")


if __name__ == "__main__":
    main()
