"""Phase 1: Classify CDKN2A deletion status across all DepMap 25Q3 cell lines.

Classifies cell lines as CDKN2A-deleted (deep deletion by copy number) vs intact.
Cross-validates with expression data. Annotates RB1 status, TP53 status,
CDKN2B co-deletion, and CCNE1 amplification.

Usage:
    uv run python -m cdkn2a_pancancer_dependency_atlas.01_cdkn2a_classifier
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
OUTPUT_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase1"

# CN threshold for deep deletion in PortalOmicsCNGeneLog2.csv.
# Data is in log2(relative CN) format but floored at ~0; values near 0 = deep deletion.
# Empirically: 330 CDKN2A deletions cluster at <=0.01, natural gap before 0.35+.
HOMDEL_CN_THRESHOLD = 0.3

# CN threshold for amplification (~4+ copies, value ~1.5+ in this scale)
AMP_CN_THRESHOLD = 1.5

# Minimum samples per group for powered analysis
MIN_DELETED = 5
MIN_INTACT = 10


def load_all_lines(depmap_dir: Path) -> pd.DataFrame:
    """Load DepMap Model.csv for all cell lines with cancer type annotation."""
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    meta = meta[meta["OncotreeLineage"].notna()].copy()
    return meta


def load_copy_number(depmap_dir: Path, gene: str) -> pd.Series:
    """Load copy number for a single gene from PortalOmicsCNGeneLog2.csv."""
    cn = load_depmap_matrix(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    if gene not in cn.columns:
        raise ValueError(f"{gene} not found in CN data columns")
    return cn[gene]


def load_expression(depmap_dir: Path, gene: str) -> pd.Series:
    """Load expression for a single gene."""
    expr = load_depmap_matrix(
        depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    if gene not in expr.columns:
        raise ValueError(f"{gene} not found in expression data columns")
    return expr[gene]


def load_lof_mutations(depmap_dir: Path, gene: str) -> set[str]:
    """Return set of ModelIDs with LOF mutations in a given gene."""
    cols = ["ModelID", "HugoSymbol", "LikelyLoF"]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    lof = mutations[
        (mutations["HugoSymbol"] == gene) & (mutations["LikelyLoF"] == True)
    ]
    return set(lof["ModelID"].unique())


def load_any_mutations(depmap_dir: Path, gene: str) -> set[str]:
    """Return set of ModelIDs with any non-silent mutation in a given gene."""
    cols = ["ModelID", "HugoSymbol", "VepImpact"]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    # Exclude LOW impact (synonymous) — keep HIGH, MODERATE, MODIFIER
    hit = mutations[
        (mutations["HugoSymbol"] == gene)
        & (mutations["VepImpact"] != "LOW")
    ]
    return set(hit["ModelID"].unique())


def load_crispr_lines(depmap_dir: Path) -> set[str]:
    """Return set of ModelIDs that have CRISPR data."""
    crispr = pd.read_csv(depmap_dir / "CRISPRGeneEffect.csv", usecols=[0])
    return set(crispr.iloc[:, 0])


def classify_cdkn2a(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Classify CDKN2A deletion status and annotate co-alterations."""
    # --- CDKN2A copy number (primary classification) ---
    cdkn2a_cn = load_copy_number(depmap_dir, "CDKN2A").rename("CDKN2A_CN_log2")
    df = df.join(cdkn2a_cn, how="left")
    df["CDKN2A_deleted"] = df["CDKN2A_CN_log2"] <= HOMDEL_CN_THRESHOLD

    # --- CDKN2A expression (cross-validation) ---
    cdkn2a_expr = load_expression(depmap_dir, "CDKN2A").rename("CDKN2A_expression")
    df = df.join(cdkn2a_expr, how="left")

    # Flag discordant: deleted by CN but still expressed (top 75th percentile)
    expr_q75 = df.loc[df["CDKN2A_deleted"] == False, "CDKN2A_expression"].quantile(0.25)
    df["expression_discordant"] = False
    has_both = df["CDKN2A_CN_log2"].notna() & df["CDKN2A_expression"].notna()
    # Deleted but expressed above the low-expression threshold
    df.loc[
        has_both & df["CDKN2A_deleted"] & (df["CDKN2A_expression"] > expr_q75),
        "expression_discordant",
    ] = True
    # Not deleted but expression near zero (potential other silencing)
    low_expr_threshold = 0.5  # log2(TPM+1)
    df.loc[
        has_both & ~df["CDKN2A_deleted"] & (df["CDKN2A_expression"] < low_expr_threshold),
        "expression_discordant",
    ] = True

    # --- CDKN2A status label ---
    df["CDKN2A_status"] = np.where(df["CDKN2A_deleted"].fillna(False), "deleted", "intact")
    # Lines with no CN data → unknown
    df.loc[df["CDKN2A_CN_log2"].isna(), "CDKN2A_status"] = "unknown"

    # --- RB1 status (CRITICAL: RB1 loss abolishes CDK4/6 dependency) ---
    rb1_cn = load_copy_number(depmap_dir, "RB1").rename("RB1_CN_log2")
    df = df.join(rb1_cn, how="left")
    rb1_lof_lines = load_lof_mutations(depmap_dir, "RB1")
    df["RB1_mutated"] = df.index.isin(rb1_lof_lines)
    df["RB1_homdel"] = df["RB1_CN_log2"] <= HOMDEL_CN_THRESHOLD
    df["RB1_status"] = np.where(
        df["RB1_mutated"] | df["RB1_homdel"].fillna(False),
        "lost",
        "intact",
    )

    # --- TP53 status (ARF/MDM2 axis context) ---
    tp53_mut_lines = load_any_mutations(depmap_dir, "TP53")
    df["TP53_status"] = np.where(df.index.isin(tp53_mut_lines), "mutant", "WT")

    # --- CDKN2B co-deletion ---
    cdkn2b_cn = load_copy_number(depmap_dir, "CDKN2B").rename("CDKN2B_CN_log2")
    df = df.join(cdkn2b_cn, how="left")
    df["CDKN2B_co_deleted"] = df["CDKN2B_CN_log2"] <= HOMDEL_CN_THRESHOLD

    # --- MTAP co-deletion (9p21 neighbour — confounds PRMT5/WDR77 dependency) ---
    mtap_cn = load_copy_number(depmap_dir, "MTAP").rename("MTAP_CN_log2")
    df = df.join(mtap_cn, how="left")
    df["MTAP_co_deleted"] = df["MTAP_CN_log2"] <= HOMDEL_CN_THRESHOLD

    # --- CCNE1 amplification (CDK2-dependent bypass) ---
    ccne1_cn = load_copy_number(depmap_dir, "CCNE1").rename("CCNE1_CN_log2")
    df = df.join(ccne1_cn, how="left")
    df["CCNE1_amplified"] = df["CCNE1_CN_log2"] >= AMP_CN_THRESHOLD

    # --- CRISPR data availability ---
    crispr_lines = load_crispr_lines(depmap_dir)
    df["has_crispr"] = df.index.isin(crispr_lines)

    return df


def cross_validate_expression(df: pd.DataFrame) -> dict:
    """Validate that CDKN2A-deleted lines have significantly lower expression."""
    known = df[df["CDKN2A_status"].isin(["deleted", "intact"])].dropna(
        subset=["CDKN2A_expression"]
    )
    deleted = known[known["CDKN2A_status"] == "deleted"]["CDKN2A_expression"]
    intact = known[known["CDKN2A_status"] == "intact"]["CDKN2A_expression"]

    if len(deleted) == 0 or len(intact) == 0:
        return {"valid": False, "reason": "No lines in one group"}

    stat, pval = stats.mannwhitneyu(deleted, intact, alternative="less")
    return {
        "valid": True,
        "n_deleted": len(deleted),
        "n_intact": len(intact),
        "median_expr_deleted": float(deleted.median()),
        "median_expr_intact": float(intact.median()),
        "mannwhitney_p": float(pval),
    }


def plot_expression_validation(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate boxplot of CDKN2A expression by deletion status."""
    known = df[df["CDKN2A_status"].isin(["deleted", "intact"])].dropna(
        subset=["CDKN2A_expression"]
    )
    deleted = known[known["CDKN2A_status"] == "deleted"]["CDKN2A_expression"]
    intact = known[known["CDKN2A_status"] == "intact"]["CDKN2A_expression"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [intact, deleted],
        tick_labels=["Intact", "CDKN2A-deleted"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")

    _, pval = stats.mannwhitneyu(deleted, intact, alternative="less")
    ax.set_title(f"CDKN2A Expression by Deletion Status\n(Mann-Whitney p={pval:.2e})")
    ax.set_ylabel("CDKN2A Expression (log2 TPM+1)")
    ax.set_xlabel(f"Intact (n={len(intact)})    Deleted (n={len(deleted)})")

    plt.tight_layout()
    fig.savefig(output_dir / "cdkn2a_expression_validation.png", dpi=150)
    plt.close(fig)


def build_cancer_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize CDKN2A status per cancer type with power analysis."""
    known = df[df["CDKN2A_status"].isin(["deleted", "intact"])]

    summary_rows = []
    for lineage, group in known.groupby("OncotreeLineage"):
        n_total = len(group)
        n_del = int((group["CDKN2A_status"] == "deleted").sum())
        n_intact = n_total - n_del
        del_freq = n_del / n_total if n_total > 0 else 0.0

        # CRISPR availability counts
        del_crispr = int(
            ((group["CDKN2A_status"] == "deleted") & group["has_crispr"]).sum()
        )
        intact_crispr = int(
            ((group["CDKN2A_status"] == "intact") & group["has_crispr"]).sum()
        )

        qualifies = del_crispr >= MIN_DELETED and intact_crispr >= MIN_INTACT

        # Co-alteration frequencies in deleted lines
        del_lines = group[group["CDKN2A_status"] == "deleted"]
        n_del_total = len(del_lines)
        rb1_co_lost = int(del_lines["RB1_status"].eq("lost").sum()) if n_del_total > 0 else 0
        tp53_co_mut = int(del_lines["TP53_status"].eq("mutant").sum()) if n_del_total > 0 else 0
        cdkn2b_co_del = int(del_lines["CDKN2B_co_deleted"].fillna(False).sum()) if n_del_total > 0 else 0
        mtap_co_del = int(del_lines["MTAP_co_deleted"].fillna(False).sum()) if n_del_total > 0 else 0
        ccne1_amp = int(del_lines["CCNE1_amplified"].fillna(False).sum()) if n_del_total > 0 else 0

        # RB1-intact deleted lines (primary analysis cohort)
        n_del_rb1_intact = int(
            ((del_lines["RB1_status"] == "intact") & del_lines["has_crispr"]).sum()
        )

        # Simple power note
        power_note = ""
        if qualifies and n_del_rb1_intact < 5:
            power_note = "low RB1-intact deleted count for stratified analysis"

        summary_rows.append({
            "cancer_type": lineage,
            "n_total": n_total,
            "n_deleted": n_del,
            "n_intact": n_intact,
            "deletion_freq": round(del_freq, 4),
            "n_deleted_crispr": del_crispr,
            "n_intact_crispr": intact_crispr,
            "n_deleted_rb1_intact_crispr": n_del_rb1_intact,
            "rb1_co_loss": rb1_co_lost,
            "tp53_co_mutation": tp53_co_mut,
            "cdkn2b_co_deletion": cdkn2b_co_del,
            "mtap_co_deletion": mtap_co_del,
            "ccne1_amplification": ccne1_amp,
            "qualifies": qualifies,
            "power_note": power_note,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("deletion_freq", ascending=False).reset_index(drop=True)
    return summary


def write_summary_txt(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    qualifying: pd.DataFrame,
    validation: dict,
    output_dir: Path,
) -> None:
    """Write human-readable summary text file."""
    known = df[df["CDKN2A_status"].isin(["deleted", "intact"])]
    n_total = len(known)
    n_del = int((known["CDKN2A_status"] == "deleted").sum())
    n_intact = n_total - n_del
    n_crispr = int(known["has_crispr"].sum())
    n_discordant = int(df["expression_discordant"].sum())

    del_lines = known[known["CDKN2A_status"] == "deleted"]
    n_rb1_lost = int(del_lines["RB1_status"].eq("lost").sum())
    n_tp53_mut = int(del_lines["TP53_status"].eq("mutant").sum())
    n_cdkn2b = int(del_lines["CDKN2B_co_deleted"].fillna(False).sum())
    n_mtap = int(del_lines["MTAP_co_deleted"].fillna(False).sum())
    n_ccne1 = int(del_lines["CCNE1_amplified"].fillna(False).sum())

    lines = [
        "=" * 60,
        "CDKN2A Pan-Cancer Dependency Atlas - Phase 1: CDKN2A Classifier",
        "=" * 60,
        "",
        "OVERVIEW",
        f"  Total cell lines with lineage + CN data: {n_total}",
        f"  Cell lines with CRISPR data: {n_crispr}",
        f"  CDKN2A-deleted (CN <= {HOMDEL_CN_THRESHOLD}): {n_del} ({n_del/n_total:.1%})",
        f"  CDKN2A-intact: {n_intact}",
        f"  Expression-discordant lines flagged: {n_discordant}",
        "",
        "EXPRESSION CROSS-VALIDATION",
    ]

    if validation.get("valid"):
        lines += [
            f"  Deleted median expression: {validation['median_expr_deleted']:.2f}",
            f"  Intact median expression: {validation['median_expr_intact']:.2f}",
            f"  Mann-Whitney U p-value: {validation['mannwhitney_p']:.2e}",
        ]
    else:
        lines.append(f"  Skipped: {validation.get('reason', 'unknown')}")

    lines += [
        "",
        "CO-ALTERATION LANDSCAPE (in CDKN2A-deleted lines)",
        f"  RB1 co-loss: {n_rb1_lost}/{n_del} ({n_rb1_lost/n_del:.1%})" if n_del > 0 else "  N/A",
        f"  TP53 mutation: {n_tp53_mut}/{n_del} ({n_tp53_mut/n_del:.1%})" if n_del > 0 else "  N/A",
        f"  CDKN2B co-deletion: {n_cdkn2b}/{n_del} ({n_cdkn2b/n_del:.1%})" if n_del > 0 else "  N/A",
        f"  MTAP co-deletion: {n_mtap}/{n_del} ({n_mtap/n_del:.1%})" if n_del > 0 else "  N/A",
        f"  CCNE1 amplification: {n_ccne1}/{n_del} ({n_ccne1/n_del:.1%})" if n_del > 0 else "  N/A",
        "",
        "CLINICAL NOTE: RB1 co-loss abolishes CDK4/6 dependency.",
        f"  {n_del - n_rb1_lost}/{n_del} deleted lines retain RB1 (eligible for CDK4/6i)." if n_del > 0 else "",
        "",
        f"QUALIFYING CANCER TYPES (>={MIN_DELETED} deleted + >={MIN_INTACT} intact with CRISPR)",
        "-" * 60,
    ]

    if len(qualifying) == 0:
        lines.append("  None qualify with current thresholds.")
    else:
        for _, row in qualifying.iterrows():
            note = f"  ** {row['power_note']}" if row["power_note"] else ""
            lines.append(
                f"  {row['cancer_type']}: {row['n_deleted_crispr']} del / "
                f"{row['n_intact_crispr']} intact (CRISPR), "
                f"{row['n_deleted']}/{row['n_total']} total ({row['deletion_freq']:.1%}), "
                f"RB1-intact-del={row['n_deleted_rb1_intact_crispr']}{note}"
            )

    lines += [
        "",
        "ALL CANCER TYPES WITH CDKN2A DELETIONS",
        "-" * 60,
    ]
    with_del = summary[summary["n_deleted"] > 0]
    for _, row in with_del.iterrows():
        q_flag = " *" if row["qualifies"] else ""
        lines.append(
            f"  {row['cancer_type']}: {row['n_deleted']} del / {row['n_intact']} intact "
            f"(CRISPR: {row['n_deleted_crispr']}/{row['n_intact_crispr']}){q_flag}"
        )

    lines += [
        "",
        "POWER ANALYSIS NOTE",
        f"  Qualifying threshold: >={MIN_DELETED} deleted + >={MIN_INTACT} intact with CRISPR.",
        "  Phase 2 primary analysis uses CDKN2A-deleted/RB1-intact stratum.",
        "  Cancer types with <5 RB1-intact deleted lines flagged for limited stratified power.",
        "",
    ]

    with open(output_dir / "cdkn2a_classifier_summary.txt", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: CDKN2A Cell Line Classifier ===\n")

    # Step 1: Load all cell lines
    print("Loading all DepMap cell lines...")
    df = load_all_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with OncotreeLineage annotation")

    # Step 2: Classify CDKN2A status and annotate co-alterations
    print("Classifying CDKN2A status and loading co-alterations...")
    print("  Loading CDKN2A copy number...")
    df = classify_cdkn2a(df, DEPMAP_DIR)

    n_del = (df["CDKN2A_status"] == "deleted").sum()
    n_intact = (df["CDKN2A_status"] == "intact").sum()
    n_unknown = (df["CDKN2A_status"] == "unknown").sum()
    print(f"  {n_del} deleted, {n_intact} intact, {n_unknown} unknown (no CN data)")

    # Co-alteration summary
    del_lines = df[df["CDKN2A_status"] == "deleted"]
    n_rb1 = del_lines["RB1_status"].eq("lost").sum()
    n_tp53 = del_lines["TP53_status"].eq("mutant").sum()
    n_cdkn2b = del_lines["CDKN2B_co_deleted"].fillna(False).sum()
    n_mtap = del_lines["MTAP_co_deleted"].fillna(False).sum()
    n_ccne1 = del_lines["CCNE1_amplified"].fillna(False).sum()
    print(f"  Co-alterations in deleted: RB1-lost={n_rb1}, TP53-mut={n_tp53}, "
          f"CDKN2B-codel={n_cdkn2b}, MTAP-codel={n_mtap}, CCNE1-amp={n_ccne1}")

    # Step 3: Cross-validate with expression
    print("\nCross-validating with CDKN2A expression...")
    validation = cross_validate_expression(df)
    if validation["valid"]:
        print(f"  Median expression: deleted={validation['median_expr_deleted']:.2f} "
              f"vs intact={validation['median_expr_intact']:.2f}")
        print(f"  Mann-Whitney p={validation['mannwhitney_p']:.2e}")
    n_discord = df["expression_discordant"].sum()
    print(f"  {n_discord} expression-discordant lines flagged")

    # Step 4: Expression validation plot
    print("Generating expression validation plot...")
    plot_expression_validation(df, OUTPUT_DIR)

    # Step 5: Cancer type summary
    print("\nBuilding cancer type summary...")
    summary = build_cancer_type_summary(df)
    qualifying = summary[summary["qualifies"]].copy()
    n_qualifying = len(qualifying)
    print(f"  {len(summary)} cancer types, {n_qualifying} qualifying")

    print(f"\nQualifying cancer types (>={MIN_DELETED} del + >={MIN_INTACT} intact with CRISPR):")
    if n_qualifying == 0:
        print("  None qualify.")
    else:
        for _, row in qualifying.iterrows():
            print(
                f"  {row['cancer_type']}: {row['n_deleted_crispr']} del / "
                f"{row['n_intact_crispr']} intact (CRISPR), "
                f"freq={row['deletion_freq']:.1%}"
            )

    # Step 6: Save outputs
    print("\nSaving outputs...")

    output_cols = [
        "OncotreeLineage", "CDKN2A_status", "CDKN2A_CN_log2", "CDKN2A_expression",
        "expression_discordant", "RB1_status", "RB1_CN_log2", "RB1_mutated", "RB1_homdel",
        "TP53_status", "CDKN2B_CN_log2", "CDKN2B_co_deleted",
        "MTAP_CN_log2", "MTAP_co_deleted",
        "CCNE1_CN_log2", "CCNE1_amplified", "has_crispr",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()
    result.to_csv(OUTPUT_DIR / "cdkn2a_classification.csv")
    print(f"  cdkn2a_classification.csv - {len(result)} lines")

    qualifying.to_csv(OUTPUT_DIR / "qualifying_cancer_types.csv", index=False)
    print(f"  qualifying_cancer_types.csv - {n_qualifying} types")

    write_summary_txt(df, summary, qualifying, validation, OUTPUT_DIR)
    print("  cdkn2a_classifier_summary.txt")

    print(f"\nSaved expression plot to cdkn2a_expression_validation.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
