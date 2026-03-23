"""Phase 1: Classify NF1-loss across all DepMap 25Q3 cell lines.

Cohort definitions:
  1. NF1-loss: truncating mutations (LikelyLoF) OR deep deletion (CN log2 <= -1.0)
  2. NF1-intact: no NF1 LOF mutation and no deep deletion

Co-mutation annotations: TP53, CDKN2A, KRAS, NRAS, HRAS.
Lines with concurrent RAS mutations (KRAS/NRAS/HRAS) are flagged for optional exclusion.
Cross-validates NF1-loss against NF1 expression levels.
Special attention to MPNST cell line availability.

Usage:
    uv run python -m nf1_loss_pancancer_dependency_atlas.01_nf1_loss_classifier
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
OUTPUT_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase1"
)

# CN threshold for deep deletion (log2 scale) — plan specifies <= -1.0
HOMDEL_CN_THRESHOLD = -1.0

# Minimum samples per group for powered analysis
MIN_ALTERED = 5
MIN_WT = 10

# Co-mutation genes to annotate
CO_MUTATION_GENES = ["TP53", "CDKN2A"]
RAS_GENES = ["KRAS", "NRAS", "HRAS"]


def load_all_lines(depmap_dir: Path) -> pd.DataFrame:
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    meta = meta[meta["OncotreeLineage"].notna()].copy()
    return meta


def load_copy_number(depmap_dir: Path, gene: str) -> pd.Series:
    cn = load_depmap_matrix(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    if gene not in cn.columns:
        raise ValueError(f"{gene} not found in CN data")
    return cn[gene]


def load_expression(depmap_dir: Path, gene: str) -> pd.Series:
    expr = load_depmap_matrix(
        depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    if gene not in expr.columns:
        raise ValueError(f"{gene} not found in expression data")
    return expr[gene]


def load_lof_mutations(depmap_dir: Path, gene: str) -> set[str]:
    cols = ["ModelID", "HugoSymbol", "LikelyLoF"]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    lof = mutations[
        (mutations["HugoSymbol"] == gene) & (mutations["LikelyLoF"] == True)  # noqa: E712
    ]
    return set(lof["ModelID"].unique())


def load_any_mutations(depmap_dir: Path, gene: str) -> set[str]:
    cols = ["ModelID", "HugoSymbol", "VepImpact"]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    hit = mutations[
        (mutations["HugoSymbol"] == gene) & (mutations["VepImpact"] != "LOW")
    ]
    return set(hit["ModelID"].unique())


def load_crispr_lines(depmap_dir: Path) -> set[str]:
    crispr = pd.read_csv(depmap_dir / "CRISPRGeneEffect.csv", usecols=[0])
    return set(crispr.iloc[:, 0])


def classify_nf1(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Classify NF1 loss status for all cell lines."""

    # --- NF1 copy number ---
    nf1_cn = load_copy_number(depmap_dir, "NF1").rename("NF1_CN_log2")
    df = df.join(nf1_cn, how="left")
    df["NF1_homdel"] = df["NF1_CN_log2"] <= HOMDEL_CN_THRESHOLD

    # --- NF1 LOF mutations ---
    nf1_lof_lines = load_lof_mutations(depmap_dir, "NF1")
    df["NF1_lof_mutation"] = df.index.isin(nf1_lof_lines)

    # --- NF1 loss (mutation OR deep deletion) ---
    df["NF1_loss"] = df["NF1_lof_mutation"] | df["NF1_homdel"].fillna(False)

    # Status label
    df["NF1_status"] = "intact"
    df.loc[df["NF1_lof_mutation"] & ~df["NF1_homdel"].fillna(False), "NF1_status"] = "LOF_mutation"
    df.loc[~df["NF1_lof_mutation"] & df["NF1_homdel"].fillna(False), "NF1_status"] = "deep_deletion"
    df.loc[df["NF1_lof_mutation"] & df["NF1_homdel"].fillna(False), "NF1_status"] = "both"

    # No data → unknown
    no_data = df["NF1_CN_log2"].isna() & ~df["NF1_lof_mutation"]
    df.loc[no_data, "NF1_status"] = "unknown"

    # --- NF1 expression for cross-validation ---
    try:
        nf1_expr = load_expression(depmap_dir, "NF1").rename("NF1_expression")
        df = df.join(nf1_expr, how="left")
    except ValueError:
        df["NF1_expression"] = np.nan

    # --- Co-mutations ---
    for gene in CO_MUTATION_GENES:
        mut_lines = load_any_mutations(depmap_dir, gene)
        df[f"{gene}_status"] = np.where(df.index.isin(mut_lines), "mutant", "WT")

    # --- RAS mutations (for exclusion flag) ---
    all_ras_lines: set[str] = set()
    for gene in RAS_GENES:
        mut_lines = load_any_mutations(depmap_dir, gene)
        df[f"{gene}_status"] = np.where(df.index.isin(mut_lines), "mutant", "WT")
        all_ras_lines |= mut_lines
    df["has_RAS_mutation"] = df.index.isin(all_ras_lines)

    # NF1+TP53 co-mutation (critical for MPNST)
    df["NF1_TP53_comut"] = df["NF1_loss"] & (df["TP53_status"] == "mutant")

    # --- CRISPR data availability ---
    crispr_lines = load_crispr_lines(depmap_dir)
    df["has_crispr"] = df.index.isin(crispr_lines)

    return df


def cross_validate_nf1_expression(df: pd.DataFrame) -> dict:
    """Validate that NF1-loss lines have lower NF1 expression."""
    known = df[df["NF1_status"].isin(["LOF_mutation", "deep_deletion", "both", "intact"])]

    lost_ids = known[known["NF1_loss"]].index
    intact_ids = known[~known["NF1_loss"]].index

    lost_expr = df.loc[df.index.isin(lost_ids), "NF1_expression"].dropna()
    intact_expr = df.loc[df.index.isin(intact_ids), "NF1_expression"].dropna()

    if len(lost_expr) < 5 or len(intact_expr) < 5:
        return {"valid": False, "reason": "Too few lines with expression data"}

    # NF1-loss lines should have LOWER expression (one-sided less)
    stat, pval = stats.mannwhitneyu(lost_expr, intact_expr, alternative="less")
    return {
        "valid": True,
        "n_lost": len(lost_expr),
        "n_intact": len(intact_expr),
        "median_lost": round(float(lost_expr.median()), 3),
        "median_intact": round(float(intact_expr.median()), 3),
        "mannwhitney_p": float(pval),
        "validates": pval < 0.05,
    }


def build_cancer_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize NF1 status per cancer type with power analysis."""
    known = df[df["NF1_status"] != "unknown"]

    summary_rows = []
    for lineage, group in known.groupby("OncotreeLineage"):
        n_total = len(group)
        n_lost = int(group["NF1_loss"].sum())
        n_intact = n_total - n_lost
        freq = n_lost / n_total if n_total > 0 else 0.0

        n_lof = int(group["NF1_lof_mutation"].sum())
        n_del = int(group["NF1_homdel"].fillna(False).sum())

        lost_crispr = int((group["NF1_loss"] & group["has_crispr"]).sum())
        intact_crispr = int((~group["NF1_loss"] & group["has_crispr"]).sum())
        qualifies = lost_crispr >= MIN_ALTERED and intact_crispr >= MIN_WT

        lost_lines = group[group["NF1_loss"]]
        n_tp53 = int((lost_lines["TP53_status"] == "mutant").sum()) if len(lost_lines) > 0 else 0
        n_cdkn2a = int((lost_lines["CDKN2A_status"] == "mutant").sum()) if len(lost_lines) > 0 else 0
        n_ras = int(lost_lines["has_RAS_mutation"].sum()) if len(lost_lines) > 0 else 0
        n_nf1_tp53 = int(lost_lines["NF1_TP53_comut"].sum()) if len(lost_lines) > 0 else 0

        # MPNST flag
        is_mpnst = "nerve" in str(lineage).lower() or "mpnst" in str(lineage).lower()

        summary_rows.append({
            "cancer_type": lineage,
            "n_total": n_total,
            "n_nf1_lost": n_lost,
            "n_intact": n_intact,
            "nf1_loss_freq": round(freq, 4),
            "n_lof_mutation": n_lof,
            "n_deep_deletion": n_del,
            "n_lost_crispr": lost_crispr,
            "n_intact_crispr": intact_crispr,
            "tp53_co_mutation": n_tp53,
            "cdkn2a_co_mutation": n_cdkn2a,
            "ras_co_mutation": n_ras,
            "nf1_tp53_comut": n_nf1_tp53,
            "is_mpnst_lineage": is_mpnst,
            "qualifies": qualifies,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("nf1_loss_freq", ascending=False).reset_index(drop=True)
    return summary


def characterize_mpnst_lines(df: pd.DataFrame) -> pd.DataFrame:
    """Individual characterization of MPNST / peripheral nerve sheath lines."""
    # Look for MPNST in lineage or primary disease
    mpnst_mask = (
        df["OncotreeLineage"].str.contains("Nerve", case=False, na=False)
        | df["OncotreeLineage"].str.contains("MPNST", case=False, na=False)
        | df.get("OncotreePrimaryDisease", pd.Series(dtype=str)).str.contains(
            "Nerve Sheath", case=False, na=False
        )
        | df.get("OncotreePrimaryDisease", pd.Series(dtype=str)).str.contains(
            "MPNST", case=False, na=False
        )
    )

    # Also include NF1-lost lines from relevant lineages (soft tissue sarcoma)
    sarcoma_nf1 = (
        df["OncotreeLineage"].str.contains("Soft Tissue", case=False, na=False)
        & df["NF1_loss"]
    )

    combined = df[mpnst_mask | sarcoma_nf1].copy()

    if len(combined) == 0:
        return pd.DataFrame()

    cols = [
        "OncotreeLineage", "OncotreePrimaryDisease",
        "NF1_status", "NF1_loss", "NF1_CN_log2", "NF1_expression",
        "TP53_status", "CDKN2A_status", "has_RAS_mutation",
        "NF1_TP53_comut", "has_crispr",
    ]
    cols = [c for c in cols if c in combined.columns]
    return combined[cols]


def plot_nf1_expression_validation(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot of NF1 expression by NF1 status."""
    if "NF1_expression" not in df.columns or df["NF1_expression"].isna().all():
        return

    known = df[df["NF1_status"].isin(["LOF_mutation", "deep_deletion", "both", "intact"])]

    lost_expr = known.loc[known["NF1_loss"], "NF1_expression"].dropna()
    intact_expr = known.loc[~known["NF1_loss"], "NF1_expression"].dropna()

    if len(lost_expr) == 0 or len(intact_expr) == 0:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    bp = ax.boxplot(
        [intact_expr, lost_expr],
        tick_labels=["NF1 Intact", "NF1 Lost"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")

    _, pval = stats.mannwhitneyu(lost_expr, intact_expr, alternative="less")
    ax.set_title(f"NF1 Expression by Status\np={pval:.2e}")
    ax.set_ylabel("NF1 Expression (log2 TPM+1)")

    plt.tight_layout()
    fig.savefig(output_dir / "nf1_expression_validation.png", dpi=150)
    plt.close(fig)


def plot_cancer_type_frequencies(summary: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of NF1-loss frequency by cancer type."""
    with_loss = summary[summary["n_nf1_lost"] > 0].sort_values(
        "nf1_loss_freq", ascending=True
    )
    if len(with_loss) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(with_loss) * 0.35)))
    colors = ["#E53935" if q else "#BBBBBB" for q in with_loss["qualifies"]]
    ax.barh(
        with_loss["cancer_type"],
        with_loss["nf1_loss_freq"],
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )
    ax.set_xlabel("NF1 Loss Frequency")
    ax.set_title(
        "NF1 Loss by Cancer Type\n(Red = qualifies for dependency screen)"
    )

    for i, (_, row) in enumerate(with_loss.iterrows()):
        ax.text(
            row["nf1_loss_freq"] + 0.005,
            i,
            f"{row['n_nf1_lost']}/{row['n_total']}",
            va="center",
            fontsize=7,
        )

    plt.tight_layout()
    fig.savefig(output_dir / "cancer_type_frequencies.png", dpi=150)
    plt.close(fig)


def write_summary_txt(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    qualifying: pd.DataFrame,
    validation: dict,
    mpnst_lines: pd.DataFrame,
    output_dir: Path,
) -> None:
    known = df[df["NF1_status"] != "unknown"]
    n_total = len(known)
    n_lost = int(known["NF1_loss"].sum())
    n_lof = int(known["NF1_lof_mutation"].sum())
    n_del = int(known["NF1_homdel"].fillna(False).sum())
    n_both = int((known["NF1_lof_mutation"] & known["NF1_homdel"].fillna(False)).sum())
    n_intact = n_total - n_lost
    n_crispr = int(known["has_crispr"].sum())

    lost = known[known["NF1_loss"]]
    n_tp53 = int((lost["TP53_status"] == "mutant").sum())
    n_cdkn2a = int((lost["CDKN2A_status"] == "mutant").sum())
    n_ras = int(lost["has_RAS_mutation"].sum())
    n_nf1_tp53 = int(lost["NF1_TP53_comut"].sum())

    lines = [
        "=" * 70,
        "NF1-Loss Pan-Cancer Dependency Atlas - Phase 1: Cohort Classifier",
        "=" * 70,
        "",
        "OVERVIEW",
        f"  Total cell lines with lineage + data: {n_total}",
        f"  Cell lines with CRISPR data: {n_crispr}",
        f"  NF1-lost (combined): {n_lost} ({n_lost/n_total:.1%})",
        f"    LOF mutation only: {n_lof - n_both}",
        f"    Deep deletion only: {n_del - n_both}",
        f"    Both: {n_both}",
        f"  NF1-intact: {n_intact}",
        "",
        "CO-ALTERATION LANDSCAPE (in NF1-lost lines)",
        f"  TP53 mutation: {n_tp53}/{n_lost} ({n_tp53/n_lost:.1%})" if n_lost > 0 else "  N/A",
        f"  CDKN2A mutation: {n_cdkn2a}/{n_lost} ({n_cdkn2a/n_lost:.1%})" if n_lost > 0 else "  N/A",
        f"  RAS mutation (KRAS/NRAS/HRAS): {n_ras}/{n_lost} ({n_ras/n_lost:.1%})" if n_lost > 0 else "  N/A",
        f"  NF1+TP53 co-mutation: {n_nf1_tp53}/{n_lost} ({n_nf1_tp53/n_lost:.1%})" if n_lost > 0 else "  N/A",
        "",
        "NF1 EXPRESSION CROSS-VALIDATION",
    ]

    if validation.get("valid"):
        status = "VALIDATES" if validation["validates"] else "does not validate"
        lines.append(
            f"  NF1 expression: lost={validation['median_lost']:.2f} vs "
            f"intact={validation['median_intact']:.2f}, "
            f"p={validation['mannwhitney_p']:.2e} ({status})"
        )
    else:
        lines.append(f"  {validation.get('reason', 'N/A')}")

    lines += [
        "",
        f"QUALIFYING CANCER TYPES (>={MIN_ALTERED} lost + >={MIN_WT} intact with CRISPR)",
        "-" * 70,
    ]

    if len(qualifying) == 0:
        lines.append("  None qualify with current thresholds.")
    else:
        for _, row in qualifying.iterrows():
            lines.append(
                f"  {row['cancer_type']}: {row['n_lost_crispr']} lost / "
                f"{row['n_intact_crispr']} intact (CRISPR), "
                f"{row['n_nf1_lost']}/{row['n_total']} total ({row['nf1_loss_freq']:.1%}), "
                f"TP53={row['tp53_co_mutation']}, RAS={row['ras_co_mutation']}"
            )

    lines += [
        "",
        "MPNST / PERIPHERAL NERVE SHEATH LINES",
        "-" * 70,
    ]

    if len(mpnst_lines) == 0:
        lines.append("  No MPNST or nerve sheath tumor lines found.")
        lines.append("  Note: MPNST cell lines are rare in DepMap. NF1-lost soft tissue sarcoma")
        lines.append("  lines were also checked.")
    else:
        for model_id, row in mpnst_lines.iterrows():
            crispr_flag = "CRISPR" if row.get("has_crispr", False) else "no CRISPR"
            lines.append(
                f"  {model_id}: {row.get('OncotreePrimaryDisease', 'N/A')} "
                f"({row.get('OncotreeLineage', 'N/A')}), "
                f"NF1={row.get('NF1_status', '?')}, TP53={row.get('TP53_status', '?')}, "
                f"{crispr_flag}"
            )

    lines += [
        "",
        "ALL CANCER TYPES WITH NF1 LOSS",
        "-" * 70,
    ]
    with_loss = summary[summary["n_nf1_lost"] > 0]
    for _, row in with_loss.iterrows():
        q_flag = " *" if row["qualifies"] else ""
        mpnst_flag = " [MPNST]" if row["is_mpnst_lineage"] else ""
        lines.append(
            f"  {row['cancer_type']}: {row['n_nf1_lost']} lost / {row['n_intact']} intact "
            f"(CRISPR: {row['n_lost_crispr']}/{row['n_intact_crispr']}){q_flag}{mpnst_flag}"
        )

    lines += ["", ""]

    with open(output_dir / "nf1_loss_classifier_summary.txt", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: NF1 Loss Cohort Classifier ===\n")

    # Step 1: Load all cell lines
    print("Loading all DepMap cell lines...")
    df = load_all_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with OncotreeLineage annotation")

    # Step 2: Classify NF1 status
    print("Classifying NF1 status...")
    df = classify_nf1(df, DEPMAP_DIR)

    n_lost = df["NF1_loss"].sum()
    n_lof = df["NF1_lof_mutation"].sum()
    n_del = df["NF1_homdel"].fillna(False).sum()
    n_both = (df["NF1_lof_mutation"] & df["NF1_homdel"].fillna(False)).sum()
    print(f"  NF1-lost: {n_lost} (LOF={n_lof}, deep_del={n_del}, both={n_both})")

    lost = df[df["NF1_loss"]]
    n_tp53 = (lost["TP53_status"] == "mutant").sum()
    n_ras = lost["has_RAS_mutation"].sum()
    n_nf1_tp53 = lost["NF1_TP53_comut"].sum()
    print(f"  Co-mutations: TP53={n_tp53}, RAS={n_ras}, NF1+TP53={n_nf1_tp53}")

    # Step 3: Cross-validate with NF1 expression
    print("\nCross-validating with NF1 expression...")
    validation = cross_validate_nf1_expression(df)
    if validation.get("valid"):
        status = "VALIDATES" if validation["validates"] else "does not validate"
        print(
            f"  NF1 expression: lost={validation['median_lost']:.2f} vs "
            f"intact={validation['median_intact']:.2f}, "
            f"p={validation['mannwhitney_p']:.2e} ({status})"
        )

    # Step 4: Plots
    print("\nGenerating plots...")
    plot_nf1_expression_validation(df, OUTPUT_DIR)

    # Step 5: Cancer type summary
    print("\nBuilding cancer type summary...")
    summary = build_cancer_type_summary(df)
    qualifying = summary[summary["qualifies"]].copy()
    print(f"  {len(summary)} cancer types, {len(qualifying)} qualifying")

    plot_cancer_type_frequencies(summary, OUTPUT_DIR)

    print(f"\nQualifying cancer types (>={MIN_ALTERED} lost + >={MIN_WT} intact with CRISPR):")
    if len(qualifying) == 0:
        print("  None qualify.")
    else:
        for _, row in qualifying.iterrows():
            print(
                f"  {row['cancer_type']}: {row['n_lost_crispr']} lost / "
                f"{row['n_intact_crispr']} intact (CRISPR), freq={row['nf1_loss_freq']:.1%}"
            )

    # Step 6: MPNST characterization
    print("\nCharacterizing MPNST / nerve sheath lines...")
    mpnst_lines = characterize_mpnst_lines(df)
    if len(mpnst_lines) > 0:
        print(f"  Found {len(mpnst_lines)} MPNST/nerve sheath/NF1-lost sarcoma lines")
        for model_id, row in mpnst_lines.iterrows():
            print(f"    {model_id}: {row.get('OncotreePrimaryDisease', 'N/A')}, NF1={row.get('NF1_status', '?')}")
    else:
        print("  No MPNST lines found in DepMap")

    # Step 7: Save outputs
    print("\nSaving outputs...")

    output_cols = [
        "OncotreeLineage", "OncotreePrimaryDisease",
        "NF1_status", "NF1_loss", "NF1_CN_log2", "NF1_homdel",
        "NF1_lof_mutation", "NF1_expression",
        "TP53_status", "CDKN2A_status",
        "KRAS_status", "NRAS_status", "HRAS_status",
        "has_RAS_mutation", "NF1_TP53_comut",
        "has_crispr",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()
    result.to_csv(OUTPUT_DIR / "nf1_loss_classification.csv")
    print(f"  nf1_loss_classification.csv - {len(result)} lines")

    summary.to_csv(OUTPUT_DIR / "cancer_type_summary.csv", index=False)
    print(f"  cancer_type_summary.csv - {len(summary)} types")

    qualifying.to_csv(OUTPUT_DIR / "qualifying_cancer_types.csv", index=False)
    print(f"  qualifying_cancer_types.csv - {len(qualifying)} types")

    if len(mpnst_lines) > 0:
        mpnst_lines.to_csv(OUTPUT_DIR / "mpnst_lines.csv")
        print(f"  mpnst_lines.csv - {len(mpnst_lines)} lines")

    write_summary_txt(df, summary, qualifying, validation, mpnst_lines, OUTPUT_DIR)
    print("  nf1_loss_classifier_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
