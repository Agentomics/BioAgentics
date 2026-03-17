"""Phase 1: Classify RB1 functional loss across all DepMap 25Q3 cell lines.

Multi-modal classification:
  1. RB1 truncating/frameshift mutations (LikelyLoF from OmicsSomaticMutations.csv)
  2. RB1 deep deletion (CN <= threshold from PortalOmicsCNGeneLog2.csv)
  3. RB1 expression loss (bottom quartile from OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv)

Cross-validates against CDK4/CDK6 CRISPR dependency: RB1-loss lines should NOT depend
on CDK4/CDK6 (no RB1 to phosphorylate = positive control).

Annotates CCNE1 amplification, TP53 co-mutation, and qualifying cancer types.

Usage:
    uv run python -m rb1_loss_pancancer_dependency_atlas.01_rb1_classifier
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
OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "rb1-loss-pancancer-dependency-atlas" / "phase1"

# CN threshold for deep deletion in PortalOmicsCNGeneLog2.csv.
# Data is log2(relative CN) floored at ~0; values near 0 = deep deletion.
# Same empirical threshold used in CDKN2A atlas.
HOMDEL_CN_THRESHOLD = 0.3

# CN threshold for amplification (~4+ copies)
AMP_CN_THRESHOLD = 1.5

# Expression threshold: bottom quartile percentile for expression-based loss
EXPR_LOSS_PERCENTILE = 25

# Minimum samples per group for powered analysis
MIN_LOSS = 5
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
    hit = mutations[
        (mutations["HugoSymbol"] == gene) & (mutations["VepImpact"] != "LOW")
    ]
    return set(hit["ModelID"].unique())


def load_crispr_lines(depmap_dir: Path) -> set[str]:
    """Return set of ModelIDs that have CRISPR data."""
    crispr = pd.read_csv(depmap_dir / "CRISPRGeneEffect.csv", usecols=[0])
    return set(crispr.iloc[:, 0])


def load_crispr_gene(depmap_dir: Path, gene: str) -> pd.Series:
    """Load CRISPR dependency scores for a single gene."""
    crispr = load_depmap_matrix(depmap_dir / "CRISPRGeneEffect.csv")
    if gene not in crispr.columns:
        raise ValueError(f"{gene} not found in CRISPR data")
    return crispr[gene]


def classify_rb1(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Classify RB1 functional status using multi-modal approach."""

    # --- RB1 copy number ---
    rb1_cn = load_copy_number(depmap_dir, "RB1").rename("RB1_CN_log2")
    df = df.join(rb1_cn, how="left")
    df["RB1_homdel"] = df["RB1_CN_log2"] <= HOMDEL_CN_THRESHOLD

    # --- RB1 LOF mutations ---
    rb1_lof_lines = load_lof_mutations(depmap_dir, "RB1")
    df["RB1_lof_mutation"] = df.index.isin(rb1_lof_lines)

    # --- RB1 expression ---
    rb1_expr = load_expression(depmap_dir, "RB1").rename("RB1_expression")
    df = df.join(rb1_expr, how="left")

    # Bottom quartile of RB1 expression = expression loss
    expr_threshold = df["RB1_expression"].quantile(EXPR_LOSS_PERCENTILE / 100)
    df["RB1_expr_low"] = df["RB1_expression"] <= expr_threshold
    df["RB1_expr_threshold"] = expr_threshold

    # --- Multi-modal RB1 status ---
    # RB1-loss: LOF mutation OR deep deletion OR expression loss
    df["RB1_loss_mutation"] = df["RB1_lof_mutation"]
    df["RB1_loss_cn"] = df["RB1_homdel"].fillna(False)
    df["RB1_loss_expression"] = df["RB1_expr_low"].fillna(False)

    df["RB1_loss"] = (
        df["RB1_loss_mutation"]
        | df["RB1_loss_cn"]
        | df["RB1_loss_expression"]
    )

    # Count evidence modalities
    df["RB1_evidence_count"] = (
        df["RB1_loss_mutation"].astype(int)
        + df["RB1_loss_cn"].astype(int)
        + df["RB1_loss_expression"].astype(int)
    )

    # Status label
    df["RB1_status"] = "intact"
    df.loc[df["RB1_loss"], "RB1_status"] = "lost"
    # Lines with no CN and no expression data → unknown
    no_data = df["RB1_CN_log2"].isna() & df["RB1_expression"].isna() & ~df["RB1_lof_mutation"]
    df.loc[no_data, "RB1_status"] = "unknown"

    # Evidence source label
    sources = []
    for idx in df.index:
        s = []
        if df.loc[idx, "RB1_loss_mutation"]:
            s.append("mutation")
        if df.loc[idx, "RB1_loss_cn"]:
            s.append("cn_deletion")
        if df.loc[idx, "RB1_loss_expression"]:
            s.append("expr_loss")
        sources.append(";".join(s) if s else "none")
    df["RB1_evidence_sources"] = sources

    # --- TP53 co-mutation ---
    tp53_mut_lines = load_any_mutations(depmap_dir, "TP53")
    df["TP53_status"] = np.where(df.index.isin(tp53_mut_lines), "mutant", "WT")

    # --- CCNE1 amplification (CDK2 dependency intensifier) ---
    ccne1_cn = load_copy_number(depmap_dir, "CCNE1").rename("CCNE1_CN_log2")
    df = df.join(ccne1_cn, how="left")
    df["CCNE1_amplified"] = df["CCNE1_CN_log2"] >= AMP_CN_THRESHOLD

    # --- MYC amplification ---
    myc_cn = load_copy_number(depmap_dir, "MYC").rename("MYC_CN_log2")
    df = df.join(myc_cn, how="left")
    df["MYC_amplified"] = df["MYC_CN_log2"] >= AMP_CN_THRESHOLD

    # --- CRISPR data availability ---
    crispr_lines = load_crispr_lines(depmap_dir)
    df["has_crispr"] = df.index.isin(crispr_lines)

    return df


def cross_validate_expression(df: pd.DataFrame) -> dict:
    """Validate that RB1-loss lines have significantly lower RB1 expression."""
    known = df[df["RB1_status"].isin(["lost", "intact"])].dropna(subset=["RB1_expression"])
    lost = known[known["RB1_status"] == "lost"]["RB1_expression"]
    intact = known[known["RB1_status"] == "intact"]["RB1_expression"]

    if len(lost) == 0 or len(intact) == 0:
        return {"valid": False, "reason": "No lines in one group"}

    stat, pval = stats.mannwhitneyu(lost, intact, alternative="less")
    return {
        "valid": True,
        "n_lost": len(lost),
        "n_intact": len(intact),
        "median_expr_lost": float(lost.median()),
        "median_expr_intact": float(intact.median()),
        "mannwhitney_p": float(pval),
    }


def cross_validate_cdk46_dependency(df: pd.DataFrame, depmap_dir: Path) -> dict:
    """Validate RB1 classifier: RB1-loss lines should NOT depend on CDK4/CDK6.

    Positive control: CDK4/6 knockout is lethal only when RB1 is present
    (CDK4/6 phosphorylates RB1 to release E2F). With RB1 lost, CDK4/6 has no
    substrate, so dependency should be abolished.
    """
    results = {}
    for gene in ["CDK4", "CDK6"]:
        try:
            dep = load_crispr_gene(depmap_dir, gene)
        except ValueError:
            results[gene] = {"valid": False, "reason": f"{gene} not in CRISPR data"}
            continue

        merged = df[["RB1_status", "has_crispr"]].join(dep.rename(f"{gene}_dep"), how="inner")
        merged = merged[merged["RB1_status"].isin(["lost", "intact"])].dropna(subset=[f"{gene}_dep"])

        lost = merged[merged["RB1_status"] == "lost"][f"{gene}_dep"]
        intact = merged[merged["RB1_status"] == "intact"][f"{gene}_dep"]

        if len(lost) < 5 or len(intact) < 5:
            results[gene] = {"valid": False, "reason": "Too few lines"}
            continue

        # RB1-loss should have LESS negative (= less dependent) scores
        stat, pval = stats.mannwhitneyu(lost, intact, alternative="greater")
        d = cohens_d(lost, intact)
        results[gene] = {
            "valid": True,
            "n_lost": len(lost),
            "n_intact": len(intact),
            "median_dep_lost": float(lost.median()),
            "median_dep_intact": float(intact.median()),
            "mannwhitney_p": float(pval),
            "cohens_d": float(d),
            "validates": pval < 0.05 and d > 0,  # lost lines are less dependent
        }
    return results


def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def plot_expression_validation(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot of RB1 expression by RB1 loss status."""
    known = df[df["RB1_status"].isin(["lost", "intact"])].dropna(subset=["RB1_expression"])
    lost = known[known["RB1_status"] == "lost"]["RB1_expression"]
    intact = known[known["RB1_status"] == "intact"]["RB1_expression"]

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot(
        [intact, lost],
        tick_labels=["RB1-intact", "RB1-loss"],
        patch_artist=True,
        widths=0.5,
    )
    bp["boxes"][0].set_facecolor("#4CAF50")
    bp["boxes"][1].set_facecolor("#F44336")

    _, pval = stats.mannwhitneyu(lost, intact, alternative="less")
    ax.set_title(f"RB1 Expression by Functional Status\n(Mann-Whitney p={pval:.2e})")
    ax.set_ylabel("RB1 Expression (log2 TPM+1)")
    ax.set_xlabel(f"Intact (n={len(intact)})    Loss (n={len(lost)})")
    plt.tight_layout()
    fig.savefig(output_dir / "rb1_expression_validation.png", dpi=150)
    plt.close(fig)


def plot_classification_modalities(df: pd.DataFrame, output_dir: Path) -> None:
    """Venn-style bar chart showing overlap of RB1-loss evidence modalities."""
    lost = df[df["RB1_status"] == "lost"]
    n_mut = lost["RB1_loss_mutation"].sum()
    n_cn = lost["RB1_loss_cn"].sum()
    n_expr = lost["RB1_loss_expression"].sum()

    # Overlaps
    n_mut_cn = (lost["RB1_loss_mutation"] & lost["RB1_loss_cn"]).sum()
    n_mut_expr = (lost["RB1_loss_mutation"] & lost["RB1_loss_expression"]).sum()
    n_cn_expr = (lost["RB1_loss_cn"] & lost["RB1_loss_expression"]).sum()
    n_all_three = (lost["RB1_loss_mutation"] & lost["RB1_loss_cn"] & lost["RB1_loss_expression"]).sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: modality counts
    labels = ["LOF\nMutation", "Deep\nDeletion", "Expression\nLoss"]
    counts = [n_mut, n_cn, n_expr]
    colors = ["#E53935", "#1E88E5", "#FDD835"]
    axes[0].bar(labels, counts, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Number of cell lines")
    axes[0].set_title("RB1-Loss Evidence by Modality")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 1, str(int(v)), ha="center", fontweight="bold")

    # Right: evidence count distribution
    ev_counts = lost["RB1_evidence_count"].value_counts().sort_index()
    axes[1].bar(ev_counts.index, ev_counts.values, color="#7E57C2", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Number of concordant evidence modalities")
    axes[1].set_ylabel("Number of cell lines")
    axes[1].set_title("Multi-Modal Evidence Concordance")
    axes[1].set_xticks([1, 2, 3])
    for i, (k, v) in enumerate(ev_counts.items()):
        axes[1].text(k, v + 1, str(int(v)), ha="center", fontweight="bold")

    plt.suptitle(
        f"RB1-Loss Classification: {len(lost)} lines total\n"
        f"Overlaps: mut+CN={int(n_mut_cn)}, mut+expr={int(n_mut_expr)}, "
        f"CN+expr={int(n_cn_expr)}, all three={int(n_all_three)}",
        fontsize=10,
    )
    plt.tight_layout()
    fig.savefig(output_dir / "rb1_classification_modalities.png", dpi=150)
    plt.close(fig)


def plot_cdk46_validation(df: pd.DataFrame, cdk46_results: dict, depmap_dir: Path, output_dir: Path) -> None:
    """Boxplot of CDK4/CDK6 dependency by RB1 status (positive control)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for i, gene in enumerate(["CDK4", "CDK6"]):
        ax = axes[i]
        res = cdk46_results.get(gene, {})
        if not res.get("valid"):
            ax.text(0.5, 0.5, f"{gene}: {res.get('reason', 'N/A')}", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(f"{gene} Dependency")
            continue

        try:
            dep = load_crispr_gene(depmap_dir, gene)
        except ValueError:
            continue

        merged = df[["RB1_status"]].join(dep.rename("dep"), how="inner")
        merged = merged[merged["RB1_status"].isin(["lost", "intact"])].dropna(subset=["dep"])

        lost_vals = merged[merged["RB1_status"] == "lost"]["dep"]
        intact_vals = merged[merged["RB1_status"] == "intact"]["dep"]

        bp = ax.boxplot(
            [intact_vals, lost_vals],
            tick_labels=["RB1-intact", "RB1-loss"],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#F44336")

        p = res["mannwhitney_p"]
        d = res["cohens_d"]
        validates = res["validates"]
        status = "VALIDATES" if validates else "DOES NOT VALIDATE"
        ax.set_title(f"{gene} CRISPR Dependency\np={p:.2e}, d={d:.2f} ({status})")
        ax.set_ylabel("CRISPR Gene Effect (more negative = more dependent)")
        ax.axhline(y=-0.5, color="gray", linestyle="--", alpha=0.5, label="dependency threshold")

    plt.suptitle("CDK4/6 Dependency by RB1 Status (Positive Control)", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "rb1_cdk46_validation.png", dpi=150)
    plt.close(fig)


def build_cancer_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize RB1 status per cancer type with power analysis."""
    known = df[df["RB1_status"].isin(["lost", "intact"])]

    summary_rows = []
    for lineage, group in known.groupby("OncotreeLineage"):
        n_total = len(group)
        n_lost = int((group["RB1_status"] == "lost").sum())
        n_intact = n_total - n_lost
        loss_freq = n_lost / n_total if n_total > 0 else 0.0

        lost_crispr = int(((group["RB1_status"] == "lost") & group["has_crispr"]).sum())
        intact_crispr = int(((group["RB1_status"] == "intact") & group["has_crispr"]).sum())
        qualifies = lost_crispr >= MIN_LOSS and intact_crispr >= MIN_INTACT

        lost_lines = group[group["RB1_status"] == "lost"]
        n_lost_total = len(lost_lines)
        tp53_co_mut = int(lost_lines["TP53_status"].eq("mutant").sum()) if n_lost_total > 0 else 0
        ccne1_amp = int(lost_lines["CCNE1_amplified"].fillna(False).sum()) if n_lost_total > 0 else 0
        myc_amp = int(lost_lines["MYC_amplified"].fillna(False).sum()) if n_lost_total > 0 else 0

        # Evidence modality breakdown in lost lines
        n_by_mutation = int(lost_lines["RB1_loss_mutation"].sum()) if n_lost_total > 0 else 0
        n_by_cn = int(lost_lines["RB1_loss_cn"].sum()) if n_lost_total > 0 else 0
        n_by_expr = int(lost_lines["RB1_loss_expression"].sum()) if n_lost_total > 0 else 0
        median_evidence = float(lost_lines["RB1_evidence_count"].median()) if n_lost_total > 0 else 0

        summary_rows.append({
            "cancer_type": lineage,
            "n_total": n_total,
            "n_lost": n_lost,
            "n_intact": n_intact,
            "loss_freq": round(loss_freq, 4),
            "n_lost_crispr": lost_crispr,
            "n_intact_crispr": intact_crispr,
            "n_by_mutation": n_by_mutation,
            "n_by_cn": n_by_cn,
            "n_by_expr": n_by_expr,
            "median_evidence_modalities": median_evidence,
            "tp53_co_mutation": tp53_co_mut,
            "ccne1_amplification": ccne1_amp,
            "myc_amplification": myc_amp,
            "qualifies": qualifies,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("loss_freq", ascending=False).reset_index(drop=True)
    return summary


def write_summary_txt(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    qualifying: pd.DataFrame,
    validation: dict,
    cdk46_results: dict,
    output_dir: Path,
) -> None:
    """Write human-readable summary text file."""
    known = df[df["RB1_status"].isin(["lost", "intact"])]
    n_total = len(known)
    n_lost = int((known["RB1_status"] == "lost").sum())
    n_intact = n_total - n_lost
    n_crispr = int(known["has_crispr"].sum())

    lost_lines = known[known["RB1_status"] == "lost"]
    n_by_mut = int(lost_lines["RB1_loss_mutation"].sum())
    n_by_cn = int(lost_lines["RB1_loss_cn"].sum())
    n_by_expr = int(lost_lines["RB1_loss_expression"].sum())
    n_multi = int((lost_lines["RB1_evidence_count"] >= 2).sum())

    n_tp53 = int(lost_lines["TP53_status"].eq("mutant").sum())
    n_ccne1 = int(lost_lines["CCNE1_amplified"].fillna(False).sum())
    n_myc = int(lost_lines["MYC_amplified"].fillna(False).sum())

    lines = [
        "=" * 65,
        "RB1-Loss Pan-Cancer Dependency Atlas - Phase 1: RB1 Classifier",
        "=" * 65,
        "",
        "OVERVIEW",
        f"  Total cell lines with lineage + data: {n_total}",
        f"  Cell lines with CRISPR data: {n_crispr}",
        f"  RB1-loss (multi-modal): {n_lost} ({n_lost/n_total:.1%})",
        f"  RB1-intact: {n_intact}",
        "",
        "CLASSIFICATION MODALITIES (in RB1-loss lines)",
        f"  LOF mutations: {n_by_mut}",
        f"  Deep deletion (CN <= {HOMDEL_CN_THRESHOLD}): {n_by_cn}",
        f"  Expression loss (bottom {EXPR_LOSS_PERCENTILE}th percentile): {n_by_expr}",
        f"  Multi-modal concordance (>=2 modalities): {n_multi}",
        "",
        "EXPRESSION CROSS-VALIDATION",
    ]

    if validation.get("valid"):
        lines += [
            f"  RB1-loss median expression: {validation['median_expr_lost']:.2f}",
            f"  RB1-intact median expression: {validation['median_expr_intact']:.2f}",
            f"  Mann-Whitney U p-value: {validation['mannwhitney_p']:.2e}",
        ]
    else:
        lines.append(f"  Skipped: {validation.get('reason', 'unknown')}")

    lines += [
        "",
        "CDK4/6 DEPENDENCY VALIDATION (positive control)",
        "  RB1-loss lines should NOT depend on CDK4/CDK6 (no target to phosphorylate).",
    ]
    for gene in ["CDK4", "CDK6"]:
        res = cdk46_results.get(gene, {})
        if res.get("valid"):
            status = "VALIDATES" if res["validates"] else "DOES NOT VALIDATE"
            lines.append(
                f"  {gene}: median dep lost={res['median_dep_lost']:.3f} vs "
                f"intact={res['median_dep_intact']:.3f}, "
                f"p={res['mannwhitney_p']:.2e}, d={res['cohens_d']:.2f} ({status})"
            )
        else:
            lines.append(f"  {gene}: {res.get('reason', 'N/A')}")

    lines += [
        "",
        "CO-ALTERATION LANDSCAPE (in RB1-loss lines)",
        f"  TP53 mutation: {n_tp53}/{n_lost} ({n_tp53/n_lost:.1%})" if n_lost > 0 else "  N/A",
        f"  CCNE1 amplification: {n_ccne1}/{n_lost} ({n_ccne1/n_lost:.1%})" if n_lost > 0 else "  N/A",
        f"  MYC amplification: {n_myc}/{n_lost} ({n_myc/n_lost:.1%})" if n_lost > 0 else "  N/A",
        "",
        "CLINICAL NOTE: RB1+TP53 co-loss defines SCLC. CCNE1 amplification",
        "  intensifies CDK2 dependency. MYC amplification drives proliferation.",
        "",
        f"QUALIFYING CANCER TYPES (>={MIN_LOSS} RB1-loss + >={MIN_INTACT} intact with CRISPR)",
        "-" * 65,
    ]

    if len(qualifying) == 0:
        lines.append("  None qualify with current thresholds.")
    else:
        for _, row in qualifying.iterrows():
            lines.append(
                f"  {row['cancer_type']}: {row['n_lost_crispr']} lost / "
                f"{row['n_intact_crispr']} intact (CRISPR), "
                f"{row['n_lost']}/{row['n_total']} total ({row['loss_freq']:.1%}), "
                f"TP53={row['tp53_co_mutation']}, CCNE1={row['ccne1_amplification']}"
            )

    lines += [
        "",
        "ALL CANCER TYPES WITH RB1 LOSS",
        "-" * 65,
    ]
    with_loss = summary[summary["n_lost"] > 0]
    for _, row in with_loss.iterrows():
        q_flag = " *" if row["qualifies"] else ""
        lines.append(
            f"  {row['cancer_type']}: {row['n_lost']} lost / {row['n_intact']} intact "
            f"(CRISPR: {row['n_lost_crispr']}/{row['n_intact_crispr']}){q_flag}"
        )

    lines += ["", ""]

    with open(output_dir / "rb1_classifier_summary.txt", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: RB1-Loss Cell Line Classifier ===\n")

    # Step 1: Load all cell lines
    print("Loading all DepMap cell lines...")
    df = load_all_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with OncotreeLineage annotation")

    # Step 2: Classify RB1 status (multi-modal)
    print("Classifying RB1 status (multi-modal: mutation + CN + expression)...")
    df = classify_rb1(df, DEPMAP_DIR)

    n_lost = (df["RB1_status"] == "lost").sum()
    n_intact = (df["RB1_status"] == "intact").sum()
    n_unknown = (df["RB1_status"] == "unknown").sum()
    print(f"  {n_lost} RB1-loss, {n_intact} intact, {n_unknown} unknown")

    lost_lines = df[df["RB1_status"] == "lost"]
    n_mut = lost_lines["RB1_loss_mutation"].sum()
    n_cn = lost_lines["RB1_loss_cn"].sum()
    n_expr = lost_lines["RB1_loss_expression"].sum()
    n_multi = (lost_lines["RB1_evidence_count"] >= 2).sum()
    print(f"  Evidence: mutation={n_mut}, CN_del={n_cn}, expr_loss={n_expr}, multi-modal={n_multi}")

    n_tp53 = lost_lines["TP53_status"].eq("mutant").sum()
    n_ccne1 = lost_lines["CCNE1_amplified"].fillna(False).sum()
    n_myc = lost_lines["MYC_amplified"].fillna(False).sum()
    print(f"  Co-alterations: TP53-mut={n_tp53}, CCNE1-amp={n_ccne1}, MYC-amp={n_myc}")

    # Step 3: Cross-validate with expression
    print("\nCross-validating with RB1 expression...")
    validation = cross_validate_expression(df)
    if validation.get("valid"):
        print(f"  Median expression: lost={validation['median_expr_lost']:.2f} "
              f"vs intact={validation['median_expr_intact']:.2f}")
        print(f"  Mann-Whitney p={validation['mannwhitney_p']:.2e}")

    # Step 4: CDK4/6 dependency validation (positive control)
    print("\nValidating with CDK4/CDK6 CRISPR dependency (positive control)...")
    cdk46_results = cross_validate_cdk46_dependency(df, DEPMAP_DIR)
    for gene in ["CDK4", "CDK6"]:
        res = cdk46_results.get(gene, {})
        if res.get("valid"):
            status = "VALIDATES" if res["validates"] else "DOES NOT VALIDATE"
            print(f"  {gene}: lost={res['median_dep_lost']:.3f} vs intact={res['median_dep_intact']:.3f}, "
                  f"p={res['mannwhitney_p']:.2e}, d={res['cohens_d']:.2f} ({status})")

    # Step 5: Plots
    print("\nGenerating plots...")
    plot_expression_validation(df, OUTPUT_DIR)
    plot_classification_modalities(df, OUTPUT_DIR)
    plot_cdk46_validation(df, cdk46_results, DEPMAP_DIR, OUTPUT_DIR)

    # Step 6: Cancer type summary
    print("\nBuilding cancer type summary...")
    summary = build_cancer_type_summary(df)
    qualifying = summary[summary["qualifies"]].copy()
    n_qualifying = len(qualifying)
    print(f"  {len(summary)} cancer types, {n_qualifying} qualifying")

    print(f"\nQualifying cancer types (>={MIN_LOSS} lost + >={MIN_INTACT} intact with CRISPR):")
    if n_qualifying == 0:
        print("  None qualify.")
    else:
        for _, row in qualifying.iterrows():
            print(
                f"  {row['cancer_type']}: {row['n_lost_crispr']} lost / "
                f"{row['n_intact_crispr']} intact (CRISPR), "
                f"freq={row['loss_freq']:.1%}"
            )

    # Step 7: Save outputs
    print("\nSaving outputs...")

    output_cols = [
        "OncotreeLineage", "RB1_status", "RB1_CN_log2", "RB1_expression",
        "RB1_lof_mutation", "RB1_homdel", "RB1_expr_low",
        "RB1_loss_mutation", "RB1_loss_cn", "RB1_loss_expression",
        "RB1_evidence_count", "RB1_evidence_sources",
        "TP53_status", "CCNE1_CN_log2", "CCNE1_amplified",
        "MYC_CN_log2", "MYC_amplified", "has_crispr",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()
    result.to_csv(OUTPUT_DIR / "rb1_classification.csv")
    print(f"  rb1_classification.csv - {len(result)} lines")

    summary.to_csv(OUTPUT_DIR / "cancer_type_summary.csv", index=False)
    print(f"  cancer_type_summary.csv - {len(summary)} types")

    qualifying.to_csv(OUTPUT_DIR / "qualifying_cancer_types.csv", index=False)
    print(f"  qualifying_cancer_types.csv - {n_qualifying} types")

    write_summary_txt(df, summary, qualifying, validation, cdk46_results, OUTPUT_DIR)
    print("  rb1_classifier_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
