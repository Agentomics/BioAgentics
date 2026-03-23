"""Phase 1: Classify KEAP1-loss and NRF2-GOF across all DepMap 25Q3 cell lines.

Cohort definitions:
  1. KEAP1-loss: LOF mutations (LikelyLoF) OR deep deletion (CN log2 <= 0.3)
  2. NRF2-GOF (NFE2L2): hotspot gain-of-function mutations in DLG/ETGE motifs
  3. Combined: KEAP1-loss OR NRF2-GOF (primary analysis group)

Co-mutation annotations: STK11, KRAS, TP53.
Cross-validates KEAP1-loss against NRF2 target gene expression (NQO1, GCLM, TXNRD1).

Usage:
    uv run python -m keap1_nrf2_pancancer_dependency_atlas.01_keap1_nrf2_classifier
"""

from __future__ import annotations

import re
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
    REPO_ROOT / "output" / "cancer" / "keap1-nrf2-pancancer-dependency-atlas" / "phase1"
)

# CN threshold for deep deletion (log2 scale, same as other atlases)
HOMDEL_CN_THRESHOLD = 0.3

# Minimum samples per group for powered analysis
MIN_ALTERED = 3
MIN_WT = 10

# Known NFE2L2 (NRF2) gain-of-function hotspot residues in the Neh2 domain.
# DLG motif: residues 29-31; ETGE motif: residues 79-82.
# These are the canonical positions where missense mutations disrupt KEAP1 binding.
NFE2L2_GOF_POSITIONS = {
    # DLG motif and flanking
    "D29", "L30", "G31", "W24", "Q26", "D27", "I28",
    # ETGE motif and flanking
    "E79", "T80", "G81", "E82", "D77", "F78", "G83",
    # Other recurrent GOF positions
    "R34", "E78", "E82",
}

# NRF2 target genes for cross-validation (should be upregulated in altered lines)
NRF2_TARGETS = ["NQO1", "GCLM", "TXNRD1", "HMOX1", "AKR1C1"]

_POS_RE = re.compile(r"p\.([A-Z])(\d+)")


def _extract_position(protein_change: str) -> str | None:
    """Extract position label like 'D29' from 'p.D29N'."""
    if not isinstance(protein_change, str):
        return None
    m = _POS_RE.search(protein_change)
    if m:
        return f"{m.group(1)}{m.group(2)}"
    return None


def load_all_lines(depmap_dir: Path) -> pd.DataFrame:
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    meta = meta[meta["OncotreeLineage"].notna()].copy()
    return meta


def load_copy_number(depmap_dir: Path, gene: str) -> pd.Series:
    cn = load_depmap_matrix(depmap_dir / "PortalOmicsCNGeneLog2.csv")
    if gene not in cn.columns:
        raise ValueError(f"{gene} not found in CN data")
    return cn[gene]


def load_expression_genes(depmap_dir: Path, genes: list[str]) -> pd.DataFrame:
    """Load expression for multiple genes at once (memory efficient)."""
    expr = load_depmap_matrix(
        depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
    )
    available = [g for g in genes if g in expr.columns]
    return expr[available]


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


def load_nfe2l2_gof_mutations(depmap_dir: Path) -> set[str]:
    """Identify cell lines with NFE2L2 GOF hotspot mutations."""
    cols = ["ModelID", "HugoSymbol", "ProteinChange", "VariantType", "VepImpact"]
    mutations = pd.read_csv(
        depmap_dir / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    nfe2l2 = mutations[mutations["HugoSymbol"] == "NFE2L2"].copy()
    # Filter to missense / in-frame (not LOF — these are GOF)
    nfe2l2 = nfe2l2[nfe2l2["VepImpact"].isin(["HIGH", "MODERATE"])]

    gof_lines = set()
    for _, row in nfe2l2.iterrows():
        pos = _extract_position(row.get("ProteinChange", ""))
        if pos and pos in NFE2L2_GOF_POSITIONS:
            gof_lines.add(row["ModelID"])

    return gof_lines


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


def classify_keap1_nrf2(df: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Classify KEAP1/NRF2 pathway alteration status."""

    # --- KEAP1 copy number ---
    keap1_cn = load_copy_number(depmap_dir, "KEAP1").rename("KEAP1_CN_log2")
    df = df.join(keap1_cn, how="left")
    df["KEAP1_homdel"] = df["KEAP1_CN_log2"] <= HOMDEL_CN_THRESHOLD

    # --- KEAP1 LOF mutations ---
    keap1_lof_lines = load_lof_mutations(depmap_dir, "KEAP1")
    df["KEAP1_lof_mutation"] = df.index.isin(keap1_lof_lines)

    # --- KEAP1 loss (mutation OR deep deletion) ---
    df["KEAP1_loss"] = df["KEAP1_lof_mutation"] | df["KEAP1_homdel"].fillna(False)

    # --- NFE2L2 GOF hotspot mutations ---
    nfe2l2_gof_lines = load_nfe2l2_gof_mutations(depmap_dir)
    df["NFE2L2_GOF"] = df.index.isin(nfe2l2_gof_lines)

    # --- Combined KEAP1/NRF2 altered cohort ---
    df["KEAP1_NRF2_altered"] = df["KEAP1_loss"] | df["NFE2L2_GOF"]

    # Status label
    df["pathway_status"] = "WT"
    df.loc[df["KEAP1_loss"] & ~df["NFE2L2_GOF"], "pathway_status"] = "KEAP1_loss"
    df.loc[~df["KEAP1_loss"] & df["NFE2L2_GOF"], "pathway_status"] = "NFE2L2_GOF"
    df.loc[df["KEAP1_loss"] & df["NFE2L2_GOF"], "pathway_status"] = "both"

    # No data → unknown
    no_data = (
        df["KEAP1_CN_log2"].isna()
        & ~df["KEAP1_lof_mutation"]
        & ~df["NFE2L2_GOF"]
    )
    df.loc[no_data, "pathway_status"] = "unknown"

    # --- Co-mutations ---
    for gene in ["STK11", "KRAS", "TP53"]:
        mut_lines = load_any_mutations(depmap_dir, gene)
        df[f"{gene}_status"] = np.where(df.index.isin(mut_lines), "mutant", "WT")

    # KL subtype: KEAP1 + STK11 co-mutation
    df["KL_subtype"] = df["KEAP1_loss"] & (df["STK11_status"] == "mutant")

    # --- CRISPR data availability ---
    crispr_lines = load_crispr_lines(depmap_dir)
    df["has_crispr"] = df.index.isin(crispr_lines)

    return df


def cross_validate_nrf2_expression(
    df: pd.DataFrame, depmap_dir: Path
) -> dict:
    """Validate that KEAP1/NRF2-altered lines have elevated NRF2 target expression."""
    expr_df = load_expression_genes(depmap_dir, NRF2_TARGETS)

    known = df[df["pathway_status"].isin(["KEAP1_loss", "NFE2L2_GOF", "both", "WT"])]
    altered_ids = known[known["KEAP1_NRF2_altered"]].index
    wt_ids = known[~known["KEAP1_NRF2_altered"]].index

    results = {}
    for gene in expr_df.columns:
        alt_vals = expr_df.loc[expr_df.index.isin(altered_ids), gene].dropna()
        wt_vals = expr_df.loc[expr_df.index.isin(wt_ids), gene].dropna()

        if len(alt_vals) < 5 or len(wt_vals) < 5:
            results[gene] = {"valid": False, "reason": "Too few lines"}
            continue

        stat, pval = stats.mannwhitneyu(alt_vals, wt_vals, alternative="greater")
        results[gene] = {
            "valid": True,
            "n_altered": len(alt_vals),
            "n_wt": len(wt_vals),
            "median_altered": round(float(alt_vals.median()), 3),
            "median_wt": round(float(wt_vals.median()), 3),
            "mannwhitney_p": float(pval),
            "validates": pval < 0.05,
        }

    return results


def build_cancer_type_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize KEAP1/NRF2 status per cancer type with power analysis."""
    known = df[df["pathway_status"] != "unknown"]

    summary_rows = []
    for lineage, group in known.groupby("OncotreeLineage"):
        n_total = len(group)
        n_altered = int(group["KEAP1_NRF2_altered"].sum())
        n_wt = n_total - n_altered
        freq = n_altered / n_total if n_total > 0 else 0.0

        n_keap1 = int(group["KEAP1_loss"].sum())
        n_nfe2l2 = int(group["NFE2L2_GOF"].sum())

        altered_crispr = int((group["KEAP1_NRF2_altered"] & group["has_crispr"]).sum())
        wt_crispr = int((~group["KEAP1_NRF2_altered"] & group["has_crispr"]).sum())
        qualifies = altered_crispr >= MIN_ALTERED and wt_crispr >= MIN_WT

        altered_lines = group[group["KEAP1_NRF2_altered"]]
        n_stk11 = int((altered_lines["STK11_status"] == "mutant").sum()) if len(altered_lines) > 0 else 0
        n_kras = int((altered_lines["KRAS_status"] == "mutant").sum()) if len(altered_lines) > 0 else 0
        n_tp53 = int((altered_lines["TP53_status"] == "mutant").sum()) if len(altered_lines) > 0 else 0
        n_kl = int(altered_lines["KL_subtype"].sum()) if len(altered_lines) > 0 else 0

        summary_rows.append({
            "cancer_type": lineage,
            "n_total": n_total,
            "n_altered": n_altered,
            "n_wt": n_wt,
            "alteration_freq": round(freq, 4),
            "n_keap1_loss": n_keap1,
            "n_nfe2l2_gof": n_nfe2l2,
            "n_altered_crispr": altered_crispr,
            "n_wt_crispr": wt_crispr,
            "stk11_co_mutation": n_stk11,
            "kras_co_mutation": n_kras,
            "tp53_co_mutation": n_tp53,
            "kl_subtype": n_kl,
            "qualifies": qualifies,
        })

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values("alteration_freq", ascending=False).reset_index(drop=True)
    return summary


def plot_nrf2_target_validation(
    df: pd.DataFrame, depmap_dir: Path, output_dir: Path
) -> None:
    """Boxplot of NRF2 target gene expression by pathway status."""
    expr_df = load_expression_genes(depmap_dir, NRF2_TARGETS)
    available_targets = list(expr_df.columns)
    if not available_targets:
        return

    n_targets = min(len(available_targets), 5)
    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]

    known = df[df["pathway_status"].isin(["KEAP1_loss", "NFE2L2_GOF", "both", "WT"])]

    for i, gene in enumerate(available_targets[:n_targets]):
        ax = axes[i]
        alt_ids = known[known["KEAP1_NRF2_altered"]].index
        wt_ids = known[~known["KEAP1_NRF2_altered"]].index

        alt_vals = expr_df.loc[expr_df.index.isin(alt_ids), gene].dropna()
        wt_vals = expr_df.loc[expr_df.index.isin(wt_ids), gene].dropna()

        if len(alt_vals) == 0 or len(wt_vals) == 0:
            continue

        bp = ax.boxplot(
            [wt_vals, alt_vals],
            tick_labels=["WT", "Altered"],
            patch_artist=True,
            widths=0.5,
        )
        bp["boxes"][0].set_facecolor("#4CAF50")
        bp["boxes"][1].set_facecolor("#F44336")

        _, pval = stats.mannwhitneyu(alt_vals, wt_vals, alternative="greater")
        ax.set_title(f"{gene}\np={pval:.2e}")
        ax.set_ylabel("Expression (log2 TPM+1)")

    plt.suptitle("NRF2 Target Gene Expression by KEAP1/NRF2 Status", fontsize=11)
    plt.tight_layout()
    fig.savefig(output_dir / "nrf2_target_validation.png", dpi=150)
    plt.close(fig)


def plot_cancer_type_frequencies(summary: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of KEAP1/NRF2 alteration frequency by cancer type."""
    with_alt = summary[summary["n_altered"] > 0].sort_values("alteration_freq", ascending=True)
    if len(with_alt) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(with_alt) * 0.35)))
    colors = ["#E53935" if q else "#BBBBBB" for q in with_alt["qualifies"]]
    ax.barh(with_alt["cancer_type"], with_alt["alteration_freq"], color=colors, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("KEAP1/NRF2 Alteration Frequency")
    ax.set_title("KEAP1/NRF2 Pathway Alteration by Cancer Type\n(Red = qualifies for dependency screen)")

    for i, (_, row) in enumerate(with_alt.iterrows()):
        ax.text(
            row["alteration_freq"] + 0.005,
            i,
            f"{row['n_altered']}/{row['n_total']}",
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
    output_dir: Path,
) -> None:
    known = df[df["pathway_status"] != "unknown"]
    n_total = len(known)
    n_altered = int(known["KEAP1_NRF2_altered"].sum())
    n_keap1 = int(known["KEAP1_loss"].sum())
    n_nfe2l2 = int(known["NFE2L2_GOF"].sum())
    n_both = int((known["KEAP1_loss"] & known["NFE2L2_GOF"]).sum())
    n_wt = n_total - n_altered
    n_crispr = int(known["has_crispr"].sum())

    altered = known[known["KEAP1_NRF2_altered"]]
    n_stk11 = int((altered["STK11_status"] == "mutant").sum())
    n_kras = int((altered["KRAS_status"] == "mutant").sum())
    n_tp53 = int((altered["TP53_status"] == "mutant").sum())
    n_kl = int(altered["KL_subtype"].sum())

    lines = [
        "=" * 70,
        "KEAP1/NRF2 Pan-Cancer Dependency Atlas - Phase 1: Cohort Classifier",
        "=" * 70,
        "",
        "OVERVIEW",
        f"  Total cell lines with lineage + data: {n_total}",
        f"  Cell lines with CRISPR data: {n_crispr}",
        f"  KEAP1/NRF2 altered (combined): {n_altered} ({n_altered/n_total:.1%})",
        f"    KEAP1-loss only: {n_keap1 - n_both}",
        f"    NFE2L2-GOF only: {n_nfe2l2 - n_both}",
        f"    Both: {n_both}",
        f"  Wild-type: {n_wt}",
        "",
        "CO-ALTERATION LANDSCAPE (in altered lines)",
        f"  STK11 mutation: {n_stk11}/{n_altered} ({n_stk11/n_altered:.1%})" if n_altered > 0 else "  N/A",
        f"  KRAS mutation: {n_kras}/{n_altered} ({n_kras/n_altered:.1%})" if n_altered > 0 else "  N/A",
        f"  TP53 mutation: {n_tp53}/{n_altered} ({n_tp53/n_altered:.1%})" if n_altered > 0 else "  N/A",
        f"  KL subtype (KEAP1+STK11): {n_kl}/{n_altered} ({n_kl/n_altered:.1%})" if n_altered > 0 else "  N/A",
        "",
        "NRF2 TARGET EXPRESSION CROSS-VALIDATION",
    ]

    for gene, res in validation.items():
        if res.get("valid"):
            status = "VALIDATES" if res["validates"] else "does not validate"
            lines.append(
                f"  {gene}: altered={res['median_altered']:.2f} vs WT={res['median_wt']:.2f}, "
                f"p={res['mannwhitney_p']:.2e} ({status})"
            )
        else:
            lines.append(f"  {gene}: {res.get('reason', 'N/A')}")

    lines += [
        "",
        f"QUALIFYING CANCER TYPES (>={MIN_ALTERED} altered + >={MIN_WT} WT with CRISPR)",
        "-" * 70,
    ]

    if len(qualifying) == 0:
        lines.append("  None qualify with current thresholds.")
    else:
        for _, row in qualifying.iterrows():
            lines.append(
                f"  {row['cancer_type']}: {row['n_altered_crispr']} altered / "
                f"{row['n_wt_crispr']} WT (CRISPR), "
                f"{row['n_altered']}/{row['n_total']} total ({row['alteration_freq']:.1%}), "
                f"STK11={row['stk11_co_mutation']}, KRAS={row['kras_co_mutation']}"
            )

    lines += [
        "",
        "ALL CANCER TYPES WITH ALTERATIONS",
        "-" * 70,
    ]
    with_alt = summary[summary["n_altered"] > 0]
    for _, row in with_alt.iterrows():
        q_flag = " *" if row["qualifies"] else ""
        lines.append(
            f"  {row['cancer_type']}: {row['n_altered']} altered / {row['n_wt']} WT "
            f"(CRISPR: {row['n_altered_crispr']}/{row['n_wt_crispr']}){q_flag}"
        )

    lines += ["", ""]

    with open(output_dir / "keap1_nrf2_classifier_summary.txt", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: KEAP1/NRF2 Cohort Classifier ===\n")

    # Step 1: Load all cell lines
    print("Loading all DepMap cell lines...")
    df = load_all_lines(DEPMAP_DIR)
    print(f"  {len(df)} cell lines with OncotreeLineage annotation")

    # Step 2: Classify KEAP1/NRF2 status
    print("Classifying KEAP1/NRF2 pathway status...")
    df = classify_keap1_nrf2(df, DEPMAP_DIR)

    n_altered = df["KEAP1_NRF2_altered"].sum()
    n_keap1 = df["KEAP1_loss"].sum()
    n_nfe2l2 = df["NFE2L2_GOF"].sum()
    n_both = (df["KEAP1_loss"] & df["NFE2L2_GOF"]).sum()
    print(f"  KEAP1/NRF2 altered: {n_altered} (KEAP1-loss={n_keap1}, NFE2L2-GOF={n_nfe2l2}, both={n_both})")

    altered = df[df["KEAP1_NRF2_altered"]]
    n_stk11 = (altered["STK11_status"] == "mutant").sum()
    n_kras = (altered["KRAS_status"] == "mutant").sum()
    n_kl = altered["KL_subtype"].sum()
    print(f"  Co-mutations: STK11={n_stk11}, KRAS={n_kras}, KL-subtype={n_kl}")

    # Step 3: Cross-validate with NRF2 target expression
    print("\nCross-validating with NRF2 target expression...")
    validation = cross_validate_nrf2_expression(df, DEPMAP_DIR)
    for gene, res in validation.items():
        if res.get("valid"):
            status = "VALIDATES" if res["validates"] else "does not validate"
            print(f"  {gene}: altered={res['median_altered']:.2f} vs WT={res['median_wt']:.2f}, "
                  f"p={res['mannwhitney_p']:.2e} ({status})")

    # Step 4: Plots
    print("\nGenerating plots...")
    plot_nrf2_target_validation(df, DEPMAP_DIR, OUTPUT_DIR)

    # Step 5: Cancer type summary
    print("\nBuilding cancer type summary...")
    summary = build_cancer_type_summary(df)
    qualifying = summary[summary["qualifies"]].copy()
    print(f"  {len(summary)} cancer types, {len(qualifying)} qualifying")

    plot_cancer_type_frequencies(summary, OUTPUT_DIR)

    print(f"\nQualifying cancer types (>={MIN_ALTERED} altered + >={MIN_WT} WT with CRISPR):")
    if len(qualifying) == 0:
        print("  None qualify.")
    else:
        for _, row in qualifying.iterrows():
            print(
                f"  {row['cancer_type']}: {row['n_altered_crispr']} altered / "
                f"{row['n_wt_crispr']} WT (CRISPR), freq={row['alteration_freq']:.1%}"
            )

    # Step 6: Save outputs
    print("\nSaving outputs...")

    output_cols = [
        "OncotreeLineage", "OncotreePrimaryDisease",
        "pathway_status", "KEAP1_NRF2_altered",
        "KEAP1_CN_log2", "KEAP1_homdel", "KEAP1_lof_mutation", "KEAP1_loss",
        "NFE2L2_GOF",
        "STK11_status", "KRAS_status", "TP53_status", "KL_subtype",
        "has_crispr",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    result = df[output_cols].copy()
    result.to_csv(OUTPUT_DIR / "keap1_nrf2_classification.csv")
    print(f"  keap1_nrf2_classification.csv - {len(result)} lines")

    summary.to_csv(OUTPUT_DIR / "cancer_type_summary.csv", index=False)
    print(f"  cancer_type_summary.csv - {len(summary)} types")

    qualifying.to_csv(OUTPUT_DIR / "qualifying_cancer_types.csv", index=False)
    print(f"  qualifying_cancer_types.csv - {len(qualifying)} types")

    write_summary_txt(df, summary, qualifying, validation, OUTPUT_DIR)
    print("  keap1_nrf2_classifier_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
