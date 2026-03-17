"""Phase 5: FSP1/AIFM2 expression layer and ferroptosis defense profile classification.

Extracts FSP1 (AIFM2) and GPX4 gene expression from DepMap OmicsExpression,
classifies cell lines by ferroptosis defense profile (FSP1-dependent,
GPX4-dependent, or dual-high), and maps FSP1 expression across KEAP1-mutant
contexts pan-cancer.

Usage:
    uv run python -m pipelines.pancancer-ferroptosis-atlas.phase5_fsp1_expression
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix, load_depmap_model_metadata

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase1"
PHASE2_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase2"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase5"

# Key genes for expression extraction
EXPRESSION_GENES = ["AIFM2", "GPX4", "SLC7A11", "GCLC", "NQO1", "FTH1", "HMOX1"]


def load_expression_data(depmap_dir: Path) -> pd.DataFrame:
    """Load ferroptosis gene expression from OmicsExpression TPM data."""
    print("Loading OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv...")
    expr = load_depmap_matrix(depmap_dir / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv")

    missing = [g for g in EXPRESSION_GENES if g not in expr.columns]
    if missing:
        print(f"  WARNING: genes not found in expression data: {missing}")

    found = [g for g in EXPRESSION_GENES if g in expr.columns]
    out = expr[found].copy()
    print(f"  Extracted expression for {len(found)} genes across {len(out)} lines")
    return out


def classify_defense_profiles(
    expr: pd.DataFrame,
    deps: pd.DataFrame,
) -> pd.DataFrame:
    """Classify cell lines by ferroptosis defense profile.

    Uses expression data for FSP1 (AIFM2) and GPX4 with median-split
    thresholds to define:
      - FSP1-low / GPX4-dependent: low FSP1 expression, respond to GPX4i alone
      - GPX4-low / FSP1-dependent: low GPX4 expression, respond to icFSP1 alone
      - Dual-high: high both, need combination FSP1i+GPX4i

    Also incorporates CRISPR dependency where available.
    """
    common = expr.index.intersection(deps.index)
    print(f"  {len(common)} lines with both expression and dependency data")

    df = pd.DataFrame(index=common)
    df["AIFM2_expression"] = expr.loc[common, "AIFM2"]
    df["GPX4_expression"] = expr.loc[common, "GPX4"]

    # Add dependency scores
    if "AIFM2" in deps.columns:
        df["AIFM2_dependency"] = deps.loc[common, "AIFM2"]
    if "GPX4" in deps.columns:
        df["GPX4_dependency"] = deps.loc[common, "GPX4"]

    # Median-split classification on expression
    fsp1_median = df["AIFM2_expression"].median()
    gpx4_median = df["GPX4_expression"].median()
    print(f"  FSP1 (AIFM2) expression median: {fsp1_median:.3f}")
    print(f"  GPX4 expression median: {gpx4_median:.3f}")

    df["FSP1_high"] = df["AIFM2_expression"] >= fsp1_median
    df["GPX4_high"] = df["GPX4_expression"] >= gpx4_median

    conditions = [
        (~df["FSP1_high"] & df["GPX4_high"]),   # FSP1-low, GPX4-high
        (df["FSP1_high"] & ~df["GPX4_high"]),    # FSP1-high, GPX4-low
        (df["FSP1_high"] & df["GPX4_high"]),     # both high
        (~df["FSP1_high"] & ~df["GPX4_high"]),   # both low
    ]
    labels = [
        "GPX4-dependent",      # respond to GPX4i alone
        "FSP1-dependent",      # respond to icFSP1 alone
        "dual-high",           # need FSP1i+GPX4i combination
        "dual-low",            # low defense — ferroptosis-sensitive
    ]
    df["defense_profile"] = np.select(conditions, labels, default="unclassified")

    # Therapeutic implication
    therapy_map = {
        "GPX4-dependent": "GPX4i monotherapy (Tier B: in vitro only)",
        "FSP1-dependent": "icFSP1 monotherapy (Tier A: in vivo validated)",
        "dual-high": "FSP1i+GPX4i combination required",
        "dual-low": "Ferroptosis-sensitive (minimal defense)",
    }
    df["therapeutic_implication"] = df["defense_profile"].map(therapy_map)

    return df


def map_fsp1_keap1_context(
    expr: pd.DataFrame,
    classification: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Map FSP1 expression across KEAP1-mutant vs WT contexts per cancer type.

    Tests whether NRF2 upregulates FSP1 in KEAP1-mutant contexts pan-cancer
    (observed in NSCLC per Nature 2025).
    """
    # Join expression + NRF2 status + cancer type
    common = expr.index.intersection(classification.index).intersection(meta.index)
    df = pd.DataFrame(index=common)
    df["AIFM2_expression"] = expr.loc[common, "AIFM2"]
    if "GPX4" in expr.columns:
        df["GPX4_expression"] = expr.loc[common, "GPX4"]
    df["nrf2_status"] = classification.loc[common, "nrf2_status"]
    df["KEAP1_mutant"] = classification.loc[common, "KEAP1_mutant"]
    df["OncotreeLineage"] = meta.loc[common, "OncotreeLineage"]

    rows = []
    for lineage, grp in df.groupby("OncotreeLineage"):
        keap1_mut = grp[grp["KEAP1_mutant"]]
        wt = grp[grp["nrf2_status"] == "WT"]

        n_keap1 = len(keap1_mut)
        n_wt = len(wt)

        row = {
            "cancer_type": lineage,
            "n_total": len(grp),
            "n_keap1_mutant": n_keap1,
            "n_wt": n_wt,
            "mean_fsp1_keap1": keap1_mut["AIFM2_expression"].mean() if n_keap1 > 0 else np.nan,
            "mean_fsp1_wt": wt["AIFM2_expression"].mean() if n_wt > 0 else np.nan,
            "mean_fsp1_all": grp["AIFM2_expression"].mean(),
        }

        # Statistical test if enough samples
        if n_keap1 >= 3 and n_wt >= 3:
            a_vals = keap1_mut["AIFM2_expression"].dropna()
            w_vals = wt["AIFM2_expression"].dropna()
            if len(a_vals) >= 3 and len(w_vals) >= 3:
                stat, pval = mannwhitneyu(a_vals, w_vals, alternative="two-sided")
                row["fsp1_keap1_vs_wt_pvalue"] = pval
                row["fsp1_diff_keap1_minus_wt"] = a_vals.mean() - w_vals.mean()
                # Effect size
                pooled_std = np.sqrt(
                    ((len(a_vals) - 1) * a_vals.std() ** 2 + (len(w_vals) - 1) * w_vals.std() ** 2)
                    / (len(a_vals) + len(w_vals) - 2)
                )
                row["cohens_d"] = (a_vals.mean() - w_vals.mean()) / pooled_std if pooled_std > 0 else 0.0
            else:
                row["fsp1_keap1_vs_wt_pvalue"] = np.nan
                row["fsp1_diff_keap1_minus_wt"] = np.nan
                row["cohens_d"] = np.nan
        else:
            row["fsp1_keap1_vs_wt_pvalue"] = np.nan
            row["fsp1_diff_keap1_minus_wt"] = np.nan
            row["cohens_d"] = np.nan

        rows.append(row)

    result = pd.DataFrame(rows)
    result = result.sort_values("mean_fsp1_all", ascending=False)
    return result


def cancer_type_defense_summary(
    profiles: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize defense profile distribution per cancer type."""
    common = profiles.index.intersection(meta.index)
    df = profiles.loc[common].copy()
    df["OncotreeLineage"] = meta.loc[common, "OncotreeLineage"]

    rows = []
    for lineage, grp in df.groupby("OncotreeLineage"):
        n = len(grp)
        counts = grp["defense_profile"].value_counts()
        row = {
            "cancer_type": lineage,
            "n_lines": n,
            "n_gpx4_dependent": counts.get("GPX4-dependent", 0),
            "n_fsp1_dependent": counts.get("FSP1-dependent", 0),
            "n_dual_high": counts.get("dual-high", 0),
            "n_dual_low": counts.get("dual-low", 0),
            "frac_gpx4_dependent": counts.get("GPX4-dependent", 0) / n,
            "frac_fsp1_dependent": counts.get("FSP1-dependent", 0) / n,
            "frac_dual_high": counts.get("dual-high", 0) / n,
            "frac_dual_low": counts.get("dual-low", 0) / n,
            "mean_fsp1_expr": grp["AIFM2_expression"].mean(),
            "mean_gpx4_expr": grp["GPX4_expression"].mean(),
        }
        # Dominant defense
        profile_fracs = {
            "GPX4-dependent": row["frac_gpx4_dependent"],
            "FSP1-dependent": row["frac_fsp1_dependent"],
            "dual-high": row["frac_dual_high"],
            "dual-low": row["frac_dual_low"],
        }
        row["dominant_profile"] = max(profile_fracs, key=profile_fracs.get)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary = summary.sort_values("frac_dual_high", ascending=False)
    return summary


def plot_defense_profiles(profiles: pd.DataFrame, meta: pd.DataFrame, out_path: Path) -> None:
    """Scatter plot of FSP1 vs GPX4 expression colored by defense profile."""
    common = profiles.index.intersection(meta.index)
    df = profiles.loc[common].copy()
    df["OncotreeLineage"] = meta.loc[common, "OncotreeLineage"]

    colors = {
        "GPX4-dependent": "#e74c3c",   # red
        "FSP1-dependent": "#3498db",   # blue
        "dual-high": "#9b59b6",        # purple
        "dual-low": "#2ecc71",         # green
    }

    fig, ax = plt.subplots(figsize=(10, 8))
    for profile, color in colors.items():
        mask = df["defense_profile"] == profile
        ax.scatter(
            df.loc[mask, "AIFM2_expression"],
            df.loc[mask, "GPX4_expression"],
            c=color, label=f"{profile} (n={mask.sum()})",
            alpha=0.4, s=15, edgecolors="none",
        )

    fsp1_med = df["AIFM2_expression"].median()
    gpx4_med = df["GPX4_expression"].median()
    ax.axvline(fsp1_med, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(gpx4_med, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_xlabel("FSP1 (AIFM2) expression (log2 TPM+1)")
    ax.set_ylabel("GPX4 expression (log2 TPM+1)")
    ax.set_title("Ferroptosis Defense Profile Classification (DepMap 25Q3)")
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter plot: {out_path}")


def plot_fsp1_keap1_boxplot(
    expr: pd.DataFrame,
    classification: pd.DataFrame,
    meta: pd.DataFrame,
    out_path: Path,
) -> None:
    """Boxplot of FSP1 expression in KEAP1-mutant vs WT for top cancer types."""
    common = expr.index.intersection(classification.index).intersection(meta.index)
    df = pd.DataFrame(index=common)
    df["AIFM2_expression"] = expr.loc[common, "AIFM2"]
    df["KEAP1_mutant"] = classification.loc[common, "KEAP1_mutant"]
    df["OncotreeLineage"] = meta.loc[common, "OncotreeLineage"]

    # Filter to cancer types with >= 3 KEAP1-mutant lines
    keap1_counts = df[df["KEAP1_mutant"]].groupby("OncotreeLineage").size()
    keep_types = keap1_counts[keap1_counts >= 3].index.tolist()

    if len(keep_types) < 2:
        print("  Too few cancer types with KEAP1-mutant lines for boxplot — skipping")
        return

    df = df[df["OncotreeLineage"].isin(keep_types)]

    # Order by mean FSP1 in KEAP1-mutant
    order = (
        df[df["KEAP1_mutant"]]
        .groupby("OncotreeLineage")["AIFM2_expression"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, max(5, len(order) * 0.6)))
    positions = list(range(len(order)))

    for i, ct in enumerate(order):
        ct_data = df[df["OncotreeLineage"] == ct]
        mut_vals = ct_data[ct_data["KEAP1_mutant"]]["AIFM2_expression"].dropna().values
        wt_vals = ct_data[~ct_data["KEAP1_mutant"]]["AIFM2_expression"].dropna().values

        bp_mut = ax.boxplot(
            [mut_vals], positions=[i - 0.15], widths=0.25,
            patch_artist=True, boxprops=dict(facecolor="#e74c3c", alpha=0.6),
            medianprops=dict(color="black"), showfliers=False,
        )
        bp_wt = ax.boxplot(
            [wt_vals], positions=[i + 0.15], widths=0.25,
            patch_artist=True, boxprops=dict(facecolor="#3498db", alpha=0.6),
            medianprops=dict(color="black"), showfliers=False,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel("FSP1 (AIFM2) expression (log2 TPM+1)")
    ax.set_title("FSP1 expression: KEAP1-mutant (red) vs WT (blue)")
    ax.invert_yaxis()

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.6, label="KEAP1-mutant"),
        Patch(facecolor="#3498db", alpha=0.6, label="WT"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved boxplot: {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 5: FSP1 expression layer and defense profiles")
    parser.add_argument("--depmap-dir", type=Path, default=DEPMAP_DIR)
    parser.add_argument("--phase1-dir", type=Path, default=PHASE1_DIR)
    parser.add_argument("--phase2-dir", type=Path, default=PHASE2_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Load expression data
    expr = load_expression_data(args.depmap_dir)

    # Load Phase 1 dependency matrix (for dependency scores)
    print("Loading Phase 1 dependency matrix...")
    deps = pd.read_csv(args.phase1_dir / "ferroptosis_dependency_matrix.csv", index_col="ModelID")

    # Load metadata
    print("Loading Model.csv...")
    meta = load_depmap_model_metadata(args.depmap_dir / "Model.csv")

    # Load Phase 2 NRF2/KEAP1 classification
    print("Loading Phase 2 NRF2/KEAP1 classification...")
    classification = pd.read_csv(args.phase2_dir / "nrf2_keap1_classification.csv", index_col="ModelID")

    # --- Step 1: Save expression matrix for ferroptosis genes ---
    expr_path = args.results_dir / "ferroptosis_expression_matrix.csv"
    # Add cancer type annotation
    common_meta = expr.index.intersection(meta.index)
    expr_out = expr.loc[common_meta].copy()
    expr_out["OncotreeLineage"] = meta.loc[common_meta, "OncotreeLineage"]
    expr_out["CellLineName"] = meta.loc[common_meta, "CellLineName"]
    expr_out.index.name = "ModelID"
    expr_out.to_csv(expr_path)
    print(f"Saved expression matrix: {expr_path} ({len(expr_out)} lines)")

    # --- Step 2: Classify defense profiles ---
    print("\nClassifying ferroptosis defense profiles...")
    profiles = classify_defense_profiles(expr, deps)
    profiles_path = args.results_dir / "defense_profile_classification.csv"
    profiles.index.name = "ModelID"
    profiles.to_csv(profiles_path)
    print(f"Saved defense profiles: {profiles_path}")
    print(f"\nDefense profile distribution:")
    print(profiles["defense_profile"].value_counts().to_string())

    # --- Step 3: Cancer type defense summary ---
    print("\nComputing cancer type defense summaries...")
    summary = cancer_type_defense_summary(profiles, meta)
    summary_path = args.results_dir / "cancer_type_defense_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved cancer type summary: {summary_path}")
    print(f"\nTop cancer types by dual-high fraction (need combination therapy):")
    top = summary[summary["n_lines"] >= 5].head(10)
    print(top[["cancer_type", "n_lines", "frac_dual_high", "frac_fsp1_dependent",
               "frac_gpx4_dependent", "dominant_profile"]].to_string(index=False))

    # --- Step 4: Map FSP1 across KEAP1-mutant contexts ---
    print("\nMapping FSP1 expression across KEAP1-mutant contexts...")
    keap1_map = map_fsp1_keap1_context(expr, classification, meta)
    keap1_path = args.results_dir / "fsp1_keap1_context_map.csv"
    keap1_map.to_csv(keap1_path, index=False)
    print(f"Saved KEAP1 context map: {keap1_path}")

    # Report on NRF2-FSP1 upregulation across cancer types
    testable = keap1_map.dropna(subset=["fsp1_keap1_vs_wt_pvalue"])
    if not testable.empty:
        print(f"\nFSP1 expression in KEAP1-mutant vs WT ({len(testable)} testable types):")
        for _, row in testable.iterrows():
            direction = "higher" if row["fsp1_diff_keap1_minus_wt"] > 0 else "lower"
            sig = "**" if row["fsp1_keap1_vs_wt_pvalue"] < 0.05 else ""
            print(f"  {row['cancer_type']}: FSP1 {direction} in KEAP1-mut "
                  f"(d={row['cohens_d']:.2f}, p={row['fsp1_keap1_vs_wt_pvalue']:.3f}){sig}")

    # --- Step 5: Plots ---
    print("\nGenerating plots...")
    scatter_path = args.results_dir / "defense_profile_scatter.png"
    plot_defense_profiles(profiles, meta, scatter_path)

    boxplot_path = args.results_dir / "fsp1_keap1_boxplot.png"
    plot_fsp1_keap1_boxplot(expr, classification, meta, boxplot_path)

    print("\nPhase 5 complete.")


if __name__ == "__main__":
    main()
