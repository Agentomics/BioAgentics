"""Inavolisib (GDC-0077) PRISM sensitivity analysis by PIK3CA allele.

Cross-references PRISM 24Q2 inavolisib sensitivity with PIK3CA mutation
status to test whether H1047R vs helical alleles show differential drug
sensitivity to this PI3Kα-selective inhibitor.

Usage:
    uv run python -m pik3ca_allele_dependencies.04_prism_inavolisib
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "pik3ca_allele_dependencies"
FIG_DIR = OUTPUT_DIR / "figures"

# GDC-0077 (inavolisib) PRISM identifier
INAVOLISIB_BRD = "BRD:BRD-K00003420-001-01-9"

KINASE_ALLELES = {"H1047R", "H1047L"}
HELICAL_ALLELES = {"E545K", "E542K"}


def load_prism_drug(prism_path: Path, drug_id: str) -> pd.Series:
    """Load PRISM sensitivity for a single drug across all cell lines."""
    prism = pd.read_csv(prism_path, index_col=0)
    if drug_id not in prism.index:
        raise ValueError(f"Drug {drug_id} not found in PRISM data")
    return prism.loc[drug_id].astype(float)


def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled)


def compare_groups(
    sensitivity: pd.Series, group1_ids: list[str], group2_ids: list[str],
    label1: str, label2: str,
) -> dict:
    """Compare drug sensitivity between two groups."""
    g1 = sensitivity.reindex(group1_ids).dropna()
    g2 = sensitivity.reindex(group2_ids).dropna()

    result = {
        "comparison": f"{label1} vs {label2}",
        "n_group1": len(g1),
        "n_group2": len(g2),
        "label1": label1,
        "label2": label2,
    }

    if len(g1) < 3 or len(g2) < 3:
        result["underpowered"] = True
        return result

    g1_arr, g2_arr = g1.values, g2.values
    stat, pval = stats.mannwhitneyu(g1_arr, g2_arr, alternative="two-sided")
    d = cohens_d(g1_arr, g2_arr)

    result.update({
        "underpowered": False,
        "median_group1": float(np.median(g1_arr)),
        "median_group2": float(np.median(g2_arr)),
        "mean_group1": float(np.mean(g1_arr)),
        "mean_group2": float(np.mean(g2_arr)),
        "cohens_d": round(d, 4),
        "mannwhitney_U": float(stat),
        "mannwhitney_p": float(pval),
    })
    return result


def plot_sensitivity_by_allele(
    sensitivity: pd.Series, classified: pd.DataFrame, out_path: Path,
) -> None:
    """Box/strip plot of inavolisib sensitivity by PIK3CA allele."""
    # Merge sensitivity with classification
    merged = classified.join(sensitivity.rename("inavolisib_sensitivity"), how="inner")
    merged = merged[merged["inavolisib_sensitivity"].notna()].copy()

    # Order alleles
    allele_order = ["WT", "H1047R", "H1047L", "E545K", "E542K",
                    "C420R", "N345K", "other_activating", "other"]
    present_alleles = [a for a in allele_order if a in merged["PIK3CA_allele"].values]

    fig, ax = plt.subplots(figsize=(10, 6))

    positions = []
    data_groups = []
    colors_list = []
    color_map = {
        "WT": "#808080", "H1047R": "#D95319", "H1047L": "#D95319",
        "E545K": "#0072BD", "E542K": "#0072BD",
        "C420R": "#77AC30", "N345K": "#77AC30",
        "other_activating": "#EDB120", "other": "#A2142F",
    }

    for i, allele in enumerate(present_alleles):
        vals = merged[merged["PIK3CA_allele"] == allele]["inavolisib_sensitivity"].values
        if len(vals) > 0:
            positions.append(i)
            data_groups.append(vals)
            colors_list.append(color_map.get(allele, "gray"))

    bp = ax.boxplot(data_groups, positions=positions, patch_artist=True, widths=0.6)
    for patch, color in zip(bp["boxes"], colors_list):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    # Add strip plot
    for pos, vals, color in zip(positions, data_groups, colors_list):
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(pos + jitter, vals, c=color, alpha=0.7, s=20, edgecolors="k",
                   linewidths=0.5, zorder=3)

    ax.set_xticks(positions)
    allele_labels = [f"{a}\n(n={len(d)})" for a, d in zip(
        [present_alleles[p] for p in range(len(positions))], data_groups
    )]
    ax.set_xticklabels(allele_labels, fontsize=9)
    ax.set_ylabel("PRISM Sensitivity (more negative = more sensitive)")
    ax.set_title("Inavolisib (GDC-0077) Sensitivity by PIK3CA Allele")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_by_status(
    sensitivity: pd.Series, classified: pd.DataFrame, out_path: Path,
) -> None:
    """Box plot: PIK3CA mutant vs WT inavolisib sensitivity."""
    merged = classified.join(sensitivity.rename("inavolisib_sensitivity"), how="inner")
    merged = merged[merged["inavolisib_sensitivity"].notna()].copy()

    mut_vals = merged[merged["PIK3CA_mutated"]]["inavolisib_sensitivity"].values
    wt_vals = merged[~merged["PIK3CA_mutated"]]["inavolisib_sensitivity"].values

    fig, ax = plt.subplots(figsize=(6, 5))
    bp = ax.boxplot([wt_vals, mut_vals], tick_labels=[
        f"WT\n(n={len(wt_vals)})", f"PIK3CA-mutant\n(n={len(mut_vals)})"
    ], patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#808080")
    bp["boxes"][1].set_facecolor("#D95319")
    for b in bp["boxes"]:
        b.set_alpha(0.5)

    # Strip
    for i, (vals, color) in enumerate([(wt_vals, "#808080"), (mut_vals, "#D95319")]):
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(vals))
        ax.scatter(i + 1 + jitter, vals, c=color, alpha=0.5, s=15, edgecolors="k",
                   linewidths=0.3, zorder=3)

    if len(mut_vals) >= 3 and len(wt_vals) >= 3:
        _, p = stats.mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")
        d = cohens_d(mut_vals, wt_vals)
        ax.set_title(f"Inavolisib Sensitivity: PIK3CA Mutant vs WT\n"
                     f"(p={p:.4f}, d={d:.2f})")
    else:
        ax.set_title("Inavolisib Sensitivity: PIK3CA Mutant vs WT")

    ax.set_ylabel("PRISM Sensitivity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading classified cell lines...")
    classified = pd.read_csv(OUTPUT_DIR / "pik3ca_classified_lines.csv", index_col=0)
    classified["PIK3CA_mutated"] = classified["PIK3CA_mutated"].astype(bool)

    print("Loading PRISM inavolisib sensitivity...")
    sensitivity = load_prism_drug(
        DEPMAP_DIR / "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
        INAVOLISIB_BRD,
    )
    print(f"  {sensitivity.notna().sum()} cell lines with inavolisib data")

    # Overlap with classified lines
    common = set(classified.index) & set(sensitivity.dropna().index)
    print(f"  {len(common)} lines with both classification and PRISM data")

    classified_common = classified.loc[classified.index.isin(common)]
    n_mut = classified_common["PIK3CA_mutated"].sum()
    print(f"  {n_mut} PIK3CA-mutant, {len(common) - n_mut} WT")

    results = {"drug": "inavolisib (GDC-0077)", "dose_uM": 2.5, "comparisons": []}

    # 1. Mutant vs WT
    print("\n--- PIK3CA Mutant vs WT ---")
    mut_ids = list(classified_common[classified_common["PIK3CA_mutated"]].index)
    wt_ids = list(classified_common[~classified_common["PIK3CA_mutated"]].index)
    comp = compare_groups(sensitivity, mut_ids, wt_ids, "PIK3CA_mutant", "WT")
    results["comparisons"].append(comp)
    if not comp.get("underpowered"):
        print(f"  Mutant median={comp['median_group1']:.3f} vs WT={comp['median_group2']:.3f}")
        print(f"  d={comp['cohens_d']:.3f}, p={comp['mannwhitney_p']:.4f}")

    # 2. Kinase vs Helical
    print("\n--- Kinase vs Helical domain ---")
    kinase_ids = list(
        classified_common[classified_common["PIK3CA_allele"].isin(KINASE_ALLELES)].index
    )
    helical_ids = list(
        classified_common[classified_common["PIK3CA_allele"].isin(HELICAL_ALLELES)].index
    )
    comp = compare_groups(sensitivity, kinase_ids, helical_ids, "kinase_domain", "helical_domain")
    results["comparisons"].append(comp)
    if not comp.get("underpowered"):
        print(f"  Kinase median={comp['median_group1']:.3f} vs Helical={comp['median_group2']:.3f}")
        print(f"  d={comp['cohens_d']:.3f}, p={comp['mannwhitney_p']:.4f}")
    else:
        print(f"  Underpowered: kinase={len(kinase_ids)}, helical={len(helical_ids)}")

    # 3. Per-allele vs WT
    print("\n--- Per-allele vs WT ---")
    for allele in ["H1047R", "H1047L", "E545K", "E542K", "C420R"]:
        allele_ids = list(
            classified_common[classified_common["PIK3CA_allele"] == allele].index
        )
        if len(allele_ids) >= 3:
            comp = compare_groups(sensitivity, allele_ids, wt_ids, allele, "WT")
            results["comparisons"].append(comp)
            if not comp.get("underpowered"):
                print(f"  {allele} (n={comp['n_group1']}): "
                      f"median={comp['median_group1']:.3f}, "
                      f"d={comp['cohens_d']:.3f}, p={comp['mannwhitney_p']:.4f}")
            else:
                print(f"  {allele}: underpowered (n={len(allele_ids)})")

    # Plots
    print("\nGenerating plots...")
    plot_sensitivity_by_allele(sensitivity, classified, FIG_DIR / "inavolisib_by_allele.png")
    plot_sensitivity_by_status(sensitivity, classified, FIG_DIR / "inavolisib_mut_vs_wt.png")

    # Save results
    out_json = OUTPUT_DIR / "prism_inavolisib_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_json.name}")


if __name__ == "__main__":
    main()
