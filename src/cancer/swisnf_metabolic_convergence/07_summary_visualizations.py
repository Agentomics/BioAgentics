"""Phase 1-5 summary visualizations for SWI/SNF metabolic convergence.

Generates three key figures:
1. OXPHOS convergence heatmap — Cohen's d across cancer types for ARID1A/SMARCA4
2. Cross-validation forest plot — targeted genes (MICOS13, HIGD2A, COX6C, HMGCR, ADCK5)
3. Drug sensitivity dot plot — IACS-010759, statins across SWI/SNF subgroups

Usage:
    PYTHONPATH=src/cancer:src uv run python src/cancer/swisnf_metabolic_convergence/07_summary_visualizations.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from bioagentics.config import REPO_ROOT

RESULTS_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence"
OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "swisnf-metabolic-convergence" / "figures"


def fig1_oxphos_heatmap() -> None:
    """OXPHOS convergence heatmap: Cohen's d for 34 OXPHOS genes across cancer types."""
    convergent = pd.read_csv(RESULTS_DIR / "phase2" / "convergent_metabolic_genes.csv")
    oxphos_genes = sorted(convergent[convergent["category"] == "OXPHOS"]["gene"].tolist())

    arid1a = pd.read_csv(RESULTS_DIR / "phase1b" / "screen_arid1a_vs_wt.csv")
    smarca4 = pd.read_csv(RESULTS_DIR / "phase1b" / "screen_smarca4_vs_wt.csv")

    # Filter to OXPHOS genes only
    arid1a_ox = arid1a[arid1a["gene"].isin(oxphos_genes)]
    smarca4_ox = smarca4[smarca4["gene"].isin(oxphos_genes)]

    # Build pivot: gene x cancer_type, values = Cohen's d
    # For ARID1A — prefix cancer types with "A: "
    arid1a_pivot = arid1a_ox.pivot_table(
        index="gene", columns="cancer_type", values="cohens_d", aggfunc="first",
    )
    arid1a_pivot.columns = ["A:" + c for c in arid1a_pivot.columns]

    # For SMARCA4 — prefix cancer types with "S: "
    smarca4_pivot = smarca4_ox.pivot_table(
        index="gene", columns="cancer_type", values="cohens_d", aggfunc="first",
    )
    smarca4_pivot.columns = ["S:" + c for c in smarca4_pivot.columns]

    # Merge side by side
    combined = arid1a_pivot.join(smarca4_pivot, how="outer")
    combined = combined.reindex(oxphos_genes)

    # Sort columns: ARID1A first, then SMARCA4
    a_cols = sorted([c for c in combined.columns if c.startswith("A:")])
    s_cols = sorted([c for c in combined.columns if c.startswith("S:")])
    combined = combined[a_cols + s_cols]

    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(
        combined,
        cmap="RdBu",
        center=0,
        vmin=-2,
        vmax=2,
        linewidths=0.3,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Cohen's d (negative = SL)", "shrink": 0.7},
    )
    ax.set_title("OXPHOS Convergent Gene Dependencies\nAcross ARID1A and SMARCA4-Mutant Cancer Types", fontsize=12)
    ax.set_ylabel("Gene")
    ax.set_xlabel("")

    # Add divider line between ARID1A and SMARCA4 columns
    ax.axvline(x=len(a_cols), color="black", linewidth=2)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_oxphos_convergence_heatmap.png", dpi=150)
    plt.close(fig)
    print("  Saved fig1_oxphos_convergence_heatmap.png")


def fig2_cross_validation_forest() -> None:
    """Cross-validation forest plot for targeted genes."""
    # Load targeted gene results
    arid1a_in_smarca4 = pd.read_csv(RESULTS_DIR / "phase2" / "targeted_arid1a_genes_in_smarca4.csv")
    smarca4_in_arid1a = pd.read_csv(RESULTS_DIR / "phase2" / "targeted_smarca4_mito_in_arid1a.csv")

    # Combine
    targeted = pd.concat([arid1a_in_smarca4, smarca4_in_arid1a], ignore_index=True)

    # Focus on key genes
    key_genes = ["MICOS13", "HIGD2A", "COX6C", "HMGCR", "ADCK5"]
    targeted = targeted[targeted["gene"].isin(key_genes)]

    fig, axes = plt.subplots(len(key_genes), 1, figsize=(10, 12), sharex=True)

    for i, gene in enumerate(key_genes):
        ax = axes[i]
        gene_data = targeted[targeted["gene"] == gene].sort_values("cohens_d")

        if len(gene_data) == 0:
            ax.set_ylabel(gene, fontsize=10, fontweight="bold")
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        colors = []
        for _, row in gene_data.iterrows():
            if row["is_sl"]:
                colors.append("#d62728")  # red for SL
            elif row["cohens_d"] < -0.3:
                colors.append("#ff7f0e")  # orange for suggestive
            else:
                colors.append("#aec7e8")  # light blue for non-SL

        y_labels = [f"{row['cancer_type']} ({row['tested_in']})" for _, row in gene_data.iterrows()]
        y_pos = range(len(gene_data))

        ax.barh(y_pos, gene_data["cohens_d"].values, color=colors, height=0.6, edgecolor="gray", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=7)
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.axvline(x=-0.3, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_ylabel("")
        ax.set_title(f"{gene} ({gene_data['source_atlas'].iloc[0]} -> {gene_data['tested_in'].iloc[0]})",
                     fontsize=10, fontweight="bold", loc="left")

    axes[-1].set_xlabel("Cohen's d (negative = synthetic lethal)")

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="SL (p<0.05, |d|>0.3)"),
        Patch(facecolor="#ff7f0e", label="Suggestive (|d|>0.3)"),
        Patch(facecolor="#aec7e8", label="Not SL"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Cross-Atlas Validation: Targeted Gene Dependencies", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_cross_validation_forest.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_cross_validation_forest.png")


def fig3_drug_sensitivity() -> None:
    """Drug sensitivity dot plot from PRISM data."""
    prism = pd.read_csv(RESULTS_DIR / "phase5" / "prism_drug_sensitivity.csv")
    drug_map = pd.read_csv(RESULTS_DIR / "phase5" / "drug_target_mapping.csv")

    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each drug-comparison combination
    comparisons = ["ARID1A-mutant", "SMARCA4-mutant", "Combined SWI/SNF"]
    comp_offsets = {c: i * 0.25 for i, c in enumerate(comparisons)}
    comp_colors = {"ARID1A-mutant": "#1f77b4", "SMARCA4-mutant": "#ff7f0e", "Combined SWI/SNF": "#2ca02c"}

    drugs = prism["drug"].unique()
    x_pos = {d: i for i, d in enumerate(drugs)}

    for _, row in prism.iterrows():
        x = x_pos[row["drug"]] + comp_offsets[row["comparison"]] - 0.25
        y = row["cohens_d"]
        color = comp_colors[row["comparison"]]
        size = max(20, min(200, abs(np.log10(max(row["p_value"], 1e-10))) * 30))

        marker = "v" if row["is_sl"] else "o"
        ax.scatter(x, y, c=color, s=size, marker=marker, edgecolors="black", linewidths=0.5, zorder=3)

    ax.set_xticks(range(len(drugs)))
    ax.set_xticklabels(drugs, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Cohen's d (negative = more sensitive)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhline(y=-0.3, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)

    # Legend for comparisons
    from matplotlib.lines import Line2D
    legend_comp = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=comp_colors[c],
               markersize=8, label=c, markeredgecolor="black", markeredgewidth=0.5)
        for c in comparisons
    ]
    legend_comp.append(
        Line2D([0], [0], marker="v", color="w", markerfacecolor="gray",
               markersize=8, label="SL (p<0.05)", markeredgecolor="black", markeredgewidth=0.5)
    )
    ax.legend(handles=legend_comp, loc="lower left", fontsize=8)

    ax.set_title("Drug Sensitivity in SWI/SNF-Mutant Cell Lines (PRISM)", fontsize=12)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_drug_sensitivity.png", dpi=150)
    plt.close(fig)
    print("  Saved fig3_drug_sensitivity.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Phase 1-5 Summary Visualizations ===\n")

    print("Figure 1: OXPHOS convergence heatmap")
    fig1_oxphos_heatmap()

    print("Figure 2: Cross-validation forest plot")
    fig2_cross_validation_forest()

    print("Figure 3: Drug sensitivity dot plot")
    fig3_drug_sensitivity()

    print(f"\nAll figures saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
