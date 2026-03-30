"""Phase 3d: Pathway-level gene-set enrichment testing.

Formal statistical test for whether the RAS upstream regulatory module
shows coordinated enhanced dependency in NF1-lost lines, beyond what's
expected by chance. Tests both uncorrected and lineage-corrected results.

Methods:
  1. Competitive gene-set test (permutation): compare mean effect size
     of set genes vs random gene sets of the same size.
  2. Wilcoxon rank-sum: compare effect size distribution of set genes
     vs all other genes.

Usage:
    PYTHONPATH=src/cancer:src uv run python -m nf1_loss_pancancer_dependency_atlas.03d_pathway_enrichment
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

OUTPUT_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase3"
)

N_PERMUTATIONS = 10000

# Gene sets to test
GENE_SETS = {
    "RAS upstream regulatory": [
        "PTPN11", "GRB2", "SHOC2", "RIT1", "SPRED1", "SPRED2", "SOS1",
    ],
    "RAS/MAPK core": [
        "BRAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
        "SOS1", "GRB2", "PTPN11", "NF1", "KRAS", "NRAS",
    ],
    "mTOR/PI3K": [
        "PIK3CA", "PIK3CB", "AKT1", "AKT2", "MTOR", "RICTOR", "RPTOR",
    ],
    "Cell cycle": ["CDK4", "CDK6", "CDK2", "CCND1", "RB1"],
    "Epigenetic/PRC2": ["EZH2", "EED", "SUZ12", "BRD4", "DOT1L"],
    "DNA damage response": ["ATR", "CHEK1", "WEE1", "PARP1"],
    "RAS feedback": [
        "SPRY1", "SPRY2", "SPRY4", "DUSP4", "DUSP6", "ERF", "RASA2",
    ],
}


def competitive_gene_set_test(
    effect_sizes: pd.Series,
    gene_set: list[str],
    n_perms: int = N_PERMUTATIONS,
) -> dict:
    """Permutation-based competitive gene-set test.

    Compares the mean effect size of genes in the set to the distribution
    of mean effect sizes from random gene sets of the same size.
    Tests for negative enrichment (gained dependencies in NF1-lost).
    """
    # Filter to genes present in the data
    present = [g for g in gene_set if g in effect_sizes.index]
    if len(present) < 2:
        return {
            "n_genes_tested": len(present),
            "n_genes_defined": len(gene_set),
            "mean_effect": float("nan"),
            "perm_p_value": float("nan"),
            "direction": "insufficient_genes",
        }

    set_effects = effect_sizes[present].values
    observed_mean = float(np.mean(set_effects))

    # All available genes for permutation
    all_genes = effect_sizes.values
    n_set = len(present)

    # Permutation: sample n_set random genes and compute mean
    rng = np.random.default_rng(42)
    perm_means = np.empty(n_perms)
    for i in range(n_perms):
        idx = rng.choice(len(all_genes), size=n_set, replace=False)
        perm_means[i] = np.mean(all_genes[idx])

    # One-sided test: is the set mean more negative (gained dependency)?
    p_left = float(np.mean(perm_means <= observed_mean))
    # Two-sided
    p_two = float(2 * min(p_left, 1 - p_left))

    return {
        "n_genes_tested": len(present),
        "n_genes_defined": len(gene_set),
        "genes_tested": ",".join(sorted(present)),
        "mean_effect": round(observed_mean, 4),
        "perm_p_left": round(p_left, 4),
        "perm_p_two": round(min(p_two, 1.0), 4),
        "perm_mean_null": round(float(np.mean(perm_means)), 4),
        "perm_sd_null": round(float(np.std(perm_means)), 4),
        "z_score": round(
            (observed_mean - np.mean(perm_means)) / max(np.std(perm_means), 1e-10), 4
        ),
        "direction": "negative" if observed_mean < 0 else "positive",
    }


def wilcoxon_gene_set_test(
    effect_sizes: pd.Series,
    gene_set: list[str],
) -> dict:
    """Wilcoxon rank-sum test: set genes vs background.

    Tests whether genes in the set have more negative effect sizes
    (gained dependencies) than the genome-wide background.
    """
    present = [g for g in gene_set if g in effect_sizes.index]
    if len(present) < 2:
        return {"wilcoxon_p": float("nan"), "wilcoxon_stat": float("nan")}

    set_vals = effect_sizes[present].values
    bg_vals = effect_sizes.drop(present, errors="ignore").values

    stat, p_two = stats.mannwhitneyu(set_vals, bg_vals, alternative="two-sided")
    _, p_less = stats.mannwhitneyu(set_vals, bg_vals, alternative="less")

    return {
        "wilcoxon_stat": round(float(stat), 2),
        "wilcoxon_p_two": round(float(p_two), 4),
        "wilcoxon_p_less": round(float(p_less), 4),
        "median_set": round(float(np.median(set_vals)), 4),
        "median_background": round(float(np.median(bg_vals)), 4),
    }


def test_all_gene_sets(
    results_df: pd.DataFrame,
    effect_col: str,
    label: str,
) -> pd.DataFrame:
    """Run both tests on all gene sets against a set of results."""
    effect_sizes = results_df.set_index("gene")[effect_col]

    rows = []
    for gs_name, gs_genes in GENE_SETS.items():
        comp = competitive_gene_set_test(effect_sizes, gs_genes)
        wilcox = wilcoxon_gene_set_test(effect_sizes, gs_genes)

        row = {"gene_set": gs_name, "analysis": label}
        row.update(comp)
        row.update(wilcox)
        rows.append(row)

    return pd.DataFrame(rows)


def plot_gene_set_effects(
    results_df: pd.DataFrame,
    effect_col: str,
    enrichment_results: pd.DataFrame,
    output_dir: Path,
    suffix: str,
) -> None:
    """Forest plot of gene-set-level effect sizes with significance."""
    effect_sizes = results_df.set_index("gene")[effect_col]

    fig, axes = plt.subplots(
        len(GENE_SETS), 1,
        figsize=(10, 2.5 * len(GENE_SETS)),
        sharex=True,
    )

    for ax, (gs_name, gs_genes) in zip(axes, GENE_SETS.items()):
        present = [g for g in gs_genes if g in effect_sizes.index]
        if not present:
            ax.set_title(f"{gs_name} (no genes found)")
            continue

        vals = effect_sizes[present].sort_values()
        colors = ["#D95319" if v < 0 else "#4DBEEE" for v in vals.values]
        ax.barh(range(len(vals)), vals.values, color=colors, alpha=0.8)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(vals.index, fontsize=8)

        # Add p-value annotation
        enr = enrichment_results[enrichment_results["gene_set"] == gs_name]
        if len(enr) > 0:
            p_left = enr.iloc[0].get("perm_p_left", float("nan"))
            mean_eff = enr.iloc[0].get("mean_effect", float("nan"))
            sig = " ***" if p_left < 0.05 else (" *" if p_left < 0.1 else "")
            ax.set_title(
                f"{gs_name}: mean={mean_eff:.3f}, p(left)={p_left:.4f}{sig}",
                fontsize=9,
            )

        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel(effect_col if ax == axes[-1] else "")

    fig.suptitle(f"Gene Set Effects ({suffix})", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / f"pathway_enrichment_{suffix}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3d: Pathway-Level Gene-Set Enrichment Testing ===\n")

    # Load results
    corrected_path = OUTPUT_DIR / "lineage_corrected_results.csv"
    uncorrected_path = OUTPUT_DIR / "genomewide_all_results.csv"

    all_enrichment = []

    # Test on lineage-corrected results (primary)
    if corrected_path.exists():
        print("Testing on lineage-corrected results (primary analysis)...")
        corrected = pd.read_csv(corrected_path)
        enr_corr = test_all_gene_sets(corrected, "cohens_d", "lineage_corrected")
        all_enrichment.append(enr_corr)

        print("\n  Lineage-corrected results:")
        for _, row in enr_corr.iterrows():
            sig = " ***" if row["perm_p_left"] < 0.05 else (" *" if row["perm_p_left"] < 0.1 else "")
            print(
                f"  {row['gene_set']:30s} mean_d={row['mean_effect']:+.3f}  "
                f"perm_p(left)={row['perm_p_left']:.4f}  "
                f"wilcox_p(less)={row['wilcoxon_p_less']:.4f}  "
                f"n={row['n_genes_tested']}/{row['n_genes_defined']}{sig}"
            )

        plot_gene_set_effects(corrected, "cohens_d", enr_corr, OUTPUT_DIR, "lineage_corrected")
    else:
        print("  WARNING: No lineage-corrected results found")

    # Test on uncorrected results (comparison)
    if uncorrected_path.exists():
        print("\nTesting on uncorrected results (comparison)...")
        uncorrected = pd.read_csv(uncorrected_path)
        # Filter to pan-cancer RAS-excluded
        uncorr_pc = uncorrected[uncorrected["cancer_type"] == "Pan-cancer (RAS-excluded)"].copy()
        enr_uncorr = test_all_gene_sets(uncorr_pc, "cohens_d", "uncorrected")
        all_enrichment.append(enr_uncorr)

        print("\n  Uncorrected (RAS-excluded) results:")
        for _, row in enr_uncorr.iterrows():
            sig = " ***" if row["perm_p_left"] < 0.05 else (" *" if row["perm_p_left"] < 0.1 else "")
            print(
                f"  {row['gene_set']:30s} mean_d={row['mean_effect']:+.3f}  "
                f"perm_p(left)={row['perm_p_left']:.4f}  "
                f"wilcox_p(less)={row['wilcoxon_p_less']:.4f}  "
                f"n={row['n_genes_tested']}/{row['n_genes_defined']}{sig}"
            )

        plot_gene_set_effects(uncorr_pc, "cohens_d", enr_uncorr, OUTPUT_DIR, "uncorrected")

    # Combine and save
    if all_enrichment:
        combined = pd.concat(all_enrichment, ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "pathway_enrichment_results.csv", index=False)

    # Summary
    summary_lines = [
        "=" * 70,
        "NF1-Loss Pan-Cancer Dependency Atlas - Phase 3d: Pathway Enrichment",
        "=" * 70,
        "",
        f"Permutations: {N_PERMUTATIONS}",
        "Tests: competitive permutation (mean effect vs random sets),",
        "       Wilcoxon rank-sum (set genes vs genome background).",
        "",
    ]

    for label, enr_df in [("LINEAGE-CORRECTED", enr_corr if corrected_path.exists() else None),
                           ("UNCORRECTED", enr_uncorr if uncorrected_path.exists() else None)]:
        if enr_df is None:
            continue
        summary_lines += [
            f"{label} RESULTS",
            "-" * 60,
        ]
        for _, row in enr_df.iterrows():
            sig = " ***" if row["perm_p_left"] < 0.05 else (" *" if row["perm_p_left"] < 0.1 else "")
            summary_lines.append(
                f"  {row['gene_set']:30s} mean_d={row['mean_effect']:+.3f}  "
                f"perm_p={row['perm_p_left']:.4f}  wilcox_p={row['wilcoxon_p_less']:.4f}  "
                f"z={row['z_score']:+.2f}  n={row['n_genes_tested']}/{row['n_genes_defined']}{sig}"
            )
            if row.get("genes_tested"):
                summary_lines.append(f"    Genes: {row['genes_tested']}")
        summary_lines.append("")

    # Interpretation
    if corrected_path.exists():
        ras_reg = enr_corr[enr_corr["gene_set"] == "RAS upstream regulatory"]
        if len(ras_reg) > 0:
            r = ras_reg.iloc[0]
            summary_lines += [
                "INTERPRETATION",
                "-" * 60,
            ]
            if r["perm_p_left"] < 0.05:
                summary_lines.append(
                    f"  The RAS upstream regulatory module shows SIGNIFICANT coordinated "
                    f"enhanced dependency (mean_d={r['mean_effect']:+.3f}, p={r['perm_p_left']:.4f})."
                )
            elif r["perm_p_left"] < 0.1:
                summary_lines.append(
                    f"  The RAS upstream regulatory module shows MARGINAL coordinated "
                    f"enhanced dependency (mean_d={r['mean_effect']:+.3f}, p={r['perm_p_left']:.4f})."
                )
            else:
                summary_lines.append(
                    f"  The RAS upstream regulatory module does NOT show significant "
                    f"coordinated enhanced dependency after lineage correction "
                    f"(mean_d={r['mean_effect']:+.3f}, p={r['perm_p_left']:.4f})."
                )
            summary_lines.append(
                f"  This is based on {r['n_genes_tested']} of {r['n_genes_defined']} "
                f"genes present in the DepMap CRISPR data."
            )
            summary_lines.append("")

    with open(OUTPUT_DIR / "pathway_enrichment_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    print("\nDone.")


if __name__ == "__main__":
    main()
