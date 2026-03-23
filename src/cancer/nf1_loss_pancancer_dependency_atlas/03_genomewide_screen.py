"""Phase 3: Genome-wide differential dependency analysis — NF1-lost vs intact.

Computes Cohen's d for ALL gene dependencies comparing NF1-lost vs NF1-intact,
both pan-cancer and per-tumor-type. Applies negative-effect filter and examines
hypothesis-driven gene sets. Ranks hits by |Cohen's d| x -log10(FDR) composite score.

Usage:
    uv run python -m nf1_loss_pancancer_dependency_atlas.03_genomewide_screen
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
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase1"
)
OUTPUT_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase3"
)

FDR_THRESHOLD = 0.1
STRICT_FDR = 0.05
EFFECT_SIZE_THRESHOLD = 0.5
HIGH_CONFIDENCE_ES = 0.8
MIN_SAMPLES = 3

# Hypothesis-driven gene sets from the plan
GENE_SETS = {
    "mTOR/PI3K": ["PIK3CA", "PIK3CB", "AKT1", "AKT2", "MTOR", "RICTOR", "RPTOR"],
    "Cell cycle": ["CDK4", "CDK6", "CDK2", "CCND1", "RB1"],
    "Epigenetic/PRC2": ["EZH2", "EED", "SUZ12", "BRD4", "DOT1L"],
    "DNA damage": ["ATR", "CHEK1", "WEE1", "PARP1"],
    "RAS feedback": ["SPRY1", "SPRY2", "SPRY4", "DUSP4", "DUSP6", "ERF", "RASA2"],
    "RAS/MAPK core": [
        "BRAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
        "SOS1", "GRB2", "PTPN11", "NF1", "KRAS", "NRAS",
    ],
}


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    n = len(pvalues)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    fdr = np.empty(n)
    for i in range(n):
        fdr[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
    fdr_sorted = fdr[sorted_idx]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[sorted_idx] = fdr_sorted
    return np.minimum(fdr, 1.0)


def gene_set_label(gene: str) -> str:
    """Return which hypothesis gene set a gene belongs to, or empty string."""
    for gs_name, gs_genes in GENE_SETS.items():
        if gene in gs_genes:
            return gs_name
    return ""


def screen_one_context(
    lost_data: pd.DataFrame,
    intact_data: pd.DataFrame,
    crispr_cols: list[str],
    context_name: str,
) -> list[dict]:
    rows = []
    pvals = []

    for gene in crispr_cols:
        lost_vals = lost_data[gene].dropna().values
        intact_vals = intact_data[gene].dropna().values

        if len(lost_vals) < MIN_SAMPLES or len(intact_vals) < MIN_SAMPLES:
            continue

        _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")
        d = cohens_d(lost_vals, intact_vals)

        rows.append({
            "cancer_type": context_name,
            "gene": gene,
            "cohens_d": round(d, 4),
            "p_value": float(pval),
            "n_lost": len(lost_vals),
            "n_intact": len(intact_vals),
            "median_dep_lost": round(float(np.median(lost_vals)), 4),
            "median_dep_intact": round(float(np.median(intact_vals)), 4),
        })
        pvals.append(pval)

    if pvals:
        fdrs = fdr_correction(np.array(pvals))
        for i, row in enumerate(rows):
            row["fdr"] = float(fdrs[i])
            # Composite score: |d| x -log10(FDR)
            fdr_val = max(fdrs[i], 1e-300)
            row["composite_score"] = round(abs(row["cohens_d"]) * -np.log10(fdr_val), 4)

    return rows


def plot_volcano(results_ct: pd.DataFrame, context_name: str, out_dir: Path) -> None:
    if "fdr" not in results_ct.columns or len(results_ct) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    x = results_ct["cohens_d"].values
    y = -np.log10(results_ct["fdr"].values.clip(min=1e-50))

    sig = (results_ct["fdr"] < FDR_THRESHOLD) & (results_ct["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ax.scatter(x[~sig], y[~sig], c="#CCCCCC", s=5, alpha=0.5)

    gained = sig & (results_ct["cohens_d"] < 0)
    lost_dep = sig & (results_ct["cohens_d"] > 0)
    ax.scatter(x[gained], y[gained], c="#D95319", s=15, alpha=0.8, label="Gained dep. in NF1-lost")
    ax.scatter(x[lost_dep], y[lost_dep], c="#4DBEEE", s=15, alpha=0.8, label="Lost dep. in NF1-lost")

    # Label top gained hits
    top = results_ct[gained].nsmallest(10, "cohens_d")
    for _, row in top.iterrows():
        ax.annotate(row["gene"], (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                     fontsize=7, ha="right")

    # Highlight hypothesis gene sets
    for gs_name, gs_genes in GENE_SETS.items():
        gs_mask = results_ct["gene"].isin(gs_genes) & sig
        if gs_mask.any():
            for _, row in results_ct[gs_mask].iterrows():
                ax.annotate(
                    f"{row['gene']}",
                    (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                    fontsize=6, ha="left", color="darkgreen",
                )

    ax.axhline(-np.log10(FDR_THRESHOLD), color="grey", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)

    ax.set_xlabel("Cohen's d (NF1-lost vs intact)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"NF1-Loss Dependency Screen: {context_name}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    safe = context_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(out_dir / f"volcano_{safe}.png", dpi=150)
    plt.close(fig)


def plot_gene_set_heatmap(all_results: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of hypothesis gene set effect sizes across contexts."""
    all_hypothesis_genes = []
    for genes in GENE_SETS.values():
        all_hypothesis_genes.extend(genes)

    hyp_results = all_results[all_results["gene"].isin(all_hypothesis_genes)]
    if len(hyp_results) == 0:
        return

    pivot = hyp_results.pivot_table(
        index="gene", columns="cancer_type", values="cohens_d", aggfunc="first"
    )
    if pivot.empty:
        return

    # Order by gene set
    ordered_genes = []
    for gs_genes in GENE_SETS.values():
        ordered_genes.extend([g for g in gs_genes if g in pivot.index])
    pivot = pivot.loc[[g for g in ordered_genes if g in pivot.index]]

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.0), max(6, len(pivot.index) * 0.3)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-1.0, vmax=1.0)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5)

    plt.colorbar(im, ax=ax, label="Cohen's d")
    ax.set_title("Hypothesis Gene Sets: NF1-Lost vs Intact")
    plt.tight_layout()
    fig.savefig(output_dir / "hypothesis_geneset_heatmap.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3: Genome-Wide NF1-Loss Dependency Screen ===\n")

    # Load Phase 1
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "nf1_loss_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()
    print(f"  {len(classified)} lines, {len(qualifying_types)} qualifying cancer types")

    # Load CRISPR
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    crispr_cols = list(crispr.columns)
    print(f"  {len(crispr_cols)} genes, {len(crispr)} cell lines")

    merged = classified.join(crispr, how="inner")
    print(f"  {len(merged)} lines with both classification and CRISPR data")

    # Exclude lines with concurrent RAS mutations
    merged_no_ras = merged[~merged["has_RAS_mutation"]]
    print(f"  {len(merged_no_ras)} lines after excluding RAS-mutant")

    # Screen each context
    all_rows = []
    contexts = qualifying_types + ["Pan-cancer (pooled)", "Pan-cancer (RAS-excluded)"]

    for context in contexts:
        if context == "Pan-cancer (pooled)":
            ct_data = merged
        elif context == "Pan-cancer (RAS-excluded)":
            ct_data = merged_no_ras
        else:
            ct_data = merged[merged["OncotreeLineage"] == context]

        lost_lines = ct_data[ct_data["NF1_loss"] == True]  # noqa: E712
        intact_lines = ct_data[ct_data["NF1_status"] == "intact"]
        print(f"  Screening {context} ({len(lost_lines)} lost, {len(intact_lines)} intact)...")

        rows = screen_one_context(lost_lines, intact_lines, crispr_cols, context)
        all_rows.extend(rows)

    all_results = pd.DataFrame(all_rows)
    print(f"\n  Total tests: {len(all_results)}")

    # Annotate gene sets
    all_results["gene_set"] = all_results["gene"].apply(gene_set_label)

    # Save full results
    all_results.to_csv(OUTPUT_DIR / "genomewide_all_results.csv", index=False)

    # Significant hits
    sig_hits = all_results[
        (all_results["fdr"] < FDR_THRESHOLD) & (all_results["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ].copy()

    gained = sig_hits[sig_hits["cohens_d"] < 0].sort_values("cohens_d")
    lost = sig_hits[sig_hits["cohens_d"] > 0].sort_values("cohens_d", ascending=False)

    print(f"  SL gained dependencies (FDR<{FDR_THRESHOLD}, |d|>{EFFECT_SIZE_THRESHOLD}): {len(gained)}")
    print(f"  Lost dependencies: {len(lost)}")

    # High-confidence
    hc = all_results[
        (all_results["fdr"] < STRICT_FDR) & (all_results["cohens_d"].abs() > HIGH_CONFIDENCE_ES)
    ]
    hc_gained = hc[hc["cohens_d"] < 0]
    print(f"  High-confidence SL (FDR<{STRICT_FDR}, |d|>{HIGH_CONFIDENCE_ES}): {len(hc_gained)}")

    # Pan-cancer gained
    pancancer_gained = gained[gained["cancer_type"] == "Pan-cancer (RAS-excluded)"].copy()
    if len(pancancer_gained) == 0:
        pancancer_gained = gained[gained["cancer_type"] == "Pan-cancer (pooled)"].copy()

    # Novel hits (not in RAS/MAPK core)
    ras_core = set(GENE_SETS.get("RAS/MAPK core", []))
    novel_gained = pancancer_gained[~pancancer_gained["gene"].isin(ras_core)]

    print(f"\nTop pan-cancer SL hits (novel, RAS-excluded):")
    for _, row in novel_gained.head(20).iterrows():
        gs = f" [{row['gene_set']}]" if row["gene_set"] else ""
        print(
            f"  {row['gene']}: d={row['cohens_d']:.3f}, FDR={row['fdr']:.3e}, "
            f"composite={row['composite_score']:.1f}{gs}"
        )

    # Hypothesis gene set analysis
    print("\nHypothesis gene set results (pan-cancer RAS-excluded):")
    ras_excl = all_results[all_results["cancer_type"] == "Pan-cancer (RAS-excluded)"]
    for gs_name, gs_genes in GENE_SETS.items():
        if gs_name == "RAS/MAPK core":
            continue
        gs_results = ras_excl[ras_excl["gene"].isin(gs_genes)].sort_values("cohens_d")
        sig_count = ((gs_results["fdr"] < FDR_THRESHOLD) & (gs_results["cohens_d"] < -EFFECT_SIZE_THRESHOLD)).sum()
        print(f"  {gs_name} ({sig_count} significant):")
        for _, row in gs_results.iterrows():
            sig_flag = " ***" if row.get("fdr", 1) < FDR_THRESHOLD and row["cohens_d"] < -EFFECT_SIZE_THRESHOLD else ""
            print(f"    {row['gene']}: d={row['cohens_d']:.3f}, FDR={row.get('fdr', 'N/A'):.3e}{sig_flag}")

    # Save significant hits
    gained.to_csv(OUTPUT_DIR / "sl_gained_dependencies.csv", index=False)
    lost.to_csv(OUTPUT_DIR / "lost_dependencies.csv", index=False)

    # Ranked pan-cancer list
    pancancer_all = all_results[all_results["cancer_type"] == "Pan-cancer (RAS-excluded)"].copy()
    pancancer_all = pancancer_all.sort_values("cohens_d").reset_index(drop=True)
    pancancer_all.to_csv(OUTPUT_DIR / "pancancer_ranked_genes.csv", index=False)

    # Volcano plots
    print("\nGenerating volcano plots...")
    for context in contexts:
        ct_res = all_results[all_results["cancer_type"] == context]
        plot_volcano(ct_res, context, OUTPUT_DIR)
    print(f"  Saved {len(contexts)} volcano plots")

    # Gene set heatmap
    plot_gene_set_heatmap(all_results, OUTPUT_DIR)

    # Summary text
    summary_lines = [
        "=" * 70,
        "NF1-Loss Pan-Cancer Dependency Atlas - Phase 3: Genome-Wide Screen",
        "=" * 70,
        "",
        f"Total tests: {len(all_results)}",
        f"Contexts: {', '.join(contexts)}",
        "",
        f"SL gained dependencies (FDR<{FDR_THRESHOLD}, |d|>{EFFECT_SIZE_THRESHOLD}): {len(gained)}",
        f"High-confidence SL (FDR<{STRICT_FDR}, |d|>{HIGH_CONFIDENCE_ES}): {len(hc_gained)}",
        f"Lost dependencies: {len(lost)}",
        "",
        "TOP PAN-CANCER SL HITS (RAS-excluded, novel)",
        "-" * 60,
    ]
    for _, row in novel_gained.head(30).iterrows():
        gs = f" [{row['gene_set']}]" if row["gene_set"] else ""
        summary_lines.append(
            f"  {row['gene']}: d={row['cohens_d']:.3f}, FDR={row['fdr']:.3e}, "
            f"composite={row['composite_score']:.1f}{gs}"
        )

    summary_lines += [
        "",
        "HYPOTHESIS GENE SET SUMMARY (RAS-excluded)",
        "-" * 60,
    ]
    for gs_name, gs_genes in GENE_SETS.items():
        if gs_name == "RAS/MAPK core":
            continue
        gs_results = ras_excl[ras_excl["gene"].isin(gs_genes)].sort_values("cohens_d")
        for _, row in gs_results.iterrows():
            sig_flag = " ***" if row.get("fdr", 1) < FDR_THRESHOLD and row["cohens_d"] < -EFFECT_SIZE_THRESHOLD else ""
            summary_lines.append(
                f"  {gs_name} > {row['gene']}: d={row['cohens_d']:.3f}, FDR={row.get('fdr', 'N/A'):.3e}{sig_flag}"
            )

    summary_lines.append("")

    with open(OUTPUT_DIR / "genomewide_screen_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("  genomewide_screen_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
