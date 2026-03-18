"""Phase 3: Genome-wide PTEN-selective dependency screen.

Screens all ~18K genes for PTEN-lost-specific dependencies per qualifying
cancer type. Includes priority target reporting, pathway enrichment, and
identification of non-PI3K dependencies.

Usage:
    uv run python -m pten_loss_pancancer_dependency_atlas.03_genomewide_screen
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
PHASE1_DIR = REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas" / "phase3"

# Significance thresholds
FDR_THRESHOLD = 0.05
EFFECT_SIZE_THRESHOLD = 0.3

# Priority targets from RD addendum
PRIORITY_TARGETS = [
    "ICMT", "CHD1", "PNKP", "SMARCA4", "BCL2L1", "ATM", "TTK", "RAD51",
]

# PI3K/AKT/mTOR pathway genes (to identify non-PI3K dependencies)
PI3K_AKT_MTOR_GENES = {
    "PIK3CA", "PIK3CB", "PIK3CD", "PIK3CG", "PIK3R1", "PIK3R2", "PIK3R3",
    "AKT1", "AKT2", "AKT3", "MTOR", "RPTOR", "RICTOR", "MLST8",
    "TSC1", "TSC2", "RHEB", "RPS6KB1", "RPS6KB2", "EIF4EBP1",
    "PTEN", "INPP4B", "PDK1", "PDPK1",
}

# Pathway gene sets for enrichment
PATHWAY_GENE_SETS = {
    "PI3K_AKT_mTOR": PI3K_AKT_MTOR_GENES,
    "Insulin_IGF": {
        "IGF1R", "INSR", "IRS1", "IRS2", "IGF2R", "GRB2", "SOS1", "GAB1",
    },
    "DNA_damage_response": {
        "ATM", "ATR", "CHEK1", "CHEK2", "BRCA1", "BRCA2", "RAD51",
        "PARP1", "PARP2", "PNKP", "NBN", "MRE11", "RAD50",
    },
    "Chromatin_remodeling": {
        "SMARCA4", "SMARCB1", "ARID1A", "ARID1B", "ARID2", "PBRM1",
        "CHD1", "CHD2", "CHD3", "CHD4", "BRD4", "BRD2",
    },
    "Protein_prenylation": {
        "ICMT", "RCE1", "FNTA", "FNTB", "PGGT1B", "RABGGTA", "RABGGTB",
    },
    "NF_kB_signaling": {
        "RELA", "RELB", "NFKB1", "NFKB2", "REL", "IKBKB", "IKBKG", "CHUK",
    },
}

# Known PTEN SL benchmark pairs
PTEN_SL_BENCHMARK = {
    "PIK3CB", "ICMT", "CHD1", "PNKP", "ATM", "TTK", "RAD51", "NUAK1",
}


def fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
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


def screen_cancer_type(
    merged: pd.DataFrame,
    cancer_type: str,
    gene_cols: list[str],
) -> pd.DataFrame:
    """Screen all genes for PTEN-selective dependency in one cancer type."""
    ct_data = merged[merged["OncotreeLineage"] == cancer_type]
    lost = ct_data[ct_data["PTEN_status"] == "lost"]
    intact = ct_data[ct_data["PTEN_status"] == "intact"]

    rows = []
    for gene in gene_cols:
        lost_vals = lost[gene].dropna().values
        intact_vals = intact[gene].dropna().values

        if len(lost_vals) < 3 or len(intact_vals) < 3:
            continue

        _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")

        # Cohen's d
        n1, n2 = len(lost_vals), len(intact_vals)
        var1, var2 = lost_vals.var(ddof=1), intact_vals.var(ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = float((lost_vals.mean() - intact_vals.mean()) / pooled_std) if pooled_std > 0 else 0.0

        rows.append({
            "gene": gene,
            "cohens_d": d,
            "p_value": float(pval),
            "n_lost": n1,
            "n_intact": n2,
            "median_dep_lost": float(np.median(lost_vals)),
            "median_dep_intact": float(np.median(intact_vals)),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["p_value"].values)
        result["cancer_type"] = cancer_type
    return result


def pathway_enrichment(
    hits: pd.DataFrame,
    all_tested: set[str],
) -> pd.DataFrame:
    """Fisher's exact test for pathway enrichment among significant hits."""
    sig_genes = set(hits["gene"])
    n_sig = len(sig_genes)
    n_total = len(all_tested)

    rows = []
    for pathway_name, pathway_genes in PATHWAY_GENE_SETS.items():
        pathway_tested = pathway_genes & all_tested
        if not pathway_tested:
            continue

        pathway_hits = sig_genes & pathway_tested
        n_pathway = len(pathway_tested)
        n_pathway_hits = len(pathway_hits)

        # Fisher's exact: pathway_hit/pathway_nothit vs nonpathway_hit/nonpathway_nothit
        table = [
            [n_pathway_hits, n_pathway - n_pathway_hits],
            [n_sig - n_pathway_hits, n_total - n_sig - n_pathway + n_pathway_hits],
        ]
        _, p_enrich = stats.fisher_exact(table, alternative="greater")

        rows.append({
            "pathway": pathway_name,
            "n_pathway_genes_tested": n_pathway,
            "n_pathway_hits": n_pathway_hits,
            "pathway_hit_genes": ";".join(sorted(pathway_hits)) if pathway_hits else "",
            "n_total_hits": n_sig,
            "n_total_tested": n_total,
            "enrichment_p": float(p_enrich),
        })

    return pd.DataFrame(rows).sort_values("enrichment_p") if rows else pd.DataFrame()


def plot_volcano(result: pd.DataFrame, cancer_type: str, output_dir: Path) -> None:
    """Volcano plot for one cancer type."""
    if len(result) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 7))
    neg_log10_fdr = -np.log10(result["fdr"].clip(lower=1e-50))

    # Color by significance
    colors = np.where(
        (result["fdr"] < FDR_THRESHOLD) & (result["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD),
        np.where(result["cohens_d"] < 0, "#D95319", "#0072BD"),
        "#CCCCCC",
    )

    ax.scatter(result["cohens_d"], neg_log10_fdr, c=colors, s=8, alpha=0.5, edgecolors="none")

    # Label priority targets
    for target in PRIORITY_TARGETS + ["PIK3CB", "PIK3CA", "AKT1", "AKT2"]:
        mask = result["gene"] == target
        if mask.any():
            row = result[mask].iloc[0]
            ax.annotate(
                target,
                (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                fontsize=7, fontweight="bold",
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
                textcoords="offset points", xytext=(5, 5),
            )

    ax.axhline(-np.log10(FDR_THRESHOLD), color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(-EFFECT_SIZE_THRESHOLD, color="gray", linestyle="--", linewidth=0.5)
    ax.axvline(EFFECT_SIZE_THRESHOLD, color="gray", linestyle="--", linewidth=0.5)

    ax.set_xlabel("Cohen's d (negative = PTEN-lost more dependent)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"PTEN-Selective Dependencies: {cancer_type}")

    fig.tight_layout()
    safe_name = cancer_type.replace("/", "_").replace(" ", "_")
    fig.savefig(output_dir / f"volcano_{safe_name}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3: Genome-Wide PTEN-Selective Dependency Screen ===\n")

    # Load Phase 1 outputs
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "pten_classification.csv", index_col=0)
    summary = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying = summary[summary["qualifies"]]["cancer_type"].tolist()
    print(f"  {len(qualifying)} qualifying cancer types")

    # Load CRISPR data
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    gene_cols = list(crispr.columns)
    print(f"  {len(gene_cols)} genes in CRISPR data")

    # Merge classification with CRISPR
    merged = classified.join(crispr, how="inner")

    # Screen each qualifying cancer type
    all_results = []
    all_hits = []
    for cancer_type in qualifying:
        print(f"\nScreening {cancer_type}...")
        result = screen_cancer_type(merged, cancer_type, gene_cols)
        if len(result) == 0:
            print(f"  No testable genes")
            continue

        # Significant hits
        sig = result[(result["fdr"] < FDR_THRESHOLD) & (result["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)]
        sig_neg = sig[sig["cohens_d"] < 0]
        sig_pos = sig[sig["cohens_d"] > 0]
        print(f"  {len(result)} genes tested, {len(sig)} significant "
              f"({len(sig_neg)} PTEN-lost deps, {len(sig_pos)} PTEN-intact deps)")

        # Top 5 strongest PTEN-lost dependencies
        if len(sig_neg) > 0:
            top5 = sig_neg.nsmallest(5, "cohens_d")
            for _, row in top5.iterrows():
                in_pi3k = " [PI3K/AKT]" if row["gene"] in PI3K_AKT_MTOR_GENES else ""
                print(f"    {row['gene']:12s} d={row['cohens_d']:.3f} FDR={row['fdr']:.3e}{in_pi3k}")

        # Save per-cancer-type hits
        if len(sig) > 0:
            safe_name = cancer_type.replace("/", "_").replace(" ", "_")
            sig.to_csv(OUTPUT_DIR / f"genomewide_hits_{safe_name}.csv", index=False)
            all_hits.append(sig)

        all_results.append(result)

        # Volcano plot
        plot_volcano(result, cancer_type, OUTPUT_DIR)

    if not all_results:
        print("\nNo results generated.")
        return

    # Combine all results
    all_results_df = pd.concat(all_results, ignore_index=True)
    all_results_df.to_csv(OUTPUT_DIR / "genomewide_all_results.csv", index=False)

    all_hits_df = pd.concat(all_hits, ignore_index=True) if all_hits else pd.DataFrame()

    # Priority targets report
    print("\n--- Priority Targets Report ---")
    priority_rows = []
    for target in PRIORITY_TARGETS:
        target_results = all_results_df[all_results_df["gene"] == target]
        if target_results.empty:
            print(f"  {target}: NOT FOUND in CRISPR data")
            continue
        for _, row in target_results.iterrows():
            sig_marker = ""
            if row["fdr"] < FDR_THRESHOLD and abs(row["cohens_d"]) > EFFECT_SIZE_THRESHOLD:
                sig_marker = " ***"
            elif row["fdr"] < 0.10:
                sig_marker = " *"
            print(f"  {target:12s} {row['cancer_type']:30s} d={row['cohens_d']:.3f} "
                  f"FDR={row['fdr']:.3e}{sig_marker}")
            priority_rows.append(row.to_dict())

    priority_df = pd.DataFrame(priority_rows)
    if len(priority_df) > 0:
        priority_df.to_csv(OUTPUT_DIR / "priority_targets_report.csv", index=False)

    # Pathway enrichment per cancer type
    print("\n--- Pathway Enrichment ---")
    enrichment_rows = []
    all_tested_genes = set(all_results_df["gene"].unique())
    for cancer_type in qualifying:
        ct_hits = all_hits_df[
            (all_hits_df["cancer_type"] == cancer_type) & (all_hits_df["cohens_d"] < 0)
        ] if len(all_hits_df) > 0 else pd.DataFrame()
        if len(ct_hits) == 0:
            continue
        enrich = pathway_enrichment(ct_hits, all_tested_genes)
        if len(enrich) > 0:
            enrich["cancer_type"] = cancer_type
            enrichment_rows.append(enrich)
            sig_enrich = enrich[enrich["enrichment_p"] < 0.05]
            if len(sig_enrich) > 0:
                for _, row in sig_enrich.iterrows():
                    print(f"  {cancer_type}: {row['pathway']} p={row['enrichment_p']:.3e} "
                          f"({row['n_pathway_hits']}/{row['n_pathway_genes_tested']})")

    if enrichment_rows:
        enrichment_df = pd.concat(enrichment_rows, ignore_index=True)
        enrichment_df.to_csv(OUTPUT_DIR / "pathway_enrichment.csv", index=False)

    # SL benchmark enrichment
    print("\n--- SL Benchmark ---")
    sl_rows = []
    for cancer_type in qualifying:
        ct_results = all_results_df[all_results_df["cancer_type"] == cancer_type]
        for sl_gene in PTEN_SL_BENCHMARK:
            gene_result = ct_results[ct_results["gene"] == sl_gene]
            if gene_result.empty:
                continue
            row = gene_result.iloc[0]
            is_hit = row["fdr"] < FDR_THRESHOLD and row["cohens_d"] < -EFFECT_SIZE_THRESHOLD
            sl_rows.append({
                "cancer_type": cancer_type,
                "gene": sl_gene,
                "cohens_d": row["cohens_d"],
                "fdr": row["fdr"],
                "is_significant": is_hit,
            })
    sl_df = pd.DataFrame(sl_rows) if sl_rows else pd.DataFrame()
    if len(sl_df) > 0:
        sl_df.to_csv(OUTPUT_DIR / "sl_benchmark_enrichment.csv", index=False)
        n_validated = sl_df[sl_df["is_significant"]].groupby("gene").size()
        if len(n_validated) > 0:
            print(f"  Validated SL partners: {dict(n_validated)}")

    # Non-PI3K dependencies (key deliverable)
    print("\n--- Non-PI3K Dependencies ---")
    if len(all_hits_df) > 0:
        non_pi3k = all_hits_df[
            (all_hits_df["cohens_d"] < 0) &
            (~all_hits_df["gene"].isin(PI3K_AKT_MTOR_GENES))
        ].copy()
        non_pi3k.to_csv(OUTPUT_DIR / "non_pi3k_dependencies.csv", index=False)
        print(f"  {len(non_pi3k)} non-PI3K PTEN-selective dependencies across all cancer types")

        # Top non-PI3K hits per cancer type
        for cancer_type in qualifying:
            ct_non_pi3k = non_pi3k[non_pi3k["cancer_type"] == cancer_type]
            if len(ct_non_pi3k) == 0:
                continue
            top3 = ct_non_pi3k.nsmallest(3, "cohens_d")
            genes = ", ".join(f"{r['gene']}(d={r['cohens_d']:.2f})" for _, r in top3.iterrows())
            print(f"  {cancer_type}: {genes}")

    # Summary text
    lines = []
    lines.append("=" * 70)
    lines.append("PTEN Loss Pan-Cancer Dependency Atlas - Phase 3")
    lines.append("Genome-Wide PTEN-Selective Dependency Screen")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Genes screened: {len(gene_cols)}")
    lines.append(f"Cancer types: {len(qualifying)}")
    lines.append(f"Significance: FDR < {FDR_THRESHOLD}, |d| > {EFFECT_SIZE_THRESHOLD}")
    lines.append("")

    for cancer_type in qualifying:
        ct_hits = all_hits_df[all_hits_df["cancer_type"] == cancer_type] if len(all_hits_df) > 0 else pd.DataFrame()
        ct_neg = ct_hits[ct_hits["cohens_d"] < 0] if len(ct_hits) > 0 else pd.DataFrame()
        lines.append(f"{cancer_type}: {len(ct_neg)} PTEN-selective dependencies")
        if len(ct_neg) > 0:
            for _, row in ct_neg.nsmallest(10, "cohens_d").iterrows():
                pi3k_tag = " [PI3K/AKT]" if row["gene"] in PI3K_AKT_MTOR_GENES else ""
                lines.append(f"  {row['gene']:15s} d={row['cohens_d']:.3f} FDR={row['fdr']:.3e}{pi3k_tag}")
        lines.append("")

    (OUTPUT_DIR / "genomewide_screen_summary.txt").write_text("\n".join(lines))
    print("\nSaved genomewide_screen_summary.txt")
    print("Done.")


if __name__ == "__main__":
    main()
