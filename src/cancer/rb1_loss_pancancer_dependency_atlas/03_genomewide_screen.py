"""Phase 3: Genome-wide RB1-loss-specific dependency screen.

Screens all ~18K genes for RB1-loss-selective dependencies per qualifying
cancer type and pan-cancer pooled. Pathway enrichment, SL benchmark
validation, comparison with CDKN2A atlas findings.

Usage:
    uv run python -m rb1_loss_pancancer_dependency_atlas.03_genomewide_screen
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
PHASE1_DIR = REPO_ROOT / "output" / "cancer" / "rb1-loss-pancancer-dependency-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "rb1-loss-pancancer-dependency-atlas" / "phase3"

# CDKN2A atlas Phase 3 output for cross-atlas comparison
CDKN2A_PHASE3_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase3"

# Significance thresholds
FDR_THRESHOLD = 0.05
EFFECT_SIZE_THRESHOLD = 0.3

MIN_SAMPLES = 3

# Pathway gene sets
CELL_CYCLE_GENES = {
    "CDK4", "CDK6", "CDK2", "CDK1", "CCND1", "CCND2", "CCND3",
    "CCNE1", "CCNE2", "CCNA1", "CCNA2", "CCNB1", "CCNB2",
    "CDC25A", "CDC25B", "CDC25C", "CDC7", "DBF4",
}

DNA_REPLICATION_GENES = {
    "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7",
    "ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6",
    "CDC6", "CDT1", "PCNA", "RPA1", "RPA2", "RFC1",
    "POLE", "POLD1", "POLA1", "PRIM1",
}

MITOTIC_CHECKPOINT_GENES = {
    "AURKA", "AURKB", "TTK", "BUB1", "BUB1B", "BUB3",
    "MAD1L1", "MAD2L1", "CDC20", "PLK1", "PLK4",
    "CENPE", "CENPF", "KNL1", "NDC80", "NUF2",
}

DDR_CHECKPOINT_GENES = {
    "CHEK1", "CHEK2", "WEE1", "ATR", "ATM", "BRCA1", "BRCA2",
    "RAD51", "RAD51C", "RAD51D", "PARP1", "PARP2",
    "FANCA", "FANCD2", "XRCC1", "NBN",
}

E2F_TARGET_GENES = {
    "E2F1", "E2F2", "E2F3", "E2F4", "E2F5",
    "FOXM1", "MYBL2", "RB1", "RBL1", "RBL2",
    "TFDP1", "TFDP2", "SKP2", "CDKN1A", "CDKN1B",
}

# Known RB1 SL benchmark genes
KNOWN_RB1_SL = {
    "CDK2", "AURKA", "AURKB", "CHEK1", "WEE1", "CSNK2A1", "TTK",
}

ALL_PATHWAY_GENES = (
    CELL_CYCLE_GENES | DNA_REPLICATION_GENES | MITOTIC_CHECKPOINT_GENES
    | DDR_CHECKPOINT_GENES | E2F_TARGET_GENES
)


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


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


def screen_one_context(
    lost_data: pd.DataFrame,
    intact_data: pd.DataFrame,
    crispr_cols: list[str],
    context_name: str,
) -> list[dict]:
    """Run genome-wide differential dependency for one context."""
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

    return rows


def pathway_enrichment(
    sig_genes: set[str],
    universe: set[str],
    pathway_name: str,
    pathway_genes: set[str],
) -> dict:
    """Fisher exact test for pathway enrichment in significant hits."""
    pathway_in_universe = pathway_genes & universe
    a = len(sig_genes & pathway_in_universe)
    b = len(sig_genes - pathway_in_universe)
    c = len(pathway_in_universe - sig_genes)
    d = len(universe - sig_genes - pathway_in_universe)

    _, pval = stats.fisher_exact([[a, b], [c, d]], alternative="greater")
    expected = len(pathway_in_universe) * len(sig_genes) / len(universe) if len(universe) > 0 else 0

    return {
        "pathway": pathway_name,
        "observed": a,
        "expected": round(expected, 2),
        "fold_enrichment": round(a / expected, 2) if expected > 0 else float("inf"),
        "fisher_p": float(pval),
        "n_hits": len(sig_genes),
        "n_pathway_in_universe": len(pathway_in_universe),
        "genes_in_overlap": ";".join(sorted(sig_genes & pathway_in_universe)),
    }


def compare_with_cdkn2a(
    rb1_results: pd.DataFrame,
) -> pd.DataFrame:
    """Compare RB1-loss hits with CDKN2A atlas genome-wide results."""
    cdkn2a_path = CDKN2A_PHASE3_DIR / "genomewide_all_results.csv"
    if not cdkn2a_path.exists():
        print("  CDKN2A Phase 3 results not found, skipping comparison")
        return pd.DataFrame()

    cdkn2a = pd.read_csv(cdkn2a_path)

    # Pan-cancer results from both
    rb1_pan = rb1_results[rb1_results["cancer_type"] == "Pan-cancer (pooled)"].set_index("gene")
    cdkn2a_pan = cdkn2a[cdkn2a["cancer_type"] == "Pan-cancer (pooled)"].set_index("gene")

    common_genes = sorted(set(rb1_pan.index) & set(cdkn2a_pan.index))
    if not common_genes:
        return pd.DataFrame()

    rows = []
    for gene in common_genes:
        rb1_d = rb1_pan.loc[gene, "cohens_d"]
        rb1_fdr = rb1_pan.loc[gene, "fdr"]
        cdkn2a_d = cdkn2a_pan.loc[gene, "cohens_d"]
        cdkn2a_fdr = cdkn2a_pan.loc[gene, "fdr"]

        rb1_sig = rb1_fdr < FDR_THRESHOLD and abs(rb1_d) > EFFECT_SIZE_THRESHOLD
        cdkn2a_sig = cdkn2a_fdr < FDR_THRESHOLD and abs(cdkn2a_d) > EFFECT_SIZE_THRESHOLD

        if rb1_sig or cdkn2a_sig:
            category = "neither"
            if rb1_sig and cdkn2a_sig:
                category = "shared"
            elif rb1_sig:
                category = "RB1-unique"
            else:
                category = "CDKN2A-unique"

            rows.append({
                "gene": gene,
                "rb1_cohens_d": round(rb1_d, 4),
                "rb1_fdr": float(rb1_fdr),
                "cdkn2a_cohens_d": round(cdkn2a_d, 4),
                "cdkn2a_fdr": float(cdkn2a_fdr),
                "category": category,
            })

    return pd.DataFrame(rows)


def plot_volcano(results_ct: pd.DataFrame, context_name: str, out_dir: Path) -> None:
    """Volcano plot for one context."""
    if "fdr" not in results_ct.columns or len(results_ct) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    x = results_ct["cohens_d"].values
    y = -np.log10(results_ct["fdr"].values.clip(min=1e-50))

    sig = (results_ct["fdr"] < FDR_THRESHOLD) & (results_ct["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ax.scatter(x[~sig], y[~sig], c="#CCCCCC", s=5, alpha=0.5)

    gained = sig & (results_ct["cohens_d"] < 0)
    lost = sig & (results_ct["cohens_d"] > 0)
    ax.scatter(x[gained], y[gained], c="#D95319", s=15, alpha=0.8, label="Gained dep. in RB1-loss")
    ax.scatter(x[lost], y[lost], c="#4DBEEE", s=15, alpha=0.8, label="Lost dep. in RB1-loss")

    # Label top gained hits
    top = results_ct[gained].nsmallest(10, "cohens_d")
    for _, row in top.iterrows():
        ax.annotate(row["gene"], (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                     fontsize=7, ha="right")

    # Highlight known SL genes
    sl_mask = results_ct["gene"].isin(KNOWN_RB1_SL) & sig
    if sl_mask.any():
        ax.scatter(x[sl_mask], y[sl_mask], c="none", edgecolors="red",
                   s=40, linewidths=1.0, label="Known RB1 SL")

    ax.axhline(-np.log10(FDR_THRESHOLD), color="grey", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)

    ax.set_xlabel("Cohen's d (RB1-loss vs intact)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Genome-wide RB1-Loss Selective Screen: {context_name}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    safe = context_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(out_dir / f"volcano_{safe}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3: Genome-Wide RB1-Loss Dependency Screen ===\n")

    # Load Phase 1
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "rb1_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()
    print(f"  {len(qualifying_types)} qualifying cancer types")

    # Load CRISPR
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    crispr_cols = list(crispr.columns)
    print(f"  {len(crispr_cols)} genes")

    merged = classified.join(crispr, how="inner")

    # Screen each cancer type + pan-cancer
    all_rows = []
    contexts = qualifying_types + ["Pan-cancer (pooled)"]

    for context in contexts:
        if context == "Pan-cancer (pooled)":
            ct_data = merged
        else:
            ct_data = merged[merged["OncotreeLineage"] == context]

        lost_lines = ct_data[ct_data["RB1_status"] == "lost"]
        intact_lines = ct_data[ct_data["RB1_status"] == "intact"]
        print(f"  Screening {context} ({len(lost_lines)} lost, {len(intact_lines)} intact)...")

        rows = screen_one_context(lost_lines, intact_lines, crispr_cols, context)
        all_rows.extend(rows)

    all_results = pd.DataFrame(all_rows)
    print(f"\n  Total tests: {len(all_results)}")

    # Save full results
    all_results.to_csv(OUTPUT_DIR / "genomewide_all_results.csv", index=False)

    # Significant hits
    sig_hits = all_results[
        (all_results["fdr"] < FDR_THRESHOLD) & (all_results["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ].copy()

    gained = sig_hits[sig_hits["cohens_d"] < 0].sort_values("cohens_d")
    lost = sig_hits[sig_hits["cohens_d"] > 0].sort_values("cohens_d", ascending=False)

    print(f"  Gained dependencies (FDR<0.05, |d|>0.3): {len(gained)}")
    print(f"  Lost dependencies (FDR<0.05, |d|>0.3): {len(lost)}")

    # Top hits
    print(f"\nTop RB1-loss gained dependencies:")
    for _, row in gained.head(20).iterrows():
        label = ""
        if row["gene"] in KNOWN_RB1_SL:
            label = " [known RB1 SL]"
        elif row["gene"] in ALL_PATHWAY_GENES:
            label = " [pathway]"
        print(f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
              f"FDR={row['fdr']:.3e}{label}")

    # Pathway enrichment (pan-cancer)
    print("\nPathway enrichment analysis (pan-cancer gained hits)...")
    pancancer_sig = sig_hits[sig_hits["cancer_type"] == "Pan-cancer (pooled)"]
    pancancer_gained_genes = set(pancancer_sig[pancancer_sig["cohens_d"] < 0]["gene"])
    universe = set(all_results[all_results["cancer_type"] == "Pan-cancer (pooled)"]["gene"])

    pathways = [
        ("Cell_cycle", CELL_CYCLE_GENES),
        ("DNA_replication", DNA_REPLICATION_GENES),
        ("Mitotic_checkpoint", MITOTIC_CHECKPOINT_GENES),
        ("DDR_checkpoint", DDR_CHECKPOINT_GENES),
        ("E2F_targets", E2F_TARGET_GENES),
    ]

    enrichment_rows = []
    for pw_name, pw_genes in pathways:
        result = pathway_enrichment(pancancer_gained_genes, universe, pw_name, pw_genes)
        enrichment_rows.append(result)
        print(f"  {pw_name}: obs={result['observed']}, expected={result['expected']}, "
              f"fold={result['fold_enrichment']}x, p={result['fisher_p']:.3e}")
        if result["genes_in_overlap"]:
            print(f"    genes: {result['genes_in_overlap']}")

    enrich_df = pd.DataFrame(enrichment_rows)
    enrich_df.to_csv(OUTPUT_DIR / "pathway_enrichment.csv", index=False)

    # SL benchmark enrichment
    print("\nSL benchmark enrichment (known RB1 SL genes)...")
    sl_result = pathway_enrichment(pancancer_gained_genes, universe, "RB1_known_SL", KNOWN_RB1_SL)
    known_in_universe = KNOWN_RB1_SL & universe
    print(f"  Known SL in hits: {sl_result['observed']}/{len(known_in_universe)} "
          f"(fold={sl_result['fold_enrichment']}x, p={sl_result['fisher_p']:.3e})")
    pd.DataFrame([sl_result]).to_csv(OUTPUT_DIR / "sl_benchmark_enrichment.csv", index=False)

    # CDKN2A comparison
    print("\nComparing with CDKN2A atlas...")
    comparison = compare_with_cdkn2a(all_results)
    if len(comparison) > 0:
        comparison.to_csv(OUTPUT_DIR / "cdkn2a_comparison.csv", index=False)
        n_shared = (comparison["category"] == "shared").sum()
        n_rb1_unique = (comparison["category"] == "RB1-unique").sum()
        n_cdkn2a_unique = (comparison["category"] == "CDKN2A-unique").sum()
        print(f"  Shared: {n_shared}, RB1-unique: {n_rb1_unique}, CDKN2A-unique: {n_cdkn2a_unique}")

        # Show shared hits
        shared = comparison[comparison["category"] == "shared"].sort_values("rb1_cohens_d")
        if len(shared) > 0:
            print("  Shared dependencies (pan-cancer):")
            for _, row in shared.head(10).iterrows():
                print(f"    {row['gene']}: RB1 d={row['rb1_cohens_d']:.3f}, "
                      f"CDKN2A d={row['cdkn2a_cohens_d']:.3f}")

    # Priority targets report
    print("\nBuilding priority targets report...")
    pancancer_gained = gained[gained["cancer_type"] == "Pan-cancer (pooled)"]
    multi_type_genes = gained.groupby("gene")["cancer_type"].nunique()
    multi_type = multi_type_genes[multi_type_genes >= 2].index.tolist()

    priority_rows = []
    for gene in set(pancancer_gained["gene"].tolist() + multi_type):
        gene_data = gained[gained["gene"] == gene]
        types_hit = gene_data["cancer_type"].unique().tolist()
        pan_d = float(pancancer_gained.loc[pancancer_gained["gene"] == gene, "cohens_d"].values[0]) if gene in pancancer_gained["gene"].values else None
        is_known = gene in KNOWN_RB1_SL
        is_pathway = gene in ALL_PATHWAY_GENES

        priority_rows.append({
            "gene": gene,
            "pancancer_d": round(pan_d, 4) if pan_d is not None else None,
            "n_cancer_types": len([t for t in types_hit if t != "Pan-cancer (pooled)"]),
            "cancer_types": ";".join([t for t in types_hit if t != "Pan-cancer (pooled)"]),
            "known_rb1_sl": is_known,
            "pathway_gene": is_pathway,
        })

    priority_df = pd.DataFrame(priority_rows)
    if len(priority_df) > 0:
        priority_df = priority_df.sort_values("pancancer_d", na_position="last").reset_index(drop=True)
        priority_df.to_csv(OUTPUT_DIR / "priority_targets.csv", index=False)
        print(f"  {len(priority_df)} priority targets identified")

    # Volcano plots
    print("\nGenerating volcano plots...")
    for context in contexts:
        ct_res = all_results[all_results["cancer_type"] == context]
        plot_volcano(ct_res, context, OUTPUT_DIR)
    print(f"  Saved {len(contexts)} volcano plots")

    # Summary text
    summary_lines = [
        "=" * 70,
        "RB1-Loss Pan-Cancer Dependency Atlas - Phase 3: Genome-Wide Screen",
        "=" * 70,
        "",
        f"Total tests: {len(all_results)}",
        f"Gained dependencies (FDR<0.05, |d|>0.3): {len(gained)}",
        f"Lost dependencies (FDR<0.05, |d|>0.3): {len(lost)}",
        "",
        "TOP GAINED DEPENDENCIES IN RB1-LOSS",
        "-" * 60,
    ]
    for _, row in gained.head(30).iterrows():
        label = ""
        if row["gene"] in KNOWN_RB1_SL:
            label = " [known SL]"
        elif row["gene"] in ALL_PATHWAY_GENES:
            label = " [pathway]"
        summary_lines.append(
            f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
            f"FDR={row['fdr']:.3e}{label}"
        )
    summary_lines += [
        "",
        "PATHWAY ENRICHMENT (pan-cancer gained)",
        "-" * 60,
    ]
    for _, row in enrich_df.iterrows():
        summary_lines.append(
            f"  {row['pathway']}: obs={row['observed']}, fold={row['fold_enrichment']}x, "
            f"p={row['fisher_p']:.3e}"
        )

    if len(comparison) > 0:
        summary_lines += [
            "",
            "CDKN2A ATLAS COMPARISON (pan-cancer)",
            "-" * 60,
            f"  Shared: {n_shared}, RB1-unique: {n_rb1_unique}, CDKN2A-unique: {n_cdkn2a_unique}",
        ]

    summary_lines += [
        "",
        f"PRIORITY TARGETS: {len(priority_df)}",
        "-" * 60,
    ]
    if len(priority_df) > 0:
        for _, row in priority_df.head(20).iterrows():
            label = " [known SL]" if row["known_rb1_sl"] else (" [pathway]" if row["pathway_gene"] else "")
            d_str = f"d={row['pancancer_d']:.3f}" if pd.notna(row["pancancer_d"]) else "d=N/A (per-type only)"
            summary_lines.append(
                f"  {row['gene']}: {d_str}, in {row['n_cancer_types']} types{label}"
            )

    summary_lines.append("")

    with open(OUTPUT_DIR / "genomewide_screen_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    print("\nDone.")


if __name__ == "__main__":
    main()
