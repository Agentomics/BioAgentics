"""Phase 3: Genome-wide CDKN2A-selective dependency screen per cancer type.

For each qualifying cancer type and pan-cancer pooled, runs differential
dependency analysis (CDKN2A-deleted vs intact) across all ~18K genes.
Primary analysis uses CDKN2A-del/RB1-intact stratum.

Includes pathway enrichment (cell cycle, RB/E2F, MDM2/p53) and volcano plots.

Usage:
    uv run python -m cdkn2a_pancancer_dependency_atlas.03_genomewide_screen
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
PHASE1_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase3"

# Significance thresholds
FDR_THRESHOLD = 0.05
EFFECT_SIZE_THRESHOLD = 0.3

MIN_SAMPLES = 3

# Pathway gene sets for enrichment
CELL_CYCLE_GENES = {
    "CDK4", "CDK6", "CDK2", "CDK1", "CCND1", "CCND2", "CCND3",
    "CCNE1", "CCNE2", "CCNA1", "CCNA2", "CCNB1", "CCNB2",
    "CDC25A", "CDC25B", "CDC25C", "CDC7", "DBF4",
}

RB_E2F_GENES = {
    "RB1", "RBL1", "RBL2", "E2F1", "E2F2", "E2F3", "E2F4", "E2F5",
    "TFDP1", "TFDP2", "SKP2", "CDKN1A", "CDKN1B",
}

MDM2_P53_GENES = {
    "MDM2", "MDM4", "TP53", "CDKN1A", "BBC3", "BAX", "PMAIP1",
    "USP7", "PPM1D", "CDKN2A",
}

CDK_SIGNALING_GENES = {
    "CDK4", "CDK6", "CDK2", "CDK1", "CDK7", "CDK9",
    "CDKN2A", "CDKN2B", "CDKN1A", "CDKN1B", "CDKN2C", "CDKN2D",
    "RB1", "CCND1", "CCNE1",
}

# Known CDKN2A SL genes from literature
KNOWN_CDKN2A_SL = {
    "CDK4", "CDK6", "PRMT5", "CDK2", "CCNE1", "E2F1", "MDM2",
}

ALL_PATHWAY_GENES = CELL_CYCLE_GENES | RB_E2F_GENES | MDM2_P53_GENES | CDK_SIGNALING_GENES


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
    del_data: pd.DataFrame,
    intact_data: pd.DataFrame,
    crispr_cols: list[str],
    context_name: str,
) -> list[dict]:
    """Run genome-wide differential dependency for one context."""
    rows = []
    pvals = []

    for gene in crispr_cols:
        del_vals = del_data[gene].dropna().values
        intact_vals = intact_data[gene].dropna().values

        if len(del_vals) < MIN_SAMPLES or len(intact_vals) < MIN_SAMPLES:
            continue

        _, pval = stats.mannwhitneyu(del_vals, intact_vals, alternative="two-sided")
        d = cohens_d(del_vals, intact_vals)

        rows.append({
            "cancer_type": context_name,
            "gene": gene,
            "cohens_d": round(d, 4),
            "p_value": float(pval),
            "n_del": len(del_vals),
            "n_intact": len(intact_vals),
            "median_dep_del": round(float(np.median(del_vals)), 4),
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
    ax.scatter(x[gained], y[gained], c="#D95319", s=15, alpha=0.8, label="Gained dep. in CDKN2A-del")
    ax.scatter(x[lost], y[lost], c="#4DBEEE", s=15, alpha=0.8, label="Lost dep. in CDKN2A-del")

    # Label top gained hits
    top = results_ct[gained].nsmallest(10, "cohens_d")
    for _, row in top.iterrows():
        ax.annotate(row["gene"], (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                     fontsize=7, ha="right")

    # Highlight pathway genes
    pathway_mask = results_ct["gene"].isin(ALL_PATHWAY_GENES) & sig
    if pathway_mask.any():
        ax.scatter(x[pathway_mask], y[pathway_mask], c="none", edgecolors="red",
                   s=40, linewidths=1.0, label="Pathway gene")

    ax.axhline(-np.log10(FDR_THRESHOLD), color="grey", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)

    ax.set_xlabel("Cohen's d (CDKN2A-del vs intact)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title(f"Genome-wide CDKN2A-Selective Screen: {context_name}")
    ax.legend(fontsize=8)

    fig.tight_layout()
    safe = context_name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    fig.savefig(out_dir / f"volcano_{safe}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3: Genome-Wide CDKN2A-Selective Dependency Screen ===\n")

    # Load Phase 1
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "cdkn2a_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()
    print(f"  {len(qualifying_types)} qualifying cancer types")

    # Load CRISPR
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    crispr_cols = list(crispr.columns)
    print(f"  {len(crispr_cols)} genes")

    # Primary analysis: CDKN2A-del/RB1-intact stratum
    # Filter: keep all intact lines + only RB1-intact deleted lines
    classified_primary = classified[
        (classified["CDKN2A_status"] == "intact") |
        ((classified["CDKN2A_status"] == "deleted") & (classified["RB1_status"] == "intact"))
    ].copy()

    merged = classified_primary.join(crispr, how="inner")

    # Screen each cancer type + pan-cancer
    all_rows = []
    contexts = qualifying_types + ["Pan-cancer (pooled)"]

    for context in contexts:
        if context == "Pan-cancer (pooled)":
            ct_data = merged
        else:
            ct_data = merged[merged["OncotreeLineage"] == context]

        del_lines = ct_data[ct_data["CDKN2A_status"] == "deleted"]
        intact_lines = ct_data[ct_data["CDKN2A_status"] == "intact"]
        print(f"  Screening {context} ({len(del_lines)} del, {len(intact_lines)} intact)...")

        rows = screen_one_context(del_lines, intact_lines, crispr_cols, context)
        all_rows.extend(rows)

    all_results = pd.DataFrame(all_rows)
    print(f"\n  Total tests: {len(all_results)}")

    # Save full results
    all_results.to_csv(OUTPUT_DIR / "genomewide_all_results.csv", index=False)

    # Save per-context hits
    for context in contexts:
        ct_res = all_results[all_results["cancer_type"] == context]
        sig = ct_res[
            (ct_res["fdr"] < FDR_THRESHOLD) & (ct_res["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
        ]
        safe = context.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
        sig.to_csv(OUTPUT_DIR / f"genomewide_hits_{safe}.csv", index=False)

    # Overall significant hits
    sig_hits = all_results[
        (all_results["fdr"] < FDR_THRESHOLD) & (all_results["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ].copy()

    gained = sig_hits[sig_hits["cohens_d"] < 0].sort_values("cohens_d")
    lost = sig_hits[sig_hits["cohens_d"] > 0].sort_values("cohens_d", ascending=False)

    print(f"  Gained dependencies (FDR<0.05, |d|>0.3): {len(gained)}")
    print(f"  Lost dependencies (FDR<0.05, |d|>0.3): {len(lost)}")

    # Top hits
    print(f"\nTop CDKN2A-selective gained dependencies:")
    for _, row in gained.head(20).iterrows():
        label = ""
        if row["gene"] in KNOWN_CDKN2A_SL:
            label = " [known CDKN2A SL]"
        elif row["gene"] in ALL_PATHWAY_GENES:
            label = " [pathway]"
        print(f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
              f"FDR={row['fdr']:.3e}{label}")

    # Pathway enrichment (pan-cancer hits)
    print("\nPathway enrichment analysis (pan-cancer hits)...")
    pancancer_sig = sig_hits[sig_hits["cancer_type"] == "Pan-cancer (pooled)"]
    pancancer_gained_genes = set(pancancer_sig[pancancer_sig["cohens_d"] < 0]["gene"])
    universe = set(all_results[all_results["cancer_type"] == "Pan-cancer (pooled)"]["gene"])

    pathways = [
        ("Cell_cycle", CELL_CYCLE_GENES),
        ("RB_E2F_signaling", RB_E2F_GENES),
        ("MDM2_p53_axis", MDM2_P53_GENES),
        ("CDK_signaling", CDK_SIGNALING_GENES),
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
    print("\nSL benchmark enrichment (known CDKN2A SL pairs)...")
    known_in_universe = KNOWN_CDKN2A_SL & universe
    sl_result = pathway_enrichment(pancancer_gained_genes, universe, "CDKN2A_known_SL", KNOWN_CDKN2A_SL)
    print(f"  Known SL in hits: {sl_result['observed']}/{len(known_in_universe)} "
          f"(fold={sl_result['fold_enrichment']}x, p={sl_result['fisher_p']:.3e})")
    pd.DataFrame([sl_result]).to_csv(OUTPUT_DIR / "sl_benchmark_enrichment.csv", index=False)

    # Volcano plots
    print("\nGenerating volcano plots...")
    for context in contexts:
        ct_res = all_results[all_results["cancer_type"] == context]
        plot_volcano(ct_res, context, OUTPUT_DIR)
    print(f"  Saved {len(contexts)} volcano plots")

    # Summary text
    summary_lines = [
        "=" * 60,
        "CDKN2A Pan-Cancer Dependency Atlas - Phase 3: Genome-Wide Screen",
        "=" * 60,
        "",
        f"Analysis: CDKN2A-del/RB1-intact vs CDKN2A-intact (primary stratum)",
        f"Total tests: {len(all_results)}",
        f"Gained dependencies (FDR<0.05, |d|>0.3): {len(gained)}",
        f"Lost dependencies (FDR<0.05, |d|>0.3): {len(lost)}",
        "",
        "TOP GAINED DEPENDENCIES IN CDKN2A-DELETED",
        "-" * 50,
    ]
    for _, row in gained.head(30).iterrows():
        label = ""
        if row["gene"] in KNOWN_CDKN2A_SL:
            label = " [known SL]"
        elif row["gene"] in ALL_PATHWAY_GENES:
            label = " [pathway]"
        summary_lines.append(
            f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
            f"FDR={row['fdr']:.3e}{label}"
        )
    summary_lines += [
        "",
        "PATHWAY ENRICHMENT",
        "-" * 50,
    ]
    for _, row in enrich_df.iterrows():
        summary_lines.append(
            f"  {row['pathway']}: obs={row['observed']}, fold={row['fold_enrichment']}x, "
            f"p={row['fisher_p']:.3e}"
        )
    summary_lines.append("")

    with open(OUTPUT_DIR / "genomewide_screen_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("  genomewide_screen_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
