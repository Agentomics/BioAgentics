"""Phase 3: Pathway enrichment analysis on SWI/SNF-selective metabolic dependencies.

Runs hypergeometric enrichment and ranked-gene-set analysis to determine
whether convergent SWI/SNF metabolic dependencies cluster in specific pathways.

Usage:
    uv run python -m swisnf_metabolic_convergence.04_pathway_enrichment
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from bioagentics.config import REPO_ROOT

PHASE1B_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase1b"
PHASE2_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase2"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase3"


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


def build_pathway_gene_sets(gene_list_path: Path) -> dict[str, set[str]]:
    """Build pathway → gene sets from the metabolic gene list.

    Each gene may belong to multiple pathways (semicolon-separated).
    Returns dict mapping pathway_name → set of gene symbols.
    """
    df = pd.read_csv(gene_list_path)
    pathway_genes: dict[str, set[str]] = {}
    for _, row in df.iterrows():
        gene = row["gene"]
        if pd.isna(row["pathways"]) or not row["pathways"]:
            continue
        for pathway in str(row["pathways"]).split("; "):
            pathway = pathway.strip()
            if pathway:
                pathway_genes.setdefault(pathway, set()).add(gene)
    return pathway_genes


def run_hypergeometric_enrichment(
    convergent_genes: set[str],
    pathway_gene_sets: dict[str, set[str]],
    background_genes: set[str],
) -> pd.DataFrame:
    """Run hypergeometric (Fisher's exact) enrichment test.

    For each pathway, tests whether convergent genes are over-represented.

    Parameters:
        convergent_genes: Set of convergent gene symbols from Phase 2.
        pathway_gene_sets: Dict mapping pathway name → set of genes in that pathway.
        background_genes: Full set of metabolic genes tested (1590 from Phase 1b).

    Returns DataFrame with enrichment results per pathway.
    """
    results = []
    for pathway_name, pathway_genes in sorted(pathway_gene_sets.items()):
        # Restrict pathway genes to those in background
        pathway_in_bg = pathway_genes & background_genes
        if len(pathway_in_bg) < 2:
            continue

        # Overlap: convergent genes in this pathway
        overlap = convergent_genes & pathway_in_bg
        k = len(overlap)  # successes in sample
        n = len(convergent_genes & background_genes)  # sample size
        K = len(pathway_in_bg)  # successes in population
        N = len(background_genes)  # population size

        # Hypergeometric test (1-sided: over-representation)
        pval = sp_stats.hypergeom.sf(k - 1, N, K, n) if k > 0 else 1.0

        # Fold enrichment
        expected = n * K / N if N > 0 else 0
        fold = k / expected if expected > 0 else 0.0

        results.append({
            "pathway": pathway_name,
            "pathway_size": len(pathway_in_bg),
            "convergent_in_pathway": k,
            "convergent_total": n,
            "background_total": N,
            "fold_enrichment": round(fold, 2),
            "p_value": pval,
            "genes": ", ".join(sorted(overlap)) if overlap else "",
        })

    df = pd.DataFrame(results)
    if len(df) > 0:
        df["fdr"] = fdr_correction(df["p_value"].values)
        df = df.sort_values("p_value").reset_index(drop=True)
    return df


def run_ranked_enrichment(
    arid1a_screen: pd.DataFrame,
    smarca4_screen: pd.DataFrame,
    pathway_gene_sets: dict[str, set[str]],
    background_genes: set[str],
) -> pd.DataFrame:
    """Run a ranked enrichment analysis (Mann-Whitney-based).

    For each pathway, tests whether genes in that pathway have more negative
    Cohen's d values (more essential in SWI/SNF-mutant) than genes outside
    the pathway. Uses the minimum d across cancer types per gene as the
    ranking metric.

    Runs separately for ARID1A and SMARCA4 screens, then combines.
    """
    results = []

    for screen_name, screen in [("ARID1A", arid1a_screen), ("SMARCA4", smarca4_screen)]:
        # Get minimum (most negative) Cohen's d per gene across cancer types
        gene_min_d = screen.groupby("gene")["cohens_d"].min()

        for pathway_name, pathway_genes in sorted(pathway_gene_sets.items()):
            pathway_in_bg = pathway_genes & background_genes
            if len(pathway_in_bg) < 3:
                continue

            in_pathway = gene_min_d[gene_min_d.index.isin(pathway_in_bg)]
            out_pathway = gene_min_d[~gene_min_d.index.isin(pathway_in_bg)]

            if len(in_pathway) < 3 or len(out_pathway) < 3:
                continue

            # Mann-Whitney: is the pathway shifted toward more negative d?
            stat, pval = sp_stats.mannwhitneyu(
                in_pathway.values, out_pathway.values, alternative="less",
            )

            results.append({
                "pathway": pathway_name,
                "screen": screen_name,
                "n_genes_in_pathway": len(in_pathway),
                "median_d_in_pathway": round(float(in_pathway.median()), 4),
                "median_d_outside": round(float(out_pathway.median()), 4),
                "p_value": pval,
            })

    df = pd.DataFrame(results)
    if len(df) > 0:
        # FDR within each screen
        for screen_name in ["ARID1A", "SMARCA4"]:
            mask = df["screen"] == screen_name
            if mask.sum() > 0:
                df.loc[mask, "fdr"] = fdr_correction(df.loc[mask, "p_value"].values)
        df = df.sort_values(["screen", "p_value"]).reset_index(drop=True)
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3: Pathway Enrichment Analysis ===\n")

    # Load Phase 2 convergent genes
    print("Loading Phase 2 convergent genes...")
    convergent_df = pd.read_csv(PHASE2_DIR / "convergent_metabolic_genes.csv")
    convergent_genes = set(convergent_df["gene"].tolist())
    print(f"  {len(convergent_genes)} convergent genes")

    # Build pathway gene sets from metabolic gene list
    print("Building pathway gene sets...")
    gene_list_path = PHASE1B_DIR / "metabolic_gene_list.csv"
    pathway_gene_sets = build_pathway_gene_sets(gene_list_path)
    print(f"  {len(pathway_gene_sets)} pathways")

    # Background: all metabolic genes tested in Phase 1b
    gene_list = pd.read_csv(gene_list_path)
    background_genes = set(gene_list["gene"].tolist())
    print(f"  {len(background_genes)} background genes")

    # === Hypergeometric enrichment ===
    print("\n--- Hypergeometric enrichment of convergent genes ---")
    hyper_results = run_hypergeometric_enrichment(
        convergent_genes, pathway_gene_sets, background_genes,
    )
    hyper_results.to_csv(OUTPUT_DIR / "hypergeometric_enrichment.csv", index=False)

    sig_hyper = hyper_results[hyper_results["fdr"] < 0.05]
    print(f"\n  {len(sig_hyper)} pathways enriched (FDR < 0.05):")
    for _, row in sig_hyper.iterrows():
        print(
            f"    {row['pathway']:55s} "
            f"fold={row['fold_enrichment']:.1f}x  "
            f"({row['convergent_in_pathway']}/{row['pathway_size']} genes)  "
            f"p={row['p_value']:.2e}  FDR={row['fdr']:.2e}"
        )

    # Top pathways by fold enrichment (nominal p < 0.05)
    nom_hyper = hyper_results[hyper_results["p_value"] < 0.05]
    print(f"\n  {len(nom_hyper)} pathways at nominal p < 0.05:")
    for _, row in nom_hyper.head(15).iterrows():
        print(
            f"    {row['pathway']:55s} "
            f"fold={row['fold_enrichment']:.1f}x  "
            f"({row['convergent_in_pathway']}/{row['pathway_size']} genes)  "
            f"p={row['p_value']:.2e}"
        )

    # === Ranked enrichment (GSEA-like) ===
    print("\n--- Ranked enrichment (Mann-Whitney) ---")
    arid1a_screen = pd.read_csv(PHASE1B_DIR / "screen_arid1a_vs_wt.csv")
    smarca4_screen = pd.read_csv(PHASE1B_DIR / "screen_smarca4_vs_wt.csv")

    ranked_results = run_ranked_enrichment(
        arid1a_screen, smarca4_screen, pathway_gene_sets, background_genes,
    )
    ranked_results.to_csv(OUTPUT_DIR / "ranked_enrichment.csv", index=False)

    for screen_name in ["ARID1A", "SMARCA4"]:
        screen_results = ranked_results[ranked_results["screen"] == screen_name]
        sig_ranked = screen_results[screen_results["fdr"] < 0.05]
        print(f"\n  {screen_name}: {len(sig_ranked)} pathways shifted toward SL (FDR < 0.05):")
        for _, row in sig_ranked.head(10).iterrows():
            print(
                f"    {row['pathway']:55s} "
                f"median_d={row['median_d_in_pathway']:+.3f} vs {row['median_d_outside']:+.3f}  "
                f"p={row['p_value']:.2e}  FDR={row['fdr']:.2e}"
            )

    # === Pathway concentration check ===
    print("\n--- Pathway concentration assessment ---")

    # For the success criterion: convergent genes belong to <=2 pathways
    # Check the top 2 enriched pathways and what fraction they cover
    if len(sig_hyper) > 0:
        top2 = sig_hyper.head(2)
        top2_genes: set[str] = set()
        for _, row in top2.iterrows():
            if row["genes"]:
                top2_genes.update(g.strip() for g in row["genes"].split(","))

        print(f"\n  Top 2 enriched pathways cover {len(top2_genes)}/{len(convergent_genes)} "
              f"convergent genes ({len(top2_genes)/len(convergent_genes):.0%})")
        for _, row in top2.iterrows():
            print(f"    {row['pathway']}: {row['convergent_in_pathway']} genes")
    elif len(nom_hyper) > 0:
        top2 = nom_hyper.head(2)
        top2_genes = set()
        for _, row in top2.iterrows():
            if row["genes"]:
                top2_genes.update(g.strip() for g in row["genes"].split(","))
        print(f"\n  Top 2 nominally enriched pathways cover {len(top2_genes)}/{len(convergent_genes)} "
              f"convergent genes ({len(top2_genes)/len(convergent_genes):.0%})")

    # === Summary ===
    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)
    print(f"\nHypergeometric: {len(sig_hyper)} pathways enriched at FDR < 0.05")
    print(f"Ranked (ARID1A): {len(ranked_results[(ranked_results['screen']=='ARID1A') & (ranked_results['fdr']<0.05)])} pathways shifted")
    print(f"Ranked (SMARCA4): {len(ranked_results[(ranked_results['screen']=='SMARCA4') & (ranked_results['fdr']<0.05)])} pathways shifted")

    # Dominant pathway theme
    if len(sig_hyper) > 0:
        dominant = sig_hyper.iloc[0]
        print(f"\nDominant pathway: {dominant['pathway']} "
              f"(fold={dominant['fold_enrichment']:.1f}x, FDR={dominant['fdr']:.2e})")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
