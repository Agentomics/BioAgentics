"""Pathway enrichment analysis (GSEA + ORA) for classifier-selected genes.

Implements Gene Set Enrichment Analysis and over-representation analysis
on classifier-selected genes using gseapy. Compares enriched pathways
with Cunningham Panel gene pathways.

Usage:
    uv run python -m bioagentics.models.pathway_enrichment ranked_genes.csv \\
        --dest output/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_sets import (
    CUNNINGHAM_PANEL_GENES,
    get_curated_gene_sets,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "transcriptomic-biomarker-panel"


def run_gsea_prerank(
    ranked_genes: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    min_size: int = 5,
    max_size: int = 500,
    permutations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run GSEA prerank analysis on a ranked gene list.

    Parameters
    ----------
    ranked_genes : pd.DataFrame
        Must have columns 'gene' and 'rank_metric' (e.g. -log10(p)*sign(logFC)).
    gene_sets : dict[str, list[str]]
        Gene sets to test.
    min_size : int
        Minimum gene set size.
    max_size : int
        Maximum gene set size.
    permutations : int
        Number of permutations.
    seed : int
        Random seed.

    Returns
    -------
    DataFrame with columns: term, es, nes, pval, fdr, geneset_size, matched_size, genes.
    """
    import gseapy

    # Build ranking Series
    rnk = ranked_genes.set_index("gene")["rank_metric"]
    rnk = rnk[~rnk.index.duplicated(keep="first")]
    rnk = rnk.dropna().sort_values(ascending=False)

    logger.info("Running GSEA prerank: %d genes, %d gene sets", len(rnk), len(gene_sets))

    pre_res = gseapy.prerank(
        rnk=rnk,
        gene_sets=gene_sets,
        min_size=min_size,
        max_size=max_size,
        permutation_num=permutations,
        seed=seed,
        no_plot=True,
        verbose=False,
    )

    results = pre_res.res2d.copy()
    results = results.rename(columns={
        "Term": "term",
        "ES": "es",
        "NES": "nes",
        "NOM p-val": "pval",
        "FDR q-val": "fdr",
        "FWER p-val": "fwer",
        "Gene %": "gene_pct",
        "Lead_genes": "genes",
    })

    # Normalize column names (gseapy versions vary)
    col_map = {}
    for c in results.columns:
        cl = c.lower().replace(" ", "_").replace("-", "_")
        if cl not in col_map.values():
            col_map[c] = cl
    results = results.rename(columns=col_map)

    results = results.sort_values("fdr").reset_index(drop=True)
    n_sig = (results["fdr"] < 0.05).sum()
    logger.info("GSEA: %d terms tested, %d significant (FDR < 0.05)", len(results), n_sig)

    return results


def run_ora(
    selected_genes: list[str],
    gene_sets: dict[str, list[str]],
    background_genes: list[str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run over-representation analysis using Fisher's exact test.

    Parameters
    ----------
    selected_genes : list[str]
        Genes selected by the classifier or DE analysis.
    gene_sets : dict[str, list[str]]
        Gene sets to test.
    background_genes : list[str], optional
        Background gene universe. If None, uses union of all gene sets + selected genes.
    alpha : float
        Significance threshold.

    Returns
    -------
    DataFrame with columns: term, overlap, term_size, fold_enrichment, pvalue, fdr, genes.
    """
    selected = set(selected_genes)

    if background_genes is not None:
        universe = set(background_genes) | selected
    else:
        all_gs_genes: set[str] = set()
        for gs in gene_sets.values():
            all_gs_genes.update(gs)
        universe = all_gs_genes | selected

    N = len(universe)
    n_selected = len(selected & universe)

    logger.info("ORA: %d selected genes, %d background, %d gene sets",
                n_selected, N, len(gene_sets))

    rows = []
    for term, gs_genes in gene_sets.items():
        gs = set(gs_genes) & universe
        K = len(gs)
        if K < 2:
            continue

        overlap_genes = sorted(selected & gs)
        k = len(overlap_genes)

        # Fisher's exact test (one-sided, enrichment)
        # Contingency table:
        #              in_set    not_in_set
        # selected     k         n_selected-k
        # not_selected K-k       N-K-(n_selected-k)
        table = [
            [k, n_selected - k],
            [K - k, N - K - (n_selected - k)],
        ]
        _, pvalue = stats.fisher_exact(table, alternative="greater")

        expected = n_selected * K / N if N > 0 else 0
        fold = k / expected if expected > 0 else 0

        rows.append({
            "term": term,
            "overlap": k,
            "term_size": K,
            "fold_enrichment": round(fold, 2),
            "pvalue": float(pvalue),
            "genes": ";".join(overlap_genes),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # BH correction
    df = df.sort_values("pvalue").reset_index(drop=True)
    n_tests = len(df)
    df["fdr"] = np.minimum(
        df["pvalue"] * n_tests / (np.arange(1, n_tests + 1)),
        1.0,
    )
    # Enforce monotonicity
    df["fdr"] = df["fdr"][::-1].cummin()[::-1]

    n_sig = (df["fdr"] < alpha).sum()
    logger.info("ORA: %d terms tested, %d significant (FDR < %.2f)", n_tests, n_sig, alpha)

    return df


def compare_with_cunningham(
    enriched_pathways: pd.DataFrame,
    selected_genes: list[str],
) -> pd.DataFrame:
    """Compare classifier gene signature with Cunningham Panel genes.

    Returns DataFrame showing overlap and novelty.
    """
    cunningham = set(CUNNINGHAM_PANEL_GENES)
    selected = set(selected_genes)

    overlap = sorted(cunningham & selected)
    novel = sorted(selected - cunningham)
    missing = sorted(cunningham - selected)

    rows = [
        {"category": "cunningham_panel_size", "count": len(cunningham), "genes": ";".join(sorted(cunningham))},
        {"category": "classifier_signature_size", "count": len(selected), "genes": ""},
        {"category": "overlap", "count": len(overlap), "genes": ";".join(overlap)},
        {"category": "novel_in_classifier", "count": len(novel), "genes": ";".join(novel[:50])},
        {"category": "cunningham_not_in_classifier", "count": len(missing), "genes": ";".join(missing)},
    ]

    logger.info(
        "Cunningham comparison: %d overlap, %d novel genes in classifier",
        len(overlap), len(novel),
    )
    return pd.DataFrame(rows)


def plot_enrichment_dotplot(
    enrichment_df: pd.DataFrame,
    title: str = "Pathway Enrichment",
    n_top: int = 20,
    save_path: Path | None = None,
) -> None:
    """Generate dotplot of enriched pathways.

    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Must have columns: term, fdr, and one of (nes, fold_enrichment).
    title : str
        Plot title.
    n_top : int
        Show top N pathways.
    save_path : Path, optional
        Save path for the figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if enrichment_df.empty:
        logger.warning("Empty enrichment results — skipping dotplot")
        return

    df = enrichment_df.head(n_top).copy()

    # Determine size and color columns
    if "nes" in df.columns:
        color_col = "nes"
        color_label = "NES"
    elif "fold_enrichment" in df.columns:
        color_col = "fold_enrichment"
        color_label = "Fold Enrichment"
    else:
        logger.warning("No NES or fold_enrichment column — skipping dotplot")
        return

    size_col = "overlap" if "overlap" in df.columns else None

    neg_log_fdr = -np.log10(df["fdr"].clip(lower=1e-300))

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.3)))

    if size_col:
        sizes = df[size_col].values * 10
        sizes = np.clip(sizes, 20, 300)
    else:
        sizes = 80

    scatter = ax.scatter(
        neg_log_fdr,
        range(len(df)),
        c=df[color_col].values,
        s=sizes,
        cmap="RdBu_r",
        edgecolors="black",
        linewidth=0.5,
        alpha=0.8,
    )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["term"].values, fontsize=7)
    ax.set_xlabel("-log10(FDR)")
    ax.set_title(title)
    ax.axvline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.5)
    ax.invert_yaxis()

    fig.colorbar(scatter, ax=ax, label=color_label, shrink=0.6)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved dotplot: %s", save_path)
    plt.close(fig)


def enrichment_pipeline(
    ranked_genes: pd.DataFrame | None = None,
    selected_genes: list[str] | None = None,
    background_genes: list[str] | None = None,
    gene_sets: dict[str, list[str]] | None = None,
    dest_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Full pathway enrichment pipeline: GSEA + ORA + Cunningham comparison.

    Parameters
    ----------
    ranked_genes : pd.DataFrame, optional
        Ranked gene list for GSEA (columns: gene, rank_metric).
    selected_genes : list[str], optional
        Selected genes for ORA. If None, uses GSEA leading edge genes.
    background_genes : list[str], optional
        Background for ORA.
    gene_sets : dict[str, list[str]], optional
        Custom gene sets. If None, loads curated sets.
    dest_dir : Path, optional
        Output directory.

    Returns
    -------
    Dict with keys: gsea, ora, cunningham_comparison.
    """
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    if gene_sets is None:
        gene_sets = get_curated_gene_sets()

    results: dict[str, pd.DataFrame] = {}

    # GSEA
    if ranked_genes is not None and not ranked_genes.empty:
        gsea_df = run_gsea_prerank(ranked_genes, gene_sets)
        gsea_df.to_csv(dest_dir / "gsea_results.csv", index=False)
        results["gsea"] = gsea_df

        plot_enrichment_dotplot(
            gsea_df[gsea_df["fdr"] < 0.25],
            title="GSEA Enrichment (FDR < 0.25)",
            save_path=dest_dir / "gsea_dotplot.png",
        )

        # Extract leading edge genes if no selected_genes provided
        if selected_genes is None and "genes" in gsea_df.columns:
            sig = gsea_df[gsea_df["fdr"] < 0.05]
            leading_genes: set[str] = set()
            for genes_str in sig["genes"].dropna():
                leading_genes.update(str(genes_str).split(";"))
            selected_genes = sorted(leading_genes) if leading_genes else None

    # ORA
    if selected_genes:
        ora_df = run_ora(selected_genes, gene_sets, background_genes)
        ora_df.to_csv(dest_dir / "ora_results.csv", index=False)
        results["ora"] = ora_df

        plot_enrichment_dotplot(
            ora_df[ora_df["fdr"] < 0.25],
            title="ORA Enrichment (FDR < 0.25)",
            save_path=dest_dir / "ora_dotplot.png",
        )

        # Cunningham comparison
        cunningham_df = compare_with_cunningham(ora_df, selected_genes)
        cunningham_df.to_csv(dest_dir / "cunningham_comparison.csv", index=False)
        results["cunningham_comparison"] = cunningham_df

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Pathway enrichment analysis (GSEA + ORA)"
    )
    parser.add_argument("input", type=Path, help="CSV with columns: gene, rank_metric (or gene, log2FoldChange, padj)")
    parser.add_argument("--selected-genes", type=Path, help="CSV with column: gene (for ORA)")
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ranked = pd.read_csv(args.input)
    # Build rank metric if not present
    if "rank_metric" not in ranked.columns:
        if "log2FoldChange" in ranked.columns and "padj" in ranked.columns:
            ranked["rank_metric"] = (
                -np.log10(ranked["padj"].clip(lower=1e-300)) *
                np.sign(ranked["log2FoldChange"])
            )
        else:
            raise ValueError("Input must have 'rank_metric' column or 'log2FoldChange' + 'padj'")

    selected = None
    if args.selected_genes:
        sel_df = pd.read_csv(args.selected_genes)
        selected = sel_df["gene"].tolist()

    results = enrichment_pipeline(
        ranked_genes=ranked,
        selected_genes=selected,
        dest_dir=args.dest,
    )

    for key, df in results.items():
        print(f"\n{key}: {len(df)} rows")
        if "fdr" in df.columns:
            print(f"  Significant (FDR < 0.05): {(df['fdr'] < 0.05).sum()}")


if __name__ == "__main__":
    main()
