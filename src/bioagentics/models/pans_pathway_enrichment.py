"""Pathway enrichment analysis for PANS ultra-rare variant genes.

Runs over-representation analysis (ORA) against KEGG, Reactome, and Gene Ontology
using gseapy/Enrichr. Includes focused testing of immune-relevant pathways
(TLR signaling, complement, Th17, cGAS-STING/type I IFN).

Usage:
    uv run python -m bioagentics.models.pans_pathway_enrichment [--dest DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/pandas_pans/pans-genetic-variant-pathway-analysis")

# Enrichr gene-set libraries to query
ENRICHR_LIBRARIES = [
    "KEGG_2021_Human",
    "Reactome_2022",
    "GO_Biological_Process_2023",
    "GO_Molecular_Function_2023",
    "GO_Cellular_Component_2023",
    "MSigDB_Hallmark_2020",
]

# Immune-focused pathways to specifically test
IMMUNE_FOCUS_TERMS = [
    "Toll-like receptor signaling",
    "Complement",
    "Th17 cell differentiation",
    "T cell activation",
    "Type I interferon",
    "Interferon alpha",
    "Interferon gamma",
    "cGAS-STING",
    "Cytosolic DNA sensing",
    "NF-kappa B signaling",
    "TNF signaling",
    "IL-17 signaling",
    "Autoimmune",
    "Blood-brain barrier",
    "Antigen processing and presentation",
    "NOD-like receptor signaling",
    "DNA damage",
    "DNA repair",
    "Mitophagy",
    "mTOR signaling",
]


def run_enrichment(gene_list: list[str],
                   libraries: list[str] | None = None) -> pd.DataFrame:
    """Run over-representation analysis via Enrichr.

    Args:
        gene_list: List of human gene symbols.
        libraries: Enrichr libraries to query (default: ENRICHR_LIBRARIES).

    Returns:
        DataFrame with columns: library, term, overlap, p_value, adj_p_value,
        genes, combined_score.
    """
    import gseapy

    if libraries is None:
        libraries = ENRICHR_LIBRARIES

    logger.info("Running Enrichr ORA with %d genes against %d libraries",
                len(gene_list), len(libraries))

    all_results = []

    for lib in libraries:
        try:
            enr = gseapy.enrich(
                gene_list=gene_list,
                gene_sets=lib,
                outdir=None,
                no_plot=True,
                verbose=False,
            )
            if enr.results is not None and not enr.results.empty:
                df = enr.results.copy()
                df["library"] = lib
                all_results.append(df)
                n_sig = (df["Adjusted P-value"] < 0.05).sum()
                logger.info("  %s: %d terms, %d significant (FDR<0.05)",
                            lib, len(df), n_sig)
        except Exception as e:
            logger.warning("  %s failed: %s", lib, e)

    if not all_results:
        logger.warning("No enrichment results obtained")
        return pd.DataFrame()

    combined = pd.concat(all_results, ignore_index=True)

    # Standardize column names
    rename_map = {
        "Term": "term",
        "Overlap": "overlap",
        "P-value": "p_value",
        "Adjusted P-value": "adj_p_value",
        "Genes": "genes",
        "Combined Score": "combined_score",
    }
    combined = combined.rename(columns=rename_map)

    # Keep relevant columns
    keep_cols = ["library", "term", "overlap", "p_value", "adj_p_value",
                 "genes", "combined_score"]
    available = [c for c in keep_cols if c in combined.columns]
    combined = combined[available]

    return combined.sort_values("p_value").reset_index(drop=True)


def filter_immune_pathways(enrichment_df: pd.DataFrame) -> pd.DataFrame:
    """Filter enrichment results for immune-focused pathways."""
    if enrichment_df.empty or "term" not in enrichment_df.columns:
        return pd.DataFrame()

    mask = enrichment_df["term"].str.lower().apply(
        lambda t: any(kw.lower() in t for kw in IMMUNE_FOCUS_TERMS)
    )
    immune_df = enrichment_df[mask].copy()
    logger.info("Immune-focused pathways: %d terms (%d significant FDR<0.05)",
                len(immune_df),
                (immune_df["adj_p_value"] < 0.05).sum() if "adj_p_value" in immune_df.columns else 0)
    return immune_df


def plot_enrichment_dotplot(enrichment_df: pd.DataFrame, dest: Path,
                            top_n: int = 20,
                            title: str = "PANS Variant Gene Pathway Enrichment") -> None:
    """Generate dot plot of top enriched pathways."""
    if enrichment_df.empty:
        logger.warning("No data to plot")
        return

    df = enrichment_df.head(top_n).copy()
    if "adj_p_value" not in df.columns:
        return

    df["-log10(padj)"] = -df["adj_p_value"].clip(lower=1e-50).apply(
        lambda x: __import__("numpy").log10(x)
    )

    # Parse gene count from overlap (e.g., "3/150")
    if "overlap" in df.columns:
        df["gene_count"] = df["overlap"].astype(str).apply(
            lambda x: int(x.split("/")[0]) if "/" in str(x) else 1
        )
    else:
        df["gene_count"] = 1

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))

    scatter = ax.scatter(
        df["-log10(padj)"],
        range(len(df)),
        s=df["gene_count"] * 50,
        c=df["-log10(padj)"],
        cmap="RdYlBu_r",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.5,
    )

    ax.set_yticks(range(len(df)))
    # Truncate long term names
    labels = [t[:60] + "..." if len(str(t)) > 60 else str(t) for t in df["term"]]
    ax.set_yticklabels(labels)
    ax.set_xlabel("-log10(adjusted p-value)")
    ax.set_title(title)
    ax.axvline(x=-__import__("numpy").log10(0.05), color="red",
               linestyle="--", alpha=0.5, label="FDR = 0.05")
    ax.legend()

    plt.colorbar(scatter, label="-log10(padj)", shrink=0.6)
    plt.tight_layout()

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved enrichment dot plot: %s", dest)


def run_pans_pathway_enrichment(dest_dir: Path | None = None) -> pd.DataFrame:
    """Run the full PANS pathway enrichment analysis pipeline.

    Returns combined enrichment results DataFrame.
    """
    from bioagentics.data.pans_variants import get_pans_gene_symbols

    if dest_dir is None:
        dest_dir = OUTPUT_DIR

    dest_dir.mkdir(parents=True, exist_ok=True)

    gene_symbols = get_pans_gene_symbols()
    logger.info("PANS variant genes: %s", ", ".join(gene_symbols))

    # Run ORA
    enrichment_df = run_enrichment(gene_symbols)

    if enrichment_df.empty:
        logger.warning("No enrichment results — check network connectivity")
        return enrichment_df

    # Save full results
    enrichment_path = dest_dir / "enrichment_results.csv"
    enrichment_df.to_csv(enrichment_path, index=False)
    logger.info("Saved full enrichment results: %s", enrichment_path)

    # Filter immune-focused pathways
    immune_df = filter_immune_pathways(enrichment_df)
    if not immune_df.empty:
        immune_path = dest_dir / "enrichment_immune_focused.csv"
        immune_df.to_csv(immune_path, index=False)
        logger.info("Saved immune-focused results: %s", immune_path)

    # Generate visualization
    sig_df = enrichment_df[enrichment_df.get("adj_p_value", pd.Series()) < 0.05]
    plot_df = sig_df if len(sig_df) >= 5 else enrichment_df
    plot_enrichment_dotplot(plot_df, dest_dir / "enrichment_dotplot.png")

    # Summary statistics
    n_sig = (enrichment_df["adj_p_value"] < 0.05).sum() if "adj_p_value" in enrichment_df.columns else 0
    logger.info("Summary: %d total terms, %d significant (FDR<0.05)",
                len(enrichment_df), n_sig)

    return enrichment_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PANS variant gene pathway enrichment analysis"
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    enrichment_df = run_pans_pathway_enrichment(dest_dir=args.dest)

    if not enrichment_df.empty:
        print(f"\nEnrichment results: {len(enrichment_df)} terms")
        n_sig = (enrichment_df["adj_p_value"] < 0.05).sum() if "adj_p_value" in enrichment_df.columns else 0
        print(f"Significant (FDR < 0.05): {n_sig}")

        # Show top 10
        print("\nTop 10 enriched terms:")
        for _, row in enrichment_df.head(10).iterrows():
            term = row.get("term", "")
            pval = row.get("adj_p_value", 1.0)
            lib = row.get("library", "")
            print(f"  {term} ({lib}) — FDR={pval:.2e}")


if __name__ == "__main__":
    main()
