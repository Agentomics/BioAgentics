"""Cross-reference PANS variant genes with GSE102482 neuroinflammation data.

Identifies which PANS candidate genes are differentially expressed under LPS
stimulation in microglia. Highlights DDR genes (TREX1, SAMHD1) showing
differential expression as key novel findings per research_director guidance.

Usage:
    uv run python -m bioagentics.models.pans_neuroinflammation_xref [--dest DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/pandas_pans/pans-genetic-variant-pathway-analysis")

# DDR genes of particular interest for novel findings
DDR_HIGHLIGHT_GENES = {"TREX1", "SAMHD1", "ADNP", "EP300", "FANCD2", "RAD54L"}

# Classification thresholds
PADJ_THRESHOLD = 0.05
LOG2FC_UP_THRESHOLD = 0.5
LOG2FC_DOWN_THRESHOLD = -0.5


def classify_de_status(log2fc: float, padj: float) -> str:
    """Classify a gene's differential expression status."""
    if padj >= PADJ_THRESHOLD:
        return "unchanged"
    if log2fc > LOG2FC_UP_THRESHOLD:
        return "upregulated"
    if log2fc < LOG2FC_DOWN_THRESHOLD:
        return "downregulated"
    return "unchanged"


def cross_reference(pans_df: pd.DataFrame, de_df: pd.DataFrame) -> pd.DataFrame:
    """Cross-reference PANS variant genes with DE results.

    Args:
        pans_df: PANS variant genes (from get_pans_variant_genes()).
        de_df: DE results with human gene symbols (from get_neuroinflammation_de()).

    Returns:
        DataFrame with PANS genes annotated with DE status.
    """
    # Index DE results by human gene symbol for fast lookup
    de_by_gene = de_df.set_index("gene_symbol") if "gene_symbol" in de_df.columns else de_df

    records = []
    for _, row in pans_df.iterrows():
        gene = row["gene_symbol"]
        record = {
            "gene_symbol": gene,
            "pathway_axis": row["pathway_axis"],
            "variant_type": row["variant_type"],
        }

        if gene in de_by_gene.index:
            de_row = de_by_gene.loc[gene]
            if isinstance(de_row, pd.DataFrame):
                de_row = de_row.iloc[0]
            record["log2fc"] = float(de_row.get("log2fc", 0))
            record["pvalue"] = float(de_row.get("pvalue", 1))
            record["padj"] = float(de_row.get("padj", 1))
            record["mouse_symbol"] = de_row.get("mouse_symbol", "")
            record["de_status"] = classify_de_status(record["log2fc"], record["padj"])
        else:
            record["log2fc"] = np.nan
            record["pvalue"] = np.nan
            record["padj"] = np.nan
            record["mouse_symbol"] = ""
            record["de_status"] = "not_detected"

        record["is_ddr_gene"] = gene in DDR_HIGHLIGHT_GENES
        records.append(record)

    return pd.DataFrame(records)


def compute_overlap_enrichment(pans_genes: list[str],
                               de_df: pd.DataFrame,
                               padj_threshold: float = PADJ_THRESHOLD) -> dict:
    """Test whether PANS variant genes are enriched among DE genes (Fisher exact).

    Returns dict with enrichment statistics.
    """
    all_genes = set(de_df["gene_symbol"].unique())
    de_genes = set(de_df[de_df["padj"] < padj_threshold]["gene_symbol"])
    pans_set = set(pans_genes)
    pans_detected = pans_set & all_genes

    if not pans_detected or not de_genes:
        return {
            "pans_in_de": 0,
            "pans_not_in_de": len(pans_detected),
            "total_de": len(de_genes),
            "total_tested": len(all_genes),
            "odds_ratio": np.nan,
            "fisher_pvalue": 1.0,
        }

    # Contingency table:
    # PANS & DE | PANS & not-DE
    # not-PANS & DE | not-PANS & not-DE
    a = len(pans_detected & de_genes)
    b = len(pans_detected - de_genes)
    c = len(de_genes - pans_detected)
    d = len(all_genes - pans_detected - de_genes)

    odds_ratio, p_value = scipy_stats.fisher_exact([[a, b], [c, d]])

    return {
        "pans_in_de": a,
        "pans_not_in_de": b,
        "total_de": len(de_genes),
        "total_tested": len(all_genes),
        "odds_ratio": odds_ratio,
        "fisher_pvalue": p_value,
    }


def plot_heatmap(xref_df: pd.DataFrame, dest: Path) -> None:
    """Generate heatmap of PANS gene expression under LPS vs control."""
    df = xref_df.dropna(subset=["log2fc"]).copy()
    if df.empty:
        logger.warning("No data for heatmap")
        return

    df = df.sort_values(["pathway_axis", "log2fc"], ascending=[True, False])

    fig, ax = plt.subplots(figsize=(8, max(4, len(df) * 0.4)))

    # Horizontal bar chart of log2FC, colored by significance
    colors = []
    for _, row in df.iterrows():
        if row["de_status"] == "upregulated":
            colors.append("#E74C3C")
        elif row["de_status"] == "downregulated":
            colors.append("#3498DB")
        else:
            colors.append("#95A5A6")

    bars = ax.barh(range(len(df)), df["log2fc"], color=colors, edgecolor="black",
                   linewidth=0.5, alpha=0.8)

    # Mark DDR genes with bold labels
    labels = []
    for _, row in df.iterrows():
        label = row["gene_symbol"]
        if row["is_ddr_gene"]:
            label = f"* {label}"
        labels.append(label)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("log2 Fold Change (LPS vs Control)")
    ax.set_title("PANS Variant Gene Expression in Neuroinflammation (GSE102482)")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=LOG2FC_UP_THRESHOLD, color="red", linestyle="--", alpha=0.3)
    ax.axvline(x=LOG2FC_DOWN_THRESHOLD, color="blue", linestyle="--", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#E74C3C", label="Upregulated (FDR<0.05)"),
        Patch(facecolor="#3498DB", label="Downregulated (FDR<0.05)"),
        Patch(facecolor="#95A5A6", label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    plt.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap: %s", dest)


def run_neuroinflammation_xref(dest_dir: Path | None = None,
                                force: bool = False) -> pd.DataFrame:
    """Run the full neuroinflammation cross-reference analysis.

    Returns the cross-reference DataFrame.
    """
    from bioagentics.data.pans_geo_expression import get_neuroinflammation_de
    from bioagentics.data.pans_variants import get_pans_gene_symbols, get_pans_variant_genes

    if dest_dir is None:
        dest_dir = OUTPUT_DIR

    dest_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path("data/pandas_pans/pans-genetic-variant-pathway-analysis")

    # Load PANS genes
    pans_df = get_pans_variant_genes()
    pans_symbols = get_pans_gene_symbols()

    # Load DE results
    de_df = get_neuroinflammation_de(dest_dir=data_dir, force=force)

    # Cross-reference
    xref_df = cross_reference(pans_df, de_df)

    # Save results
    xref_path = dest_dir / "neuroinflammation_xref.csv"
    xref_df.to_csv(xref_path, index=False)
    logger.info("Saved cross-reference: %s", xref_path)

    # Overlap enrichment
    enrichment = compute_overlap_enrichment(pans_symbols, de_df)
    logger.info("Overlap enrichment: %d/%d PANS genes are DE, "
                "OR=%.2f, p=%.4f",
                enrichment["pans_in_de"],
                enrichment["pans_in_de"] + enrichment["pans_not_in_de"],
                enrichment.get("odds_ratio", 0),
                enrichment["fisher_pvalue"])

    # Save enrichment stats
    enrich_df = pd.DataFrame([enrichment])
    enrich_df.to_csv(dest_dir / "neuroinflammation_overlap_enrichment.csv", index=False)

    # Highlight DDR genes
    ddr_de = xref_df[(xref_df["is_ddr_gene"]) & (xref_df["de_status"] != "not_detected")]
    if not ddr_de.empty:
        logger.info("DDR genes with DE data: %s",
                    ", ".join(f"{r['gene_symbol']}({r['de_status']})"
                              for _, r in ddr_de.iterrows()))

    # Plot
    plot_heatmap(xref_df, dest_dir / "neuroinflammation_heatmap.png")

    return xref_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="PANS neuroinflammation cross-reference analysis"
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download of GEO data")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    xref_df = run_neuroinflammation_xref(dest_dir=args.dest, force=args.force)

    print(f"\nCross-reference results: {len(xref_df)} PANS genes")
    for status in ["upregulated", "downregulated", "unchanged", "not_detected"]:
        count = (xref_df["de_status"] == status).sum()
        print(f"  {status}: {count}")

    ddr = xref_df[xref_df["is_ddr_gene"]]
    if not ddr.empty:
        print("\nDDR highlight genes:")
        for _, row in ddr.iterrows():
            fc = f"log2FC={row['log2fc']:.2f}" if not np.isnan(row["log2fc"]) else "not detected"
            print(f"  {row['gene_symbol']}: {row['de_status']} ({fc})")


if __name__ == "__main__":
    main()
