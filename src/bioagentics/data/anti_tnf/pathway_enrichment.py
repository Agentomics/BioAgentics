"""Pathway enrichment analysis for anti-TNF response DE results.

Runs GSEA with Hallmark and Reactome gene sets, maps findings to known
anti-TNF response biology (TNF signaling, Th17, innate immunity, OSMR,
fibrosis/ECM), and generates enrichment plots.

Usage:
    uv run python -m bioagentics.data.anti_tnf.pathway_enrichment
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

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_DE_DIR = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "differential_expression"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "pathway_enrichment"

# Curated gene sets for anti-TNF response biology mapping
ANTI_TNF_BIOLOGY = {
    "TNF_signaling": {
        "genes": [
            "TNF", "TNFRSF1A", "TNFRSF1B", "TRADD", "TRAF2", "TRAF5",
            "RIPK1", "NFKB1", "NFKB2", "RELA", "RELB", "BIRC2", "BIRC3",
            "MAP3K7", "IKBKB", "IKBKG", "CHUK",
        ],
        "description": "Core TNF-NFkB signaling cascade",
    },
    "Th17_differentiation": {
        "genes": [
            "RORC", "IL17A", "IL17F", "IL23R", "IL6ST", "STAT3", "CCR6",
            "IL22", "IL21", "BATF",
        ],
        "description": "Th17 cell differentiation and IL-23/IL-17 axis",
    },
    "Innate_immunity_TREM1": {
        "genes": [
            "TREM1", "S100A8", "S100A9", "S100A12", "LCN2", "CXCL1",
            "CXCL2", "CXCL3", "CXCL5", "CXCL8", "IL1B", "IL6",
        ],
        "description": "Innate immune activation / TREM1 signaling",
    },
    "OSMR_oncostatin_M": {
        "genes": [
            "OSMR", "OSM", "IL6ST", "IL13RA2", "LIFR", "JAK1", "STAT3",
        ],
        "description": "Oncostatin M / OSMR signaling (non-response marker)",
    },
    "Fibrosis_ECM": {
        "genes": [
            "MMP2", "MMP9", "COL1A1", "COL3A1", "CD81", "FAP", "ACTA2",
            "FN1", "TGFB1", "CTGF",
        ],
        "description": "Fibrosis and ECM remodeling",
    },
    "JNK_MAPK_pathway": {
        "genes": [
            "MAPK8", "MAPK9", "MAPK14", "MAP2K4", "MAP2K7", "JUN",
            "FOS", "ATF2",
        ],
        "description": "JNK/MAPK signaling (pathogenic Th17 driver)",
    },
    "NAMPT_NAD_salvage": {
        "genes": [
            "NAMPT", "NMNAT1", "NMNAT2", "NAPRT", "IDO1", "KMO", "TDO2",
        ],
        "description": "NAD+ salvage pathway (F. nucleatum resistance mechanism)",
    },
}


def load_de_results(de_dir: Path) -> pd.DataFrame:
    """Load differential expression results."""
    return pd.read_csv(de_dir / "de_results.csv")


def run_gsea(de: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Run GSEA using pre-ranked gene list against MSigDB collections.

    Returns combined GSEA results from Hallmark and Reactome.
    """
    import gseapy

    ranking = de.set_index("gene")["t_statistic"].dropna().sort_values(ascending=False)

    gene_sets = ["MSigDB_Hallmark_2020", "Reactome_2022"]
    all_results = []

    for gs_name in gene_sets:
        try:
            res = gseapy.prerank(
                rnk=ranking,
                gene_sets=gs_name,
                outdir=None,
                min_size=10,
                max_size=500,
                permutation_num=1000,
                seed=42,
                verbose=False,
            )
            df = res.res2d.copy()
            df["gene_set_collection"] = gs_name
            all_results.append(df)
            logger.info("GSEA %s: %d pathways tested", gs_name, len(df))
        except Exception:
            logger.exception("GSEA failed for %s", gs_name)

    if not all_results:
        logger.warning("No GSEA results generated")
        return pd.DataFrame()

    gsea_results = pd.concat(all_results, ignore_index=True)
    gsea_results = gsea_results.sort_values("NES", key=abs, ascending=False)
    gsea_results.to_csv(output_dir / "gsea_results.csv", index=False)
    return gsea_results


def map_to_anti_tnf_biology(de: pd.DataFrame) -> pd.DataFrame:
    """Map DE results to curated anti-TNF response biology gene sets.

    Returns per-pathway summary with enrichment statistics.
    """
    de_idx = de.set_index("gene")
    all_de_genes = set(de_idx.index)
    sig_genes = set(de[de["adj_p_value"] < 0.05]["gene"])

    records = []
    for pathway_name, info in ANTI_TNF_BIOLOGY.items():
        pathway_genes = set(info["genes"])
        present = pathway_genes & all_de_genes
        significant = pathway_genes & sig_genes

        # Get stats for present genes
        gene_stats = []
        for g in sorted(present):
            row = de_idx.loc[g]
            gene_stats.append({
                "gene": g,
                "logFC": row["logFC"],
                "adj_p_value": row["adj_p_value"],
                "significant": g in significant,
            })

        mean_logfc = np.mean([gs["logFC"] for gs in gene_stats]) if gene_stats else np.nan
        direction = "up_in_NR" if mean_logfc > 0 else "up_in_R" if mean_logfc < 0 else "mixed"

        records.append({
            "pathway": pathway_name,
            "description": info["description"],
            "total_genes": len(pathway_genes),
            "present_in_data": len(present),
            "significant_de": len(significant),
            "mean_logFC": mean_logfc,
            "direction": direction,
            "significant_genes": ", ".join(sorted(significant)) if significant else "none",
        })

    return pd.DataFrame(records)


def plot_biology_mapping(biology_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot anti-TNF biology pathway enrichment summary."""
    df = biology_df.sort_values("significant_de", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if d == "up_in_NR" else "#1f77b4" if d == "up_in_R" else "#7f7f7f"
              for d in df["direction"]]

    bars = ax.barh(range(len(df)), df["significant_de"], color=colors, alpha=0.8)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["pathway"], fontsize=9)
    ax.set_xlabel("Number of Significant DE Genes")
    ax.set_title("Anti-TNF Response Biology: Pathway Activation")

    # Add gene count annotations
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["significant_de"] + 0.1, i,
                f"{row['significant_de']}/{row['present_in_data']}",
                va="center", fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="Up in Non-Responders"),
        Patch(facecolor="#1f77b4", label="Up in Responders"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_dir / "biology_mapping.png", dpi=150)
    plt.close(fig)
    logger.info("Saved biology mapping plot")


def plot_gsea_barplot(gsea_results: pd.DataFrame, output_dir: Path, top_n: int = 20) -> None:
    """Plot top enriched pathways from GSEA as a bar plot."""
    if gsea_results.empty:
        return

    # Get NES column
    nes_col = "NES"
    fdr_col = None
    for c in ["FDR q-val", "fdr", "NOM p-val"]:
        if c in gsea_results.columns:
            fdr_col = c
            break

    top = gsea_results.head(top_n).copy()
    term_col = "Term" if "Term" in top.columns else "Name"

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top[nes_col]]

    ax.barh(range(len(top)), top[nes_col].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(top)))

    # Truncate long names
    labels = [str(t)[:50] for t in top[term_col]]
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Normalized Enrichment Score (NES)")
    ax.set_title(f"Top {top_n} Enriched Pathways (GSEA)")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_dir / "gsea_barplot.png", dpi=150)
    plt.close(fig)
    logger.info("Saved GSEA barplot")


def run_pathway_enrichment(
    de_dir: Path = DEFAULT_DE_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full pathway enrichment pipeline.

    Returns (gsea_results, biology_mapping).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    de = load_de_results(de_dir)
    logger.info("Loaded DE results: %d genes", len(de))

    # GSEA
    gsea_results = run_gsea(de, output_dir)
    if not gsea_results.empty:
        logger.info("GSEA: %d pathways", len(gsea_results))
        plot_gsea_barplot(gsea_results, output_dir)

    # Biology mapping
    biology = map_to_anti_tnf_biology(de)
    biology.to_csv(output_dir / "biology_mapping.csv", index=False)
    plot_biology_mapping(biology, output_dir)

    for _, row in biology.iterrows():
        if row["significant_de"] > 0:
            logger.info(
                "  %s: %d sig DE genes (%s) — %s",
                row["pathway"], row["significant_de"],
                row["significant_genes"], row["direction"],
            )

    return gsea_results, biology


def main():
    parser = argparse.ArgumentParser(
        description="Pathway enrichment analysis for anti-TNF response"
    )
    parser.add_argument("--de-dir", type=Path, default=DEFAULT_DE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_pathway_enrichment(de_dir=args.de_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
