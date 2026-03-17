"""Differential expression analysis for anti-TNF response prediction.

Identifies genes differentially expressed between responders and non-responders
using linear models with study as covariate (limma-style approach in Python).
Generates volcano/MA plots and runs GSEA pathway enrichment.

Usage:
    uv run python -m bioagentics.data.anti_tnf.differential_expression
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
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_INPUT = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "batch_correction"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "differential_expression"


def load_batch_corrected(
    input_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load batch-corrected expression matrix and metadata."""
    expr = pd.read_csv(input_dir / "expression_combat.csv", index_col=0)
    metadata = pd.read_csv(input_dir / "metadata.csv")
    return expr, metadata


def run_de_analysis(
    expr: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Run limma-style differential expression: responder vs non-responder.

    Uses OLS with response_status as main effect and study as covariate.
    Returns DE results sorted by adjusted p-value.
    """
    # Build design matrix
    meta = metadata.set_index("sample_id").loc[expr.columns]
    response = (meta["response_status"] == "non_responder").astype(float).values
    study_dummies = pd.get_dummies(meta["study"], drop_first=True, dtype=float).values

    # Design: intercept + response + study covariates
    X = np.column_stack([np.ones(len(response)), response, study_dummies])

    results = []
    for gene in expr.index:
        y = expr.loc[gene].values.astype(float)

        # Skip genes with zero variance
        if np.std(y) == 0:
            continue

        try:
            model = sm.OLS(y, X).fit()
            # Coefficient for response (index 1): positive = higher in non-responders
            logfc = model.params[1]
            pval = model.pvalues[1]
            tstat = model.tvalues[1]
            results.append({
                "gene": gene,
                "logFC": logfc,
                "t_statistic": tstat,
                "p_value": pval,
                "mean_expr": np.mean(y),
            })
        except Exception:
            continue

    de = pd.DataFrame(results)

    # Multiple testing correction (BH)
    _, adj_pvals, _, _ = multipletests(de["p_value"], method="fdr_bh")
    de["adj_p_value"] = adj_pvals

    de = de.sort_values("adj_p_value").reset_index(drop=True)

    logger.info(
        "DE analysis: %d genes tested, %d significant (adj.p < 0.05), %d at adj.p < 0.01",
        len(de),
        (de["adj_p_value"] < 0.05).sum(),
        (de["adj_p_value"] < 0.01).sum(),
    )
    return de


def plot_volcano(de: pd.DataFrame, output_dir: Path) -> None:
    """Generate volcano plot: logFC vs -log10(adj.p.value)."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sig = de["adj_p_value"] < 0.05
    up = sig & (de["logFC"] > 0)
    down = sig & (de["logFC"] < 0)
    ns = ~sig

    ax.scatter(de.loc[ns, "logFC"], -np.log10(de.loc[ns, "p_value"]),
               c="gray", alpha=0.3, s=8, label=f"NS ({ns.sum()})")
    ax.scatter(de.loc[up, "logFC"], -np.log10(de.loc[up, "p_value"]),
               c="#d62728", alpha=0.5, s=12, label=f"Up in NR ({up.sum()})")
    ax.scatter(de.loc[down, "logFC"], -np.log10(de.loc[down, "p_value"]),
               c="#1f77b4", alpha=0.5, s=12, label=f"Up in R ({down.sum()})")

    # Label top genes
    top = de.head(15)
    for _, row in top.iterrows():
        ax.annotate(row["gene"], (row["logFC"], -np.log10(row["p_value"])),
                     fontsize=7, alpha=0.8,
                     xytext=(5, 5), textcoords="offset points")

    ax.axhline(-np.log10(0.05), ls="--", c="gray", alpha=0.5)
    ax.set_xlabel("log2 Fold Change (NR vs R)")
    ax.set_ylabel("-log10(p-value)")
    ax.set_title("Differential Expression: Non-Responders vs Responders")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "volcano_plot.png", dpi=150)
    plt.close(fig)
    logger.info("Saved volcano plot")


def plot_ma(de: pd.DataFrame, output_dir: Path) -> None:
    """Generate MA plot: mean expression vs logFC."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sig = de["adj_p_value"] < 0.05

    ax.scatter(de.loc[~sig, "mean_expr"], de.loc[~sig, "logFC"],
               c="gray", alpha=0.3, s=8, label=f"NS ({(~sig).sum()})")
    ax.scatter(de.loc[sig, "mean_expr"], de.loc[sig, "logFC"],
               c="#d62728", alpha=0.5, s=12, label=f"Significant ({sig.sum()})")

    ax.axhline(0, ls="--", c="black", alpha=0.5)
    ax.set_xlabel("Mean Expression")
    ax.set_ylabel("log2 Fold Change (NR vs R)")
    ax.set_title("MA Plot: Anti-TNF Response")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_dir / "ma_plot.png", dpi=150)
    plt.close(fig)
    logger.info("Saved MA plot")


def run_gsea(de: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    """Run GSEA using pre-ranked gene list against MSigDB gene sets.

    Returns combined GSEA results from Hallmark and Reactome collections.
    """
    import gseapy

    # Create ranked gene list (by t-statistic for GSEA)
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

    # Save
    gsea_results.to_csv(output_dir / "gsea_results.csv", index=False)
    logger.info("GSEA complete: %d total pathways", len(gsea_results))

    # Report key pathways
    key_terms = ["TNF", "Th17", "innate", "fibr", "oncostatin", "OSMR", "inflam", "interferon"]
    for _, row in gsea_results.iterrows():
        term_lower = str(row.get("Term", row.get("Name", ""))).lower()
        if any(k.lower() in term_lower for k in key_terms):
            nes = row.get("NES", "?")
            pval = row.get("FDR q-val", row.get("FWER p-val", "?"))
            logger.info("  Key pathway: %s (NES=%.2f, FDR=%.4f)", row.get("Term", row.get("Name", "")), float(nes), float(pval))

    return gsea_results


def check_expected_genes(de: pd.DataFrame) -> None:
    """Check if known response genes appear in top DE results."""
    expected = ["TREM1", "OSMR", "IL13RA2"]
    top200 = set(de.head(200)["gene"])
    for gene in expected:
        rank = de[de["gene"] == gene].index
        if len(rank) > 0:
            r = rank[0] + 1
            sig = de.loc[rank[0], "adj_p_value"] < 0.05
            logger.info(
                "  Expected gene %s: rank %d, adj.p=%.4f %s",
                gene, r, de.loc[rank[0], "adj_p_value"],
                "(significant)" if sig else "(not significant)",
            )
        else:
            logger.warning("  Expected gene %s: NOT FOUND in DE results", gene)


def run_de_pipeline(
    input_dir: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full differential expression pipeline.

    Returns (de_results, gsea_results).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    expr, metadata = load_batch_corrected(input_dir)
    logger.info("Loaded: %d genes x %d samples", *expr.shape)

    # Run DE
    de = run_de_analysis(expr, metadata)
    de.to_csv(output_dir / "de_results.csv", index=False)
    logger.info("Saved DE results: %d genes", len(de))

    # Check expected genes
    check_expected_genes(de)

    # Generate plots
    plot_volcano(de, output_dir)
    plot_ma(de, output_dir)

    # Run GSEA
    gsea_results = run_gsea(de, output_dir)

    return de, gsea_results


def main():
    parser = argparse.ArgumentParser(
        description="Differential expression analysis for anti-TNF response"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_de_pipeline(input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
