"""Striatal cell-type deconvolution of TS risk gene expression.

Maps Tourette syndrome risk gene expression onto striatal cell types using
publicly available reference data. Computes cell-type specificity scores
(tau index) and tests enrichment of TS risk genes in specific cell types
using a marker-based deconvolution approach.

Strategy:
  - Phase 1 (current): Use AHBA bulk microarray expression of canonical
    cell-type marker genes to build a reference specificity profile, then
    score TS risk genes against this profile using correlation-based
    deconvolution and tau specificity index.
  - Phase 2 (future): When Wang et al. TS caudate snRNA-seq becomes
    available (NDA #290), load single-cell data for direct TS vs control
    cell-type-resolved differential expression.

Cell types profiled:
  D1 MSNs, D2 MSNs, cholinergic interneurons, PV interneurons,
  SST interneurons, astrocytes, oligodendrocytes, microglia.

Dependencies:
  - bioagentics.data.tourettes.gene_sets (TS risk genes + cell-type markers)
  - bioagentics.data.tourettes.hmba_reference (HMBA taxonomy annotations)
  - bioagentics.analysis.tourettes.ahba_spatial (AHBA expression fetching)

Usage:
    uv run python -m bioagentics.analysis.tourettes.striatal_celltypes
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.tourettes.gene_sets import (
    get_celltype_markers,
    get_gene_set,
    list_celltype_markers,
    list_gene_sets,
)
from bioagentics.data.tourettes.hmba_reference import (
    get_cstc_region_cell_types,
    get_taxonomy,
    map_to_hmba,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "cstc-circuit-expression-atlas"

# Striatal cell types for deconvolution (matching gene_sets marker panels)
STRIATAL_CELL_TYPES: list[str] = [
    "d1_msn",
    "d2_msn",
    "cholinergic_interneuron",
    "pv_interneuron",
    "sst_interneuron",
    "astrocyte",
    "oligodendrocyte",
    "microglia",
]


def build_marker_expression_matrix(
    expression_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a cell-type x gene marker expression matrix from AHBA data.

    Uses canonical marker genes for each striatal cell type and extracts
    their mean AHBA expression across striatal regions (caudate + putamen).

    Parameters
    ----------
    expression_df
        AHBA expression DataFrame from ``ahba_spatial.fetch_gene_expression``.
        Expected columns: gene_symbol, cstc_region, mean_zscore.

    Returns
    -------
    pd.DataFrame
        Rows = cell types, columns = marker genes, values = mean z-scored
        expression in striatal regions.
    """
    # Filter to striatal regions only
    striatal_regions = ["caudate", "putamen"]
    striatal = expression_df[expression_df["cstc_region"].isin(striatal_regions)]

    if striatal.empty:
        logger.warning("No striatal expression data found")
        return pd.DataFrame()

    # Average across donors and striatal regions per gene
    gene_avg: pd.Series = (
        striatal.groupby("gene_symbol")["mean_zscore"].mean()
    )
    gene_means: dict[str, float] = dict(zip(gene_avg.index, gene_avg.values))

    # Build cell-type x marker matrix
    records: list[dict] = []
    for ct_name in STRIATAL_CELL_TYPES:
        markers = get_celltype_markers(ct_name)
        row: dict = {"cell_type": ct_name}
        for gene in markers:
            row[gene] = gene_means.get(gene, np.nan)
        records.append(row)

    df = pd.DataFrame(records).set_index("cell_type")
    return df


def compute_tau_specificity(expression_vector: np.ndarray) -> float:
    """Compute the tau specificity index for a gene expression profile.

    Tau ranges from 0 (ubiquitous) to 1 (tissue/cell-type specific).
    Based on Yanai et al. (2005) Bioinformatics.

    Parameters
    ----------
    expression_vector
        Expression values across cell types (or tissues). NaN values
        are excluded from the computation.
    """
    vals = expression_vector[~np.isnan(expression_vector)]
    if len(vals) < 2:
        return np.nan
    # Shift to non-negative
    vals = vals - vals.min()
    max_val = vals.max()
    if max_val == 0:
        return 0.0
    n = len(vals)
    hat = vals / max_val
    tau = np.sum(1 - hat) / (n - 1)
    return float(tau)


def compute_gene_celltype_specificity(
    expression_df: pd.DataFrame,
    gene_symbols: list[str],
) -> pd.DataFrame:
    """Score each TS risk gene for cell-type specificity using marker correlation.

    For each risk gene, compute its expression correlation with each cell
    type's marker panel across AHBA striatal samples. High correlation
    indicates the gene's spatial expression pattern resembles that cell type.

    Parameters
    ----------
    expression_df
        AHBA expression DataFrame with columns: gene_symbol, cstc_region,
        donor_id, mean_zscore.
    gene_symbols
        TS risk gene symbols to score.

    Returns
    -------
    pd.DataFrame
        Rows = risk genes, columns = cell types, values = Pearson correlation
        with cell-type marker profile. Additional column 'tau' = specificity
        index. Column 'top_cell_type' = most correlated cell type.
    """
    # Filter to striatal regions
    striatal_regions = ["caudate", "putamen"]
    striatal = expression_df[expression_df["cstc_region"].isin(striatal_regions)]

    if striatal.empty:
        return pd.DataFrame()

    # Build per-donor per-region expression vectors for each gene
    pivot = striatal.pivot_table(
        index=["donor_id", "cstc_region"],
        columns="gene_symbol",
        values="mean_zscore",
        aggfunc="mean",
    )

    # Build marker profiles: for each cell type, average its marker genes
    ct_profiles: dict[str, pd.Series] = {}
    for ct_name in STRIATAL_CELL_TYPES:
        markers = list(get_celltype_markers(ct_name).keys())
        available = [m for m in markers if m in pivot.columns]
        if available:
            ct_profiles[ct_name] = pivot[available].mean(axis=1)

    if not ct_profiles:
        logger.warning("No cell-type marker profiles could be built from AHBA data")
        return pd.DataFrame()

    # Score each risk gene against each cell type profile
    records: list[dict] = []
    for gene in gene_symbols:
        if gene not in pivot.columns:
            continue
        gene_expr = pivot[gene].dropna()
        row: dict = {"gene_symbol": gene}
        corr_values: list[float] = []

        for ct_name, ct_profile in ct_profiles.items():
            # Align indices
            common = gene_expr.index.intersection(ct_profile.dropna().index)
            if len(common) < 3:
                row[ct_name] = np.nan
                corr_values.append(np.nan)
                continue
            corr = np.corrcoef(
                gene_expr.loc[common].values,
                ct_profile.loc[common].values,
            )[0, 1]
            row[ct_name] = float(corr)
            corr_values.append(float(corr))

        # Tau specificity of the correlation profile
        corr_arr = np.array(corr_values)
        row["tau"] = compute_tau_specificity(corr_arr)

        # Top cell type
        valid_cts = {
            k: v for k, v in row.items()
            if k in ct_profiles and not (isinstance(v, float) and np.isnan(v))
        }
        if valid_cts:
            row["top_cell_type"] = max(valid_cts, key=lambda k: valid_cts[k])
        else:
            row["top_cell_type"] = "none"

        records.append(row)

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records).set_index("gene_symbol")
    return result


def compute_celltype_enrichment(
    specificity_df: pd.DataFrame,
    n_permutations: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Test whether TS risk genes are enriched in specific cell types.

    Uses a permutation-based approach: for each cell type, compare the mean
    correlation of TS risk genes against randomly sampled gene sets of the
    same size from all available AHBA genes.

    Since we don't have the full AHBA gene universe loaded (memory
    constraints), we use a parametric z-test against the null hypothesis
    that the mean correlation is zero, plus bootstrap resampling of the
    observed correlations.

    Parameters
    ----------
    specificity_df
        Output of ``compute_gene_celltype_specificity``. Rows = genes,
        columns include cell-type correlation scores.
    n_permutations
        Number of bootstrap resamples for p-value estimation.
    seed
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Rows = cell types, columns: mean_corr, std_corr, z_score,
        p_value_bootstrap, n_genes, significant (Bonferroni-corrected).
    """
    rng = np.random.default_rng(seed)
    ct_cols = [c for c in specificity_df.columns if c in STRIATAL_CELL_TYPES]

    records: list[dict] = []
    n_tests = len(ct_cols)

    for ct in ct_cols:
        vals = specificity_df[ct].dropna().values
        n = len(vals)
        if n < 2:
            continue

        mean_corr = float(np.mean(vals))
        std_corr = float(np.std(vals, ddof=1))

        # Bootstrap: resample with replacement, count how often
        # the bootstrap mean exceeds the observed mean under H0 (mean=0)
        boot_means = np.array([
            np.mean(rng.choice(vals, size=n, replace=True))
            for _ in range(n_permutations)
        ])
        # P-value: proportion of bootstrap means <= 0 (one-sided test
        # for positive enrichment)
        p_bootstrap = float(np.mean(boot_means <= 0))

        # Z-score against null of zero mean
        se = std_corr / np.sqrt(n) if n > 0 else np.inf
        z_score = mean_corr / se if se > 0 else 0.0

        records.append({
            "cell_type": ct,
            "mean_corr": mean_corr,
            "std_corr": std_corr,
            "z_score": z_score,
            "p_value_bootstrap": p_bootstrap,
            "n_genes": n,
            "significant": p_bootstrap < (0.05 / n_tests),  # Bonferroni
        })

    return pd.DataFrame(records).set_index("cell_type")


def annotate_with_hmba(
    specificity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add HMBA taxonomy annotations to the specificity results.

    Maps the cell-type columns to canonical HMBA labels and adds metadata
    (cell class, region, description) for each cell type.

    Returns a summary DataFrame with HMBA annotations per cell type.
    """
    taxonomy = get_taxonomy()
    hmba_map = map_to_hmba(STRIATAL_CELL_TYPES)

    records: list[dict] = []
    for ct_name in STRIATAL_CELL_TYPES:
        hmba_label = hmba_map.get(ct_name)
        ct_info = taxonomy.get(hmba_label) if hmba_label else None

        # Aggregate specificity stats for this cell type
        if ct_name in specificity_df.columns:
            vals = specificity_df[ct_name].dropna()
            mean_val = float(vals.mean()) if len(vals) > 0 else np.nan
            max_val = float(vals.max()) if len(vals) > 0 else np.nan
        else:
            mean_val = np.nan
            max_val = np.nan

        record: dict = {
            "cell_type": ct_name,
            "hmba_label": hmba_label or "unmapped",
            "cell_class": ct_info.cell_class if ct_info else "unknown",
            "region": ct_info.region if ct_info else "unknown",
            "description": ct_info.description if ct_info else "",
            "mean_specificity": mean_val,
            "max_specificity": max_val,
            "n_markers": len(get_celltype_markers(ct_name)),
        }
        records.append(record)

    return pd.DataFrame(records).set_index("cell_type")


def generate_specificity_heatmap(
    specificity_df: pd.DataFrame,
    output_path: Path,
    title: str = "TS Risk Gene Cell-Type Specificity (Striatum)",
) -> Path:
    """Generate and save a heatmap of gene x cell-type specificity scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Select only cell-type columns for the heatmap
    ct_cols = [c for c in STRIATAL_CELL_TYPES if c in specificity_df.columns]
    plot_data = specificity_df[ct_cols].copy()

    # Clean column names for display
    display_cols = [c.replace("_", " ").title() for c in ct_cols]
    plot_data.columns = display_cols

    fig, ax = plt.subplots(figsize=(12, max(6, len(plot_data) * 0.4)))
    sns.heatmap(
        plot_data,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel("TS Risk Gene")
    ax.set_xlabel("Striatal Cell Type")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved specificity heatmap to %s", output_path)
    return output_path


def generate_enrichment_barplot(
    enrichment_df: pd.DataFrame,
    output_path: Path,
    title: str = "TS Risk Gene Enrichment by Cell Type",
) -> Path:
    """Generate bar plot of cell-type enrichment z-scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    cell_types = enrichment_df.index.tolist()
    z_scores = enrichment_df["z_score"].values
    significant = enrichment_df["significant"].values

    colors = ["#d32f2f" if sig else "#1976d2" for sig in significant]

    bars = ax.barh(range(len(cell_types)), z_scores, color=colors)
    ax.set_yticks(range(len(cell_types)))
    ax.set_yticklabels([ct.replace("_", " ").title() for ct in cell_types])
    ax.set_xlabel("Z-score (enrichment)")
    ax.set_title(title, fontsize=14)
    ax.axvline(x=0, color="black", linewidth=0.5)

    # Add significance markers
    for i, (z, sig) in enumerate(zip(z_scores, significant)):
        if sig:
            ax.text(z + 0.1, i, "*", ha="left", va="center", fontsize=14,
                    fontweight="bold", color="#d32f2f")

    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, fc="#d32f2f", label="Significant (Bonferroni)"),
            plt.Rectangle((0, 0), 1, 1, fc="#1976d2", label="Not significant"),
        ],
        loc="lower right",
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved enrichment barplot to %s", output_path)
    return output_path


def run_analysis(
    gene_set_name: str = "ts_combined",
    output_dir: Path = OUTPUT_DIR,
    cache: bool = True,
) -> dict:
    """Run the full striatal cell-type deconvolution analysis.

    Steps:
    1. Fetch AHBA expression for TS risk genes + all cell-type markers
    2. Build marker expression profiles per cell type
    3. Compute gene-cell type specificity correlations
    4. Test enrichment significance
    5. Annotate with HMBA taxonomy
    6. Generate figures and save results

    Returns dict with results summary and output paths.
    """
    from bioagentics.analysis.tourettes.ahba_spatial import fetch_gene_expression

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect all genes needed: TS risk genes + all cell-type markers
    logger.info("Loading gene set: %s", gene_set_name)
    risk_genes = get_gene_set(gene_set_name)
    risk_symbols = sorted(risk_genes.keys())
    logger.info("  %d TS risk genes", len(risk_symbols))

    marker_symbols: set[str] = set()
    for ct_name in STRIATAL_CELL_TYPES:
        markers = get_celltype_markers(ct_name)
        marker_symbols.update(markers.keys())
    logger.info("  %d unique cell-type marker genes", len(marker_symbols))

    all_genes = sorted(set(risk_symbols) | marker_symbols)
    logger.info("  %d total genes to query", len(all_genes))

    # 2. Fetch AHBA expression
    logger.info("Fetching AHBA expression data...")
    expr_df = fetch_gene_expression(all_genes, cache=cache)

    if expr_df.empty:
        logger.error("No expression data retrieved. Check API connectivity.")
        return {"error": "No expression data"}

    # 3. Build marker expression matrix
    logger.info("Building marker expression matrix...")
    marker_matrix = build_marker_expression_matrix(expr_df)
    if not marker_matrix.empty:
        marker_path = output_dir / "striatal_marker_expression.csv"
        marker_matrix.to_csv(marker_path)
        logger.info("Saved marker expression matrix: %s", marker_path)

    # 4. Compute gene-cell type specificity
    logger.info("Computing cell-type specificity scores...")
    specificity = compute_gene_celltype_specificity(expr_df, risk_symbols)

    if specificity.empty:
        logger.error("Could not compute specificity scores")
        return {"error": "No specificity results"}

    specificity_path = output_dir / f"striatal_specificity_{gene_set_name}.csv"
    specificity.to_csv(specificity_path)
    logger.info("Saved specificity matrix: %s", specificity_path)

    # 5. Test enrichment
    logger.info("Testing cell-type enrichment...")
    enrichment = compute_celltype_enrichment(specificity)
    enrichment_path = output_dir / f"striatal_enrichment_{gene_set_name}.csv"
    enrichment.to_csv(enrichment_path)
    logger.info("Saved enrichment results: %s", enrichment_path)

    # 6. HMBA annotation
    logger.info("Annotating with HMBA taxonomy...")
    hmba_annot = annotate_with_hmba(specificity)
    hmba_path = output_dir / f"striatal_hmba_annotation_{gene_set_name}.csv"
    hmba_annot.to_csv(hmba_path)

    # 7. Generate figures
    heatmap_path = output_dir / f"striatal_specificity_heatmap_{gene_set_name}.png"
    generate_specificity_heatmap(
        specificity, heatmap_path,
        title=f"TS Risk Gene Striatal Cell-Type Specificity ({gene_set_name})",
    )

    barplot_path = output_dir / f"striatal_enrichment_barplot_{gene_set_name}.png"
    if not enrichment.empty:
        generate_enrichment_barplot(
            enrichment, barplot_path,
            title=f"TS Risk Gene Cell-Type Enrichment ({gene_set_name})",
        )

    # 8. Summary
    sig_types = enrichment[enrichment["significant"]].index.tolist() if not enrichment.empty else []
    top_genes_per_ct: dict[str, list[str]] = {}
    for ct in STRIATAL_CELL_TYPES:
        if ct in specificity.columns:
            top = specificity[ct].dropna().nlargest(3).index.tolist()
            if top:
                top_genes_per_ct[ct] = top

    summary = {
        "gene_set": gene_set_name,
        "n_risk_genes_scored": len(specificity),
        "n_cell_types": len(STRIATAL_CELL_TYPES),
        "significant_cell_types": sig_types,
        "top_genes_per_cell_type": top_genes_per_ct,
        "mean_tau": float(specificity["tau"].mean()) if "tau" in specificity.columns else None,
        "outputs": {
            "specificity_csv": str(specificity_path),
            "enrichment_csv": str(enrichment_path),
            "hmba_annotation_csv": str(hmba_path),
            "heatmap_png": str(heatmap_path),
            "barplot_png": str(barplot_path),
        },
    }

    summary_path = output_dir / f"striatal_deconvolution_summary_{gene_set_name}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Striatal cell-type deconvolution of TS risk gene expression"
    )
    parser.add_argument(
        "--gene-set",
        default="ts_combined",
        choices=list_gene_sets(),
        help="Gene set to analyze (default: ts_combined)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh data download (ignore cache)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = run_analysis(
        gene_set_name=args.gene_set,
        output_dir=args.output,
        cache=not args.no_cache,
    )

    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    print(f"\nStriatal cell-type deconvolution complete: {results['gene_set']}")
    print(f"  Risk genes scored: {results['n_risk_genes_scored']}")
    print(f"  Cell types: {results['n_cell_types']}")
    print(f"  Mean tau specificity: {results.get('mean_tau', 'N/A'):.3f}")
    if results["significant_cell_types"]:
        print(f"  Significant enrichment in: {', '.join(results['significant_cell_types'])}")
    else:
        print("  No significant cell-type enrichment (Bonferroni-corrected)")
    print("\nOutputs:")
    for name, path in results["outputs"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
