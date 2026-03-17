"""Differential expression analysis with mandatory sex-stratified mode.

Implements DE analysis using pydeseq2. MANDATORY: supports three analysis
modes per the research plan — (1) combined (all samples), (2) male-only,
(3) female-only. Rahman SS et al. 2025 showed sex-associated differences
in monocyte phenotypes in PANS.

Usage:
    uv run python -m bioagentics.models.differential_expression input.h5ad \\
        --condition-key condition --sex-key sex --dest output/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "transcriptomic-biomarker-panel"


def run_deseq2(
    adata: ad.AnnData,
    condition_key: str = "condition",
    ref_level: str | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run DESeq2 differential expression on an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Must have raw integer counts in adata.layers["counts"] or adata.X.
        Must have condition_key in adata.obs.
    condition_key : str
        Column in obs defining the groups to compare.
    ref_level : str, optional
        Reference level for the contrast (e.g., "control"). If None, uses
        the alphabetically first level.
    alpha : float
        Significance threshold for adjusted p-values.

    Returns
    -------
    DataFrame with columns: gene, log2FoldChange, pvalue, padj, baseMean, stat.
    Sorted by padj ascending.
    """
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    if condition_key not in adata.obs.columns:
        raise ValueError(f"'{condition_key}' not in adata.obs: {list(adata.obs.columns)}")

    conditions = adata.obs[condition_key].unique()
    if len(conditions) < 2:
        raise ValueError(f"Need >= 2 groups in '{condition_key}', got: {list(conditions)}")

    # Get counts — prefer layers["counts"], fall back to X
    if "counts" in adata.layers:
        counts_arr = np.array(adata.layers["counts"])
    else:
        counts_arr = np.array(adata.X)

    counts_df = pd.DataFrame(
        counts_arr, index=adata.obs_names, columns=adata.var_names
    ).round().astype(int)

    # Remove genes with zero total
    gene_sums = counts_df.sum(axis=0)
    counts_df = counts_df.loc[:, gene_sums > 0]

    meta = adata.obs[[condition_key]].copy()
    meta[condition_key] = meta[condition_key].astype(str)

    if ref_level is None:
        ref_level = sorted(meta[condition_key].unique())[0]

    logger.info(
        "Running DESeq2: %d samples, %d genes, groups=%s, ref=%s",
        counts_df.shape[0], counts_df.shape[1],
        list(meta[condition_key].unique()), ref_level,
    )

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=meta,
        design=f"~{condition_key}",
        quiet=True,
    )
    dds.deseq2()

    # Get the contrast — compare each non-ref level to reference
    non_ref = [c for c in meta[condition_key].unique() if c != ref_level]
    if not non_ref:
        raise ValueError("All samples are in the reference group")

    all_results = []
    for level in non_ref:
        stat = DeseqStats(
            dds,
            contrast=[condition_key, level, ref_level],
            alpha=alpha,
            quiet=True,
        )
        stat.summary()

        results = stat.results_df.copy()
        results["gene"] = results.index
        results["contrast"] = f"{level}_vs_{ref_level}"
        results = results.rename(columns={
            "log2FoldChange": "log2FoldChange",
            "pvalue": "pvalue",
            "padj": "padj",
            "baseMean": "baseMean",
            "stat": "stat",
        })
        all_results.append(results)

    de_df = pd.concat(all_results, ignore_index=True)
    de_df = de_df.sort_values("padj").reset_index(drop=True)

    n_sig = (de_df["padj"] < alpha).sum()
    logger.info("DE complete: %d genes tested, %d significant (padj < %.2f)",
                len(de_df), n_sig, alpha)

    return de_df


def run_sex_stratified_de(
    adata: ad.AnnData,
    condition_key: str = "condition",
    sex_key: str = "sex",
    ref_level: str | None = None,
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """Run DE analysis in three modes: combined, male-only, female-only.

    Parameters
    ----------
    adata : AnnData
        Input data with condition and sex columns in obs.
    condition_key : str
        Group comparison column.
    sex_key : str
        Sex column (values should contain 'M'/'male' or 'F'/'female').
    ref_level : str, optional
        Reference level for the condition contrast.
    alpha : float
        Significance threshold.

    Returns
    -------
    Dict with keys "combined", "male", "female" -> DE result DataFrames.
    """
    if sex_key not in adata.obs.columns:
        raise ValueError(f"Sex key '{sex_key}' not in adata.obs: {list(adata.obs.columns)}")

    results: dict[str, pd.DataFrame] = {}

    # 1. Combined (all samples)
    logger.info("=== DE mode: COMBINED ===")
    results["combined"] = run_deseq2(adata, condition_key, ref_level, alpha)

    # Classify sex values
    sex_vals = adata.obs[sex_key].astype(str).str.upper().str.strip()
    male_mask = sex_vals.isin(["M", "MALE", "1"])
    female_mask = sex_vals.isin(["F", "FEMALE", "2"])

    # 2. Male-only
    n_male = male_mask.sum()
    if n_male >= 4:
        logger.info("=== DE mode: MALE-ONLY (%d samples) ===", n_male)
        male_conditions = adata.obs.loc[male_mask, condition_key].unique()
        if len(male_conditions) >= 2:
            results["male"] = run_deseq2(adata[male_mask].copy(), condition_key, ref_level, alpha)
        else:
            logger.warning("Male subset has only 1 condition group — skipping male-only DE")
            results["male"] = pd.DataFrame()
    else:
        logger.warning("Only %d male samples — skipping male-only DE (need >= 4)", n_male)
        results["male"] = pd.DataFrame()

    # 3. Female-only
    n_female = female_mask.sum()
    if n_female >= 4:
        logger.info("=== DE mode: FEMALE-ONLY (%d samples) ===", n_female)
        female_conditions = adata.obs.loc[female_mask, condition_key].unique()
        if len(female_conditions) >= 2:
            results["female"] = run_deseq2(adata[female_mask].copy(), condition_key, ref_level, alpha)
        else:
            logger.warning("Female subset has only 1 condition group — skipping female-only DE")
            results["female"] = pd.DataFrame()
    else:
        logger.warning("Only %d female samples — skipping female-only DE (need >= 4)", n_female)
        results["female"] = pd.DataFrame()

    return results


def plot_volcano(
    de_df: pd.DataFrame,
    title: str = "Volcano Plot",
    alpha: float = 0.05,
    lfc_threshold: float = 1.0,
    save_path: Path | None = None,
) -> None:
    """Generate a volcano plot from DE results."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if de_df.empty:
        logger.warning("Empty DE results — skipping volcano plot for '%s'", title)
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    log2fc = de_df["log2FoldChange"].values
    neg_log_p = -np.log10(de_df["padj"].clip(lower=1e-300).values)

    sig = de_df["padj"] < alpha
    up = sig & (de_df["log2FoldChange"] > lfc_threshold)
    down = sig & (de_df["log2FoldChange"] < -lfc_threshold)
    ns = ~(up | down)

    ax.scatter(log2fc[ns], neg_log_p[ns], c="grey", alpha=0.3, s=5, label="NS")
    ax.scatter(log2fc[up], neg_log_p[up], c="red", alpha=0.5, s=10, label=f"Up ({up.sum()})")
    ax.scatter(log2fc[down], neg_log_p[down], c="blue", alpha=0.5, s=10, label=f"Down ({down.sum()})")

    ax.axhline(-np.log10(alpha), color="black", linestyle="--", linewidth=0.5)
    ax.axvline(lfc_threshold, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(-lfc_threshold, color="black", linestyle="--", linewidth=0.5)

    ax.set_xlabel("log2 Fold Change")
    ax.set_ylabel("-log10(padj)")
    ax.set_title(title)
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved volcano plot: %s", save_path)
    plt.close(fig)


def de_pipeline(
    adata: ad.AnnData,
    condition_key: str = "condition",
    sex_key: str = "sex",
    ref_level: str | None = None,
    dest_dir: Path | None = None,
    alpha: float = 0.05,
) -> dict[str, pd.DataFrame]:
    """Full DE pipeline: sex-stratified analysis with volcano plots.

    Returns dict of mode -> DE results DataFrame.
    Saves CSVs and volcano plots to dest_dir.
    """
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    results = run_sex_stratified_de(
        adata, condition_key, sex_key, ref_level, alpha
    )

    for mode, de_df in results.items():
        if not de_df.empty:
            csv_path = dest_dir / f"de_results_{mode}.csv"
            de_df.to_csv(csv_path, index=False)
            logger.info("Saved %s DE results (%d genes) to %s", mode, len(de_df), csv_path)

            plot_volcano(
                de_df,
                title=f"DE: {mode}",
                alpha=alpha,
                save_path=dest_dir / f"volcano_{mode}.png",
            )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run sex-stratified differential expression analysis"
    )
    parser.add_argument("input", type=Path, help="Input h5ad file")
    parser.add_argument("--condition-key", default="condition")
    parser.add_argument("--sex-key", default="sex")
    parser.add_argument("--ref-level", default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    adata = ad.read_h5ad(args.input)
    results = de_pipeline(
        adata,
        condition_key=args.condition_key,
        sex_key=args.sex_key,
        ref_level=args.ref_level,
        dest_dir=args.dest,
        alpha=args.alpha,
    )

    for mode, de_df in results.items():
        n_sig = (de_df["padj"] < args.alpha).sum() if not de_df.empty else 0
        print(f"{mode}: {len(de_df)} genes, {n_sig} significant")


if __name__ == "__main__":
    main()
