"""Differential expression at interneuron subtype resolution.

Runs pseudobulk DE analysis between TS and control samples for each
interneuron subtype using PyDESeq2.  Falls back to Wilcoxon rank-sum
(scanpy) when pseudobulk replication is insufficient.

Input:  classified h5ad with 'interneuron_subclass' and 'condition' columns
Output: per-subtype DE results (TSV) and summary statistics

Usage:
    uv run python -m bioagentics.tourettes.striatal_interneuron.de_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

ad.settings.allow_write_nullable_strings = True

from bioagentics.tourettes.striatal_interneuron.config import (
    DE_DIR,
    DE_PARAMS,
    OUTPUT_DIR,
    ensure_dirs,
)


def _pseudobulk(
    adata: ad.AnnData,
    groupby: list[str],
    layer: str | None = "raw_counts",
) -> ad.AnnData:
    """Aggregate single-cell counts into pseudobulk samples.

    Groups cells by *groupby* columns (e.g. [sample, condition, subtype])
    and sums raw counts per gene.
    """
    X = adata.layers[layer] if layer and layer in adata.layers else adata.X

    records: list[dict] = []
    matrices: list[np.ndarray] = []

    groups = adata.obs.groupby(groupby, observed=True)
    for keys, idx in groups.groups.items():
        if not isinstance(keys, tuple):
            keys = (keys,)
        subset = X[idx]
        if hasattr(subset, "toarray"):
            subset = subset.toarray()
        summed = np.asarray(subset.sum(axis=0)).ravel()
        matrices.append(summed)
        meta = dict(zip(groupby, keys))
        meta["n_cells"] = len(idx)
        records.append(meta)

    obs = pd.DataFrame(records)
    bulk = ad.AnnData(
        X=np.vstack(matrices),
        obs=obs,
        var=adata.var[[]].copy(),
    )
    bulk.obs_names = [f"pb_{i}" for i in range(bulk.n_obs)]
    return bulk


def run_pseudobulk_de(
    adata: ad.AnnData,
    cell_type: str,
    condition_key: str,
    group1: str,
    group2: str,
    sample_key: str = "sample",
) -> pd.DataFrame | None:
    """Run PyDESeq2 pseudobulk DE for a single cell type.

    Returns a DataFrame with gene-level results or None if there are too
    few replicates.
    """
    cell_type_key = DE_PARAMS["cell_type_key"]
    mask = adata.obs[cell_type_key] == cell_type
    sub = adata[mask].copy()

    # Check sample-level replication
    cond_counts = sub.obs.groupby(condition_key, observed=True)[sample_key].nunique()
    for label in [group1, group2]:
        if label not in cond_counts or cond_counts[label] < 2:
            print(f"    {cell_type}: <2 replicates for {label}, skipping pseudobulk")
            return None

    # Build pseudobulk
    groupby = [sample_key, condition_key]
    pb = _pseudobulk(sub, groupby=groupby)

    # Filter low-count genes
    gene_sums = np.asarray(pb.X.sum(axis=0)).ravel()
    keep = gene_sums >= 10
    if keep.sum() < 50:
        print(f"    {cell_type}: <50 genes pass count filter, skipping")
        return None
    pb = pb[:, keep].copy()

    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError:
        print("    PyDESeq2 not installed — skipping pseudobulk DE")
        return None

    dds = DeseqDataSet(
        counts=pd.DataFrame(
            pb.X.astype(int),
            index=pb.obs_names,
            columns=pb.var_names,
        ),
        metadata=pb.obs[[condition_key]].copy(),
        design=f"~{condition_key}",
    )
    dds.deseq2()
    stat = DeseqStats(dds, contrast=[condition_key, group1, group2])
    stat.summary()

    results = stat.results_df.copy()
    results["cell_type"] = cell_type
    results["method"] = "pseudobulk_pydeseq2"
    return results


def run_wilcoxon_de(
    adata: ad.AnnData,
    cell_type: str,
    condition_key: str,
    group1: str,
    group2: str,
) -> pd.DataFrame | None:
    """Run Wilcoxon rank-sum DE for a single cell type via scanpy.

    Fallback when pseudobulk replication is insufficient.
    """
    cell_type_key = DE_PARAMS["cell_type_key"]
    min_cells = DE_PARAMS["min_cells_per_group"]

    mask = adata.obs[cell_type_key] == cell_type
    sub = adata[mask].copy()

    counts = sub.obs[condition_key].value_counts()
    if counts.get(group1, 0) < min_cells or counts.get(group2, 0) < min_cells:
        print(f"    {cell_type}: <{min_cells} cells in a group, skipping Wilcoxon")
        return None

    sc.tl.rank_genes_groups(
        sub,
        groupby=condition_key,
        groups=[group1],
        reference=group2,
        method="wilcoxon",
        pts=True,
    )

    result = sc.get.rank_genes_groups_df(sub, group=group1)
    result["cell_type"] = cell_type
    result["method"] = "wilcoxon"
    return result


def run_de(
    classified_path: Path | None = None,
    output_dir: Path | None = None,
) -> Path | None:
    """Run DE analysis across all interneuron subtypes.

    Tries pseudobulk (PyDESeq2) first; falls back to Wilcoxon if
    replication is insufficient.  Saves per-subtype result TSVs and a
    combined summary.

    Returns path to combined results TSV, or None if no input found.
    """
    if output_dir is None:
        output_dir = DE_DIR

    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / "de_combined.tsv"

    if combined_path.exists():
        print(f"  DE already done: {combined_path.name}")
        return combined_path

    # Find classified input
    if classified_path is None:
        candidates = sorted((OUTPUT_DIR / "classification").glob("classified*.h5ad"))
        if candidates:
            classified_path = candidates[0]
        else:
            print("  No classified file found. Run classification phase first.")
            return None

    print(f"\n=== Differential Expression: {classified_path.name} ===")
    adata = ad.read_h5ad(classified_path)
    print(f"  Input: {adata.n_obs} cells x {adata.n_vars} genes")

    condition_key = DE_PARAMS["condition_key"]
    cell_type_key = DE_PARAMS["cell_type_key"]
    group1 = DE_PARAMS["group1_label"]
    group2 = DE_PARAMS["group2_label"]
    fdr = DE_PARAMS["fdr_threshold"]

    # Verify required columns
    if condition_key not in adata.obs:
        print(f"  ERROR: '{condition_key}' column missing — cannot run DE")
        print("  Ensure the input h5ad has condition annotations (TS vs control)")
        return None
    if cell_type_key not in adata.obs:
        print(f"  ERROR: '{cell_type_key}' column missing — run classification first")
        return None

    subtypes = sorted(adata.obs[cell_type_key].unique())
    print(f"  Subtypes: {len(subtypes)}")

    all_results: list[pd.DataFrame] = []
    has_sample_key = "sample" in adata.obs

    for ct in subtypes:
        print(f"\n  --- {ct} ---")
        result = None

        # Try pseudobulk first (requires sample-level replication)
        if has_sample_key:
            result = run_pseudobulk_de(adata, ct, condition_key, group1, group2)

        # Fallback to Wilcoxon
        if result is None:
            result = run_wilcoxon_de(adata, ct, condition_key, group1, group2)

        if result is not None and len(result) > 0:
            # Save per-subtype results
            ct_path = output_dir / f"de_{ct}.tsv"
            result.to_csv(ct_path, sep="\t", index=True)
            print(f"    Saved: {ct_path.name} ({len(result)} genes)")

            # Count significant
            if "padj" in result.columns:
                n_sig = (result["padj"] < fdr).sum()
                print(f"    Significant (FDR<{fdr}): {n_sig}")
            elif "pvals_adj" in result.columns:
                n_sig = (result["pvals_adj"] < fdr).sum()
                print(f"    Significant (FDR<{fdr}): {n_sig}")

            all_results.append(result)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(combined_path, sep="\t", index=False)
        print(f"\n  Combined: {combined_path.name} ({len(combined)} total gene-tests)")
    else:
        print("\n  WARNING: No DE results produced for any subtype")
        return None

    return combined_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Differential expression at interneuron subtype resolution"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input classified h5ad file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DE_DIR,
        help="Output directory for DE results",
    )
    args = parser.parse_args()

    run_de(args.input, args.output_dir)


if __name__ == "__main__":
    main()
