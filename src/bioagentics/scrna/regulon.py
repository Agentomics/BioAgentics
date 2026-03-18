"""Transcription factor regulon analysis for IL-23-responsive cell populations.

Uses decoupleR with the CollecTRI TF-target network to infer transcription
factor activity per cell, then identifies master regulators active in
IL-23-responsive cell types (Th17, ILC3, inflammatory macrophages).

This is a lightweight alternative to full pySCENIC that uses curated
literature-based TF-target interactions rather than de novo motif inference.

Usage:
    from bioagentics.scrna.regulon import run_regulon_analysis
    adata, tf_ranking = run_regulon_analysis(adata)
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import decoupler as dc
import numpy as np
import pandas as pd


# IL-23-responsive cell types to focus on
IL23_RESPONSIVE_TYPES = [
    "Th17",
    "ILC3",
    "Inflammatory_Mac",
    "Macrophage",
    "Dendritic_cell",
]

# Known IL-23/Th17 axis TFs for context (not novel)
KNOWN_IL23_TFS = {
    "RORC",  # Master Th17 TF
    "STAT3",  # IL-23 signaling
    "RORA",  # Th17 co-factor
}


@dataclass
class TFActivity:
    """Per-cell-type transcription factor activity summary."""

    tf_name: str
    cell_type: str
    mean_activity: float
    median_activity: float
    pct_active: float  # % cells with activity > 0
    n_cells: int
    n_targets: int  # number of targets in network
    is_known_il23: bool


@dataclass
class RegulonResult:
    """Summary of regulon analysis."""

    n_cells_analyzed: int
    n_tfs_tested: int
    n_cell_types: int
    top_tfs_per_celltype: dict[str, list[str]]
    novel_tfs: list[str]  # TFs not in KNOWN_IL23_TFS that rank highly

    def summary(self) -> str:
        lines = [
            f"Regulon Analysis Summary:",
            f"  Cells analyzed: {self.n_cells_analyzed:,}",
            f"  TFs tested: {self.n_tfs_tested}",
            f"  Cell types: {self.n_cell_types}",
        ]
        if self.novel_tfs:
            lines.append(f"  Novel TFs (beyond RORC/STAT3/RORA): {', '.join(self.novel_tfs[:10])}")
        for ct, tfs in self.top_tfs_per_celltype.items():
            lines.append(f"  {ct} top TFs: {', '.join(tfs[:5])}")
        return "\n".join(lines)


def get_tf_network(organism: str = "human") -> pd.DataFrame:
    """Get CollecTRI TF-target network from OmniPath.

    Returns DataFrame with source (TF), target (gene), weight columns.
    """
    net = dc.op.collectri(organism=organism)
    print(f"  CollecTRI network: {net.shape[0]:,} interactions, {net['source'].nunique()} TFs")
    return net


def infer_tf_activity(
    adata: ad.AnnData,
    net: pd.DataFrame,
    method: str = "ulm",
    min_targets: int = 5,
) -> ad.AnnData:
    """Infer per-cell transcription factor activity using decoupleR.

    Stores activity scores in adata.obsm['score_{method}'] and
    p-values in adata.obsm['padj_{method}'].

    Args:
        adata: Log-normalized AnnData.
        net: TF-target network (source, target, weight columns).
        method: Inference method — 'ulm' (recommended) or 'mlm'.
        min_targets: Minimum targets per TF to include.
    """
    # Filter network to genes present in the dataset
    available_genes = list(adata.var_names)
    net_filtered = net[net["target"].isin(available_genes)].copy()
    tf_counts = net_filtered.groupby("source").size()
    valid_tfs = tf_counts[tf_counts >= min_targets].index.tolist()
    net_filtered = net_filtered[net_filtered["source"].isin(valid_tfs)]

    print(
        f"  Network after filtering: {net_filtered.shape[0]:,} interactions, "
        f"{net_filtered['source'].nunique()} TFs (min {min_targets} targets)"
    )

    # Run activity inference (decoupler v2 API: no source/target/weight kwargs)
    run_fn = getattr(dc.mt, method, None)
    if run_fn is None:
        raise ValueError(f"Unknown method: {method}. Use 'ulm' or 'mlm'.")
    run_fn(adata, net_filtered, tmin=min_targets)

    # decoupler v2 stores results as score_{method} and padj_{method}
    key_score = f"score_{method}"
    key_padj = f"padj_{method}"

    if key_score in adata.obsm:
        n_tfs = adata.obsm[key_score].shape[1]
        print(f"  Inferred activity for {n_tfs} TFs across {adata.n_obs:,} cells")
    else:
        print(f"  Warning: {key_score} not found in adata.obsm")

    return adata


def compute_celltype_tf_activity(
    adata: ad.AnnData,
    net: pd.DataFrame,
    cell_type_key: str = "cell_type",
    method: str = "ulm",
    il23_types_only: bool = True,
) -> list[TFActivity]:
    """Compute per-cell-type TF activity summaries.

    Args:
        adata: AnnData with TF activity in obsm.
        net: TF-target network for target counts.
        cell_type_key: Column with cell type labels.
        method: Method used (to find the right obsm key).
        il23_types_only: If True, only analyze IL-23-responsive cell types.

    Returns:
        List of TFActivity records.
    """
    key_estimate = f"score_{method}"
    if key_estimate not in adata.obsm:
        print(f"  No TF activity found in adata.obsm['{key_estimate}']")
        return []

    act_df = adata.obsm[key_estimate]
    if isinstance(act_df, np.ndarray):
        return []

    # Target counts per TF
    tf_target_counts = net.groupby("source").size().to_dict()

    # Determine cell types to analyze
    all_types = adata.obs[cell_type_key].unique()
    if il23_types_only:
        cell_types = [ct for ct in IL23_RESPONSIVE_TYPES if ct in all_types]
        if not cell_types:
            print("  No IL-23-responsive cell types found, using all types")
            cell_types = [ct for ct in all_types if ct != "Unassigned"]
    else:
        cell_types = [ct for ct in all_types if ct != "Unassigned"]

    results = []
    for ct in cell_types:
        ct_mask = adata.obs[cell_type_key] == ct
        ct_act = act_df[ct_mask]

        for tf in act_df.columns:
            scores = np.asarray(ct_act[tf])
            results.append(
                TFActivity(
                    tf_name=tf,
                    cell_type=ct,
                    mean_activity=float(np.mean(scores)),
                    median_activity=float(np.median(scores)),
                    pct_active=float((scores > 0).sum() / len(scores) * 100) if len(scores) > 0 else 0.0,
                    n_cells=len(scores),
                    n_targets=tf_target_counts.get(tf, 0),
                    is_known_il23=tf in KNOWN_IL23_TFS,
                )
            )

    return results


def rank_tfs_per_celltype(
    activities: list[TFActivity],
    metric: str = "mean_activity",
    top_n: int = 20,
) -> dict[str, pd.DataFrame]:
    """Rank TFs by activity within each cell type.

    Returns dict mapping cell_type -> DataFrame of ranked TFs.
    """
    if not activities:
        return {}

    df = pd.DataFrame([
        {
            "tf": a.tf_name,
            "cell_type": a.cell_type,
            "mean_activity": a.mean_activity,
            "median_activity": a.median_activity,
            "pct_active": a.pct_active,
            "n_cells": a.n_cells,
            "n_targets": a.n_targets,
            "is_known_il23": a.is_known_il23,
        }
        for a in activities
    ])

    rankings = {}
    for ct, group in df.groupby("cell_type"):
        ranked = group.sort_values(metric, ascending=False).head(top_n).reset_index(drop=True)
        rankings[str(ct)] = ranked

    return rankings


def identify_novel_regulons(
    rankings: dict[str, pd.DataFrame],
    metric: str = "mean_activity",
    top_n: int = 10,
) -> list[str]:
    """Identify TFs that rank highly but are not known IL-23/Th17 TFs.

    A TF is "novel" if it's in the top_n for any IL-23-responsive cell type
    and is NOT in KNOWN_IL23_TFS.
    """
    novel = set()
    for ct, df in rankings.items():
        top_tfs = df.head(top_n)
        for _, row in top_tfs.iterrows():
            if not row["is_known_il23"] and row[metric] > 0:
                novel.add(row["tf"])
    return sorted(novel)


def run_regulon_analysis(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    method: str = "ulm",
    organism: str = "human",
    min_targets: int = 5,
    max_cells_per_type: int | None = None,
    il23_types_only: bool = True,
) -> tuple[ad.AnnData, RegulonResult]:
    """Full regulon analysis pipeline using decoupleR + CollecTRI.

    Steps:
    1. Load CollecTRI TF-target network
    2. Optionally subsample cells per type (for tractability)
    3. Infer per-cell TF activity
    4. Compute per-cell-type activity summaries
    5. Rank TFs and identify novel regulons

    Args:
        adata: Annotated, log-normalized AnnData with cell_type column.
        cell_type_key: Column in .obs with cell type labels.
        method: Activity inference method ('ulm' or 'mlm').
        organism: Organism for network ('human' or 'mouse').
        min_targets: Minimum targets per TF.
        max_cells_per_type: If set, subsample to this many cells per type.
        il23_types_only: Focus analysis on IL-23-responsive cell types.

    Returns:
        Tuple of (AnnData with TF activity, RegulonResult).
    """
    print(f"Running regulon analysis on {adata.n_obs:,} cells...")

    # Step 1: Get TF-target network
    net = get_tf_network(organism=organism)

    # Step 2: Optional subsampling for large datasets
    if max_cells_per_type and cell_type_key in adata.obs.columns:
        sampled_idx = []
        for ct in adata.obs[cell_type_key].unique():
            ct_idx = adata.obs.index[adata.obs[cell_type_key] == ct]
            if len(ct_idx) > max_cells_per_type:
                rng = np.random.default_rng(42)
                ct_idx = rng.choice(ct_idx, max_cells_per_type, replace=False)
            sampled_idx.extend(ct_idx)
        adata_sub = adata[sampled_idx].copy()
        print(f"  Subsampled to {adata_sub.n_obs:,} cells (max {max_cells_per_type}/type)")
    else:
        adata_sub = adata

    # Step 3: Infer TF activity
    adata_sub = infer_tf_activity(adata_sub, net, method=method, min_targets=min_targets)

    # Copy results back to original if subsampled
    key_estimate = f"score_{method}"
    if adata_sub is not adata and key_estimate in adata_sub.obsm:
        # Store subsampled results in original — only for analyzed cells
        pass  # Keep results in adata_sub for now

    # Step 4: Per-cell-type activity
    activities = compute_celltype_tf_activity(
        adata_sub, net, cell_type_key=cell_type_key,
        method=method, il23_types_only=il23_types_only,
    )

    # Step 5: Rank and identify novel regulons
    rankings = rank_tfs_per_celltype(activities)
    novel_tfs = identify_novel_regulons(rankings)

    top_tfs_per_ct = {
        ct: df["tf"].head(5).tolist() for ct, df in rankings.items()
    }

    n_tfs = adata_sub.obsm[key_estimate].shape[1] if key_estimate in adata_sub.obsm else 0

    result = RegulonResult(
        n_cells_analyzed=adata_sub.n_obs,
        n_tfs_tested=n_tfs,
        n_cell_types=len(rankings),
        top_tfs_per_celltype=top_tfs_per_ct,
        novel_tfs=novel_tfs,
    )

    print(f"\n{result.summary()}")

    # Store TF activity in original adata if not subsampled
    if adata_sub is adata:
        return adata, result
    return adata_sub, result
