"""Multi-dataset integration and batch correction for scRNA-seq.

Integrates scRNA-seq datasets from multiple GEO studies using Harmony
batch correction, with LISI score evaluation of integration quality.

Usage:
    from bioagentics.scrna.integration import integrate_datasets
    adata_integrated = integrate_datasets(adatas, batch_key="dataset")
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import anndata as ad
import harmonypy as hm
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


@dataclass
class IntegrationStats:
    """Integration quality statistics."""

    n_datasets: int
    n_cells_total: int
    cells_per_dataset: dict[str, int]
    n_hvgs: int
    batch_lisi_median: float
    batch_lisi_mean: float

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        ds_lines = "\n".join(f"    {k}: {v:,}" for k, v in self.cells_per_dataset.items())
        return (
            f"Integration Summary:\n"
            f"  Datasets: {self.n_datasets}\n"
            f"  Total cells: {self.n_cells_total:,}\n"
            f"  Cells per dataset:\n{ds_lines}\n"
            f"  HVGs used: {self.n_hvgs:,}\n"
            f"  Batch LISI (median): {self.batch_lisi_median:.3f}\n"
            f"  Batch LISI (mean): {self.batch_lisi_mean:.3f}"
        )


def compute_lisi(
    X: np.ndarray,
    labels: np.ndarray,
    perplexity: float = 30.0,
) -> np.ndarray:
    """Compute Local Inverse Simpson's Index (LISI) for batch mixing.

    Higher LISI = better mixing. Perfect mixing across N batches gives LISI ≈ N.
    Uses a KNN-based approximation for efficiency.
    """
    from sklearn.neighbors import NearestNeighbors

    k = min(int(perplexity * 3), X.shape[0] - 1)
    if k < 1:
        return np.ones(X.shape[0])

    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    unique_labels = np.unique(labels)
    n_labels = len(unique_labels)
    label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
    label_ints = np.array([label_to_idx[lab] for lab in labels])

    lisi_scores = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        neighbor_labels = label_ints[indices[i]]
        counts = np.bincount(neighbor_labels, minlength=n_labels)
        freqs = counts / counts.sum()
        # Simpson's index = sum(p^2), LISI = 1/Simpson
        simpsons = np.sum(freqs**2)
        lisi_scores[i] = 1.0 / simpsons if simpsons > 0 else 1.0

    return lisi_scores


def normalize_and_hvg(
    adata: ad.AnnData,
    n_top_genes: int = 2000,
    target_sum: float = 1e4,
) -> ad.AnnData:
    """Normalize, log-transform, and select highly variable genes.

    Stores raw counts in adata.layers["counts"] before normalization.
    """
    # Preserve raw counts
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)

    # HVG selection using batch-aware method
    batch_key = "dataset" if "dataset" in adata.obs.columns else None
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        batch_key=batch_key,
        flavor="seurat",
    )

    print(f"  Selected {adata.var['highly_variable'].sum()} highly variable genes")
    return adata


def run_pca(adata: ad.AnnData, n_comps: int = 50) -> ad.AnnData:
    """Scale data and compute PCA on HVGs."""
    # Subset to HVGs for PCA
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=min(n_comps, adata_hvg.n_vars - 1))

    # Store PCA in original object
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.uns["pca"] = adata_hvg.uns["pca"]

    print(f"  PCA: {adata.obsm['X_pca'].shape[1]} components")
    return adata


def run_harmony(
    adata: ad.AnnData,
    batch_key: str = "dataset",
    max_iter: int = 20,
    random_state: int = 0,
) -> ad.AnnData:
    """Run Harmony batch correction on PCA embeddings.

    Stores corrected embedding in adata.obsm["X_pca_harmony"].
    """
    if "X_pca" not in adata.obsm:
        raise ValueError("PCA must be computed before Harmony. Call run_pca() first.")

    pca_data = adata.obsm["X_pca"]
    meta = adata.obs[[batch_key]].copy()

    print(f"  Running Harmony on {pca_data.shape[1]} PCs across {meta[batch_key].nunique()} batches...")
    ho = hm.run_harmony(
        pca_data,
        meta,
        batch_key,
        max_iter_harmony=max_iter,
        random_state=random_state,
    )

    # Z_corr shape is (d, N) — transpose to (N, d) for obsm
    z_corr = ho.Z_corr
    if hasattr(z_corr, "numpy"):
        z_corr = z_corr.numpy()  # PyTorch tensor -> numpy
    z_corr = np.asarray(z_corr, dtype=np.float32)
    if z_corr.shape[0] != adata.n_obs:
        z_corr = z_corr.T
    adata.obsm["X_pca_harmony"] = z_corr
    print(f"  Harmony corrected embedding shape: {adata.obsm['X_pca_harmony'].shape}")
    return adata


def compute_neighbors_and_umap(
    adata: ad.AnnData,
    use_rep: str = "X_pca_harmony",
    n_neighbors: int = 15,
    random_state: int = 0,
) -> ad.AnnData:
    """Compute neighbor graph and UMAP from corrected embedding."""
    sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors, random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)

    sc.tl.leiden(adata, resolution=1.0, random_state=random_state, flavor="igraph", n_iterations=2, directed=False)
    cluster_key = "leiden"
    print(f"  UMAP computed, {adata.obs[cluster_key].nunique()} clusters")
    return adata


def integrate_datasets(
    adatas: list[ad.AnnData] | ad.AnnData,
    batch_key: str = "dataset",
    n_top_genes: int = 2000,
    n_pcs: int = 50,
    harmony_max_iter: int = 20,
    random_state: int = 0,
) -> tuple[ad.AnnData, IntegrationStats]:
    """Full integration pipeline: concat, normalize, HVG, PCA, Harmony, UMAP.

    Args:
        adatas: List of AnnData objects or a single concatenated AnnData.
        batch_key: Column in .obs identifying batch/dataset.
        n_top_genes: Number of highly variable genes.
        n_pcs: Number of PCA components.
        harmony_max_iter: Maximum Harmony iterations.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (integrated AnnData, IntegrationStats).
    """
    # Concatenate if list
    if isinstance(adatas, list):
        print(f"Concatenating {len(adatas)} datasets...")
        adata = ad.concat(adatas, join="outer", index_unique="-")
    else:
        adata = adatas

    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")

    # Ensure sparse CSR
    if sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    n_datasets = adata.obs[batch_key].nunique()
    cells_per_ds = adata.obs[batch_key].value_counts().to_dict()
    print(f"Integration: {adata.n_obs:,} cells from {n_datasets} datasets")

    # Normalize and select HVGs
    adata = normalize_and_hvg(adata, n_top_genes=n_top_genes)
    n_hvgs = int(adata.var["highly_variable"].sum())

    # PCA
    adata = run_pca(adata, n_comps=n_pcs)

    # Harmony batch correction
    adata = run_harmony(
        adata,
        batch_key=batch_key,
        max_iter=harmony_max_iter,
        random_state=random_state,
    )

    # Neighbors + UMAP + clustering
    adata = compute_neighbors_and_umap(adata, random_state=random_state)

    # Evaluate batch mixing with LISI
    print("  Computing batch LISI scores...")
    batch_labels = adata.obs[batch_key].values
    lisi_scores = compute_lisi(adata.obsm["X_pca_harmony"], batch_labels)
    adata.obs["batch_lisi"] = lisi_scores

    stats = IntegrationStats(
        n_datasets=n_datasets,
        n_cells_total=adata.n_obs,
        cells_per_dataset=cells_per_ds,
        n_hvgs=n_hvgs,
        batch_lisi_median=float(np.median(lisi_scores)),
        batch_lisi_mean=float(np.mean(lisi_scores)),
    )

    print(f"\n{stats.summary()}")
    return adata, stats
