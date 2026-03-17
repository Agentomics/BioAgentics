"""Cell type annotation pipeline for IVIG scRNA-seq analysis.

Fine-resolution annotation of PBMCs from PANS patients (Han VX et al.),
focusing on immune cell subtypes relevant to IVIG mechanism of action.

Supports two annotation strategies:
1. Marker-based: Canonical PBMC marker scoring with hierarchical assignment
2. Reference-based: Label transfer from PBMCpedia or other reference datasets

Usage:
    from bioagentics.pandas_pans.ivig_cell_annotation import annotate_ivig_cells
    adata, stats = annotate_ivig_cells(adata)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from bioagentics.config import DATA_DIR

REFERENCE_DIR = DATA_DIR / "pandas_pans" / "ivig-mechanism-single-cell-analysis" / "reference"

# PBMC cell type markers for fine-resolution annotation.
# Focused on immune subtypes relevant to IVIG mechanism.
PBMC_MARKERS: dict[str, list[str]] = {
    # --- Monocyte subsets ---
    "Classical_Mono": ["CD14", "LYZ", "S100A8", "S100A9", "VCAN", "FCN1"],
    "Intermediate_Mono": ["CD14", "FCGR3A", "HLA-DRA", "CSF1R", "CX3CR1"],
    "NonClassical_Mono": ["FCGR3A", "MS4A7", "CDKN1C", "LILRB2", "LST1"],
    # --- T cell subsets ---
    "CD4_Naive": ["CCR7", "SELL", "LEF1", "TCF7", "IL7R", "CD4"],
    "CD4_Memory": ["IL7R", "CD4", "S100A4", "GPR183", "LMNA"],
    "Th1": ["IFNG", "TBX21", "CXCR3", "TNF", "IL12RB2"],
    "Th2": ["GATA3", "IL4", "IL5", "IL13", "CCR4", "PTGDR2"],
    "Th17": ["RORC", "IL17A", "IL17F", "CCR6", "IL23R"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT"],
    "CD8_Effector": ["CD8A", "CD8B", "GZMB", "PRF1", "NKG7", "GNLY"],
    "CD8_Naive": ["CD8A", "CD8B", "CCR7", "SELL", "LEF1"],
    # --- NK cell subtypes ---
    "NK_CD56bright": ["NCAM1", "KLRC1", "XCL1", "XCL2", "GZMK", "SELL"],
    "NK_CD56dim": ["FCGR3A", "NCAM1", "GZMB", "PRF1", "FGFBP2", "CX3CR1"],
    # --- B cell maturation ---
    "B_Naive": ["MS4A1", "CD79A", "IGHD", "FCER2", "TCL1A", "IL4R"],
    "B_Memory": ["MS4A1", "CD79A", "CD27", "IGHG1", "AIM2"],
    "Plasmablast": ["JCHAIN", "MZB1", "XBP1", "PRDM1", "SDC1", "IGHG1"],
    # --- Other ---
    "pDC": ["LILRA4", "IRF7", "CLEC4C", "IL3RA", "JCHAIN"],
    "cDC": ["CLEC9A", "FLT3", "BATF3", "IRF8", "THBD"],
    "Neutrophil": ["FCGR3B", "CXCR2", "CSF3R", "S100A12", "MMP9"],
    "Platelet": ["PPBP", "PF4", "GP9", "ITGA2B"],
}

# Broad lineage markers for initial classification
PBMC_LINEAGE_MARKERS: dict[str, list[str]] = {
    "T_cell": ["CD3D", "CD3E", "CD3G", "TRAC"],
    "B_cell": ["CD79A", "MS4A1", "CD19"],
    "Monocyte": ["CD14", "LYZ", "CST3", "S100A8"],
    "NK": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "DC": ["FLT3", "IRF8", "LILRA4"],
    "Neutrophil": ["FCGR3B", "CXCR2", "S100A12"],
    "Platelet": ["PPBP", "PF4"],
}

# Genes of particular interest for IVIG mechanism analysis
IVIG_FOCUS_GENES: dict[str, list[str]] = {
    "autophagy": ["ATG7", "UVRAG", "BECN1", "MAP1LC3B", "SQSTM1", "MTOR"],
    "s100a12_axis": ["S100A12", "TLR4", "MYD88", "NFKB1"],
    "histone_modification": ["KDM6A", "KDM6B", "EZH2", "KAT2A", "HDAC1", "HDAC2"],
    "defense_response": ["IFITM1", "IFITM3", "MX1", "OAS1", "ISG15"],
}


@dataclass
class AnnotationStats:
    """Cell type annotation summary."""

    n_cells: int = 0
    n_clusters: int = 0
    n_cell_types: int = 0
    cell_type_counts: dict[str, int] | None = None
    lineage_counts: dict[str, int] | None = None
    unassigned_count: int = 0
    pct_unassigned: float = 0.0
    method: str = "marker_based"
    resolution: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            f"Cell Type Annotation Summary (method={self.method}):",
            f"  Total cells: {self.n_cells:,}",
            f"  Clusters: {self.n_clusters}",
            f"  Cell types assigned: {self.n_cell_types}",
            f"  Unassigned: {self.unassigned_count:,} ({self.pct_unassigned:.1f}%)",
        ]
        if self.lineage_counts:
            lines.append("  Lineages:")
            for k, v in sorted(self.lineage_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {k}: {v:,}")
        if self.cell_type_counts:
            lines.append("  Cell types:")
            for k, v in sorted(self.cell_type_counts.items(), key=lambda x: -x[1]):
                lines.append(f"    {k}: {v:,}")
        return "\n".join(lines)


def preprocess_for_clustering(
    adata: ad.AnnData,
    n_top_genes: int = 3000,
    n_pcs: int = 50,
    n_neighbors: int = 15,
) -> ad.AnnData:
    """Normalize, find HVGs, PCA, and build neighbor graph for clustering.

    Expects raw counts in adata.X (stores normalized in layers['normalized']).
    """
    # Store raw counts if not already saved
    if "raw_counts" not in adata.layers:
        adata.layers["raw_counts"] = adata.X.copy()

    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.layers["normalized"] = adata.X.copy()

    # HVGs (using seurat flavor on log-normalized data)
    n_hvg = min(n_top_genes, adata.n_vars - 1)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)

    # PCA on HVGs
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    n_pcs_actual = min(n_pcs, adata_hvg.n_vars - 1, adata_hvg.n_obs - 1)
    sc.tl.pca(adata_hvg, n_comps=n_pcs_actual)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]
    adata.uns["pca"] = adata_hvg.uns["pca"]

    # Neighbor graph
    sc.pp.neighbors(adata, n_pcs=n_pcs_actual, n_neighbors=n_neighbors)

    # UMAP
    sc.tl.umap(adata)

    return adata


def cluster_cells(
    adata: ad.AnnData,
    resolution: float = 1.0,
    key_added: str = "leiden",
) -> ad.AnnData:
    """Leiden clustering at specified resolution."""
    sc.tl.leiden(adata, resolution=resolution, key_added=key_added)
    n_clusters = adata.obs[key_added].nunique()
    print(f"  Leiden clustering (res={resolution}): {n_clusters} clusters")
    return adata


def score_cell_types(
    adata: ad.AnnData,
    markers: dict[str, list[str]] | None = None,
    prefix: str = "score_",
) -> ad.AnnData:
    """Score cells for each cell type using marker gene sets."""
    if markers is None:
        markers = PBMC_MARKERS

    for cell_type, genes in markers.items():
        available = [g for g in genes if g in adata.var_names]
        if len(available) < 2:
            adata.obs[f"{prefix}{cell_type}"] = 0.0
            continue
        sc.tl.score_genes(adata, gene_list=available, score_name=f"{prefix}{cell_type}")

    return adata


def assign_lineage(adata: ad.AnnData) -> ad.AnnData:
    """Assign broad lineage from lineage marker scores."""
    adata = score_cell_types(adata, markers=PBMC_LINEAGE_MARKERS, prefix="lineage_")

    lineage_cols = [f"lineage_{lin}" for lin in PBMC_LINEAGE_MARKERS]
    available_cols = [c for c in lineage_cols if c in adata.obs.columns]
    if not available_cols:
        adata.obs["lineage"] = "Unknown"
        return adata

    scores_df = adata.obs[available_cols]
    best = scores_df.idxmax(axis=1).str.replace("lineage_", "", regex=False)
    max_scores = scores_df.max(axis=1)
    best[max_scores < 0.1] = "Unknown"
    adata.obs["lineage"] = best.values
    return adata


def assign_cell_types(
    adata: ad.AnnData,
    min_score: float = 0.15,
) -> ad.AnnData:
    """Assign fine cell types from marker scores with confidence."""
    score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    if not score_cols:
        adata.obs["cell_type"] = "Unassigned"
        adata.obs["cell_type_confidence"] = 0.0
        return adata

    scores_df = adata.obs[score_cols]
    best_type = scores_df.idxmax(axis=1).str.replace("score_", "", regex=False)
    max_scores = scores_df.max(axis=1)

    # Confidence: gap between best and second-best score
    sorted_scores = np.sort(scores_df.values, axis=1)
    confidence = sorted_scores[:, -1] - sorted_scores[:, -2] if sorted_scores.shape[1] > 1 else sorted_scores[:, -1]

    best_type[max_scores < min_score] = "Unassigned"

    adata.obs["cell_type"] = best_type.values
    adata.obs["cell_type_confidence"] = confidence
    return adata


def score_ivig_focus_genes(adata: ad.AnnData) -> ad.AnnData:
    """Score cells for IVIG mechanism-relevant gene modules."""
    for module_name, genes in IVIG_FOCUS_GENES.items():
        available = [g for g in genes if g in adata.var_names]
        if len(available) < 2:
            adata.obs[f"ivig_{module_name}"] = 0.0
            continue
        sc.tl.score_genes(
            adata, gene_list=available, score_name=f"ivig_{module_name}"
        )
    return adata


def transfer_labels_from_reference(
    adata: ad.AnnData,
    ref_path: Path,
    ref_label_key: str = "cell_type",
) -> ad.AnnData:
    """Transfer cell type labels from a reference dataset using ingest.

    Args:
        adata: Query dataset (preprocessed with PCA/neighbors).
        ref_path: Path to reference h5ad file.
        ref_label_key: obs column in reference with cell type labels.

    Returns:
        adata with 'ref_cell_type' column added.
    """
    if not ref_path.exists():
        print(f"  Reference not found at {ref_path}, skipping label transfer")
        adata.obs["ref_cell_type"] = "N/A"
        return adata

    print(f"  Loading reference from {ref_path}...")
    ref = ad.read_h5ad(ref_path)

    if ref_label_key not in ref.obs.columns:
        print(f"  Reference missing '{ref_label_key}' column, skipping")
        adata.obs["ref_cell_type"] = "N/A"
        return adata

    # Align to shared genes
    shared_genes = list(set(adata.var_names) & set(ref.var_names))
    if len(shared_genes) < 100:
        print(f"  Only {len(shared_genes)} shared genes, skipping label transfer")
        adata.obs["ref_cell_type"] = "N/A"
        return adata

    print(f"  Label transfer using {len(shared_genes)} shared genes...")
    sc.tl.ingest(adata, ref, obs=ref_label_key)
    adata.obs["ref_cell_type"] = adata.obs[ref_label_key]
    return adata


def annotate_ivig_cells(
    adata: ad.AnnData,
    resolution: float = 1.0,
    min_score: float = 0.15,
    n_top_genes: int = 3000,
    reference_path: Path | None = None,
    sample_key: str | None = None,
) -> tuple[ad.AnnData, AnnotationStats]:
    """Full cell type annotation pipeline for IVIG dataset.

    Args:
        adata: QC'd AnnData with raw counts.
        resolution: Leiden clustering resolution.
        min_score: Minimum marker score for cell type assignment.
        n_top_genes: Number of HVGs for dimensionality reduction.
        reference_path: Optional path to reference h5ad for label transfer.
        sample_key: obs column for sample identity.

    Returns:
        Tuple of (annotated AnnData, AnnotationStats).
    """
    print(f"Annotating {adata.n_obs:,} cells x {adata.n_vars:,} genes...")

    # Step 1: Preprocess
    adata = preprocess_for_clustering(adata, n_top_genes=n_top_genes)

    # Step 2: Cluster
    adata = cluster_cells(adata, resolution=resolution)

    # Step 3: Score and assign lineages
    adata = assign_lineage(adata)

    # Step 4: Score and assign fine cell types
    adata = score_cell_types(adata)
    adata = assign_cell_types(adata, min_score=min_score)

    # Step 5: Score IVIG-relevant gene modules
    adata = score_ivig_focus_genes(adata)

    # Step 6: Reference-based transfer (if available)
    if reference_path:
        adata = transfer_labels_from_reference(adata, reference_path)

    # Compile stats
    ct_counts = dict(adata.obs["cell_type"].value_counts())
    lin_counts = dict(adata.obs["lineage"].value_counts())
    unassigned = ct_counts.get("Unassigned", 0)

    stats = AnnotationStats(
        n_cells=adata.n_obs,
        n_clusters=adata.obs["leiden"].nunique(),
        n_cell_types=len([k for k in ct_counts if k != "Unassigned"]),
        cell_type_counts=ct_counts,
        lineage_counts=lin_counts,
        unassigned_count=unassigned,
        pct_unassigned=unassigned / adata.n_obs * 100 if adata.n_obs > 0 else 0,
        method="marker_based" if not reference_path else "marker_and_reference",
        resolution=resolution,
    )

    adata.uns["annotation_stats"] = stats.to_dict()
    print(f"\n{stats.summary()}")
    return adata, stats
