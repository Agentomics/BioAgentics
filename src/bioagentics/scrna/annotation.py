"""Cell-type annotation for IL-23/Th17 single-cell atlas.

Marker-based annotation using canonical gut immune and stromal markers,
with focus on IL-23-responsive cell types: Th17, ILC3, macrophages, DCs,
fibroblasts.

Usage:
    from bioagentics.scrna.annotation import annotate_cell_types
    adata = annotate_cell_types(adata)
"""

from __future__ import annotations

from dataclasses import dataclass

import anndata as ad
import scanpy as sc

# Canonical marker genes for gut immune and stromal cell types.
# Each cell type has positive markers (expected high) used for scoring.
CELL_TYPE_MARKERS: dict[str, list[str]] = {
    # --- T cell subsets ---
    "Th17": ["IL17A", "IL17F", "RORC", "CCR6", "IL23R", "IL22", "AHR"],
    "Th1": ["IFNG", "TBX21", "CXCR3", "TNF", "IL12RB2"],
    "Treg": ["FOXP3", "IL2RA", "CTLA4", "IKZF2", "TIGIT"],
    "CD8_T": ["CD8A", "CD8B", "GZMB", "PRF1", "NKG7"],
    "CD4_T_naive": ["CCR7", "SELL", "LEF1", "TCF7", "IL7R"],
    # --- Innate lymphoid ---
    "ILC3": ["RORC", "IL23R", "IL22", "NCR2", "KIT", "IL1R1"],
    "NK": ["NKG7", "GNLY", "KLRD1", "KLRB1", "NCAM1"],
    # --- Myeloid ---
    "Macrophage": ["CD68", "CD14", "FCGR1A", "CSF1R", "MARCO", "C1QA", "C1QB"],
    "Inflammatory_Mac": ["FCGR1A", "IL23A", "IL1B", "TNF", "CXCL10", "CD14"],
    "Dendritic_cell": ["CLEC9A", "XCR1", "BATF3", "IRF8", "FLT3"],
    "pDC": ["LILRA4", "IRF7", "CLEC4C", "IL3RA"],
    "Monocyte": ["CD14", "VCAN", "S100A8", "S100A9", "FCN1"],
    # --- B cells / Plasma ---
    "B_cell": ["CD79A", "MS4A1", "CD19", "PAX5", "BANK1"],
    "Plasma": ["JCHAIN", "MZB1", "XBP1", "PRDM1", "SDC1"],
    # --- Stromal ---
    "Fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM", "PDGFRA"],
    "Myofibroblast": ["ACTA2", "TAGLN", "MYH11", "CNN1"],
    "Endothelial": ["PECAM1", "VWF", "CDH5", "CLDN5", "FLT1"],
    # --- Epithelial ---
    "Epithelial": ["EPCAM", "KRT18", "KRT19", "CDH1", "MUC2"],
    "Goblet": ["MUC2", "TFF3", "FCGBP", "SPINK4", "CLCA1"],
    "Paneth": ["DEFA5", "DEFA6", "LYZ", "REG3A"],
}

# Broad lineage groupings for hierarchical annotation
LINEAGE_MARKERS: dict[str, list[str]] = {
    "T_cell": ["CD3D", "CD3E", "CD3G", "TRAC"],
    "B_lineage": ["CD79A", "MS4A1", "CD19", "JCHAIN"],
    "Myeloid": ["LYZ", "CD14", "CST3", "AIF1"],
    "Stromal": ["COL1A1", "COL1A2", "DCN", "VIM"],
    "Epithelial": ["EPCAM", "KRT18", "KRT19"],
}


@dataclass
class AnnotationStats:
    """Cell-type annotation summary."""

    n_cells: int
    n_cell_types: int
    n_lineages: int
    cell_type_counts: dict[str, int]
    lineage_counts: dict[str, int]
    unassigned_count: int
    pct_unassigned: float

    def summary(self) -> str:
        ct_lines = "\n".join(
            f"    {k}: {v:,}" for k, v in sorted(self.cell_type_counts.items(), key=lambda x: -x[1])
        )
        lin_lines = "\n".join(
            f"    {k}: {v:,}" for k, v in sorted(self.lineage_counts.items(), key=lambda x: -x[1])
        )
        return (
            f"Annotation Summary:\n"
            f"  Total cells: {self.n_cells:,}\n"
            f"  Cell types: {self.n_cell_types}\n"
            f"  Unassigned: {self.unassigned_count:,} ({self.pct_unassigned:.1f}%)\n"
            f"  Lineages:\n{lin_lines}\n"
            f"  Cell types:\n{ct_lines}"
        )


def score_cell_types(
    adata: ad.AnnData,
    markers: dict[str, list[str]] | None = None,
    score_prefix: str = "score_",
) -> ad.AnnData:
    """Score each cell for each cell type using scanpy's score_genes.

    Stores scores in adata.obs['{score_prefix}{cell_type}'].
    """
    if markers is None:
        markers = CELL_TYPE_MARKERS

    for cell_type, genes in markers.items():
        # Only use genes present in the dataset
        available = [g for g in genes if g in adata.var_names]
        if len(available) < 2:
            adata.obs[f"{score_prefix}{cell_type}"] = 0.0
            continue

        sc.tl.score_genes(adata, gene_list=available, score_name=f"{score_prefix}{cell_type}")

    return adata


def assign_lineage(adata: ad.AnnData) -> ad.AnnData:
    """Assign broad lineage based on lineage marker scores."""
    adata = score_cell_types(adata, markers=LINEAGE_MARKERS, score_prefix="lineage_")

    lineage_cols = [f"lineage_{lin}" for lin in LINEAGE_MARKERS]
    available_cols = [c for c in lineage_cols if c in adata.obs.columns]

    if not available_cols:
        adata.obs["lineage"] = "Unknown"
        return adata

    scores_df = adata.obs[available_cols]
    best_lineage = scores_df.idxmax(axis=1).str.replace("lineage_", "", regex=False)

    # Mark as Unknown if max score is below threshold
    max_scores = scores_df.max(axis=1)
    best_lineage[max_scores < 0.1] = "Unknown"

    adata.obs["lineage"] = best_lineage.values
    return adata


def assign_cell_types(
    adata: ad.AnnData,
    min_score: float = 0.2,
) -> ad.AnnData:
    """Assign fine-grained cell types based on marker scores.

    Uses the highest scoring cell type per cell, with a minimum score threshold.
    """
    score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    if not score_cols:
        adata.obs["cell_type"] = "Unassigned"
        return adata

    scores_df = adata.obs[score_cols]
    best_type = scores_df.idxmax(axis=1).str.replace("score_", "", regex=False)
    max_scores = scores_df.max(axis=1)

    # Apply minimum score threshold
    best_type[max_scores < min_score] = "Unassigned"

    adata.obs["cell_type"] = best_type.values
    return adata


def annotate_cell_types(
    adata: ad.AnnData,
    markers: dict[str, list[str]] | None = None,
    min_score: float = 0.2,
) -> tuple[ad.AnnData, AnnotationStats]:
    """Full annotation pipeline: lineage assignment + fine cell type scoring.

    Args:
        adata: Integrated AnnData (should be log-normalized).
        markers: Custom marker dict, or None for default IL-23/gut markers.
        min_score: Minimum score to assign a cell type.

    Returns:
        Tuple of (annotated AnnData, AnnotationStats).
    """
    print(f"Annotating {adata.n_obs:,} cells...")

    # Check gene overlap
    all_markers = set()
    for genes in (markers or CELL_TYPE_MARKERS).values():
        all_markers.update(genes)
    overlap = len(all_markers & set(adata.var_names))
    print(f"  Marker gene overlap: {overlap}/{len(all_markers)}")

    # Score cell types
    adata = score_cell_types(adata, markers=markers)

    # Assign lineage
    adata = assign_lineage(adata)

    # Assign fine cell types
    adata = assign_cell_types(adata, min_score=min_score)

    # Stats
    ct_counts = adata.obs["cell_type"].value_counts().to_dict()
    lin_counts = adata.obs["lineage"].value_counts().to_dict()
    unassigned = ct_counts.get("Unassigned", 0)

    stats = AnnotationStats(
        n_cells=adata.n_obs,
        n_cell_types=len([k for k in ct_counts if k != "Unassigned"]),
        n_lineages=len([k for k in lin_counts if k != "Unknown"]),
        cell_type_counts=ct_counts,
        lineage_counts=lin_counts,
        unassigned_count=unassigned,
        pct_unassigned=unassigned / adata.n_obs * 100 if adata.n_obs > 0 else 0,
    )

    print(f"\n{stats.summary()}")
    return adata, stats
