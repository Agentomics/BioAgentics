"""Tests for scRNA-seq cell-type annotation module."""

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from bioagentics.scrna.annotation import (
    CELL_TYPE_MARKERS,
    AnnotationStats,
    annotate_cell_types,
    assign_cell_types,
    assign_lineage,
    score_cell_types,
)


def make_annotatable_adata(n_cells: int = 200, seed: int = 42) -> ad.AnnData:
    """Create synthetic AnnData with known marker genes for annotation testing."""
    rng = np.random.default_rng(seed)

    # Include a subset of known markers
    markers = ["IL17A", "RORC", "CCR6", "CD68", "FCGR1A", "COL1A1", "EPCAM", "CD3D", "CD3E", "CD79A"]
    filler = [f"GENE{i}" for i in range(90)]
    gene_names = markers + filler
    n_genes = len(gene_names)

    # Background expression
    data = rng.negative_binomial(2, 0.5, size=(n_cells, n_genes)).astype(np.float32)

    # Make some cells express Th17 markers highly (first 50 cells)
    for i, g in enumerate(gene_names):
        if g in ["IL17A", "RORC", "CCR6"]:
            data[:50, i] = rng.negative_binomial(20, 0.3, size=50)

    # Make cells 50-100 express macrophage markers
    for i, g in enumerate(gene_names):
        if g in ["CD68", "FCGR1A"]:
            data[50:100, i] = rng.negative_binomial(20, 0.3, size=50)

    X = sp.csr_matrix(data)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(index=[f"CELL_{i}" for i in range(n_cells)]),
        var=pd.DataFrame(index=gene_names),
    )

    # Normalize and log-transform (required for score_genes)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    return adata


class TestScoreCellTypes:
    def test_adds_score_columns(self):
        adata = make_annotatable_adata()
        markers = {"Th17": ["IL17A", "RORC", "CCR6"], "Mac": ["CD68", "FCGR1A"]}
        adata = score_cell_types(adata, markers=markers)
        assert "score_Th17" in adata.obs.columns
        assert "score_Mac" in adata.obs.columns

    def test_missing_genes_handled(self):
        adata = make_annotatable_adata()
        markers = {"NoGenes": ["NONEXISTENT1", "NONEXISTENT2"]}
        adata = score_cell_types(adata, markers=markers)
        assert "score_NoGenes" in adata.obs.columns
        assert (adata.obs["score_NoGenes"] == 0.0).all()


class TestAssignLineage:
    def test_assigns_lineage_column(self):
        adata = make_annotatable_adata()
        adata = assign_lineage(adata)
        assert "lineage" in adata.obs.columns
        assert adata.obs["lineage"].nunique() >= 1


class TestAssignCellTypes:
    def test_assigns_cell_type_column(self):
        adata = make_annotatable_adata()
        markers = {"Th17": ["IL17A", "RORC", "CCR6"], "Mac": ["CD68", "FCGR1A"]}
        adata = score_cell_types(adata, markers=markers)
        adata = assign_cell_types(adata, min_score=0.0)
        assert "cell_type" in adata.obs.columns

    def test_min_score_threshold(self):
        adata = make_annotatable_adata()
        markers = {"Th17": ["IL17A", "RORC", "CCR6"]}
        adata = score_cell_types(adata, markers=markers)
        adata = assign_cell_types(adata, min_score=100.0)  # Very high threshold
        assert (adata.obs["cell_type"] == "Unassigned").all()


class TestAnnotateCellTypes:
    def test_full_pipeline(self):
        adata = make_annotatable_adata(n_cells=200)
        markers = {"Th17": ["IL17A", "RORC", "CCR6"], "Mac": ["CD68", "FCGR1A"]}
        result, stats = annotate_cell_types(adata, markers=markers, min_score=0.1)

        assert isinstance(stats, AnnotationStats)
        assert stats.n_cells == 200
        assert "cell_type" in result.obs.columns
        assert "lineage" in result.obs.columns

    def test_stats_summary(self):
        adata = make_annotatable_adata(n_cells=100)
        markers = {"Th17": ["IL17A", "RORC"]}
        _, stats = annotate_cell_types(adata, markers=markers)
        summary = stats.summary()
        assert "Annotation Summary" in summary
