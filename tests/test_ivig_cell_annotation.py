"""Tests for IVIG cell type annotation pipeline."""

import numpy as np
import anndata as ad
import scipy.sparse as sp

from bioagentics.pandas_pans.ivig_cell_annotation import (
    PBMC_MARKERS,
    PBMC_LINEAGE_MARKERS,
    IVIG_FOCUS_GENES,
    AnnotationStats,
    assign_cell_types,
    assign_lineage,
    cluster_cells,
    preprocess_for_clustering,
    score_cell_types,
    score_ivig_focus_genes,
    annotate_ivig_cells,
)


def _make_pbmc_adata(n_cells: int = 300, seed: int = 42) -> ad.AnnData:
    """Create synthetic PBMC-like AnnData with known marker genes."""
    rng = np.random.default_rng(seed)

    # Collect all marker genes
    all_genes = set()
    for genes in PBMC_MARKERS.values():
        all_genes.update(genes)
    for genes in PBMC_LINEAGE_MARKERS.values():
        all_genes.update(genes)
    for genes in IVIG_FOCUS_GENES.values():
        all_genes.update(genes)
    # Add some filler genes
    n_filler = 200
    gene_names = sorted(all_genes) + [f"FILLER{i}" for i in range(n_filler)]
    n_genes = len(gene_names)

    # Generate count matrix
    X = rng.negative_binomial(n=2, p=0.3, size=(n_cells, n_genes)).astype(np.float32)

    # Boost monocyte markers in first cluster
    mono_indices = [gene_names.index(g) for g in ["CD14", "LYZ", "S100A8", "S100A9"] if g in gene_names]
    X[:50, mono_indices] *= 10

    # Boost T cell markers in second cluster
    t_indices = [gene_names.index(g) for g in ["CD3D", "CD3E"] if g in gene_names]
    X[50:150, t_indices] *= 10

    adata = ad.AnnData(
        X=sp.csr_matrix(X),
        var={"gene_symbols": gene_names},
    )
    adata.var_names = gene_names
    adata.obs_names = [f"cell_{i}" for i in range(n_cells)]
    adata.obs["sample"] = [f"sample_{i % 3}" for i in range(n_cells)]
    adata.obs["condition"] = ["PANS" if i < 150 else "Control" for i in range(n_cells)]
    return adata


class TestAnnotationStats:
    def test_to_dict(self):
        stats = AnnotationStats(n_cells=1000, n_clusters=15)
        d = stats.to_dict()
        assert d["n_cells"] == 1000
        assert d["n_clusters"] == 15

    def test_summary(self):
        stats = AnnotationStats(
            n_cells=1000,
            n_clusters=15,
            n_cell_types=8,
            cell_type_counts={"Monocyte": 300, "T_cell": 500},
            lineage_counts={"Myeloid": 400, "Lymphoid": 600},
        )
        s = stats.summary()
        assert "1,000" in s
        assert "15" in s


class TestPreprocessing:
    def test_normalize_and_pca(self):
        adata = _make_pbmc_adata()
        processed = preprocess_for_clustering(adata, n_top_genes=100)
        assert "X_pca" in processed.obsm
        assert "X_umap" in processed.obsm
        assert "raw_counts" in processed.layers
        assert "normalized" in processed.layers

    def test_preserves_obs(self):
        adata = _make_pbmc_adata()
        processed = preprocess_for_clustering(adata)
        assert "sample" in processed.obs.columns
        assert "condition" in processed.obs.columns


class TestClustering:
    def test_leiden(self):
        adata = _make_pbmc_adata()
        adata = preprocess_for_clustering(adata)
        adata = cluster_cells(adata, resolution=1.0)
        assert "leiden" in adata.obs.columns
        assert adata.obs["leiden"].nunique() >= 1

    def test_higher_resolution_more_clusters(self):
        adata = _make_pbmc_adata()
        adata = preprocess_for_clustering(adata)
        low = cluster_cells(adata.copy(), resolution=0.3, key_added="leiden_low")
        high = cluster_cells(adata.copy(), resolution=2.0, key_added="leiden_high")
        assert high.obs["leiden_high"].nunique() >= low.obs["leiden_low"].nunique()


class TestScoring:
    def test_score_cell_types(self):
        adata = _make_pbmc_adata()
        adata = preprocess_for_clustering(adata)
        adata = score_cell_types(adata)
        # Should have score columns
        score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
        assert len(score_cols) > 0

    def test_score_lineage(self):
        adata = _make_pbmc_adata()
        adata = preprocess_for_clustering(adata)
        adata = assign_lineage(adata)
        assert "lineage" in adata.obs.columns

    def test_assign_cell_types(self):
        adata = _make_pbmc_adata()
        adata = preprocess_for_clustering(adata)
        adata = score_cell_types(adata)
        adata = assign_cell_types(adata)
        assert "cell_type" in adata.obs.columns
        assert "cell_type_confidence" in adata.obs.columns

    def test_ivig_focus_genes(self):
        adata = _make_pbmc_adata()
        adata = preprocess_for_clustering(adata)
        adata = score_ivig_focus_genes(adata)
        for module in IVIG_FOCUS_GENES:
            assert f"ivig_{module}" in adata.obs.columns


class TestAnnotationPipeline:
    def test_full_pipeline(self):
        adata = _make_pbmc_adata()
        annotated, stats = annotate_ivig_cells(adata, resolution=0.5)

        assert "cell_type" in annotated.obs.columns
        assert "lineage" in annotated.obs.columns
        assert "leiden" in annotated.obs.columns
        assert "cell_type_confidence" in annotated.obs.columns
        assert "annotation_stats" in annotated.uns
        assert stats.n_cells == annotated.n_obs
        assert stats.n_clusters > 0

    def test_missing_reference_handled(self):
        from pathlib import Path

        adata = _make_pbmc_adata()
        annotated, stats = annotate_ivig_cells(
            adata,
            reference_path=Path("/nonexistent/ref.h5ad"),
        )
        assert annotated.obs["ref_cell_type"].eq("N/A").all()
