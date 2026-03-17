"""Tests for scRNA-seq differential expression module."""

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

from bioagentics.scrna.differential import DEResult, run_de_per_celltype, wilcoxon_de


def make_de_adata(n_cells: int = 400, n_genes: int = 100, seed: int = 42) -> ad.AnnData:
    """Create synthetic AnnData with two conditions and cell types."""
    rng = np.random.default_rng(seed)

    data = rng.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(np.float32)

    # Make first 10 genes DE in Th17 cells (inflamed > non-inflamed)
    # Th17 cells: 0-199, inflamed: 0-99, non-inflamed: 100-199
    data[:100, :10] = rng.negative_binomial(15, 0.3, size=(100, 10))  # inflamed Th17
    data[100:200, :10] = rng.negative_binomial(3, 0.3, size=(100, 10))  # non-inflamed Th17

    gene_names = [f"GENE{i}" for i in range(n_genes)]
    cell_types = ["Th17"] * 200 + ["Macrophage"] * 200
    conditions = (["inflamed"] * 100 + ["non-inflamed"] * 100) * 2

    X = sp.csr_matrix(data)
    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"cell_type": cell_types, "condition": conditions},
            index=[f"CELL_{i}" for i in range(n_cells)],
        ),
        var=pd.DataFrame(index=gene_names),
    )

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


class TestWilcoxonDE:
    def test_detects_de_genes(self):
        adata = make_de_adata()
        g1_mask = np.array(adata.obs["condition"] == "inflamed")
        g2_mask = np.array(adata.obs["condition"] == "non-inflamed")

        # Test on Th17 subset
        th17 = adata[adata.obs["cell_type"] == "Th17"]
        g1 = np.array(th17.obs["condition"] == "inflamed")
        g2 = np.array(th17.obs["condition"] == "non-inflamed")

        de_df = wilcoxon_de(th17, g1, g2)
        assert not de_df.empty
        assert "gene" in de_df.columns
        assert "logFC" in de_df.columns
        assert "padj" in de_df.columns

    def test_too_few_cells(self):
        adata = make_de_adata(n_cells=400)
        # Create masks with too few cells
        g1_mask = np.zeros(200, dtype=bool)
        g1_mask[:5] = True
        g2_mask = np.zeros(200, dtype=bool)
        g2_mask[5:10] = True

        th17 = adata[adata.obs["cell_type"] == "Th17"]
        de_df = wilcoxon_de(th17, g1_mask, g2_mask, min_cells=10)
        assert de_df.empty


class TestRunDEPerCelltype:
    def test_full_pipeline(self):
        adata = make_de_adata()
        results = run_de_per_celltype(
            adata,
            condition_key="condition",
            cell_type_key="cell_type",
            group1_label="inflamed",
            group2_label="non-inflamed",
        )
        assert len(results) > 0
        assert all(isinstance(r, DEResult) for r in results)

    def test_th17_has_de_genes(self):
        adata = make_de_adata()
        results = run_de_per_celltype(
            adata,
            condition_key="condition",
            group1_label="inflamed",
            group2_label="non-inflamed",
        )
        th17_result = next((r for r in results if r.cell_type == "Th17"), None)
        assert th17_result is not None
        assert th17_result.n_significant > 0

    def test_missing_condition_key(self):
        adata = make_de_adata()
        try:
            run_de_per_celltype(adata, condition_key="nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_summary(self):
        adata = make_de_adata()
        results = run_de_per_celltype(adata)
        for r in results:
            s = r.summary()
            assert r.cell_type in s
