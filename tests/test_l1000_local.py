"""Tests for local L1000 connectivity scoring module."""

import numpy as np
import pytest

from bioagentics.data.cd_fibrosis.l1000_local import (
    BATCH_SIZE,
    FIBROBLAST_CELLS,
    GEO_GCTX_URLS,
    GEO_META_URLS,
    compute_connectivity_score,
    map_genes_to_rows,
)


class TestConstants:
    def test_geo_urls_defined(self):
        assert "phase2" in GEO_GCTX_URLS
        assert "phase1" in GEO_GCTX_URLS

    def test_meta_urls_defined(self):
        assert "phase2_siginfo" in GEO_META_URLS
        assert "phase2_geneinfo" in GEO_META_URLS

    def test_fibroblast_cells_defined(self):
        assert "IMR90" in FIBROBLAST_CELLS
        assert len(FIBROBLAST_CELLS) >= 3

    def test_batch_size_memory_safe(self):
        """Batch size should keep memory well under 2GB."""
        # 978 genes x batch_size x 4 bytes (float32)
        estimated_mb = 978 * BATCH_SIZE * 4 / 1e6
        assert estimated_mb < 500, f"Batch too large: ~{estimated_mb:.0f}MB"


class TestConnectivityScore:
    def test_perfect_reversal_negative(self):
        """When UP genes are most downregulated, score should be negative."""
        # Profile where first genes have lowest z-scores
        n = 100
        zscores = np.linspace(-3, 3, n)
        # UP genes at indices 0-4 (lowest z-scores = most downregulated by drug)
        up_indices = list(range(5))
        score = compute_connectivity_score(zscores, up_indices, [])
        assert score < 0, "Perfect reversal of UP genes should give negative score"

    def test_no_reversal_positive(self):
        """When UP genes are most upregulated, score should be positive."""
        n = 100
        zscores = np.linspace(-3, 3, n)
        # UP genes at indices 95-99 (highest z-scores = also upregulated)
        up_indices = list(range(95, 100))
        score = compute_connectivity_score(zscores, up_indices, [])
        assert score > 0, "No reversal should give positive score"

    def test_random_near_zero(self):
        """Random placement of UP genes should give score near zero."""
        np.random.seed(42)
        n = 1000
        zscores = np.random.randn(n)
        up_indices = list(range(0, 50))
        score = compute_connectivity_score(zscores, up_indices, [])
        assert abs(score) < 0.3, f"Random should be near zero, got {score}"

    def test_empty_returns_zero(self):
        score = compute_connectivity_score(np.array([]), [], [])
        assert score == 0.0

    def test_both_up_and_down(self):
        """Combined UP + DOWN scoring."""
        n = 100
        zscores = np.linspace(-3, 3, n)
        # UP genes at low end (reversed), DOWN genes at high end (reversed)
        up_indices = [0, 1, 2]
        down_indices = [97, 98, 99]
        score = compute_connectivity_score(zscores, up_indices, down_indices)
        assert score < 0, "Both sets reversed should be negative"


class TestMapGenesToRows:
    def test_maps_correctly(self):
        import pandas as pd

        gctx_row_ids = ["200814", "10357", "55811"]
        gene_info = pd.DataFrame({
            "pr_gene_id": [200814, 10357, 55811],
            "pr_gene_symbol": ["SERPINE1", "HDAC1", "ACTA2"],
        })
        query_genes = ["SERPINE1", "HDAC1", "FGF2"]

        result = map_genes_to_rows(gctx_row_ids, gene_info, query_genes)
        assert "SERPINE1" in result
        assert "HDAC1" in result
        assert "FGF2" not in result  # not in GCTX

    def test_empty_query(self):
        import pandas as pd

        gctx_row_ids = ["200814"]
        gene_info = pd.DataFrame({
            "pr_gene_id": [200814],
            "pr_gene_symbol": ["SERPINE1"],
        })
        result = map_genes_to_rows(gctx_row_ids, gene_info, [])
        assert len(result) == 0

    def test_case_insensitive(self):
        import pandas as pd

        gctx_row_ids = ["200814"]
        gene_info = pd.DataFrame({
            "pr_gene_id": [200814],
            "pr_gene_symbol": ["Serpine1"],
        })
        result = map_genes_to_rows(gctx_row_ids, gene_info, ["serpine1"])
        assert "SERPINE1" in result
