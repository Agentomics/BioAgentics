"""Tests for bioagentics.models.batch_correction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import anndata as ad

from bioagentics.models.batch_correction import combat_correct, plot_pca_comparison


def _make_batched_adata(n_genes: int = 100, n_per_batch: int = 10, seed: int = 42) -> ad.AnnData:
    """Create synthetic AnnData with two batches having a systematic offset."""
    rng = np.random.default_rng(seed)
    n_total = n_per_batch * 2

    # Batch 1: baseline expression
    batch1 = rng.poisson(50, (n_per_batch, n_genes)).astype(float)
    # Batch 2: same biology + systematic offset
    batch2 = rng.poisson(50, (n_per_batch, n_genes)).astype(float) + 20.0

    X = np.vstack([batch1, batch2]).astype(np.float32)
    obs = pd.DataFrame({
        "dataset": ["batch_A"] * n_per_batch + ["batch_B"] * n_per_batch,
        "condition": (["case", "control"] * n_total)[:n_total],
    }, index=[f"S{i}" for i in range(n_total)])
    var = pd.DataFrame(index=[f"G{i}" for i in range(n_genes)])

    return ad.AnnData(X=X, obs=obs, var=var)


class TestCombatCorrect:
    def test_basic_correction(self):
        adata = _make_batched_adata()
        result = combat_correct(adata, batch_key="dataset")

        assert result.n_obs == adata.n_obs
        assert result.n_vars == adata.n_vars
        assert "pre_combat" in result.layers

    def test_reduces_batch_difference(self):
        adata = _make_batched_adata()
        result = combat_correct(adata, batch_key="dataset")

        # Mean difference between batches should be smaller after correction
        mask_a = result.obs["dataset"] == "batch_A"
        mask_b = result.obs["dataset"] == "batch_B"

        before_diff = abs(adata.X[mask_a].mean() - adata.X[mask_b].mean())
        after_diff = abs(result.X[mask_a].mean() - result.X[mask_b].mean())

        assert after_diff < before_diff

    def test_single_batch_skips(self):
        adata = _make_batched_adata()
        adata.obs["dataset"] = "single_batch"
        result = combat_correct(adata, batch_key="dataset")
        # Should return unchanged
        np.testing.assert_array_equal(result.X, adata.X)

    def test_missing_batch_key_raises(self):
        adata = _make_batched_adata()
        try:
            combat_correct(adata, batch_key="nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)


class TestPCAPlot:
    def test_plot_runs(self, tmp_path):
        adata = _make_batched_adata()
        corrected = combat_correct(adata, batch_key="dataset")
        before = corrected.copy()
        before.X = before.layers["pre_combat"]

        plot_path = tmp_path / "pca.png"
        plot_pca_comparison(before, corrected, batch_key="dataset", save_path=plot_path)
        assert plot_path.exists()
