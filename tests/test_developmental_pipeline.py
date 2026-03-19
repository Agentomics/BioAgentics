"""Tests for developmental trajectory pipeline steps 03-06.

Tests core logic functions using synthetic data (no network access required).
"""

import importlib
import numpy as np
import pandas as pd

from bioagentics.analysis.tourettes.brainspan_trajectories import (
    DEV_STAGES,
    CSTC_REGIONS,
)


# ---------------------------------------------------------------------------
# Import helpers for numeric-prefixed modules
# ---------------------------------------------------------------------------

def _import_step(step: str):
    """Import a pipeline step module by name (e.g. '03_expression_trajectories')."""
    return importlib.import_module(
        f"bioagentics.tourettes.developmental_trajectory.{step}"
    )


step03 = _import_step("03_expression_trajectories")
step04 = _import_step("04_temporal_clustering")
step05 = _import_step("05_enrichment_testing")
step06 = _import_step("06_wgcna_brainspan")


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

STAGE_ORDER = list(DEV_STAGES.keys())
REGION_LIST = list(CSTC_REGIONS.keys())


def _make_trajectory_df(
    genes: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic trajectory data matching extract_trajectories output."""
    rng = np.random.default_rng(seed)
    if genes is None:
        genes = ["BCL11B", "NDFIP2", "HDC", "SLITRK1", "CNTN6",
                 "NRXN1", "PPP5C", "EXOC1", "AR", "ESR1"]
    records = []
    for gene in genes:
        for stage in STAGE_ORDER:
            for region in REGION_LIST:
                rpkm = rng.lognormal(3, 1)
                records.append({
                    "gene_symbol": gene,
                    "dev_stage": stage,
                    "cstc_region": region,
                    "mean_rpkm": rpkm,
                    "mean_log2_rpkm": np.log2(rpkm + 1),
                    "n_samples": rng.integers(2, 6),
                })
    return pd.DataFrame(records)


# ===========================================================================
# Step 03 — Expression trajectories
# ===========================================================================


class TestComputeStageMeans:
    """Tests for 03_expression_trajectories.compute_stage_means."""

    def test_returns_expected_columns(self):
        df = _make_trajectory_df()
        result = step03.compute_stage_means(df)
        assert not result.empty
        for col in ("dev_stage", "cstc_region", "mean_rpkm", "mean_log2_rpkm",
                     "n_genes", "stage_order"):
            assert col in result.columns, f"missing column: {col}"

    def test_correct_stage_region_combos(self):
        df = _make_trajectory_df()
        result = step03.compute_stage_means(df)
        combos = set(zip(result["dev_stage"], result["cstc_region"]))
        expected = {(s, r) for s in STAGE_ORDER for r in REGION_LIST}
        assert combos == expected

    def test_empty_input(self):
        result = step03.compute_stage_means(pd.DataFrame())
        assert result.empty

    def test_stage_order_monotonic(self):
        df = _make_trajectory_df()
        result = step03.compute_stage_means(df)
        for region in REGION_LIST:
            sub = result[result["cstc_region"] == region]
            orders = sub["stage_order"].tolist()
            assert orders == sorted(orders), f"stage_order not sorted for {region}"


# ===========================================================================
# Step 04 — Temporal clustering
# ===========================================================================


class TestBuildExpressionMatrix:
    """Tests for 04_temporal_clustering.build_expression_matrix."""

    def test_shape(self):
        df = _make_trajectory_df()
        matrix = step04.build_expression_matrix(df)
        assert matrix.shape[0] == 10  # 10 genes
        assert matrix.shape[1] <= len(STAGE_ORDER)

    def test_columns_in_stage_order(self):
        df = _make_trajectory_df()
        matrix = step04.build_expression_matrix(df)
        cols = list(matrix.columns)
        stage_positions = [STAGE_ORDER.index(c) for c in cols]
        assert stage_positions == sorted(stage_positions)


class TestZscoreMatrix:
    """Tests for 04_temporal_clustering.zscore_matrix."""

    def test_zero_mean_per_gene(self):
        df = _make_trajectory_df()
        matrix = step04.build_expression_matrix(df)
        zs = step04.zscore_matrix(matrix)
        row_means = zs.mean(axis=1)
        np.testing.assert_allclose(row_means, 0, atol=1e-10)

    def test_unit_std_per_gene(self):
        df = _make_trajectory_df()
        matrix = step04.build_expression_matrix(df)
        zs = step04.zscore_matrix(matrix)
        row_stds = zs.std(axis=1)
        variable = row_stds[row_stds > 0]
        np.testing.assert_allclose(variable, 1.0, atol=1e-10)


class TestClusterGenes:
    """Tests for 04_temporal_clustering.cluster_genes."""

    def test_assignments_shape(self):
        df = _make_trajectory_df()
        matrix = step04.build_expression_matrix(df)
        zs = step04.zscore_matrix(matrix)
        assignments, centroids, _km = step04.cluster_genes(zs, n_clusters=3)
        assert len(assignments) == len(zs)
        assert set(assignments["cluster"].unique()).issubset({0, 1, 2})
        assert centroids.shape[0] == 3

    def test_all_genes_assigned(self):
        df = _make_trajectory_df()
        matrix = step04.build_expression_matrix(df)
        zs = step04.zscore_matrix(matrix)
        assignments, _, _ = step04.cluster_genes(zs, n_clusters=2)
        assert set(assignments["gene_symbol"]) == set(zs.index)


class TestCharacterizeClusters:
    """Tests for 04_temporal_clustering.characterize_clusters."""

    def test_characterization_count(self):
        centroids = np.array([
            [2, 1, 0, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, 0, 1, 2, 1, 0, -1],
            [-1, -1, -1, -1, 0, 1, 2, 1],
        ], dtype=float)
        result = step04.characterize_clusters(centroids, STAGE_ORDER)
        assert len(result) == 4

    def test_early_peak_detected(self):
        centroids = np.array([
            [2, 1.5, 0.5, 0, -0.5, -1, -1.5, -1.5],
        ], dtype=float)
        result = step04.characterize_clusters(centroids, STAGE_ORDER)
        assert result[0]["temporal_pattern"] == "early_peak"

    def test_flat_detected(self):
        centroids = np.array([
            [0.1, 0.15, 0.05, 0.12, 0.08, 0.11, 0.09, 0.13],
        ], dtype=float)
        result = step04.characterize_clusters(centroids, STAGE_ORDER)
        assert result[0]["temporal_pattern"] == "flat_constitutive"

    def test_fields_present(self):
        centroids = np.array([[1, 0, -1, 0, 1, 0, -1, 0]], dtype=float)
        result = step04.characterize_clusters(centroids, STAGE_ORDER)
        info = result[0]
        for key in ("cluster", "peak_stage", "trough_stage",
                     "peak_biological_window", "temporal_pattern",
                     "amplitude", "centroid"):
            assert key in info, f"missing key: {key}"


# ===========================================================================
# Step 05 — Enrichment testing
# ===========================================================================


class TestClusterEnrichment:
    """Tests for 05_enrichment_testing.test_cluster_enrichment."""

    def _make_assignments(self) -> pd.DataFrame:
        genes = [f"GENE{i}" for i in range(40)]
        clusters = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10
        return pd.DataFrame({"gene_symbol": genes, "cluster": clusters})

    def test_returns_result_per_cluster(self):
        assignments = self._make_assignments()
        test_genes = {"GENE0", "GENE1", "GENE2", "GENE3"}
        results = step05.test_cluster_enrichment(
            assignments, test_genes, n_permutations=200,
        )
        assert len(results) == 4
        assert all("fold_enrichment" in r for r in results)

    def test_enrichment_detected(self):
        assignments = self._make_assignments()
        test_genes = {"GENE0", "GENE1", "GENE2", "GENE3"}
        results = step05.test_cluster_enrichment(
            assignments, test_genes, n_permutations=500,
        )
        c0 = next(r for r in results if r["cluster"] == 0)
        assert c0["fold_enrichment"] > 1.0
        assert c0["test_genes_in_cluster"] == 4

    def test_no_test_genes_returns_empty(self):
        assignments = self._make_assignments()
        results = step05.test_cluster_enrichment(
            assignments, {"NOTEXIST"}, n_permutations=100,
        )
        assert results == []

    def test_result_fields(self):
        assignments = self._make_assignments()
        test_genes = {"GENE5", "GENE15", "GENE25"}
        results = step05.test_cluster_enrichment(
            assignments, test_genes, n_permutations=100,
        )
        for r in results:
            for key in ("cluster", "cluster_size", "test_genes_in_cluster",
                         "fold_enrichment", "hypergeometric_p", "permutation_p",
                         "significant_005", "genes_found"):
                assert key in r, f"missing key: {key}"


class TestStageEnrichment:
    """Tests for 05_enrichment_testing.test_stage_enrichment."""

    def test_returns_result_per_stage(self):
        genes = [f"GENE{i}" for i in range(20)]
        assignments = pd.DataFrame({
            "gene_symbol": genes,
            "cluster": [0] * 10 + [1] * 10,
        })
        rng = np.random.default_rng(42)
        zscore = pd.DataFrame(
            rng.standard_normal((20, 4)),
            index=genes,
            columns=STAGE_ORDER[:4],
        )
        test_genes = set(genes[:5])
        results = step05.test_stage_enrichment(
            assignments, zscore, test_genes, n_permutations=100,
        )
        assert len(results) == 4
        assert all(r["stage"] in STAGE_ORDER[:4] for r in results)


# ===========================================================================
# Step 06 — WGCNA module dynamics characterization
# ===========================================================================


class TestCharacterizeModuleDynamics:
    """Tests for 06_wgcna_brainspan.characterize_module_dynamics."""

    def _make_inputs(self):
        rng = np.random.default_rng(42)
        n_samples = 24
        genes = [f"G{i}" for i in range(30)]

        expr = pd.DataFrame(
            rng.lognormal(3, 1, (n_samples, len(genes))),
            columns=genes,
        )

        modules = pd.DataFrame({
            "gene_symbol": genes,
            "module": [1] * 15 + [2] * 15,
        })

        stages_per_sample = (STAGE_ORDER * 3)[:n_samples]
        sample_meta = pd.DataFrame({
            "sample_idx": range(n_samples),
            "dev_stage": stages_per_sample,
            "cstc_region": ["striatum"] * n_samples,
        })

        return expr, modules, sample_meta

    def test_returns_dynamics_per_module(self):
        expr, modules, meta = self._make_inputs()
        dynamics = step06.characterize_module_dynamics(expr, modules, meta)
        assert len(dynamics) == 2

    def test_dynamics_fields(self):
        expr, modules, meta = self._make_inputs()
        dynamics = step06.characterize_module_dynamics(expr, modules, meta)
        for d in dynamics:
            for key in ("module", "n_genes", "variance_explained",
                         "peak_stage", "trough_stage", "amplitude",
                         "temporal_pattern", "stage_eigengene"):
                assert key in d, f"missing key: {key}"

    def test_skips_module_zero(self):
        expr, modules, meta = self._make_inputs()
        extra = pd.DataFrame({"gene_symbol": ["EXTRA1", "EXTRA2"], "module": [0, 0]})
        modules = pd.concat([modules, extra], ignore_index=True)
        dynamics = step06.characterize_module_dynamics(expr, modules, meta)
        module_ids = {d["module"] for d in dynamics}
        assert 0 not in module_ids

    def test_temporal_pattern_valid(self):
        expr, modules, meta = self._make_inputs()
        dynamics = step06.characterize_module_dynamics(expr, modules, meta)
        valid_patterns = {"stable", "early_declining", "childhood_peak",
                          "adolescent_rising", "adult_peak"}
        for d in dynamics:
            assert d["temporal_pattern"] in valid_patterns
