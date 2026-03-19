"""Tests for developmental trajectory pipeline steps 03-09.

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
step07 = _import_step("07_critical_period_modules")


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


# ===========================================================================
# Step 07 — Critical period gene module analysis (Phase 2)
# ===========================================================================


def _make_cluster_info() -> list[dict]:
    """Create synthetic cluster characterization matching step 04 output."""
    return [
        {"cluster": 0, "temporal_pattern": "early_peak",
         "peak_stage": "early_prenatal", "amplitude": 1.5},
        {"cluster": 1, "temporal_pattern": "onset_window_peak",
         "peak_stage": "late_childhood", "amplitude": 1.2},
        {"cluster": 2, "temporal_pattern": "adolescent_peak",
         "peak_stage": "adolescence", "amplitude": 0.9},
        {"cluster": 3, "temporal_pattern": "flat_constitutive",
         "peak_stage": "adulthood", "amplitude": 0.3},
    ]


def _make_cluster_assignments(n_genes: int = 40) -> pd.DataFrame:
    """Create synthetic cluster assignments matching step 04 output."""
    genes = [f"GENE{i}" for i in range(n_genes)]
    clusters = [i % 4 for i in range(n_genes)]
    return pd.DataFrame({"gene_symbol": genes, "cluster": clusters})


def _make_wgcna_dynamics() -> list[dict]:
    """Create synthetic WGCNA module dynamics matching step 06 output."""
    return [
        {"module": 1, "temporal_pattern": "childhood_peak",
         "peak_stage": "late_childhood", "amplitude": 1.3,
         "n_genes": 25, "variance_explained": 0.45},
        {"module": 2, "temporal_pattern": "adolescent_rising",
         "peak_stage": "adolescence", "amplitude": 0.8,
         "n_genes": 20, "variance_explained": 0.38},
        {"module": 3, "temporal_pattern": "stable",
         "peak_stage": "adulthood", "amplitude": 0.2,
         "n_genes": 30, "variance_explained": 0.52},
    ]


def _make_wgcna_modules(n_genes: int = 75) -> pd.DataFrame:
    """Create synthetic WGCNA module assignments matching step 06 output."""
    genes = [f"WGENE{i}" for i in range(n_genes)]
    modules = [1] * 25 + [2] * 20 + [3] * 30
    return pd.DataFrame({"gene_symbol": genes, "module": modules})


def _make_wgcna_enrichment() -> list[dict]:
    """Create synthetic WGCNA enrichment results matching step 06 output."""
    return [
        {"module": 1, "significant": True, "fold_enrichment": 2.5},
        {"module": 2, "significant": False, "fold_enrichment": 0.8},
        {"module": 3, "significant": False, "fold_enrichment": 1.1},
    ]


class TestMatchTemporalClustersToWindows:
    """Tests for 07_critical_period_modules.match_temporal_clusters_to_windows."""

    def test_returns_all_windows(self):
        info = _make_cluster_info()
        assignments = _make_cluster_assignments()
        matched = step07.match_temporal_clusters_to_windows(info, assignments)
        assert "onset" in matched
        assert "peak_severity" in matched
        assert "remission" in matched

    def test_onset_matches_onset_window_pattern(self):
        info = _make_cluster_info()
        assignments = _make_cluster_assignments()
        matched = step07.match_temporal_clusters_to_windows(info, assignments)
        onset = matched["onset"]
        assert onset["n_genes"] > 0
        cluster_patterns = [c["temporal_pattern"] for c in onset["matched_clusters"]]
        assert "onset_window_peak" in cluster_patterns

    def test_remission_matches_adolescent_pattern(self):
        info = _make_cluster_info()
        assignments = _make_cluster_assignments()
        matched = step07.match_temporal_clusters_to_windows(info, assignments)
        remission = matched["remission"]
        assert remission["n_genes"] > 0
        cluster_patterns = [c["temporal_pattern"] for c in remission["matched_clusters"]]
        assert "adolescent_peak" in cluster_patterns

    def test_genes_come_from_matched_clusters(self):
        info = _make_cluster_info()
        assignments = _make_cluster_assignments()
        matched = step07.match_temporal_clusters_to_windows(info, assignments)
        # Onset cluster (1 = onset_window_peak) should contain genes from cluster 1
        onset_genes = set(matched["onset"]["genes"])
        cluster1_genes = set(
            assignments[assignments["cluster"] == 1]["gene_symbol"]
        )
        assert onset_genes & cluster1_genes  # Should overlap


class TestMatchWgcnaModulesToWindows:
    """Tests for 07_critical_period_modules.match_wgcna_modules_to_windows."""

    def test_returns_all_windows(self):
        dynamics = _make_wgcna_dynamics()
        modules = _make_wgcna_modules()
        enrichment = _make_wgcna_enrichment()
        matched = step07.match_wgcna_modules_to_windows(dynamics, modules, enrichment)
        assert set(matched.keys()) == {"onset", "peak_severity", "remission"}

    def test_marks_ts_enriched_modules(self):
        dynamics = _make_wgcna_dynamics()
        modules = _make_wgcna_modules()
        enrichment = _make_wgcna_enrichment()
        matched = step07.match_wgcna_modules_to_windows(dynamics, modules, enrichment)
        # Module 1 is childhood_peak (matches onset/peak) and is enriched
        onset = matched["onset"]
        enriched_mods = [m for m in onset["matched_modules"] if m["ts_enriched"]]
        assert len(enriched_mods) > 0


class TestCombineWindowGenes:
    """Tests for 07_critical_period_modules.combine_window_genes."""

    def test_union_includes_both_sources(self):
        temporal = {
            "onset": {"genes": ["A", "B", "C"]},
            "peak_severity": {"genes": ["D"]},
            "remission": {"genes": ["E"]},
        }
        wgcna = {
            "onset": {"genes": ["B", "C", "D"]},
            "peak_severity": {"genes": ["F"]},
            "remission": {"genes": ["G"]},
        }
        combined = step07.combine_window_genes(temporal, wgcna)
        assert set(combined["onset"]["genes_union"]) == {"A", "B", "C", "D"}
        assert set(combined["onset"]["genes_both"]) == {"B", "C"}

    def test_counts_are_correct(self):
        temporal = {
            "onset": {"genes": ["A", "B"]},
            "peak_severity": {"genes": []},
            "remission": {"genes": ["X"]},
        }
        wgcna = {
            "onset": {"genes": ["B", "C"]},
            "peak_severity": {"genes": ["Y"]},
            "remission": {"genes": []},
        }
        combined = step07.combine_window_genes(temporal, wgcna)
        assert combined["onset"]["n_temporal"] == 2
        assert combined["onset"]["n_wgcna"] == 2
        assert combined["onset"]["n_union"] == 3
        assert combined["onset"]["n_intersection"] == 1


class TestTsEnrichmentInWindow:
    """Tests for 07_critical_period_modules.test_ts_enrichment_in_window."""

    def test_returns_expected_fields(self):
        bg = [f"G{i}" for i in range(100)]
        window = bg[:30]
        ts_genes = set(bg[:10])
        result = step07.test_ts_enrichment_in_window(
            window, bg, ts_genes, n_permutations=100,
        )
        for key in ("observed", "expected", "fold_enrichment",
                     "fisher_p", "permutation_p", "significant", "ts_genes_found"):
            assert key in result, f"missing key: {key}"

    def test_detects_enrichment(self):
        # All TS genes are in the window — should be enriched
        bg = [f"G{i}" for i in range(100)]
        window = bg[:20]
        ts_genes = set(bg[:15])  # 15 of 20 window genes are TS genes
        result = step07.test_ts_enrichment_in_window(
            window, bg, ts_genes, n_permutations=500,
        )
        assert result["fold_enrichment"] > 1.0
        assert result["observed"] == 15

    def test_empty_window(self):
        bg = [f"G{i}" for i in range(50)]
        result = step07.test_ts_enrichment_in_window(
            [], bg, set(bg[:5]), n_permutations=100,
        )
        assert result["observed"] == 0
        assert result["fold_enrichment"] == 0


class TestPathwayEnrichment:
    """Tests for 07_critical_period_modules.test_pathway_enrichment."""

    def test_returns_results(self):
        # Build background that includes some pathway genes
        pathway_genes = set()
        for pw in step07.NEURODEVEL_PATHWAYS.values():
            pathway_genes.update(pw.keys())
        bg = sorted(pathway_genes) + [f"BG{i}" for i in range(200)]
        # Window includes some dopamine genes
        da_genes = list(step07.NEURODEVEL_PATHWAYS["dopamine_signaling"].keys())
        window = da_genes + [f"BG{i}" for i in range(20)]
        results = step07.test_pathway_enrichment(window, bg)
        assert len(results) > 0
        assert all("pathway" in r for r in results)
        assert all("fisher_p" in r for r in results)

    def test_sorted_by_pvalue(self):
        pathway_genes = set()
        for pw in step07.NEURODEVEL_PATHWAYS.values():
            pathway_genes.update(pw.keys())
        bg = sorted(pathway_genes) + [f"BG{i}" for i in range(200)]
        window = list(step07.NEURODEVEL_PATHWAYS["dopamine_signaling"].keys())
        results = step07.test_pathway_enrichment(window, bg)
        pvals = [r["fisher_p"] for r in results]
        assert pvals == sorted(pvals)

    def test_empty_window_returns_empty(self):
        bg = [f"G{i}" for i in range(50)]
        results = step07.test_pathway_enrichment([], bg)
        assert results == []


# ===========================================================================
# Step 08 — Cell-type developmental dynamics (Phase 3)
# ===========================================================================

step08 = _import_step("08_celltype_deconvolution")


def _make_celltype_trajectory_df(
    include_markers: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic trajectory data with cell-type marker genes.

    If include_markers is True, includes canonical marker genes that step 08
    will recognise. Generates expression patterns that mimic expected biology:
    - Cholinergic markers peak in late_childhood
    - PV markers peak in adolescence
    - D1/D2 markers are present across stages
    """
    rng = np.random.default_rng(seed)

    if include_markers:
        genes = [
            # D1-MSN markers
            "DRD1", "TAC1", "PDYN", "CHRM4", "ISL1",
            # D2-MSN markers
            "DRD2", "PENK", "ADORA2A", "GPR6", "SP9",
            # Cholinergic markers
            "CHAT", "SLC5A7", "SLC18A3", "LHX8", "GBX2",
            # PV markers
            "PVALB", "KCNC1", "KCNC2", "EYA1", "TAC3",
            # Microglia markers
            "CX3CR1", "P2RY12", "TMEM119", "AIF1", "CSF1R",
            # Background genes
            "BG1", "BG2", "BG3", "BG4", "BG5",
        ]
    else:
        genes = [f"GENE{i}" for i in range(30)]

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


class TestComputeCelltypeScores:
    """Tests for 08_celltype_deconvolution.compute_celltype_scores."""

    def test_returns_expected_columns(self):
        df = _make_celltype_trajectory_df()
        result = step08.compute_celltype_scores(df)
        assert not result.empty
        for col in ("celltype", "dev_stage", "cstc_region", "score",
                     "n_markers_found", "n_markers_total", "stage_order"):
            assert col in result.columns, f"missing column: {col}"

    def test_all_celltypes_with_markers_present(self):
        df = _make_celltype_trajectory_df()
        result = step08.compute_celltype_scores(df)
        celltypes = set(result["celltype"].unique())
        # Should find at least D1, D2, cholinergic, PV, microglia
        for expected in ["d1_msn", "d2_msn", "cholinergic_interneuron",
                          "pv_interneuron", "microglia"]:
            assert expected in celltypes, f"missing celltype: {expected}"

    def test_stage_order_monotonic(self):
        df = _make_celltype_trajectory_df()
        result = step08.compute_celltype_scores(df)
        for ct in result["celltype"].unique():
            for region in result["cstc_region"].unique():
                sub = result[
                    (result["celltype"] == ct) & (result["cstc_region"] == region)
                ]
                orders = sub["stage_order"].tolist()
                assert orders == sorted(orders), (
                    f"stage_order not sorted for {ct}/{region}"
                )

    def test_empty_input(self):
        result = step08.compute_celltype_scores(pd.DataFrame())
        assert result.empty

    def test_no_marker_genes_returns_empty(self):
        df = _make_celltype_trajectory_df(include_markers=False)
        result = step08.compute_celltype_scores(df)
        assert result.empty

    def test_custom_markers(self):
        df = _make_celltype_trajectory_df()
        custom = {"test_type": {"DRD1": "test", "DRD2": "test"}}
        result = step08.compute_celltype_scores(df, celltype_markers=custom)
        assert set(result["celltype"].unique()) == {"test_type"}


class TestComputeMsnRatio:
    """Tests for 08_celltype_deconvolution.compute_msn_ratio."""

    def test_returns_expected_columns(self):
        df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(df)
        ratio = step08.compute_msn_ratio(scores)
        assert not ratio.empty
        for col in ("dev_stage", "d1_score", "d2_score", "d2_d1_ratio",
                     "stage_order"):
            assert col in ratio.columns, f"missing column: {col}"

    def test_ratio_positive(self):
        df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(df)
        ratio = step08.compute_msn_ratio(scores)
        assert (ratio["d2_d1_ratio"] > 0).all()

    def test_empty_input(self):
        result = step08.compute_msn_ratio(pd.DataFrame())
        assert result.empty


class TestOnsetCholinergicPeak:
    """Tests for 08_celltype_deconvolution.test_onset_cholinergic_peak."""

    def test_returns_expected_fields(self):
        df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(df)
        result = step08.test_onset_cholinergic_peak(scores)
        for key in ("hypothesis", "celltype", "window_mean", "other_mean",
                     "effect_size", "elevated_in_window", "p_value"):
            assert key in result, f"missing key: {key}"

    def test_celltype_is_cholinergic(self):
        df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(df)
        result = step08.test_onset_cholinergic_peak(scores)
        assert result["celltype"] == "cholinergic_interneuron"

    def test_empty_scores(self):
        result = step08.test_onset_cholinergic_peak(pd.DataFrame())
        assert "error" in result


class TestRemissionPvIncrease:
    """Tests for 08_celltype_deconvolution.test_remission_pv_increase."""

    def test_returns_expected_fields(self):
        df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(df)
        result = step08.test_remission_pv_increase(scores)
        for key in ("hypothesis", "celltype", "window_mean", "other_mean",
                     "effect_size", "p_value"):
            assert key in result, f"missing key: {key}"

    def test_celltype_is_pv(self):
        df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(df)
        result = step08.test_remission_pv_increase(scores)
        assert result["celltype"] == "pv_interneuron"


class TestMicrogliaDevelopmental:
    """Tests for 08_celltype_deconvolution.test_microglia_developmental_component."""

    def test_returns_expected_fields(self):
        df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(df)
        result = step08.test_microglia_developmental_component(scores)
        for key in ("hypothesis", "spearman_rho", "spearman_p",
                     "peak_stage", "trough_stage"):
            assert key in result, f"missing key: {key}"

    def test_empty_scores(self):
        result = step08.test_microglia_developmental_component(pd.DataFrame())
        assert "error" in result


class TestMsnRatioTrajectory:
    """Tests for 08_celltype_deconvolution.test_msn_ratio_trajectory."""

    def test_returns_expected_fields(self):
        df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(df)
        result = step08.test_msn_ratio_trajectory(scores)
        for key in ("hypothesis", "spearman_rho", "spearman_p",
                     "peak_ratio_stage", "trajectory"):
            assert key in result, f"missing key: {key}"

    def test_trajectory_has_stages(self):
        df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(df)
        result = step08.test_msn_ratio_trajectory(scores)
        assert len(result["trajectory"]) > 0

    def test_empty_scores(self):
        result = step08.test_msn_ratio_trajectory(pd.DataFrame())
        assert "error" in result


class TestRunAllHypotheses:
    """Tests for 08_celltype_deconvolution.run_all_hypotheses."""

    def test_returns_four_results(self):
        df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(df)
        results = step08.run_all_hypotheses(scores)
        assert len(results) == 4

    def test_each_result_has_hypothesis(self):
        df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(df)
        results = step08.run_all_hypotheses(scores)
        assert all("hypothesis" in r for r in results)


# ===========================================================================
# Step 09 — Persistence vs. remission model (Phase 4)
# ===========================================================================

step09 = _import_step("09_persistence_remission_model")


def _make_phase4_trajectory_df(seed: int = 42) -> pd.DataFrame:
    """Create synthetic trajectory data with genes needed for Phase 4.

    Includes PV interneuron markers, GABA signaling genes, glutamate/DA genes,
    dorsal zone markers, hormone receptors, and TS risk genes.
    """
    rng = np.random.default_rng(seed)
    genes = [
        # PV interneuron markers
        "PVALB", "KCNC1", "KCNC2", "EYA1", "TAC3",
        # GABA signaling
        "GAD1", "GAD2", "SLC32A1", "GABRA1", "GABRG2",
        # Excitatory / dopamine
        "GRIN1", "GRIN2A", "GRIA1", "DRD1", "DRD2", "TH",
        # Dorsal zone markers
        "FOXP2", "EPHA4", "MET", "SEMA3A", "CACNA1A",
        "KCND2", "GRIA3", "GRID2",
        # Hormone receptors
        "AR", "ESR1", "ESR2",
        # TS risk genes
        "BCL11B", "NDFIP2", "SLITRK1", "HDC", "NRXN1",
    ]
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


class TestComputeRemissionScore:
    """Tests for 09_persistence_remission_model.compute_remission_score."""

    def test_returns_expected_columns(self):
        df = _make_phase4_trajectory_df()
        result = step09.compute_remission_score(df)
        assert not result.empty
        for col in ("dev_stage", "pv_score", "gaba_score",
                     "remission_score", "stage_order"):
            assert col in result.columns, f"missing column: {col}"

    def test_all_stages_present(self):
        df = _make_phase4_trajectory_df()
        result = step09.compute_remission_score(df)
        assert set(result["dev_stage"]) == set(STAGE_ORDER)

    def test_stage_order_monotonic(self):
        df = _make_phase4_trajectory_df()
        result = step09.compute_remission_score(df)
        orders = result["stage_order"].tolist()
        assert orders == sorted(orders)

    def test_empty_input(self):
        result = step09.compute_remission_score(pd.DataFrame())
        assert result.empty

    def test_no_matching_genes(self):
        df = _make_trajectory_df(genes=["FAKEGENE1", "FAKEGENE2"])
        result = step09.compute_remission_score(df)
        assert result.empty

    def test_remission_score_is_average(self):
        df = _make_phase4_trajectory_df()
        result = step09.compute_remission_score(df)
        for _, row in result.iterrows():
            expected = (row["pv_score"] + row["gaba_score"]) / 2
            np.testing.assert_allclose(row["remission_score"], expected, atol=1e-10)


class TestComputePersistenceScore:
    """Tests for 09_persistence_remission_model.compute_persistence_score."""

    def test_returns_expected_columns(self):
        df = _make_phase4_trajectory_df()
        result = step09.compute_persistence_score(df)
        assert not result.empty
        for col in ("dev_stage", "excitatory_score", "inhibitory_deficit",
                     "persistence_score", "stage_order"):
            assert col in result.columns, f"missing column: {col}"

    def test_all_stages_present(self):
        df = _make_phase4_trajectory_df()
        result = step09.compute_persistence_score(df)
        assert set(result["dev_stage"]) == set(STAGE_ORDER)

    def test_persistence_score_is_average(self):
        df = _make_phase4_trajectory_df()
        result = step09.compute_persistence_score(df)
        for _, row in result.iterrows():
            expected = (row["excitatory_score"] + row["inhibitory_deficit"]) / 2
            np.testing.assert_allclose(row["persistence_score"], expected, atol=1e-10)

    def test_empty_input(self):
        result = step09.compute_persistence_score(pd.DataFrame())
        assert result.empty


class TestRemissionAdolescentIncrease:
    """Tests for 09_persistence_remission_model.test_remission_adolescent_increase."""

    def test_returns_expected_fields(self):
        df = _make_phase4_trajectory_df()
        rem = step09.compute_remission_score(df)
        result = step09.test_remission_adolescent_increase(rem)
        for key in ("hypothesis", "adolescent_mean", "pre_remission_mean",
                     "effect_size", "increases_in_adolescence",
                     "spearman_rho", "spearman_p", "trajectory"):
            assert key in result, f"missing key: {key}"

    def test_trajectory_has_postnatal_stages(self):
        df = _make_phase4_trajectory_df()
        rem = step09.compute_remission_score(df)
        result = step09.test_remission_adolescent_increase(rem)
        postnatal = set(step09.POSTNATAL_STAGES)
        assert set(result["trajectory"].keys()) == postnatal

    def test_empty_input(self):
        result = step09.test_remission_adolescent_increase(pd.DataFrame())
        assert "error" in result


class TestPersistencePlateau:
    """Tests for 09_persistence_remission_model.test_persistence_plateau."""

    def test_returns_expected_fields(self):
        df = _make_phase4_trajectory_df()
        pers = step09.compute_persistence_score(df)
        result = step09.test_persistence_plateau(pers)
        for key in ("hypothesis", "adolescent_mean", "pre_remission_mean",
                     "does_not_decline", "spearman_rho", "trajectory"):
            assert key in result, f"missing key: {key}"

    def test_empty_input(self):
        result = step09.test_persistence_plateau(pd.DataFrame())
        assert "error" in result


class TestZonationAttenuation:
    """Tests for 09_persistence_remission_model.compute_zonation_attenuation."""

    def test_returns_expected_columns(self):
        df = _make_phase4_trajectory_df()
        result = step09.compute_zonation_attenuation(df)
        assert not result.empty
        for col in ("dev_stage", "mean_expression", "cv", "n_markers",
                     "stage_order"):
            assert col in result.columns, f"missing column: {col}"

    def test_cv_is_positive(self):
        df = _make_phase4_trajectory_df()
        result = step09.compute_zonation_attenuation(df)
        assert (result["cv"] >= 0).all()

    def test_custom_markers(self):
        df = _make_phase4_trajectory_df()
        custom = {"PVALB": "test", "GAD1": "test", "GRIN1": "test"}
        result = step09.compute_zonation_attenuation(df, dorsal_markers=custom)
        assert not result.empty

    def test_empty_input(self):
        result = step09.compute_zonation_attenuation(pd.DataFrame())
        assert result.empty

    def test_no_matching_markers(self):
        df = _make_trajectory_df(genes=["NOGENE1", "NOGENE2"])
        result = step09.compute_zonation_attenuation(df)
        assert result.empty


class TestZonationAttenuationTrend:
    """Tests for 09_persistence_remission_model.test_zonation_attenuation_trend."""

    def test_returns_expected_fields(self):
        df = _make_phase4_trajectory_df()
        zon = step09.compute_zonation_attenuation(df)
        result = step09.test_zonation_attenuation_trend(zon)
        for key in ("hypothesis", "spearman_rho", "spearman_p",
                     "cv_decreases", "cv_trajectory"):
            assert key in result, f"missing key: {key}"

    def test_empty_input(self):
        result = step09.test_zonation_attenuation_trend(pd.DataFrame())
        assert "error" in result


class TestHormoneModulation:
    """Tests for 09_persistence_remission_model.analyze_hormone_modulation."""

    def test_returns_expected_fields(self):
        df = _make_phase4_trajectory_df()
        result = step09.analyze_hormone_modulation(df)
        for key in ("hypothesis", "hormone_genes_found",
                     "pubertal_elevation", "pubertal_effect_size"):
            assert key in result, f"missing key: {key}"

    def test_finds_hormone_genes(self):
        df = _make_phase4_trajectory_df()
        result = step09.analyze_hormone_modulation(df)
        assert len(result["hormone_genes_found"]) > 0
        assert "AR" in result["hormone_genes_found"]

    def test_with_celltype_scores(self):
        df = _make_phase4_trajectory_df()
        ct_df = _make_celltype_trajectory_df()
        scores = step08.compute_celltype_scores(ct_df)
        result = step09.analyze_hormone_modulation(df, scores)
        # Should attempt PV correlation
        assert "pv_correlation" in result

    def test_empty_input(self):
        result = step09.analyze_hormone_modulation(pd.DataFrame())
        assert "error" in result

    def test_no_hormone_genes(self):
        df = _make_trajectory_df(genes=["FAKEGENE1", "FAKEGENE2"])
        result = step09.analyze_hormone_modulation(df)
        assert "error" in result


class TestScoreGenes:
    """Tests for 09_persistence_remission_model.score_genes_persistence_remission."""

    def test_returns_expected_columns(self):
        df = _make_phase4_trajectory_df()
        result = step09.score_genes_persistence_remission(df)
        assert not result.empty
        for col in ("gene_symbol", "adolescent_change", "remission_direction",
                     "discrimination_score", "is_ts_gene"):
            assert col in result.columns, f"missing column: {col}"

    def test_sorted_by_discrimination_score(self):
        df = _make_phase4_trajectory_df()
        result = step09.score_genes_persistence_remission(df)
        scores = result["discrimination_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_direction_values(self):
        df = _make_phase4_trajectory_df()
        result = step09.score_genes_persistence_remission(df)
        assert set(result["remission_direction"].unique()) <= {"remission", "persistence"}

    def test_ts_gene_flag(self):
        df = _make_phase4_trajectory_df()
        result = step09.score_genes_persistence_remission(df)
        ts_rows = result[result["is_ts_gene"]]
        # BCL11B, NDFIP2, SLITRK1, HDC, NRXN1 should be flagged
        assert len(ts_rows) > 0

    def test_empty_input(self):
        result = step09.score_genes_persistence_remission(pd.DataFrame())
        assert result.empty


class TestGeneratePredictions:
    """Tests for 09_persistence_remission_model.generate_predictions."""

    def _run_full_pipeline(self):
        df = _make_phase4_trajectory_df()
        rem = step09.compute_remission_score(df)
        pers = step09.compute_persistence_score(df)
        zon = step09.compute_zonation_attenuation(df)
        rem_test = step09.test_remission_adolescent_increase(rem)
        pers_test = step09.test_persistence_plateau(pers)
        zon_test = step09.test_zonation_attenuation_trend(zon)
        horm_test = step09.analyze_hormone_modulation(df)
        genes = step09.score_genes_persistence_remission(df)
        return rem_test, pers_test, zon_test, horm_test, genes

    def test_returns_list(self):
        results = self._run_full_pipeline()
        predictions = step09.generate_predictions(*results)
        assert isinstance(predictions, list)

    def test_prediction_structure(self):
        results = self._run_full_pipeline()
        predictions = step09.generate_predictions(*results)
        for p in predictions:
            assert "id" in p
            assert "prediction" in p
            assert "evidence" in p
            assert "validation" in p
            assert "confidence" in p
            assert p["id"].startswith("P4.")

    def test_empty_inputs_returns_empty(self):
        empty_test = {"error": "no data"}
        predictions = step09.generate_predictions(
            empty_test, empty_test, empty_test, empty_test, pd.DataFrame(),
        )
        assert isinstance(predictions, list)
        assert len(predictions) == 0
