"""Tests for MAGMA cell-type enrichment analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.tourettes.ts_gwas_functional_annotation.celltype_enrichment import (
    BRAIN_CELLTYPE_MARKERS,
    CELLTYPE_LABELS,
    CelltypeResult,
    _apply_fdr,
    _compute_enrichment,
    conditional_celltype_analysis,
    load_gene_results,
    load_specificity_matrix,
    marginal_celltype_analysis,
    markers_to_specificity,
    run_celltype_enrichment,
    write_celltype_results,
    write_summary,
)


# --- Fixtures ---


@pytest.fixture
def gene_results_df():
    """Gene results with known enrichment pattern for striatal markers."""
    # Create genes: type_a markers get high Z, type_b markers get low Z
    type_a_markers = ["DRD1", "DRD2", "PPP1R1B", "PENK", "TAC1",
                      "ADORA2A", "GPR88", "FOXP1", "ISL1", "EBF1"]
    type_b_markers = ["GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1",
                      "GJA1", "SOX9", "S100B", "GLUL", "NDRG2"]
    background = [f"BG{i}" for i in range(80)]
    genes = type_a_markers + type_b_markers + background

    z_scores = ([3.5] * len(type_a_markers)
                + [0.0] * len(type_b_markers)
                + [0.0] * len(background))
    p_values = ([2e-4] * len(type_a_markers)
                + [0.5] * len(type_b_markers)
                + [0.5] * len(background))
    n_snps = [10, 15, 20, 8, 12, 25, 11, 14, 9, 18,
              10, 15, 20, 8, 12, 25, 11, 14, 9, 18] + [10] * 80

    return pd.DataFrame({
        "GENE": genes,
        "Z": z_scores,
        "P": p_values,
        "N_SNPS": n_snps,
    })


@pytest.fixture
def gene_results_df_mixed():
    """Gene results with differential enrichment across cell types."""
    # D1 MSN markers get high Z, D2 MSN markers get moderate Z, glia low
    d1_genes = ["DRD1", "TAC1", "PDYN", "ISL1", "EBF1"]
    d2_genes = ["DRD2", "PENK", "ADORA2A", "GPR6", "SP9"]
    astro_genes = ["GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1"]
    bg = [f"BG{i}" for i in range(85)]

    genes = d1_genes + d2_genes + astro_genes + bg
    z = ([4.0] * 5 + [2.0] * 5 + [0.0] * 5 + [0.0] * 85)

    return pd.DataFrame({
        "GENE": genes,
        "Z": z,
        "P": [float(1e-4)] * 5 + [0.02] * 5 + [0.5] * 90,
        "N_SNPS": [10] * len(genes),
    })


@pytest.fixture
def small_markers():
    """Small marker gene sets for testing."""
    return {
        "type_a": ["DRD1", "DRD2", "PPP1R1B", "PENK", "TAC1",
                    "ADORA2A", "GPR88", "FOXP1", "ISL1", "EBF1"],
        "type_b": ["GFAP", "AQP4", "SLC1A2", "SLC1A3", "ALDH1L1",
                    "GJA1", "SOX9", "S100B", "GLUL", "NDRG2"],
    }


# --- Marker Gene Definitions ---


class TestBrainCelltypeMarkers:
    def test_all_cell_types_have_markers(self):
        for ct, genes in BRAIN_CELLTYPE_MARKERS.items():
            assert len(genes) >= 5, f"{ct} has fewer than 5 markers"

    def test_all_cell_types_have_labels(self):
        for ct in BRAIN_CELLTYPE_MARKERS:
            assert ct in CELLTYPE_LABELS, f"{ct} missing label"

    def test_key_cell_types_present(self):
        expected = {"msn", "d1_msn", "d2_msn", "cortical_excitatory",
                    "astrocyte", "microglia", "oligodendrocyte"}
        assert expected.issubset(set(BRAIN_CELLTYPE_MARKERS.keys()))

    def test_d1_d2_markers_differ(self):
        d1 = set(BRAIN_CELLTYPE_MARKERS["d1_msn"])
        d2 = set(BRAIN_CELLTYPE_MARKERS["d2_msn"])
        # D1 and D2 should share some genes but not be identical
        assert d1 != d2
        assert len(d1 & d2) > 0  # some overlap expected (PPP1R1B, GPR88)


# --- Specificity Matrix ---


class TestMarkersToSpecificity:
    def test_correct_shape(self, small_markers):
        genes = ["DRD1", "DRD2", "GFAP", "AQP4", "OTHER"]
        spec = markers_to_specificity(small_markers, genes)
        assert spec.shape == (5, 2)
        assert list(spec.columns) == ["type_a", "type_b"]
        assert list(spec.index) == genes

    def test_marker_values_are_one(self, small_markers):
        genes = ["DRD1", "DRD2", "GFAP", "OTHER"]
        spec = markers_to_specificity(small_markers, genes)
        assert spec.loc["DRD1", "type_a"] == 1.0
        assert spec.loc["GFAP", "type_b"] == 1.0

    def test_nonmarker_values_are_zero(self, small_markers):
        genes = ["DRD1", "GFAP", "OTHER"]
        spec = markers_to_specificity(small_markers, genes)
        assert spec.loc["OTHER", "type_a"] == 0.0
        assert spec.loc["OTHER", "type_b"] == 0.0
        assert spec.loc["DRD1", "type_b"] == 0.0


# --- Loading Gene Results ---


class TestLoadGeneResults:
    def test_loads_valid_file(self, tmp_path):
        path = tmp_path / "genes.tsv"
        df = pd.DataFrame({
            "GENE": ["A", "B", "C"],
            "Z": [3.0, 1.0, 0.0],
            "P": [0.001, 0.1, 0.5],
            "N_SNPS": [10, 20, 5],
        })
        df.to_csv(path, sep="\t", index=False)
        result = load_gene_results(path)
        assert len(result) == 3

    def test_drops_nan_z(self, tmp_path):
        path = tmp_path / "genes.tsv"
        df = pd.DataFrame({
            "GENE": ["A", "B"],
            "Z": [3.0, float("nan")],
            "P": [0.001, 0.5],
        })
        df.to_csv(path, sep="\t", index=False)
        result = load_gene_results(path)
        assert len(result) == 1

    def test_returns_empty_for_missing(self, tmp_path):
        result = load_gene_results(tmp_path / "nonexistent.tsv")
        assert result.empty

    def test_returns_empty_for_bad_columns(self, tmp_path):
        path = tmp_path / "bad.tsv"
        pd.DataFrame({"X": [1]}).to_csv(path, sep="\t", index=False)
        result = load_gene_results(path)
        assert result.empty


# --- Loading Specificity Matrix ---


class TestLoadSpecificityMatrix:
    def test_loads_valid_matrix(self, tmp_path):
        path = tmp_path / "spec.tsv"
        df = pd.DataFrame({
            "GENE": ["A", "B"],
            "type1": [0.8, 0.1],
            "type2": [0.2, 0.9],
        })
        df.to_csv(path, sep="\t", index=False)
        result = load_specificity_matrix(path)
        assert result is not None
        assert result.shape == (2, 2)
        assert result.index.tolist() == ["A", "B"]

    def test_returns_none_for_missing(self, tmp_path):
        result = load_specificity_matrix(tmp_path / "nope.tsv")
        assert result is None

    def test_returns_none_for_bad_format(self, tmp_path):
        path = tmp_path / "bad.tsv"
        pd.DataFrame({"X": [1]}).to_csv(path, sep="\t", index=False)
        result = load_specificity_matrix(path)
        assert result is None


# --- Enrichment Regression ---


class TestComputeEnrichment:
    def test_positive_enrichment(self):
        np.random.seed(42)
        n = 100
        z = np.zeros(n)
        indicator = np.zeros(n)
        # First 10 genes are "markers" with high Z
        z[:10] = 3.5
        indicator[:10] = 1.0
        beta, se, z_score, p = _compute_enrichment(z, indicator)
        assert beta > 0
        assert z_score > 0
        assert p < 0.05

    def test_no_enrichment(self):
        np.random.seed(42)
        n = 100
        z = np.random.normal(0, 1, n)
        indicator = np.zeros(n)
        indicator[:10] = 1.0
        beta, se, z_score, p = _compute_enrichment(z, indicator)
        # p should not be very significant with random data
        assert p > 0.001

    def test_with_covariate(self):
        np.random.seed(42)
        n = 100
        z = np.zeros(n)
        z[:10] = 3.0
        indicator = np.zeros(n)
        indicator[:10] = 1.0
        cov = np.random.normal(0, 1, (n, 1))
        beta, se, z_score, p = _compute_enrichment(z, indicator, cov)
        assert beta > 0
        assert p < 0.05

    def test_too_few_genes(self):
        z = np.array([1.0, 2.0])
        indicator = np.array([1.0, 0.0])
        beta, se, z_score, p = _compute_enrichment(z, indicator)
        assert p == 1.0  # too few genes


# --- FDR Correction ---


class TestApplyFDR:
    def test_fdr_greater_than_pvalue(self):
        results = [
            CelltypeResult("a", "A", "marginal", 10, 8, 0.5, 0.1, 2.0, 0.01),
            CelltypeResult("b", "B", "marginal", 10, 8, 0.3, 0.1, 1.0, 0.1),
            CelltypeResult("c", "C", "marginal", 10, 8, 0.1, 0.1, 0.5, 0.3),
        ]
        _apply_fdr(results)
        for r in results:
            assert r.fdr_q >= r.p_value

    def test_fdr_monotonic(self):
        results = [
            CelltypeResult("a", "A", "marginal", 10, 8, 0.5, 0.1, 2.0, 0.01),
            CelltypeResult("b", "B", "marginal", 10, 8, 0.3, 0.1, 1.0, 0.05),
            CelltypeResult("c", "C", "marginal", 10, 8, 0.1, 0.1, 0.5, 0.2),
        ]
        _apply_fdr(results)
        for i in range(len(results) - 1):
            assert results[i].fdr_q <= results[i + 1].fdr_q + 1e-10

    def test_empty_list(self):
        results: list[CelltypeResult] = []
        _apply_fdr(results)  # should not raise


# --- Marginal Analysis ---


class TestMarginalCelltypeAnalysis:
    def test_enriched_type_detected(self, gene_results_df, small_markers):
        spec = markers_to_specificity(small_markers, gene_results_df["GENE"].tolist())
        results = marginal_celltype_analysis(gene_results_df, spec)
        # type_a markers (striatal) have high Z, should be enriched
        type_a = [r for r in results if r.cell_type == "type_a"]
        assert len(type_a) == 1
        assert type_a[0].p_value < 0.05
        assert type_a[0].beta > 0
        assert type_a[0].analysis_type == "marginal"

    def test_nonenriched_type_not_significant(self, gene_results_df, small_markers):
        spec = markers_to_specificity(small_markers, gene_results_df["GENE"].tolist())
        results = marginal_celltype_analysis(gene_results_df, spec)
        type_b = [r for r in results if r.cell_type == "type_b"]
        assert len(type_b) == 1
        # type_b (astrocyte-like) markers have Z=0, should not be enriched
        assert type_b[0].p_value > type_a_p(results)

    def test_min_genes_filter(self, gene_results_df):
        tiny = {"type_x": ["DRD1", "DRD2"]}  # only 2 markers
        spec = markers_to_specificity(tiny, gene_results_df["GENE"].tolist())
        results = marginal_celltype_analysis(gene_results_df, spec, min_genes=5)
        assert len(results) == 0

    def test_top_genes_reported(self, gene_results_df, small_markers):
        spec = markers_to_specificity(small_markers, gene_results_df["GENE"].tolist())
        results = marginal_celltype_analysis(gene_results_df, spec)
        for r in results:
            if r.n_genes_tested > 0:
                assert len(r.top_genes) > 0

    def test_fdr_applied(self, gene_results_df, small_markers):
        spec = markers_to_specificity(small_markers, gene_results_df["GENE"].tolist())
        results = marginal_celltype_analysis(gene_results_df, spec)
        for r in results:
            assert r.fdr_q >= r.p_value


def type_a_p(results: list[CelltypeResult]) -> float:
    """Helper: get type_a p-value."""
    return next(r.p_value for r in results if r.cell_type == "type_a")


# --- Conditional Analysis ---


class TestConditionalCelltypeAnalysis:
    def test_returns_conditional_type(self, gene_results_df, small_markers):
        spec = markers_to_specificity(small_markers, gene_results_df["GENE"].tolist())
        results = conditional_celltype_analysis(gene_results_df, spec)
        for r in results:
            assert r.analysis_type == "conditional"

    def test_enriched_survives_conditioning(self, gene_results_df, small_markers):
        spec = markers_to_specificity(small_markers, gene_results_df["GENE"].tolist())
        results = conditional_celltype_analysis(gene_results_df, spec)
        type_a = [r for r in results if r.cell_type == "type_a"]
        assert len(type_a) == 1
        # Strong enrichment in type_a should survive conditioning on type_b
        assert type_a[0].p_value < 0.05

    def test_conditional_vs_marginal(self, gene_results_df, small_markers):
        spec = markers_to_specificity(small_markers, gene_results_df["GENE"].tolist())
        marginal = marginal_celltype_analysis(gene_results_df, spec)
        conditional = conditional_celltype_analysis(gene_results_df, spec)
        # Conditional p-values should generally be >= marginal for truly enriched types
        # (but not always, so just check both analyses ran)
        assert len(marginal) == len(conditional)


# --- Output Writing ---


class TestWriteCelltypeResults:
    def test_writes_tsv(self, tmp_path):
        results = [CelltypeResult(
            cell_type="msn", label="Medium Spiny Neurons",
            analysis_type="marginal", n_marker_genes=10,
            n_genes_tested=8, beta=0.5, beta_se=0.1,
            z_score=5.0, p_value=1e-6, fdr_q=1e-5,
            top_genes=["DRD1", "DRD2"],
        )]
        path = write_celltype_results(results, tmp_path / "test.tsv")
        assert path.exists()
        df = pd.read_csv(path, sep="\t")
        assert len(df) == 1
        assert df.iloc[0]["CELL_TYPE"] == "msn"
        assert df.iloc[0]["ANALYSIS"] == "marginal"

    def test_top_genes_semicolon_separated(self, tmp_path):
        results = [CelltypeResult(
            cell_type="ct", label="CT", analysis_type="marginal",
            n_marker_genes=5, n_genes_tested=5, beta=0.1, beta_se=0.1,
            z_score=1.0, p_value=0.1, top_genes=["A", "B", "C"],
        )]
        path = write_celltype_results(results, tmp_path / "test.tsv")
        df = pd.read_csv(path, sep="\t")
        assert df.iloc[0]["TOP_GENES"] == "A;B;C"


class TestWriteSummary:
    def test_writes_summary_file(self, tmp_path):
        marginal = [CelltypeResult(
            cell_type="msn", label="MSN", analysis_type="marginal",
            n_marker_genes=10, n_genes_tested=8, beta=0.5, beta_se=0.1,
            z_score=3.0, p_value=0.001, fdr_q=0.01, top_genes=["DRD1"],
        )]
        conditional = [CelltypeResult(
            cell_type="msn", label="MSN", analysis_type="conditional",
            n_marker_genes=10, n_genes_tested=8, beta=0.4, beta_se=0.1,
            z_score=2.5, p_value=0.005, fdr_q=0.05, top_genes=["DRD1"],
        )]
        path = write_summary(marginal, conditional, tmp_path)
        assert path.exists()
        text = path.read_text()
        assert "Marginal" in text
        assert "Conditional" in text
        assert "MSN" in text


# --- Pipeline Integration ---


class TestRunCelltypeEnrichment:
    def test_full_pipeline(self, gene_results_df, small_markers, tmp_path):
        # Write gene results to file
        gene_path = tmp_path / "gene_results.tsv"
        gene_results_df.to_csv(gene_path, sep="\t", index=False)

        marginal, conditional = run_celltype_enrichment(
            gene_results_path=gene_path,
            markers=small_markers,
            output_dir=tmp_path / "output",
            min_genes=5,
        )
        assert len(marginal) > 0
        assert len(conditional) > 0

        # Check output files
        out = tmp_path / "output"
        assert (out / "magma_celltype.tsv").exists()
        assert (out / "magma_celltype_marginal.tsv").exists()
        assert (out / "magma_celltype_conditional.tsv").exists()
        assert (out / "celltype_summary.md").exists()

    def test_empty_gene_results(self, tmp_path):
        gene_path = tmp_path / "empty.tsv"
        pd.DataFrame({"GENE": [], "Z": [], "P": []}).to_csv(
            gene_path, sep="\t", index=False
        )
        marginal, conditional = run_celltype_enrichment(
            gene_results_path=gene_path,
            output_dir=tmp_path / "output",
        )
        assert len(marginal) == 0
        assert len(conditional) == 0

    def test_with_builtin_markers(self, gene_results_df, tmp_path):
        gene_path = tmp_path / "gene_results.tsv"
        gene_results_df.to_csv(gene_path, sep="\t", index=False)

        # Using default (builtin) markers — most won't overlap with our
        # small gene set, but should not crash
        marginal, conditional = run_celltype_enrichment(
            gene_results_path=gene_path,
            output_dir=tmp_path / "output",
            min_genes=3,
        )
        # At least msn should be testable since our fixture has MSN markers
        msn_marginal = [r for r in marginal if r.cell_type == "msn"]
        assert len(msn_marginal) > 0
