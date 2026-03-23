"""Tests for Cepo + Cauchy combination module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.tourettes.ts_gwas_functional_annotation.cepo_cauchy import (
    CauchyCombinedResult,
    CepoCelltypeResult,
    CepoGeneScore,
    cauchy_combination,
    cepo_from_markers,
    cepo_magma_gsea,
    combine_celltype_methods,
    compute_cepo_scores,
    filter_protein_coding,
    write_cauchy_results,
    write_cepo_results,
    write_summary,
    _apply_fdr_cepo,
    _regression_test,
)


# --- Fixtures ---


@pytest.fixture
def gene_results_df():
    """MAGMA gene results DataFrame."""
    np.random.seed(42)
    genes = [f"GENE_{i}" for i in range(50)]
    return pd.DataFrame({
        "GENE": genes,
        "Z": np.random.normal(0, 1, 50),
        "P": np.random.uniform(0, 1, 50),
        "N_SNPS": np.random.randint(5, 100, 50),
    })


@pytest.fixture
def marker_genes():
    """Marker gene definitions for 3 cell types."""
    return {
        "type_a": ["GENE_0", "GENE_1", "GENE_2", "GENE_3", "GENE_4",
                    "GENE_5", "GENE_6", "GENE_7"],
        "type_b": ["GENE_10", "GENE_11", "GENE_12", "GENE_13", "GENE_14",
                    "GENE_15", "GENE_16"],
        "type_c": ["GENE_20", "GENE_21", "GENE_22", "GENE_23", "GENE_24"],
    }


@pytest.fixture
def expression_matrix():
    """Simulated scRNA-seq expression matrix (genes x cells)."""
    np.random.seed(123)
    n_genes = 30
    n_cells = 60
    genes = [f"GENE_{i}" for i in range(n_genes)]
    cells = [f"cell_{i}" for i in range(n_cells)]

    # Base expression
    data = np.random.exponential(0.5, (n_genes, n_cells))

    # Make some genes cell-type-specific
    # Type A cells (0-19): high GENE_0-4
    data[0:5, 0:20] += 3.0
    # Type B cells (20-39): high GENE_10-14
    data[10:15, 20:40] += 3.0
    # Type C cells (40-59): high GENE_20-24
    data[20:25, 40:60] += 3.0

    return pd.DataFrame(data, index=genes, columns=cells)


@pytest.fixture
def cell_labels():
    """Cell-type labels for 60 cells."""
    return pd.Series(
        ["type_a"] * 20 + ["type_b"] * 20 + ["type_c"] * 20,
        index=[f"cell_{i}" for i in range(60)],
    )


# --- Cepo score tests ---


class TestComputeCepoScores:
    def test_basic(self, expression_matrix, cell_labels):
        scores = compute_cepo_scores(expression_matrix, cell_labels, top_n=10)
        assert len(scores) == 3
        for ct in ["type_a", "type_b", "type_c"]:
            assert ct in scores
            assert len(scores[ct]) <= 10
            assert all(isinstance(s, CepoGeneScore) for s in scores[ct])

    def test_ranks_monotonic(self, expression_matrix, cell_labels):
        scores = compute_cepo_scores(expression_matrix, cell_labels, top_n=10)
        for ct, ct_scores in scores.items():
            for i, s in enumerate(ct_scores):
                assert s.rank == i + 1

    def test_scores_descending(self, expression_matrix, cell_labels):
        scores = compute_cepo_scores(expression_matrix, cell_labels, top_n=10)
        for ct, ct_scores in scores.items():
            cepo_vals = [s.cepo_score for s in ct_scores]
            assert cepo_vals == sorted(cepo_vals, reverse=True)

    def test_min_cells_filter(self, expression_matrix, cell_labels):
        scores = compute_cepo_scores(
            expression_matrix, cell_labels, min_cells=25, top_n=5,
        )
        # All cell types have 20 cells, so none pass min_cells=25
        assert len(scores) == 0

    def test_top_n_limit(self, expression_matrix, cell_labels):
        scores = compute_cepo_scores(expression_matrix, cell_labels, top_n=3)
        for ct_scores in scores.values():
            assert len(ct_scores) <= 3

    def test_cell_type_in_scores(self, expression_matrix, cell_labels):
        scores = compute_cepo_scores(expression_matrix, cell_labels, top_n=5)
        for ct, ct_scores in scores.items():
            for s in ct_scores:
                assert s.cell_type == ct


class TestCepoFromMarkers:
    def test_basic(self, marker_genes):
        all_genes = [f"GENE_{i}" for i in range(50)]
        scores = cepo_from_markers(marker_genes, all_genes)
        assert len(scores) == 3
        assert "type_a" in scores
        assert len(scores["type_a"]) == 8

    def test_filters_missing_genes(self, marker_genes):
        all_genes = [f"GENE_{i}" for i in range(5)]  # Only 0-4
        scores = cepo_from_markers(marker_genes, all_genes)
        assert len(scores["type_a"]) == 5  # Only GENE_0 to GENE_4
        assert len(scores["type_b"]) == 0  # No overlap
        assert len(scores["type_c"]) == 0  # No overlap

    def test_rank_based_scores(self, marker_genes):
        all_genes = [f"GENE_{i}" for i in range(50)]
        scores = cepo_from_markers(marker_genes, all_genes)
        for ct_scores in scores.values():
            for i, s in enumerate(ct_scores):
                assert s.cepo_score == pytest.approx(1.0 / (i + 1))
                assert s.rank == i + 1


class TestFilterProteinCoding:
    def test_no_filter(self, marker_genes):
        all_genes = [f"GENE_{i}" for i in range(50)]
        scores = cepo_from_markers(marker_genes, all_genes)
        filtered = filter_protein_coding(scores, None)
        assert filtered is scores  # Returns same dict

    def test_with_filter(self, marker_genes):
        all_genes = [f"GENE_{i}" for i in range(50)]
        scores = cepo_from_markers(marker_genes, all_genes)
        pc = {"GENE_0", "GENE_1", "GENE_10", "GENE_20"}
        filtered = filter_protein_coding(scores, pc)
        assert len(filtered["type_a"]) == 2
        assert len(filtered["type_b"]) == 1
        assert len(filtered["type_c"]) == 1

    def test_reranks(self, marker_genes):
        all_genes = [f"GENE_{i}" for i in range(50)]
        scores = cepo_from_markers(marker_genes, all_genes)
        pc = {"GENE_0", "GENE_2", "GENE_4"}
        filtered = filter_protein_coding(scores, pc)
        for s in filtered["type_a"]:
            assert s.rank <= len(filtered["type_a"])


# --- MAGMA-GSEA tests ---


class TestCepoMagmaGSEA:
    def test_basic(self, gene_results_df, marker_genes):
        all_genes = gene_results_df["GENE"].tolist()
        cepo_scores = cepo_from_markers(marker_genes, all_genes)
        results = cepo_magma_gsea(gene_results_df, cepo_scores)
        assert len(results) == 3
        assert all(isinstance(r, CepoCelltypeResult) for r in results)

    def test_min_genes_filter(self, gene_results_df, marker_genes):
        all_genes = gene_results_df["GENE"].tolist()
        cepo_scores = cepo_from_markers(marker_genes, all_genes)
        results = cepo_magma_gsea(gene_results_df, cepo_scores, min_genes=10)
        # No cell type has >= 10 marker genes
        assert len(results) == 0

    def test_fdr_applied(self, gene_results_df, marker_genes):
        all_genes = gene_results_df["GENE"].tolist()
        cepo_scores = cepo_from_markers(marker_genes, all_genes)
        results = cepo_magma_gsea(gene_results_df, cepo_scores)
        for r in results:
            assert 0 <= r.fdr_q <= 1.0

    def test_labels_used(self, gene_results_df, marker_genes):
        all_genes = gene_results_df["GENE"].tolist()
        cepo_scores = cepo_from_markers(marker_genes, all_genes)
        labels = {"type_a": "Type A Neurons"}
        results = cepo_magma_gsea(gene_results_df, cepo_scores, labels=labels)
        type_a = [r for r in results if r.cell_type == "type_a"]
        assert type_a[0].label == "Type A Neurons"

    def test_empty_gene_df(self, marker_genes):
        empty_df = pd.DataFrame(columns=["GENE", "Z", "P", "N_SNPS"])
        cepo_scores = cepo_from_markers(marker_genes, [])
        results = cepo_magma_gsea(empty_df, cepo_scores)
        assert results == []

    def test_top_genes_populated(self, gene_results_df, marker_genes):
        all_genes = gene_results_df["GENE"].tolist()
        cepo_scores = cepo_from_markers(marker_genes, all_genes)
        results = cepo_magma_gsea(gene_results_df, cepo_scores)
        for r in results:
            assert len(r.top_genes) <= 5


# --- Regression test ---


class TestRegressionTest:
    def test_basic(self):
        np.random.seed(42)
        z = np.random.normal(0, 1, 100)
        indicator = np.zeros(100)
        indicator[:20] = 1.0
        # Make indicator group have higher Z
        z[:20] += 1.0

        beta, se, z_score, p = _regression_test(z, indicator)
        assert beta > 0
        assert se > 0
        assert z_score > 0
        assert 0 < p < 1

    def test_no_effect(self):
        np.random.seed(42)
        z = np.random.normal(0, 1, 100)
        indicator = np.zeros(100)
        indicator[:50] = 1.0

        beta, se, z_score, p = _regression_test(z, indicator)
        assert abs(beta) < 1.0  # No systematic effect
        assert p > 0.01  # Should not be significant

    def test_small_n(self):
        z = np.array([1.0, 2.0, 3.0])
        indicator = np.array([0.0, 1.0, 1.0])
        beta, se, z_score, p = _regression_test(z, indicator)
        assert p == 1.0  # n < 10

    def test_with_covariates(self):
        np.random.seed(42)
        z = np.random.normal(0, 1, 100)
        indicator = np.zeros(100)
        indicator[:20] = 1.0
        z[:20] += 1.0
        cov = np.random.normal(0, 1, 100)

        beta, se, z_score, p = _regression_test(z, indicator, cov)
        assert beta > 0
        assert 0 < p < 1


# --- Cauchy combination tests ---


class TestCauchyCombination:
    def test_single_pvalue(self):
        assert cauchy_combination([0.05]) == pytest.approx(0.05)

    def test_two_small_pvalues(self):
        combined = cauchy_combination([0.01, 0.01])
        assert combined <= 0.01 + 1e-12

    def test_one_significant(self):
        combined = cauchy_combination([0.001, 0.5])
        assert combined < 0.5
        assert combined < 0.01  # One very small p drives the combination

    def test_all_nonsignificant(self):
        combined = cauchy_combination([0.5, 0.6, 0.7])
        assert combined > 0.1

    def test_empty(self):
        assert cauchy_combination([]) == 1.0

    def test_bounds(self):
        combined = cauchy_combination([0.01, 0.02, 0.03])
        assert 0 <= combined <= 1

    def test_weights(self):
        # Higher weight on smaller p should give smaller combined
        p_vals = [0.001, 0.5]
        combined_heavy_small = cauchy_combination(p_vals, [10.0, 1.0])
        combined_heavy_large = cauchy_combination(p_vals, [1.0, 10.0])
        assert combined_heavy_small < combined_heavy_large

    def test_equal_weights_default(self):
        p_vals = [0.01, 0.05]
        default = cauchy_combination(p_vals)
        explicit = cauchy_combination(p_vals, [1.0, 1.0])
        assert default == pytest.approx(explicit)

    def test_invalid_pvalues_filtered(self):
        combined = cauchy_combination([0.05, float("nan"), 0.01])
        assert 0 < combined < 1

    def test_monotonic_in_p(self):
        # Smaller p-values should give smaller combined p
        p1 = cauchy_combination([0.01, 0.01])
        p2 = cauchy_combination([0.05, 0.05])
        p3 = cauchy_combination([0.1, 0.1])
        assert p1 < p2 < p3


class TestCombineCelltypeMethods:
    def test_basic(self):
        methods = {
            "method_a": {"type1": 0.01, "type2": 0.5},
            "method_b": {"type1": 0.02, "type2": 0.3},
        }
        results = combine_celltype_methods(methods)
        assert len(results) == 2
        assert all(isinstance(r, CauchyCombinedResult) for r in results)

    def test_sorted_by_p(self):
        methods = {
            "method_a": {"type1": 0.5, "type2": 0.01},
        }
        results = combine_celltype_methods(methods)
        assert results[0].combined_p <= results[1].combined_p

    def test_fdr_applied(self):
        methods = {
            "method_a": {"type1": 0.01, "type2": 0.5, "type3": 0.1},
            "method_b": {"type1": 0.02, "type2": 0.3, "type3": 0.2},
        }
        results = combine_celltype_methods(methods)
        for r in results:
            assert 0 <= r.fdr_q <= 1.0

    def test_labels(self):
        methods = {"m": {"type1": 0.01}}
        labels = {"type1": "Nice Label"}
        results = combine_celltype_methods(methods, cell_type_labels=labels)
        assert results[0].label == "Nice Label"

    def test_unequal_methods(self):
        methods = {
            "method_a": {"type1": 0.01, "type2": 0.05},
            "method_b": {"type1": 0.02},  # type2 missing
        }
        results = combine_celltype_methods(methods)
        type1 = [r for r in results if r.cell_type == "type1"][0]
        type2 = [r for r in results if r.cell_type == "type2"][0]
        assert type1.n_methods == 2
        assert type2.n_methods == 1

    def test_method_weights(self):
        methods = {
            "method_a": {"type1": 0.001},
            "method_b": {"type1": 0.5},
        }
        heavy_a = combine_celltype_methods(
            methods, method_weights={"method_a": 10.0, "method_b": 1.0}
        )
        heavy_b = combine_celltype_methods(
            methods, method_weights={"method_a": 1.0, "method_b": 10.0}
        )
        assert heavy_a[0].combined_p < heavy_b[0].combined_p


# --- FDR correction ---


class TestApplyFdrCepo:
    def test_basic(self):
        results = [
            CepoCelltypeResult("a", "A", 10, 8, 0.5, 0.1, 2.0, 0.01),
            CepoCelltypeResult("b", "B", 10, 8, 0.3, 0.1, 1.5, 0.05),
            CepoCelltypeResult("c", "C", 10, 8, 0.1, 0.1, 0.5, 0.3),
        ]
        _apply_fdr_cepo(results)
        for r in results:
            assert 0 <= r.fdr_q <= 1.0
        # FDR should be >= raw p-value
        for r in results:
            assert r.fdr_q >= r.p_value or r.fdr_q == pytest.approx(r.p_value)

    def test_monotonic(self):
        results = [
            CepoCelltypeResult("a", "A", 10, 8, 0.5, 0.1, 2.0, 0.01),
            CepoCelltypeResult("b", "B", 10, 8, 0.3, 0.1, 1.5, 0.02),
            CepoCelltypeResult("c", "C", 10, 8, 0.1, 0.1, 0.5, 0.5),
        ]
        _apply_fdr_cepo(results)
        fdrs = [r.fdr_q for r in results]
        for i in range(len(fdrs) - 1):
            assert fdrs[i] <= fdrs[i + 1]

    def test_empty(self):
        _apply_fdr_cepo([])  # Should not raise


# --- Output writer tests ---


class TestWriteCepoResults:
    def test_basic(self, tmp_path):
        results = [
            CepoCelltypeResult("msn", "MSN", 100, 50, 0.5, 0.1, 3.0, 0.001,
                               0.01, ["GENE_A", "GENE_B"]),
        ]
        path = write_cepo_results(results, tmp_path / "test.tsv")
        assert path.exists()
        df = pd.read_csv(path, sep="\t")
        assert len(df) == 1
        assert "CELL_TYPE" in df.columns
        assert df.iloc[0]["CELL_TYPE"] == "msn"

    def test_top_genes_semicolon(self, tmp_path):
        results = [
            CepoCelltypeResult("t", "T", 10, 5, 0.1, 0.1, 1.0, 0.1,
                               0.2, ["G1", "G2", "G3"]),
        ]
        path = write_cepo_results(results, tmp_path / "test.tsv")
        df = pd.read_csv(path, sep="\t")
        assert df.iloc[0]["TOP_GENES"] == "G1;G2;G3"


class TestWriteCauchyResults:
    def test_basic(self, tmp_path):
        results = [
            CauchyCombinedResult(
                "msn", "MSN", 2, {"cepo": 0.01, "magma": 0.02},
                5.0, 0.005, 0.01,
            ),
        ]
        path = write_cauchy_results(results, tmp_path / "test.tsv")
        assert path.exists()
        df = pd.read_csv(path, sep="\t")
        assert len(df) == 1
        assert "COMBINED_P" in df.columns
        assert "P_cepo" in df.columns
        assert "P_magma" in df.columns


class TestWriteSummary:
    def test_basic(self, tmp_path):
        cepo = [
            CepoCelltypeResult("msn", "MSN", 100, 50, 0.5, 0.1, 3.0, 0.001,
                               0.01, ["G1"]),
        ]
        cauchy = [
            CauchyCombinedResult("msn", "MSN", 2, {"c": 0.01}, 5.0, 0.005, 0.01),
        ]
        path = write_summary(cepo, cauchy, tmp_path)
        assert path.exists()
        text = path.read_text()
        assert "Cepo" in text
        assert "Cauchy" in text
        assert "MSN" in text

    def test_empty(self, tmp_path):
        path = write_summary([], [], tmp_path)
        assert path.exists()
