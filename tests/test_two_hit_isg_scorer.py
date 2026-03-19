"""Tests for the ISG scorer using AGS 6-gene panel."""

import json

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.two_hit_isg_scorer import (
    ISG_PANEL,
    compute_isg_scores,
    compare_groups,
    generate_synthetic_demo,
    run_isg_scoring,
)


def _make_expression_df(n_samples=10, genes=None, seed=42):
    """Helper to create a simple expression DataFrame."""
    rng = np.random.default_rng(seed)
    if genes is None:
        genes = ISG_PANEL
    samples = [f"S{i}" for i in range(n_samples)]
    data = rng.normal(loc=8, scale=2, size=(len(genes), n_samples))
    return pd.DataFrame(data, index=genes, columns=samples)


class TestComputeIsgScores:
    def test_basic_scoring(self):
        expr_df = _make_expression_df()
        scores = compute_isg_scores(expr_df)
        assert len(scores) == 10
        assert "isg_score" in scores.columns
        assert "n_isg_genes" in scores.columns

    def test_all_genes_found(self):
        expr_df = _make_expression_df()
        scores = compute_isg_scores(expr_df)
        assert scores.iloc[0]["n_isg_genes"] == 6

    def test_missing_genes_fallback(self):
        """Should fall back to extended panel if < 3 panel genes present."""
        genes = ["IFI27", "MX1", "OAS1", "IFIT3"]  # Only 1 panel gene + extended
        expr_df = _make_expression_df(genes=genes)
        scores = compute_isg_scores(expr_df)
        assert len(scores) > 0
        assert scores.iloc[0]["n_isg_genes"] >= 3

    def test_too_few_genes(self):
        """Should return empty if fewer than MIN_ISG_GENES found."""
        genes = ["GAPDH", "ACTB"]  # No ISG genes
        expr_df = _make_expression_df(genes=genes)
        scores = compute_isg_scores(expr_df)
        assert scores.empty

    def test_gene_col_parameter(self):
        """Should work when genes are in a named column."""
        expr_df = _make_expression_df()
        expr_df = expr_df.reset_index().rename(columns={"index": "gene_name"})
        scores = compute_isg_scores(expr_df, gene_col="gene_name")
        assert len(scores) > 0

    def test_case_insensitive(self):
        """Gene matching should be case-insensitive."""
        genes = [g.lower() for g in ISG_PANEL]
        expr_df = _make_expression_df(genes=genes)
        scores = compute_isg_scores(expr_df)
        assert len(scores) > 0

    def test_scores_are_finite(self):
        expr_df = _make_expression_df()
        scores = compute_isg_scores(expr_df)
        assert scores["isg_score"].notna().all()
        assert np.isfinite(scores["isg_score"]).all()


class TestCompareGroups:
    def test_two_groups(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "sample": [f"S{i}" for i in range(20)],
            "isg_score": np.concatenate([
                rng.normal(0, 1, 10),
                rng.normal(2, 1, 10),
            ]),
            "group": ["control"] * 10 + ["pans"] * 10,
        })
        result = compare_groups(df)
        assert "group_summary" in result
        assert "pairwise_comparisons" in result
        assert len(result["pairwise_comparisons"]) == 1
        comp = list(result["pairwise_comparisons"].values())[0]
        assert comp["p_value"] < 0.05  # Should be significant

    def test_three_groups(self):
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "sample": [f"S{i}" for i in range(30)],
            "isg_score": rng.normal(0, 1, 30),
            "group": ["A"] * 10 + ["B"] * 10 + ["C"] * 10,
        })
        result = compare_groups(df)
        assert len(result["group_summary"]) == 3
        assert len(result["pairwise_comparisons"]) == 3  # A-B, A-C, B-C

    def test_single_group_no_comparisons(self):
        df = pd.DataFrame({
            "sample": [f"S{i}" for i in range(5)],
            "isg_score": [1.0, 2.0, 3.0, 4.0, 5.0],
            "group": ["A"] * 5,
        })
        result = compare_groups(df)
        assert len(result["pairwise_comparisons"]) == 0


class TestSyntheticDemo:
    def test_generates_data(self):
        expr_df, meta_df = generate_synthetic_demo(n_per_group=5)
        assert len(ISG_PANEL) == expr_df.shape[0]  # genes in rows
        assert expr_df.shape[1] == 20  # 4 groups * 5 samples
        assert len(meta_df) == 20

    def test_four_groups(self):
        _, meta_df = generate_synthetic_demo()
        assert meta_df["group"].nunique() == 4

    def test_pans_higher_than_healthy(self):
        """PANS group should have elevated ISG expression."""
        expr_df, meta_df = generate_synthetic_demo(n_per_group=30, seed=42)
        scores = compute_isg_scores(expr_df)
        scores = scores.merge(meta_df, on="sample")
        pans_median = scores.loc[scores["group"] == "pans", "isg_score"].median()
        healthy_median = scores.loc[scores["group"] == "healthy", "isg_score"].median()
        assert pans_median > healthy_median


class TestRunPipeline:
    def test_creates_output_files(self, tmp_path):
        scores_df, stats = run_isg_scoring(dest_dir=tmp_path, use_demo=True)
        assert (tmp_path / "isg_scores.csv").exists()
        assert (tmp_path / "isg_comparison_stats.json").exists()
        assert (tmp_path / "isg_scores_boxplot.png").exists()

    def test_scores_have_groups(self, tmp_path):
        scores_df, _ = run_isg_scoring(dest_dir=tmp_path, use_demo=True)
        assert "group" in scores_df.columns
        assert scores_df["group"].nunique() == 4

    def test_stats_json_valid(self, tmp_path):
        run_isg_scoring(dest_dir=tmp_path, use_demo=True)
        with open(tmp_path / "isg_comparison_stats.json") as f:
            data = json.load(f)
        assert "group_summary" in data
        assert "pairwise_comparisons" in data

    def test_with_expression_file(self, tmp_path):
        """Test loading expression from CSV file."""
        expr_df, meta_df = generate_synthetic_demo(n_per_group=5)
        expr_path = tmp_path / "expr.csv"
        meta_path = tmp_path / "meta.csv"
        expr_df.to_csv(expr_path)
        meta_df.to_csv(meta_path, index=False)

        scores_df, stats = run_isg_scoring(
            expression_path=expr_path,
            metadata_path=meta_path,
            dest_dir=tmp_path,
        )
        assert len(scores_df) == 20
