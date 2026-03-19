"""Tests for bioagentics.models.pathway_enrichment."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bioagentics.models.pathway_enrichment import (
    run_ora,
    compare_with_cunningham,
    plot_enrichment_dotplot,
    enrichment_pipeline,
)


def _make_gene_sets() -> dict[str, list[str]]:
    """Simple gene sets for testing."""
    return {
        "pathway_A": [f"GENE{i}" for i in range(20)],
        "pathway_B": [f"GENE{i}" for i in range(15, 35)],
        "pathway_C": [f"GENE{i}" for i in range(50, 70)],
    }


def _make_ranked_genes(n: int = 100) -> pd.DataFrame:
    """Create a ranked gene list with injected signal."""
    rng = np.random.default_rng(42)
    genes = [f"GENE{i}" for i in range(n)]
    # First 20 genes have strong positive signal (should enrich pathway_A)
    rank_metric = rng.normal(0, 1, n)
    rank_metric[:20] += 3
    return pd.DataFrame({"gene": genes, "rank_metric": rank_metric})


class TestRunORA:
    def test_detects_enrichment(self):
        gene_sets = _make_gene_sets()
        # Select genes overlapping heavily with pathway_A
        selected = [f"GENE{i}" for i in range(15)]
        background = [f"GENE{i}" for i in range(100)]

        result = run_ora(selected, gene_sets, background)
        assert not result.empty
        assert "term" in result.columns
        assert "fdr" in result.columns

        pa = result[result["term"] == "pathway_A"]
        assert not pa.empty
        assert pa["overlap"].values[0] >= 10

    def test_no_overlap_gives_high_pvalue(self):
        gene_sets = {"no_match": [f"X{i}" for i in range(20)]}
        selected = [f"GENE{i}" for i in range(10)]
        background = [f"GENE{i}" for i in range(100)] + [f"X{i}" for i in range(20)]

        result = run_ora(selected, gene_sets, background)
        if not result.empty:
            assert result["pvalue"].values[0] >= 0.5

    def test_empty_gene_sets(self):
        result = run_ora(["A", "B"], {}, ["A", "B", "C"])
        assert result.empty

    def test_fdr_bounded(self):
        gene_sets = _make_gene_sets()
        selected = [f"GENE{i}" for i in range(10)]
        result = run_ora(selected, gene_sets)
        if not result.empty:
            assert (result["fdr"] <= 1.0).all()
            assert (result["fdr"] >= 0.0).all()


class TestCompareWithCunningham:
    def test_comparison_structure(self):
        enriched = pd.DataFrame({"term": ["A"], "fdr": [0.01]})
        selected = ["DRD1", "DRD2", "NOVEL1", "NOVEL2"]

        result = compare_with_cunningham(enriched, selected)
        assert len(result) == 5
        assert "category" in result.columns

        overlap_row = result[result["category"] == "overlap"]
        assert overlap_row["count"].values[0] == 2

        novel_row = result[result["category"] == "novel_in_classifier"]
        assert novel_row["count"].values[0] == 2

    def test_no_overlap(self):
        enriched = pd.DataFrame({"term": ["A"], "fdr": [0.01]})
        selected = ["NOVEL1", "NOVEL2"]
        result = compare_with_cunningham(enriched, selected)
        overlap_row = result[result["category"] == "overlap"]
        assert overlap_row["count"].values[0] == 0


class TestPlotEnrichmentDotplot:
    def test_saves_plot(self, tmp_path):
        df = pd.DataFrame({
            "term": ["pathway_A", "pathway_B"],
            "fdr": [0.001, 0.05],
            "fold_enrichment": [3.5, 1.8],
            "overlap": [10, 5],
        })
        save_path = tmp_path / "dotplot.png"
        plot_enrichment_dotplot(df, save_path=save_path)
        assert save_path.exists()

    def test_handles_empty_df(self, tmp_path):
        save_path = tmp_path / "empty.png"
        plot_enrichment_dotplot(pd.DataFrame(), save_path=save_path)
        assert not save_path.exists()

    def test_nes_column(self, tmp_path):
        df = pd.DataFrame({
            "term": ["A", "B"],
            "fdr": [0.01, 0.04],
            "nes": [2.1, -1.5],
        })
        save_path = tmp_path / "nes_dotplot.png"
        plot_enrichment_dotplot(df, save_path=save_path)
        assert save_path.exists()


class TestEnrichmentPipeline:
    def test_ora_only(self, tmp_path):
        gene_sets = _make_gene_sets()
        selected = [f"GENE{i}" for i in range(15)]

        results = enrichment_pipeline(
            selected_genes=selected,
            gene_sets=gene_sets,
            dest_dir=tmp_path,
        )

        assert "ora" in results
        assert (tmp_path / "ora_results.csv").exists()
        assert (tmp_path / "cunningham_comparison.csv").exists()

    def test_no_inputs_returns_empty(self, tmp_path):
        results = enrichment_pipeline(dest_dir=tmp_path)
        assert len(results) == 0
