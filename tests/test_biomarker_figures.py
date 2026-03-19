"""Tests for bioagentics.models.figures (publication figure generation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bioagentics.models.figures import (
    plot_volcano,
    plot_roc_multi,
    plot_gene_heatmap,
    plot_feature_importance,
    plot_enrichment_dotplot,
    generate_all_figures,
)


def _make_de_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Synthetic DE results."""
    rng = np.random.default_rng(seed)
    genes = [f"GENE{i}" for i in range(n)]
    lfc = rng.normal(0, 2, n)
    pval = rng.uniform(1e-10, 1, n)
    # Make some genes significant
    pval[:10] = rng.uniform(1e-30, 1e-5, 10)
    return pd.DataFrame({
        "gene": genes,
        "log2FoldChange": lfc,
        "padj": pval,
    })


class TestPlotVolcano:
    def test_saves_figure(self, tmp_path):
        de_df = _make_de_df()
        path = tmp_path / "volcano.png"
        plot_volcano(de_df, save_path=path)
        assert path.exists()
        assert path.stat().st_size > 1000  # non-trivial image

    def test_empty_df_skips(self, tmp_path):
        path = tmp_path / "empty_volcano.png"
        plot_volcano(pd.DataFrame(), save_path=path)
        assert not path.exists()

    def test_no_labels_when_n_label_zero(self, tmp_path):
        de_df = _make_de_df()
        path = tmp_path / "no_labels.png"
        plot_volcano(de_df, n_label=0, save_path=path)
        assert path.exists()


class TestPlotROCMulti:
    def test_saves_multiple_curves(self, tmp_path):
        rng = np.random.default_rng(42)
        roc_data = [
            {"name": "RF", "y_true": np.array([0]*20 + [1]*20),
             "y_prob": rng.uniform(0, 0.4, 20).tolist() + rng.uniform(0.6, 1, 20).tolist(),
             "auc": 0.92},
            {"name": "LR", "y_true": np.array([0]*20 + [1]*20),
             "y_prob": rng.uniform(0.1, 0.5, 20).tolist() + rng.uniform(0.5, 0.9, 20).tolist(),
             "auc": 0.78},
        ]
        path = tmp_path / "roc.png"
        plot_roc_multi(roc_data, save_path=path)
        assert path.exists()

    def test_empty_list(self, tmp_path):
        path = tmp_path / "empty_roc.png"
        plot_roc_multi([], save_path=path)
        assert not path.exists()


class TestPlotGeneHeatmap:
    def test_saves_heatmap(self, tmp_path):
        rng = np.random.default_rng(42)
        n_samples, n_genes = 40, 20
        genes = [f"G{i}" for i in range(n_genes)]
        expr = pd.DataFrame(
            rng.normal(0, 1, (n_samples, n_genes)),
            columns=genes,
            index=[f"S{i}" for i in range(n_samples)],
        )
        groups = pd.Series(
            ["case"] * 20 + ["control"] * 20,
            index=expr.index,
        )
        path = tmp_path / "heatmap.png"
        plot_gene_heatmap(expr, genes[:10], groups, save_path=path)
        assert path.exists()

    def test_too_few_genes_skips(self, tmp_path):
        expr = pd.DataFrame({"G0": [1, 2, 3]}, index=["S0", "S1", "S2"])
        groups = pd.Series(["A", "A", "B"], index=expr.index)
        path = tmp_path / "skip.png"
        plot_gene_heatmap(expr, ["G0"], groups, save_path=path)
        assert not path.exists()


class TestPlotFeatureImportance:
    def test_saves_barplot(self, tmp_path):
        genes = [f"G{i}" for i in range(15)]
        imp = np.random.default_rng(42).uniform(0, 1, 15)
        path = tmp_path / "importance.png"
        plot_feature_importance(genes, imp, save_path=path)
        assert path.exists()

    def test_empty_features(self, tmp_path):
        path = tmp_path / "empty.png"
        plot_feature_importance([], np.array([]), save_path=path)
        assert not path.exists()


class TestPlotEnrichmentDotplot:
    def test_saves_dotplot(self, tmp_path):
        df = pd.DataFrame({
            "term": ["Pathway_A", "Pathway_B", "Pathway_C"],
            "fdr": [0.001, 0.01, 0.1],
            "fold_enrichment": [4.2, 2.1, 1.3],
            "overlap": [8, 5, 3],
        })
        path = tmp_path / "dotplot.png"
        plot_enrichment_dotplot(df, save_path=path)
        assert path.exists()

    def test_empty_skips(self, tmp_path):
        path = tmp_path / "empty_dot.png"
        plot_enrichment_dotplot(pd.DataFrame(), save_path=path)
        assert not path.exists()


class TestGenerateAllFigures:
    def test_generates_volcano(self, tmp_path):
        de_results = {"combined": _make_de_df()}
        saved = generate_all_figures(de_results=de_results, dest_dir=tmp_path)
        assert len(saved) >= 1
        assert any("volcano" in str(p) for p in saved)

    def test_generates_roc_and_importance(self, tmp_path):
        rng = np.random.default_rng(42)
        roc_data = [{
            "name": "RF", "auc": 0.85,
            "y_true": np.array([0]*15 + [1]*15),
            "y_prob": rng.uniform(0, 0.4, 15).tolist() + rng.uniform(0.6, 1, 15).tolist(),
        }]
        feat_names = [f"G{i}" for i in range(10)]
        feat_imp = {"RF": rng.uniform(0, 1, 10)}

        saved = generate_all_figures(
            classifier_roc_data=roc_data,
            feature_importances=feat_imp,
            feature_names=feat_names,
            dest_dir=tmp_path,
        )
        assert len(saved) >= 2
        assert any("roc" in str(p) for p in saved)
        assert any("importance" in str(p) for p in saved)

    def test_no_inputs_returns_empty(self, tmp_path):
        saved = generate_all_figures(dest_dir=tmp_path)
        assert saved == []
