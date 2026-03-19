"""Tests for SWI/SNF metabolic convergence Phase 3 pathway enrichment."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "cancer"))

_mod = importlib.import_module(
    "swisnf_metabolic_convergence.04_pathway_enrichment"
)
run_hypergeometric_enrichment = _mod.run_hypergeometric_enrichment
build_pathway_gene_sets = _mod.build_pathway_gene_sets
fdr_correction = _mod.fdr_correction


# ---------- Test hypergeometric enrichment ----------


class TestHypergeometricEnrichment:
    """Test hypergeometric (Fisher's exact) enrichment."""

    def test_perfect_enrichment(self):
        """All convergent genes in one pathway → strong enrichment."""
        convergent = {"A", "B", "C"}
        pathways = {"PathwayX": {"A", "B", "C", "D"}}
        background = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"}

        result = run_hypergeometric_enrichment(convergent, pathways, background)
        assert len(result) == 1
        assert result.iloc[0]["convergent_in_pathway"] == 3
        assert result.iloc[0]["p_value"] < 0.05

    def test_no_enrichment(self):
        """Convergent genes not over-represented → high p-value."""
        convergent = {"X", "Y", "Z"}
        pathways = {"PathwayX": {"A", "B", "C", "D", "E"}}
        background = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

        result = run_hypergeometric_enrichment(convergent, pathways, background)
        assert len(result) == 1
        assert result.iloc[0]["convergent_in_pathway"] == 0
        assert result.iloc[0]["p_value"] > 0.5

    def test_fold_enrichment(self):
        """Fold enrichment computed correctly."""
        # 5 convergent out of 20 background, pathway has 3 genes in bg
        # All 3 convergent genes in pathway
        convergent = {"A", "B", "C", "D", "E"}
        pathways = {"Pw": {"A", "B", "C", "X", "Y"}}
        background = set("ABCDEFGHIJKLMNOPQRST")
        # X, Y not in background → pathway_in_bg = {A, B, C} = 3 genes
        # Expected = (5/20) * 3 = 0.75, fold = 3/0.75 = 4.0

        result = run_hypergeometric_enrichment(convergent, pathways, background)
        assert result.iloc[0]["fold_enrichment"] == pytest.approx(4.0, rel=0.01)

    def test_multiple_pathways_fdr(self):
        """FDR correction applied across multiple pathways."""
        convergent = {"A", "B"}
        pathways = {
            "Pw1": {"A", "B", "C"},
            "Pw2": {"D", "E", "F"},
            "Pw3": {"G", "H", "I"},
        }
        background = set("ABCDEFGHIJKLMNO")

        result = run_hypergeometric_enrichment(convergent, pathways, background)
        assert "fdr" in result.columns
        assert len(result) == 3


# ---------- Test pathway gene set building ----------


class TestBuildPathwayGeneSets:
    """Test construction of pathway gene sets from gene list CSV."""

    def test_parses_semicolon_delimited(self, tmp_path):
        """Genes in multiple pathways are assigned to each."""
        csv = tmp_path / "genes.csv"
        csv.write_text("gene,pathways\nGENE1,PathA; PathB\nGENE2,PathA\nGENE3,PathC\n")

        result = build_pathway_gene_sets(csv)
        assert "PathA" in result
        assert "PathB" in result
        assert "PathC" in result
        assert result["PathA"] == {"GENE1", "GENE2"}
        assert result["PathB"] == {"GENE1"}
        assert result["PathC"] == {"GENE3"}

    def test_empty_pathways_skipped(self, tmp_path):
        """Genes with no pathway annotation are skipped."""
        csv = tmp_path / "genes.csv"
        csv.write_text("gene,pathways\nGENE1,PathA\nGENE2,\n")

        result = build_pathway_gene_sets(csv)
        assert len(result) == 1
        assert "GENE2" not in result.get("PathA", set())


# ---------- Test FDR ----------


class TestFDR:
    """Test FDR correction."""

    def test_capped_at_one(self):
        pvals = np.array([0.8, 0.9, 0.95])
        fdrs = fdr_correction(pvals)
        assert all(f <= 1.0 for f in fdrs)

    def test_preserves_order(self):
        pvals = np.array([0.001, 0.01, 0.05])
        fdrs = fdr_correction(pvals)
        assert fdrs[0] <= fdrs[1] <= fdrs[2]
