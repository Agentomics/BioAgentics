"""Tests for SWI/SNF metabolic convergence Phase 2 cross-atlas validation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src/cancer to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "cancer"))

_mod = importlib.import_module(
    "swisnf_metabolic_convergence.03_cross_atlas_validation"
)
cohens_d = _mod.cohens_d
fdr_correction = _mod.fdr_correction
find_convergent_genes_from_screens = _mod.find_convergent_genes_from_screens
build_convergence_matrix = _mod.build_convergence_matrix
categorize_convergent_genes = _mod.categorize_convergent_genes


# ---------- Fixtures ----------


def _make_screen_df(genes, cancer_types, comparison, seed=42):
    """Create synthetic screen results resembling Phase 1b output."""
    rng = np.random.default_rng(seed)
    rows = []
    for ct in cancer_types:
        for gene in genes:
            d = rng.normal(-0.5, 1.0)
            p = max(1e-10, min(0.99, abs(rng.normal(0.1, 0.2))))
            rows.append({
                "cancer_type": ct,
                "gene": gene,
                "comparison": comparison,
                "cohens_d": round(d, 4),
                "p_value": p,
                "n_mut": rng.integers(5, 20),
                "n_wt": rng.integers(20, 100),
                "median_dep_mut": round(rng.normal(-0.3, 0.5), 4),
                "median_dep_wt": round(rng.normal(0.0, 0.3), 4),
                "fdr": round(p * len(genes), 4),
            })
    return pd.DataFrame(rows)


# ---------- Test convergence detection ----------


class TestFindConvergentGenes:
    """Test identification of genes with SL signal in both screens."""

    def test_finds_overlap(self):
        """Genes present in both screens are found."""
        shared_genes = ["NDUFA2", "COX6C", "SDHB"]
        arid1a_only = ["HMGCR"]
        smarca4_only = ["MTERF4"]

        # Make ARID1A screen with shared + arid1a_only all having SL signal
        a_screen = pd.DataFrame([
            {"cancer_type": "Lung", "gene": g, "comparison": "ARID1A",
             "cohens_d": -0.8, "p_value": 0.01, "fdr": 0.05,
             "n_mut": 10, "n_wt": 50, "median_dep_mut": -0.5, "median_dep_wt": -0.1}
            for g in shared_genes + arid1a_only
        ])
        s_screen = pd.DataFrame([
            {"cancer_type": "Ovary", "gene": g, "comparison": "SMARCA4",
             "cohens_d": -0.9, "p_value": 0.005, "fdr": 0.03,
             "n_mut": 8, "n_wt": 40, "median_dep_mut": -0.6, "median_dep_wt": -0.1}
            for g in shared_genes + smarca4_only
        ])

        result = find_convergent_genes_from_screens(a_screen, s_screen)
        assert len(result) == 3
        assert set(result["gene"].tolist()) == set(shared_genes)

    def test_no_overlap(self):
        """No convergent genes when screens have no overlapping SL hits."""
        a_screen = pd.DataFrame([{
            "cancer_type": "Lung", "gene": "HMGCR", "comparison": "ARID1A",
            "cohens_d": -0.8, "p_value": 0.01, "fdr": 0.05,
            "n_mut": 10, "n_wt": 50, "median_dep_mut": -0.5, "median_dep_wt": -0.1,
        }])
        s_screen = pd.DataFrame([{
            "cancer_type": "Ovary", "gene": "MTERF4", "comparison": "SMARCA4",
            "cohens_d": -0.9, "p_value": 0.005, "fdr": 0.03,
            "n_mut": 8, "n_wt": 40, "median_dep_mut": -0.6, "median_dep_wt": -0.1,
        }])

        result = find_convergent_genes_from_screens(a_screen, s_screen)
        assert len(result) == 0

    def test_respects_thresholds(self):
        """Genes with non-significant p-values are excluded."""
        a_screen = pd.DataFrame([{
            "cancer_type": "Lung", "gene": "NDUFA2", "comparison": "ARID1A",
            "cohens_d": -0.8, "p_value": 0.01, "fdr": 0.05,
            "n_mut": 10, "n_wt": 50, "median_dep_mut": -0.5, "median_dep_wt": -0.1,
        }])
        # SMARCA4 screen: gene has high p-value (not significant)
        s_screen = pd.DataFrame([{
            "cancer_type": "Ovary", "gene": "NDUFA2", "comparison": "SMARCA4",
            "cohens_d": -0.8, "p_value": 0.2, "fdr": 0.4,
            "n_mut": 8, "n_wt": 40, "median_dep_mut": -0.3, "median_dep_wt": -0.1,
        }])

        result = find_convergent_genes_from_screens(a_screen, s_screen)
        assert len(result) == 0

    def test_excludes_positive_d(self):
        """Genes with positive d (resistance, not SL) are excluded."""
        a_screen = pd.DataFrame([{
            "cancer_type": "Lung", "gene": "GENEX", "comparison": "ARID1A",
            "cohens_d": 0.8, "p_value": 0.01, "fdr": 0.05,
            "n_mut": 10, "n_wt": 50, "median_dep_mut": 0.2, "median_dep_wt": -0.1,
        }])
        s_screen = pd.DataFrame([{
            "cancer_type": "Ovary", "gene": "GENEX", "comparison": "SMARCA4",
            "cohens_d": 0.9, "p_value": 0.005, "fdr": 0.03,
            "n_mut": 8, "n_wt": 40, "median_dep_mut": 0.3, "median_dep_wt": -0.1,
        }])

        result = find_convergent_genes_from_screens(a_screen, s_screen)
        assert len(result) == 0


# ---------- Test convergence matrix ----------


class TestBuildConvergenceMatrix:
    """Test the gene × cancer_type convergence matrix."""

    def test_matrix_structure(self):
        """Matrix has expected columns and rows."""
        a_screen = pd.DataFrame([
            {"cancer_type": "Lung", "gene": "SDHB", "cohens_d": -0.8,
             "p_value": 0.01, "comparison": "ARID1A"},
            {"cancer_type": "Breast", "gene": "SDHB", "cohens_d": -0.3,
             "p_value": 0.2, "comparison": "ARID1A"},
        ])
        s_screen = pd.DataFrame([
            {"cancer_type": "Lung", "gene": "SDHB", "cohens_d": -0.9,
             "p_value": 0.005, "comparison": "SMARCA4"},
        ])

        matrix = build_convergence_matrix(a_screen, s_screen, ["SDHB"])
        assert "gene" in matrix.columns
        assert "cancer_type" in matrix.columns
        assert "arid1a_d" in matrix.columns
        assert "smarca4_d" in matrix.columns
        assert "both_sl" in matrix.columns
        # 2 cancer types for SDHB
        assert len(matrix) == 2

    def test_both_sl_flag(self):
        """both_sl is True only when both screens show SL."""
        a_screen = pd.DataFrame([
            {"cancer_type": "Lung", "gene": "SDHB", "cohens_d": -0.8,
             "p_value": 0.01, "comparison": "ARID1A"},
        ])
        s_screen = pd.DataFrame([
            {"cancer_type": "Lung", "gene": "SDHB", "cohens_d": -0.9,
             "p_value": 0.005, "comparison": "SMARCA4"},
        ])

        matrix = build_convergence_matrix(a_screen, s_screen, ["SDHB"])
        assert matrix.iloc[0]["both_sl"] == True


# ---------- Test pathway categorization ----------


class TestCategorizeConvergentGenes:
    """Test pathway category assignment."""

    def test_oxphos_genes(self):
        """OXPHOS pathway genes categorized correctly."""
        convergent = pd.DataFrame({"gene": ["NDUFA2", "COX6C"]})
        gene_list = pd.DataFrame({
            "gene": ["NDUFA2", "COX6C"],
            "pathways": ["Oxidative phosphorylation", "Oxidative phosphorylation"],
        })
        result = categorize_convergent_genes(convergent, gene_list)
        assert (result["category"] == "OXPHOS").all()

    def test_missing_pathway_info(self):
        """Genes without pathway info get 'Other metabolism'."""
        convergent = pd.DataFrame({"gene": ["UNKNOWNGENE"]})
        gene_list = pd.DataFrame({"gene": ["OTHERGENE"], "pathways": ["something"]})
        result = categorize_convergent_genes(convergent, gene_list)
        assert result.iloc[0]["category"] == "Other metabolism"
