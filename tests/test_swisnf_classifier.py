"""Tests for SWI/SNF metabolic convergence Phase 1a classifier and Phase 1b screen."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src/cancer to path for module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "cancer"))

# Numeric-prefixed modules require importlib
_classifier_mod = importlib.import_module(
    "swisnf_metabolic_convergence.01_swisnf_classifier"
)
classify_swisnf_status = _classifier_mod.classify_swisnf_status
build_cancer_type_summary = _classifier_mod.build_cancer_type_summary

_screen_mod = importlib.import_module(
    "swisnf_metabolic_convergence.02_metabolic_dependency_screen"
)
cohens_d = _screen_mod.cohens_d
fdr_correction = _screen_mod.fdr_correction


# ---------- Phase 1a: classifier tests ----------

def _make_classified_df(n_lines=20):
    """Create a synthetic cell line DataFrame (pre-classification).

    Only includes columns that exist BEFORE classify_swisnf_status is called:
    OncotreeLineage plus copy number homdel flags (added by add_gene_copy_number).
    """
    rng = np.random.default_rng(42)
    model_ids = [f"ACH-{i:06d}" for i in range(n_lines)]

    df = pd.DataFrame({
        "OncotreeLineage": rng.choice(["Lung", "Ovary", "Breast"], n_lines),
        "ARID1A_has_homdel": [False] * n_lines,
        "SMARCA4_has_homdel": [False] * n_lines,
    }, index=model_ids)
    df.index.name = "ModelID"
    return df


class TestClassifySwisnfStatus:
    """Test the SWI/SNF status classification logic."""

    def test_all_wt(self):
        """All lines WT when no mutations or deletions."""
        df = _make_classified_df(10)
        # Provide empty LOF DataFrames
        arid1a_lof = pd.DataFrame(columns=[
            "ModelID", "ARID1A_has_lof", "ARID1A_mutation_type",
            "ARID1A_n_lof", "ARID1A_protein_changes",
        ])
        smarca4_lof = pd.DataFrame(columns=[
            "ModelID", "SMARCA4_has_lof", "SMARCA4_mutation_type",
            "SMARCA4_n_lof", "SMARCA4_protein_changes",
        ])
        result = classify_swisnf_status(df, arid1a_lof, smarca4_lof)
        assert (result["swisnf_status"] == "WT").all()
        assert not result["swisnf_any_mutant"].any()

    def test_arid1a_only_mutant(self):
        """Lines with only ARID1A LOF classified as ARID1A_mutant."""
        df = _make_classified_df(5)
        arid1a_lof = pd.DataFrame({
            "ModelID": [df.index[0]],
            "ARID1A_has_lof": [True],
            "ARID1A_mutation_type": ["frameshift"],
            "ARID1A_n_lof": [1],
            "ARID1A_protein_changes": ["p.R1234fs"],
        })
        smarca4_lof = pd.DataFrame(columns=[
            "ModelID", "SMARCA4_has_lof", "SMARCA4_mutation_type",
            "SMARCA4_n_lof", "SMARCA4_protein_changes",
        ])
        result = classify_swisnf_status(df, arid1a_lof, smarca4_lof)
        assert result.loc[df.index[0], "swisnf_status"] == "ARID1A_mutant"
        assert result.loc[df.index[0], "swisnf_any_mutant"] == True
        assert (result.loc[df.index[1:], "swisnf_status"] == "WT").all()

    def test_smarca4_only_mutant(self):
        """Lines with only SMARCA4 LOF classified as SMARCA4_mutant."""
        df = _make_classified_df(5)
        arid1a_lof = pd.DataFrame(columns=[
            "ModelID", "ARID1A_has_lof", "ARID1A_mutation_type",
            "ARID1A_n_lof", "ARID1A_protein_changes",
        ])
        smarca4_lof = pd.DataFrame({
            "ModelID": [df.index[2]],
            "SMARCA4_has_lof": [True],
            "SMARCA4_mutation_type": ["nonsense"],
            "SMARCA4_n_lof": [1],
            "SMARCA4_protein_changes": ["p.Q500*"],
        })
        result = classify_swisnf_status(df, arid1a_lof, smarca4_lof)
        assert result.loc[df.index[2], "swisnf_status"] == "SMARCA4_mutant"

    def test_dual_mutant(self):
        """Lines with both ARID1A and SMARCA4 LOF classified as dual_mutant."""
        df = _make_classified_df(5)
        target = df.index[0]
        arid1a_lof = pd.DataFrame({
            "ModelID": [target],
            "ARID1A_has_lof": [True],
            "ARID1A_mutation_type": ["frameshift"],
            "ARID1A_n_lof": [1],
            "ARID1A_protein_changes": ["p.R100fs"],
        })
        smarca4_lof = pd.DataFrame({
            "ModelID": [target],
            "SMARCA4_has_lof": [True],
            "SMARCA4_mutation_type": ["nonsense"],
            "SMARCA4_n_lof": [1],
            "SMARCA4_protein_changes": ["p.E200*"],
        })
        result = classify_swisnf_status(df, arid1a_lof, smarca4_lof)
        assert result.loc[target, "swisnf_status"] == "dual_mutant"
        assert result.loc[target, "swisnf_any_mutant"] == True

    def test_homdel_classified_as_disrupted(self):
        """Lines with homozygous deletion (no LOF mutation) classified correctly."""
        df = _make_classified_df(5)
        df.loc[df.index[1], "ARID1A_has_homdel"] = True
        arid1a_lof = pd.DataFrame(columns=[
            "ModelID", "ARID1A_has_lof", "ARID1A_mutation_type",
            "ARID1A_n_lof", "ARID1A_protein_changes",
        ])
        smarca4_lof = pd.DataFrame(columns=[
            "ModelID", "SMARCA4_has_lof", "SMARCA4_mutation_type",
            "SMARCA4_n_lof", "SMARCA4_protein_changes",
        ])
        result = classify_swisnf_status(df, arid1a_lof, smarca4_lof)
        assert result.loc[df.index[1], "ARID1A_disrupted"] == True
        assert result.loc[df.index[1], "swisnf_status"] == "ARID1A_mutant"


class TestBuildCancerTypeSummary:
    """Test cancer type summary generation."""

    def test_summary_counts(self):
        """Summary correctly counts mutant/WT per cancer type."""
        df = pd.DataFrame({
            "OncotreeLineage": ["Lung"] * 10 + ["Ovary"] * 5,
            "swisnf_status": (
                ["ARID1A_mutant"] * 3 + ["SMARCA4_mutant"] * 2
                + ["dual_mutant"] * 1 + ["WT"] * 4
                + ["ARID1A_mutant"] * 2 + ["WT"] * 3
            ),
            "swisnf_any_mutant": (
                [True] * 6 + [False] * 4 + [True] * 2 + [False] * 3
            ),
        })
        summary = build_cancer_type_summary(df)

        lung = summary[summary["cancer_type"] == "Lung"].iloc[0]
        assert lung["n_ARID1A_mutant"] == 3
        assert lung["n_SMARCA4_mutant"] == 2
        assert lung["n_dual_mutant"] == 1
        assert lung["n_WT"] == 4
        assert lung["n_any_swisnf_mutant"] == 6

    def test_qualification_flags(self):
        """Qualification flags based on minimum sample thresholds."""
        df = pd.DataFrame({
            "OncotreeLineage": ["TypeA"] * 12 + ["TypeB"] * 8,
            "swisnf_status": (
                ["ARID1A_mutant"] * 6 + ["WT"] * 6
                + ["ARID1A_mutant"] * 2 + ["WT"] * 6
            ),
            "swisnf_any_mutant": (
                [True] * 6 + [False] * 6
                + [True] * 2 + [False] * 6
            ),
        })
        summary = build_cancer_type_summary(df)

        type_a = summary[summary["cancer_type"] == "TypeA"].iloc[0]
        assert type_a["qualifies_arid1a"] == True  # 6 mutant >= 5
        assert type_a["qualifies_combined"] == True

        type_b = summary[summary["cancer_type"] == "TypeB"].iloc[0]
        assert type_b["qualifies_arid1a"] == False  # 2 mutant < 5


# ---------- Phase 1b: screen utility tests ----------


class TestCohensD:
    """Test Cohen's d computation."""

    def test_identical_groups(self):
        """Cohen's d is 0 for identical groups."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert cohens_d(a, a) == 0.0

    def test_known_effect(self):
        """Cohen's d has correct sign and approximate magnitude."""
        rng = np.random.default_rng(42)
        group1 = rng.normal(0, 1, 100)
        group2 = rng.normal(1, 1, 100)
        d = cohens_d(group1, group2)
        assert d < 0  # group1 mean < group2 mean
        assert -1.5 < d < -0.5  # approximately -1

    def test_zero_variance(self):
        """Cohen's d handles zero variance gracefully."""
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        assert cohens_d(a, b) == 0.0


class TestFDRCorrection:
    """Test Benjamini-Hochberg FDR correction."""

    def test_empty_array(self):
        """FDR correction handles empty array."""
        result = fdr_correction(np.array([]))
        assert len(result) == 0

    def test_single_pvalue(self):
        """Single p-value unchanged by FDR."""
        result = fdr_correction(np.array([0.03]))
        assert result[0] == pytest.approx(0.03)

    def test_monotonicity(self):
        """FDR-adjusted p-values maintain relative ordering."""
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        fdrs = fdr_correction(pvals)
        assert all(fdrs[i] <= fdrs[i + 1] for i in range(len(fdrs) - 1))

    def test_no_value_exceeds_one(self):
        """FDR-adjusted p-values capped at 1.0."""
        pvals = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        fdrs = fdr_correction(pvals)
        assert all(f <= 1.0 for f in fdrs)

    def test_significant_stays_significant(self):
        """Very small p-values remain significant after FDR."""
        pvals = np.array([1e-10, 0.5, 0.9])
        fdrs = fdr_correction(pvals)
        assert fdrs[0] < 0.05
