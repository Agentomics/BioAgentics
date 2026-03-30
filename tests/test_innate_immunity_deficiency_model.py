"""Tests for innate-immunity-deficiency-model pipeline modules.

Covers pure-logic functions from composite_deficiency_score, cytokine_classification,
innate_immunity_modules, and variant_status_extraction without requiring data files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# innate_immunity_modules tests
# ---------------------------------------------------------------------------


class TestInnateImmunityModules:
    """Tests for gene module definitions and helper functions."""

    def test_get_all_innate_genes_nonempty(self):
        from bioagentics.pandas_pans.innate_immunity_modules import get_all_innate_genes

        genes = get_all_innate_genes()
        assert len(genes) > 0
        assert isinstance(genes, list)
        assert all(isinstance(g, str) for g in genes)

    def test_get_all_innate_genes_deduplicated(self):
        from bioagentics.pandas_pans.innate_immunity_modules import get_all_innate_genes

        genes = get_all_innate_genes()
        assert len(genes) == len(set(genes)), "Innate gene list should be deduplicated"

    def test_get_all_adaptive_genes_nonempty(self):
        from bioagentics.pandas_pans.innate_immunity_modules import get_all_adaptive_genes

        genes = get_all_adaptive_genes()
        assert len(genes) > 0
        assert isinstance(genes, list)

    def test_get_all_adaptive_genes_deduplicated(self):
        from bioagentics.pandas_pans.innate_immunity_modules import get_all_adaptive_genes

        genes = get_all_adaptive_genes()
        assert len(genes) == len(set(genes)), "Adaptive gene list should be deduplicated"

    def test_innate_adaptive_no_overlap(self):
        from bioagentics.pandas_pans.innate_immunity_modules import (
            get_all_adaptive_genes,
            get_all_innate_genes,
        )

        innate = set(get_all_innate_genes())
        adaptive = set(get_all_adaptive_genes())
        assert innate.isdisjoint(adaptive), (
            f"Innate and adaptive gene lists should not overlap: {innate & adaptive}"
        )

    def test_get_innate_adaptive_ratio_genes_returns_tuple(self):
        from bioagentics.pandas_pans.innate_immunity_modules import (
            get_innate_adaptive_ratio_genes,
        )

        innate, adaptive = get_innate_adaptive_ratio_genes()
        assert isinstance(innate, list)
        assert isinstance(adaptive, list)
        assert len(innate) > 0
        assert len(adaptive) > 0

    def test_lectin_complement_genes_present(self):
        from bioagentics.pandas_pans.innate_immunity_modules import LECTIN_COMPLEMENT_GENES

        expected = {"MBL2", "MASP1", "MASP2", "FCN1", "FCN2", "FCN3", "COLEC11"}
        assert expected.issubset(set(LECTIN_COMPLEMENT_GENES))

    def test_cgas_sting_genes_present(self):
        from bioagentics.pandas_pans.innate_immunity_modules import CGAS_STING_GENES

        expected = {"CGAS", "STING1", "TBK1", "IRF3", "TREX1", "SAMHD1"}
        assert expected.issubset(set(CGAS_STING_GENES))

    def test_innate_modules_keys(self):
        from bioagentics.pandas_pans.innate_immunity_modules import INNATE_MODULES

        expected_keys = {
            "lectin_complement",
            "lectin_complement_downstream",
            "nk_cell_effector",
            "neutrophil_defense",
            "monocyte_defense",
            "pattern_recognition_receptors",
            "trained_immunity",
            "cgas_sting_pathway",
        }
        assert set(INNATE_MODULES.keys()) == expected_keys

    def test_adaptive_modules_keys(self):
        from bioagentics.pandas_pans.innate_immunity_modules import ADAPTIVE_MODULES

        expected_keys = {"t_cell_core", "b_cell_core", "th_signatures"}
        assert set(ADAPTIVE_MODULES.keys()) == expected_keys

    def test_get_cytokine_genes_valid(self):
        from bioagentics.pandas_pans.innate_immunity_modules import get_cytokine_genes

        for cat in ("innate", "adaptive", "regulatory"):
            genes = get_cytokine_genes(cat)
            assert len(genes) > 0
            assert isinstance(genes[0], str)

    def test_get_cytokine_genes_invalid(self):
        from bioagentics.pandas_pans.innate_immunity_modules import get_cytokine_genes

        with pytest.raises(ValueError, match="Unknown category"):
            get_cytokine_genes("nonexistent")

    def test_get_cytokine_proteins_valid(self):
        from bioagentics.pandas_pans.innate_immunity_modules import get_cytokine_proteins

        proteins = get_cytokine_proteins("innate")
        assert "IL-1β" in proteins
        assert "TNF-α" in proteins

    def test_get_lectin_variant_genes(self):
        from bioagentics.pandas_pans.innate_immunity_modules import get_lectin_variant_genes

        genes = get_lectin_variant_genes()
        assert set(genes) == {"MBL2", "MASP1", "MASP2"}

    def test_cytokine_classification_structure(self):
        from bioagentics.pandas_pans.innate_immunity_modules import CYTOKINE_CLASSIFICATION

        for cat, info in CYTOKINE_CLASSIFICATION.items():
            assert "genes" in info, f"Missing 'genes' key in {cat}"
            assert "proteins" in info, f"Missing 'proteins' key in {cat}"
            assert len(info["genes"]) == len(info["proteins"]), (
                f"Mismatch in {cat}: {len(info['genes'])} genes vs {len(info['proteins'])} proteins"
            )


# ---------------------------------------------------------------------------
# composite_deficiency_score tests
# ---------------------------------------------------------------------------


class TestCompositeDeficiencyScore:
    """Tests for scoring logic in composite_deficiency_score module."""

    def test_zscore_normal(self):
        from pandas_pans.innate_immunity_deficiency_model.composite_deficiency_score import (
            _zscore,
        )

        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = _zscore(values)
        assert len(z) == 5
        assert abs(np.mean(z)) < 1e-10, "Z-scored mean should be ~0"
        assert abs(np.std(z, ddof=1) - 1.0) < 1e-10, "Z-scored std should be ~1"

    def test_zscore_constant(self):
        from pandas_pans.innate_immunity_deficiency_model.composite_deficiency_score import (
            _zscore,
        )

        values = np.array([5.0, 5.0, 5.0])
        z = _zscore(values)
        assert np.all(z == 0), "Constant values should z-score to all zeros"

    def test_zscore_single_value(self):
        from pandas_pans.innate_immunity_deficiency_model.composite_deficiency_score import (
            _zscore,
        )

        values = np.array([42.0])
        z = _zscore(values)
        assert z[0] == 0.0, "Single value should z-score to 0"

    def test_assign_condition(self):
        from pandas_pans.innate_immunity_deficiency_model.composite_deficiency_score import (
            _assign_condition,
        )

        assert _assign_condition("Control_1") == "control"
        assert _assign_condition("PreIVIG_P001") == "pre"
        assert _assign_condition("PostIVIG_P001") == "post"
        assert _assign_condition("Secondpost_P001") == "second_post"
        assert _assign_condition("Something_else") == "unknown"

    def test_weights_sum_to_one(self):
        from pandas_pans.innate_immunity_deficiency_model.composite_deficiency_score import (
            WEIGHTS,
        )

        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-10, "WEIGHTS should sum to 1.0"

    def test_score_group_comparison_basic(self):
        from pandas_pans.innate_immunity_deficiency_model.composite_deficiency_score import (
            score_group_comparison,
        )

        df = pd.DataFrame({
            "condition": ["pre", "pre", "pre", "control", "control", "control"],
            "deficiency_score": [1.5, 2.0, 1.8, 0.5, 0.3, 0.7],
        })
        result = score_group_comparison(df, "test")
        assert len(result) == 1
        assert result.iloc[0]["comparison"] == "pre_vs_control"
        assert result.iloc[0]["data_source"] == "test"
        assert result.iloc[0]["direction"] == "pans_more_deficient"

    def test_score_group_comparison_post_vs_pre(self):
        from pandas_pans.innate_immunity_deficiency_model.composite_deficiency_score import (
            score_group_comparison,
        )

        df = pd.DataFrame({
            "condition": ["pre", "pre", "pre", "post", "post", "post"],
            "deficiency_score": [2.0, 1.8, 2.2, 0.5, 0.6, 0.4],
        })
        result = score_group_comparison(df, "test")
        assert len(result) == 1
        assert result.iloc[0]["comparison"] == "post_vs_pre"
        assert result.iloc[0]["direction"] == "post_less_deficient"

    def test_score_group_comparison_insufficient_samples(self):
        from pandas_pans.innate_immunity_deficiency_model.composite_deficiency_score import (
            score_group_comparison,
        )

        df = pd.DataFrame({
            "condition": ["pre", "control"],
            "deficiency_score": [1.5, 0.5],
        })
        result = score_group_comparison(df, "test")
        assert result.empty, "Should return empty for n<2 per group"


# ---------------------------------------------------------------------------
# cytokine_classification tests
# ---------------------------------------------------------------------------


class TestCytokineClassification:
    """Tests for cytokine classification logic."""

    def test_protein_to_category_mapping(self):
        from pandas_pans.innate_immunity_deficiency_model.cytokine_classification import (
            _PROTEIN_TO_CATEGORY,
        )

        assert _PROTEIN_TO_CATEGORY["IL-1β"] == "innate"
        assert _PROTEIN_TO_CATEGORY["TNF-α"] == "innate"
        assert _PROTEIN_TO_CATEGORY["IL-4"] == "adaptive"
        assert _PROTEIN_TO_CATEGORY["IFN-γ"] == "adaptive"
        assert _PROTEIN_TO_CATEGORY["IL-10"] == "regulatory"

    def test_analyte_overrides(self):
        from pandas_pans.innate_immunity_deficiency_model.cytokine_classification import (
            _ANALYTE_OVERRIDES,
        )

        assert _ANALYTE_OVERRIDES["S100B"] == "innate"
        assert _ANALYTE_OVERRIDES["TGF-β1"] == "regulatory"

    def test_compute_category_summary_basic(self):
        from pandas_pans.innate_immunity_deficiency_model.cytokine_classification import (
            compute_category_summary,
        )

        classified = pd.DataFrame({
            "analyte": ["IL-1β", "TNF-α", "IL-6", "IL-4", "IL-10"],
            "category": ["innate", "innate", "innate", "adaptive", "regulatory"],
            "pooled_g": [0.8, 0.5, -0.3, 0.4, -0.1],
            "direction": ["up", "up", "down", "up", "down"],
            "significant": [True, True, True, True, False],
        })

        summary = compute_category_summary(classified)
        assert len(summary) == 3  # innate, adaptive, regulatory

        innate_row = summary[summary["category"] == "innate"].iloc[0]
        assert innate_row["n_cytokines"] == 3
        assert innate_row["n_up"] == 2
        assert innate_row["n_down"] == 1
        assert innate_row["dominant_direction"] == "up"

        reg_row = summary[summary["category"] == "regulatory"].iloc[0]
        assert reg_row["n_ns"] == 1
        assert reg_row["dominant_direction"] == "mixed"  # 0 up, 0 down (only NS)

    def test_compute_category_summary_empty(self):
        from pandas_pans.innate_immunity_deficiency_model.cytokine_classification import (
            compute_category_summary,
        )

        classified = pd.DataFrame(
            columns=["analyte", "category", "pooled_g", "direction", "significant"]
        )
        summary = compute_category_summary(classified)
        assert summary.empty

    def test_classify_cytokines_missing_file(self, tmp_path):
        from pandas_pans.innate_immunity_deficiency_model.cytokine_classification import (
            classify_cytokines,
        )

        result = classify_cytokines(meta_path=tmp_path / "nonexistent.csv")
        assert result.empty
