"""Tests for positive control validation module."""

import pandas as pd
import pytest

from bioagentics.data.cd_fibrosis.positive_controls import (
    PATHWAY_CLASSES,
    POSITIVE_CONTROLS,
    PositiveControl,
    check_success_criteria,
    generate_validation_report,
    match_compound_name,
    validate_by_pathway_class,
    validate_positive_controls,
)


class TestPositiveControlDefinitions:
    def test_controls_defined(self):
        assert len(POSITIVE_CONTROLS) >= 10

    def test_all_controls_have_required_fields(self):
        for ctrl in POSITIVE_CONTROLS:
            assert ctrl.name, f"Control missing name"
            assert len(ctrl.aliases) > 0, f"{ctrl.name} has no aliases"
            assert ctrl.mechanism, f"{ctrl.name} missing mechanism"
            assert ctrl.pathway, f"{ctrl.name} missing pathway"
            assert ctrl.evidence, f"{ctrl.name} missing evidence"

    def test_pathway_classes_cover_all_controls(self):
        all_in_classes = set()
        for compounds in PATHWAY_CLASSES.values():
            all_in_classes.update(compounds)

        for ctrl in POSITIVE_CONTROLS:
            assert ctrl.name in all_in_classes, (
                f"{ctrl.name} not in any pathway class"
            )

    def test_plan_specified_classes_exist(self):
        """The research plan specifies 4 specific pathway classes."""
        plan_classes = ["miR-124", "TL1A-DR3", "TGF-beta/anti-fibrotic", "CD38/NAD+"]
        for cls in plan_classes:
            assert cls in PATHWAY_CLASSES, f"Plan-specified class '{cls}' not defined"


class TestMatchCompoundName:
    def test_exact_match(self):
        ctrl = match_compound_name("pirfenidone")
        assert ctrl is not None
        assert ctrl.name == "pirfenidone"

    def test_case_insensitive(self):
        ctrl = match_compound_name("PIRFENIDONE")
        assert ctrl is not None
        assert ctrl.name == "pirfenidone"

    def test_alias_match(self):
        ctrl = match_compound_name("esbriet")
        assert ctrl is not None
        assert ctrl.name == "pirfenidone"

    def test_substring_match(self):
        ctrl = match_compound_name("vorinostat (SAHA)")
        assert ctrl is not None
        assert ctrl.name == "vorinostat"

    def test_no_match(self):
        ctrl = match_compound_name("aspirin")
        assert ctrl is None

    def test_tl1a_compounds(self):
        ctrl = match_compound_name("duvakitug")
        assert ctrl is not None
        assert ctrl.pathway == "TL1A-DR3"

    def test_jak_inhibitors(self):
        ctrl = match_compound_name("upadacitinib")
        assert ctrl is not None
        assert ctrl.pathway == "JAK-STAT"


class TestValidatePositiveControls:
    def _make_ranked_results(self) -> pd.DataFrame:
        """Create mock ranked results with some positive controls present."""
        compounds = [
            ("vorinostat", -0.85, 3, 3, True),
            ("some-novel-compound", -0.80, 2, 2, False),
            ("pirfenidone", -0.75, 4, 4, True),
            ("another-compound", -0.70, 2, 2, False),
            ("trichostatin-a", -0.65, 3, 3, False),
            ("upadacitinib", -0.60, 2, 2, False),
            ("compound-x", -0.55, 1, 1, False),
            ("compound-y", -0.50, 1, 1, False),
            ("daratumumab", -0.45, 2, 2, False),
            ("compound-z", -0.40, 1, 1, False),
        ]
        return pd.DataFrame(
            compounds,
            columns=[
                "compound", "mean_concordance", "n_signatures_queried",
                "n_negative_hits", "has_fibroblast_hit",
            ],
        )

    def test_returns_dataframe(self):
        ranked = self._make_ranked_results()
        result = validate_positive_controls(ranked)
        assert isinstance(result, pd.DataFrame)

    def test_all_controls_represented(self):
        ranked = self._make_ranked_results()
        result = validate_positive_controls(ranked)
        assert len(result) == len(POSITIVE_CONTROLS)

    def test_found_controls_have_rank(self):
        ranked = self._make_ranked_results()
        result = validate_positive_controls(ranked)
        vor = result[result["control_name"] == "vorinostat"].iloc[0]
        assert vor["rank"] == 1
        assert vor["percentile"] == pytest.approx(0.1)
        assert vor["passes_threshold"]

    def test_missing_controls_marked(self):
        ranked = self._make_ranked_results()
        result = validate_positive_controls(ranked)
        obeaz = result[result["control_name"] == "obefazimod"].iloc[0]
        assert pd.isna(obeaz["rank"])
        assert not obeaz["passes_threshold"]

    def test_empty_results(self):
        ranked = pd.DataFrame(columns=["compound", "mean_concordance"])
        result = validate_positive_controls(ranked)
        assert len(result) == len(POSITIVE_CONTROLS)
        assert not any(result["passes_threshold"])

    def test_percentile_calculation(self):
        ranked = self._make_ranked_results()
        result = validate_positive_controls(ranked)
        pirf = result[result["control_name"] == "pirfenidone"].iloc[0]
        # pirfenidone is at index 2 (rank 3) out of 10 compounds
        assert pirf["rank"] == 3
        assert pirf["percentile"] == pytest.approx(0.3)


class TestValidateByPathwayClass:
    def test_pathway_summary(self):
        ranked = pd.DataFrame([
            ("vorinostat", -0.85, 3, 3, True),
            ("pirfenidone", -0.75, 4, 4, True),
        ] * 5, columns=[
            "compound", "mean_concordance", "n_signatures_queried",
            "n_negative_hits", "has_fibroblast_hit",
        ])
        validation_df = validate_positive_controls(ranked)
        results = validate_by_pathway_class(validation_df)

        assert "epigenetic/HDAC" in results
        assert "TGF-beta/anti-fibrotic" in results
        assert results["epigenetic/HDAC"]["n_total"] == 2
        assert results["TGF-beta/anti-fibrotic"]["n_total"] == 2


class TestCheckSuccessCriteria:
    def test_passes_when_enough_classes(self):
        # Create results where 4 pathway classes pass
        compounds = []
        for i, name in enumerate([
            "pirfenidone", "duvakitug", "daratumumab", "obefazimod",
            "vorinostat", "upadacitinib",
        ]):
            compounds.append((name, -(0.9 - i * 0.05), 3, 3, True))
        # Add filler to make controls in top 20%
        for i in range(24):
            compounds.append((f"compound-{i}", -0.1 + i * 0.01, 1, 1, False))

        ranked = pd.DataFrame(
            compounds,
            columns=[
                "compound", "mean_concordance", "n_signatures_queried",
                "n_negative_hits", "has_fibroblast_hit",
            ],
        )
        validation_df = validate_positive_controls(ranked)
        criteria = check_success_criteria(validation_df)
        assert criteria["overall_pass"] is True
        assert criteria["classes_passing"] >= 3

    def test_fails_when_insufficient_classes(self):
        # Only 1 pathway class represented
        ranked = pd.DataFrame([
            ("pirfenidone", -0.85, 3, 3, True),
            ("compound-a", -0.50, 1, 1, False),
            ("compound-b", -0.40, 1, 1, False),
        ], columns=[
            "compound", "mean_concordance", "n_signatures_queried",
            "n_negative_hits", "has_fibroblast_hit",
        ])
        validation_df = validate_positive_controls(ranked)
        criteria = check_success_criteria(validation_df)
        assert criteria["overall_pass"] is False
        assert criteria["classes_passing"] <= 1


class TestGenerateReport:
    def test_report_with_results(self, tmp_path):
        ranked = pd.DataFrame([
            ("vorinostat", -0.85, 3, 3, True),
            ("pirfenidone", -0.75, 4, 4, True),
            ("compound-a", -0.50, 1, 1, False),
        ], columns=[
            "compound", "mean_concordance", "n_signatures_queried",
            "n_negative_hits", "has_fibroblast_hit",
        ])

        val_df, criteria = generate_validation_report(ranked, output_dir=tmp_path)
        assert isinstance(val_df, pd.DataFrame)
        assert isinstance(criteria, dict)
        assert "overall_pass" in criteria
        assert (tmp_path / "positive_control_validation.tsv").exists()

    def test_report_with_empty_results(self, tmp_path):
        ranked = pd.DataFrame(columns=[
            "compound", "mean_concordance", "n_signatures_queried",
            "n_negative_hits", "has_fibroblast_hit",
        ])
        val_df, criteria = generate_validation_report(ranked, output_dir=tmp_path)
        assert criteria["overall_pass"] is False
