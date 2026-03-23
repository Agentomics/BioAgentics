"""Tests for the therapeutic model for cGAS-STING inhibitor prediction."""

import json

import numpy as np

from bioagentics.models.two_hit_therapeutic_model import (
    COMPOUND_PROFILES,
    INHIBITORS,
    Inhibitor,
    simulate_inhibitor_effect,
    predict_therapy,
    run_therapeutic_predictions,
)


class TestInhibitorPanel:
    def test_inhibitor_count(self):
        """INHIBITORS list includes all 4 inhibitors (H-151, RU.521, C-178, SN-011)."""
        assert len(INHIBITORS) == 4

    def test_sn011_in_panel(self):
        names = [inh.name for inh in INHIBITORS]
        assert "SN-011" in names

    def test_sn011_targets_sting(self):
        sn011 = next(inh for inh in INHIBITORS if inh.name == "SN-011")
        assert sn011.target == "sting"
        assert sn011.efficacy == 0.92

    def test_sn011_highest_sting_efficacy(self):
        sting_inhibitors = [inh for inh in INHIBITORS if inh.target == "sting"]
        best = max(sting_inhibitors, key=lambda i: i.efficacy)
        assert best.name == "SN-011"


class TestSimulateInhibitorEffect:
    def test_sting_inhibitor_reduces_ifn(self):
        h151 = INHIBITORS[0]  # H-151 targets STING
        result = simulate_inhibitor_effect(0.1, 0.1, 0.5, h151)
        assert result["inhibited_ifn"] < result["baseline_ifn"]
        assert result["ifn_reduction"] > 0

    def test_cgas_inhibitor_reduces_ifn(self):
        ru521 = INHIBITORS[1]  # RU.521 targets cGAS
        result = simulate_inhibitor_effect(0.1, 0.1, 0.5, ru521)
        assert result["inhibited_ifn"] < result["baseline_ifn"]

    def test_no_reduction_at_zero_dna(self):
        """No IFN to reduce when there's no DNA input."""
        result = simulate_inhibitor_effect(0.1, 0.1, 0.0, INHIBITORS[0])
        assert result["baseline_ifn"] == 0.0
        assert result["ifn_reduction"] == 0.0

    def test_result_fields(self):
        result = simulate_inhibitor_effect(0.5, 0.5, 0.5, INHIBITORS[0])
        assert "inhibitor" in result
        assert "target" in result
        assert "baseline_ifn" in result
        assert "inhibited_ifn" in result
        assert "pct_reduction" in result

    def test_higher_efficacy_more_reduction(self):
        low_eff = Inhibitor("low", "sting", 0.3)
        high_eff = Inhibitor("high", "sting", 0.9)
        r_low = simulate_inhibitor_effect(0.1, 0.1, 0.5, low_eff)
        r_high = simulate_inhibitor_effect(0.1, 0.1, 0.5, high_eff)
        assert r_high["ifn_reduction"] >= r_low["ifn_reduction"]

    def test_sn011_reduces_ifn(self):
        sn011 = next(inh for inh in INHIBITORS if inh.name == "SN-011")
        result = simulate_inhibitor_effect(0.1, 0.1, 0.5, sn011)
        assert result["inhibited_ifn"] < result["baseline_ifn"]
        assert result["pct_reduction"] > 0

    def test_sn011_outperforms_h151(self):
        """SN-011 (efficacy 0.92) should reduce IFN more than H-151 (0.85)."""
        h151 = next(inh for inh in INHIBITORS if inh.name == "H-151")
        sn011 = next(inh for inh in INHIBITORS if inh.name == "SN-011")
        r_h151 = simulate_inhibitor_effect(0.1, 0.1, 0.5, h151)
        r_sn011 = simulate_inhibitor_effect(0.1, 0.1, 0.5, sn011)
        assert r_sn011["ifn_reduction"] >= r_h151["ifn_reduction"]


class TestPredictTherapy:
    def test_lectin_only_recommends_complement(self):
        profile = COMPOUND_PROFILES[0]  # lectin_only
        pred = predict_therapy(profile)
        assert "complement" in pred["recommended_therapy"].lower() or \
               "monitoring" in pred["recommended_therapy"].lower()

    def test_cgas_sting_only_recommends_inhibitor(self):
        profile = COMPOUND_PROFILES[1]  # cgas_sting_only
        pred = predict_therapy(profile)
        # Should recommend an inhibitor, not complement replacement
        assert "complement" not in pred["recommended_therapy"].lower() or \
               "H-151" in pred["recommended_therapy"] or \
               "RU.521" in pred["recommended_therapy"]

    def test_two_hit_severe_recommends_combined(self):
        profile = COMPOUND_PROFILES[3]  # two_hit_severe
        pred = predict_therapy(profile)
        assert "combined" in pred["recommended_therapy"].lower() or \
               "complement" in pred["recommended_therapy"].lower()

    def test_prediction_has_all_inhibitor_results(self):
        pred = predict_therapy(COMPOUND_PROFILES[1])
        assert len(pred["inhibitor_results"]) == len(INHIBITORS)

    def test_best_inhibitor_is_valid(self):
        pred = predict_therapy(COMPOUND_PROFILES[1])
        valid_names = [inh.name for inh in INHIBITORS]
        assert pred["best_single_inhibitor"] in valid_names

    def test_sn011_preferred_sting_for_ddr_profiles(self):
        """SN-011 should be the preferred STING inhibitor for DDR/cGAS-STING profiles."""
        for profile in COMPOUND_PROFILES:
            if profile["trex1_activity"] < 0.8 or profile["samhd1_activity"] < 0.8:
                pred = predict_therapy(profile)
                assert pred["preferred_sting_inhibitor"] == "SN-011"

    def test_sn011_in_rationale_for_ddr_profiles(self):
        """Profiles with cGAS-STING variants should mention SN-011 in rationale."""
        for profile in COMPOUND_PROFILES:
            if profile["trex1_activity"] < 0.8 or profile["samhd1_activity"] < 0.8:
                pred = predict_therapy(profile)
                assert "SN-011" in pred["rationale"]


class TestRunPipeline:
    def test_creates_output_files(self, tmp_path):
        results = run_therapeutic_predictions(dest_dir=tmp_path)
        assert (tmp_path / "therapeutic_predictions.json").exists()
        assert (tmp_path / "therapeutic_comparison.png").exists()

    def test_json_output_valid(self, tmp_path):
        run_therapeutic_predictions(dest_dir=tmp_path)
        with open(tmp_path / "therapeutic_predictions.json") as f:
            data = json.load(f)
        assert "metadata" in data
        assert "predictions" in data
        assert len(data["predictions"]) == len(COMPOUND_PROFILES)

    def test_all_profiles_have_predictions(self, tmp_path):
        results = run_therapeutic_predictions(dest_dir=tmp_path)
        for pred in results["predictions"]:
            assert "recommended_therapy" in pred
            assert "rationale" in pred
            assert "baseline_ifn" in pred

    def test_metadata_has_inhibitors(self, tmp_path):
        results = run_therapeutic_predictions(dest_dir=tmp_path)
        assert len(results["metadata"]["inhibitors"]) == len(INHIBITORS)
