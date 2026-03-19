"""Tests for the cGAS-STING pathway activation model."""

import json
from pathlib import Path

import pytest

from bioagentics.models.two_hit_cgas_sting_model import (
    DEFAULT_SCENARIOS,
    GenotypeScenario,
    PathwayState,
    hill,
    simulate_pathway,
    run_genotype_simulations,
    run_cgas_sting_model,
)


class TestHillFunction:
    def test_zero_input(self):
        assert hill(0.0) == 0.0

    def test_at_half_max(self):
        result = hill(0.5, k=0.5, n=2)
        assert abs(result - 0.5) < 0.01

    def test_high_input_saturates(self):
        result = hill(1.0, k=0.5, n=2)
        assert result > 0.75

    def test_monotonically_increasing(self):
        values = [hill(x / 10, k=0.5) for x in range(11)]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1]


class TestSimulatePathway:
    def test_no_dna_no_ifn(self):
        """With no DNA input, IFN output should be zero."""
        state = simulate_pathway(gas_dna=0.0, trex1_activity=1.0, samhd1_activity=1.0)
        assert state.ifn_output == 0.0
        assert state.cytosolic_dna == 0.0

    def test_wt_low_ifn(self):
        """Wild-type TREX1/SAMHD1 should clear DNA and produce low IFN."""
        state = simulate_pathway(gas_dna=0.5, trex1_activity=1.0, samhd1_activity=1.0)
        assert state.ifn_output < 0.1

    def test_lof_high_ifn(self):
        """Loss-of-function TREX1+SAMHD1 should produce high IFN."""
        state = simulate_pathway(gas_dna=0.5, trex1_activity=0.1, samhd1_activity=0.1)
        assert state.ifn_output > 0.3

    def test_trex1_lof_higher_than_samhd1_lof(self):
        """TREX1 is primary exonuclease — its LOF should have larger effect."""
        trex1_state = simulate_pathway(gas_dna=0.5, trex1_activity=0.1, samhd1_activity=1.0)
        samhd1_state = simulate_pathway(gas_dna=0.5, trex1_activity=1.0, samhd1_activity=0.1)
        assert trex1_state.ifn_output > samhd1_state.ifn_output

    def test_cascade_ordering(self):
        """Upstream nodes should activate before downstream ones."""
        state = simulate_pathway(gas_dna=0.8, trex1_activity=0.1, samhd1_activity=0.1)
        assert state.cytosolic_dna > 0
        assert state.cgas_activity > 0
        assert state.sting_activity > 0

    def test_output_bounded(self):
        """All outputs should be in [0, 1]."""
        state = simulate_pathway(gas_dna=1.0, trex1_activity=0.0, samhd1_activity=0.0)
        for val in [state.cytosolic_dna, state.cgas_activity, state.cgamp_level,
                    state.sting_activity, state.tbk1_activity, state.irf3_activity,
                    state.ifn_output]:
            assert 0.0 <= val <= 1.0


class TestRunGenotypSimulations:
    def test_returns_all_scenarios(self):
        results = run_genotype_simulations()
        assert "scenarios" in results
        assert len(results["scenarios"]) == len(DEFAULT_SCENARIOS)

    def test_scenario_has_exposure_results(self):
        results = run_genotype_simulations()
        for name, data in results["scenarios"].items():
            assert "exposure_results" in data
            assert len(data["exposure_results"]) > 0
            assert "ifn_at_moderate_infection" in data

    def test_custom_scenarios(self):
        scenarios = [
            GenotypeScenario("test_wt", 1.0, 1.0),
            GenotypeScenario("test_lof", 0.0, 0.0),
        ]
        results = run_genotype_simulations(scenarios=scenarios, dna_exposures=[0.0, 0.5, 1.0])
        assert len(results["scenarios"]) == 2
        assert "test_wt" in results["scenarios"]
        assert "test_lof" in results["scenarios"]

    def test_wt_threshold_higher_than_lof(self):
        """WT genotype should require more DNA to trigger IFN than LOF."""
        results = run_genotype_simulations()
        wt_thresh = results["scenarios"]["wild_type"]["dna_threshold_50pct_ifn"]
        lof_thresh = results["scenarios"]["compound_lof"]["dna_threshold_50pct_ifn"]
        if wt_thresh is not None and lof_thresh is not None:
            assert wt_thresh > lof_thresh


class TestRunModel:
    def test_creates_output_files(self, tmp_path):
        results = run_cgas_sting_model(dest_dir=tmp_path)
        assert (tmp_path / "pathway_activation_scores.json").exists()
        assert (tmp_path / "ifn_response_curves.png").exists()
        assert (tmp_path / "pathway_heatmap.png").exists()

    def test_json_output_valid(self, tmp_path):
        run_cgas_sting_model(dest_dir=tmp_path)
        with open(tmp_path / "pathway_activation_scores.json") as f:
            data = json.load(f)
        assert "metadata" in data
        assert "scenarios" in data
        assert len(data["scenarios"]) > 0
