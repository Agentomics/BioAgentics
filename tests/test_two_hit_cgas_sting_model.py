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
    run_pep_age_sweep,
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
                    state.ifn_output, state.complement_activation]:
            assert 0.0 <= val <= 1.0

    def test_ddr_impairment_increases_cytosolic_dna(self):
        """DDR impairment should increase cytosolic DNA even without infection."""
        wt = simulate_pathway(gas_dna=0.0, trex1_activity=1.0, samhd1_activity=1.0, ddr_activity=1.0)
        ddr_lof = simulate_pathway(gas_dna=0.0, trex1_activity=1.0, samhd1_activity=1.0, ddr_activity=0.0)
        assert ddr_lof.cytosolic_dna > wt.cytosolic_dna

    def test_mito_impairment_increases_cytosolic_dna(self):
        """Mitochondrial impairment should increase cytosolic DNA even without infection."""
        wt = simulate_pathway(gas_dna=0.0, trex1_activity=1.0, samhd1_activity=1.0, mito_activity=1.0)
        mito_lof = simulate_pathway(gas_dna=0.0, trex1_activity=1.0, samhd1_activity=1.0, mito_activity=0.0)
        assert mito_lof.cytosolic_dna > wt.cytosolic_dna

    def test_ddr_leakage_magnitude(self):
        """Full DDR loss should contribute 0.3 to cytosolic DNA."""
        state = simulate_pathway(gas_dna=0.0, trex1_activity=1.0, samhd1_activity=1.0, ddr_activity=0.0)
        assert abs(state.cytosolic_dna - 0.3) < 0.01

    def test_mito_leakage_magnitude(self):
        """Full mito loss should contribute 0.25 to cytosolic DNA."""
        state = simulate_pathway(gas_dna=0.0, trex1_activity=1.0, samhd1_activity=1.0, mito_activity=0.0)
        assert abs(state.cytosolic_dna - 0.25) < 0.01

    def test_triple_hit_higher_than_single(self):
        """Triple hit (TREX1+DDR+mito) should produce more IFN than any single axis."""
        trex1_only = simulate_pathway(gas_dna=0.5, trex1_activity=0.5, samhd1_activity=1.0)
        ddr_only = simulate_pathway(gas_dna=0.5, trex1_activity=1.0, samhd1_activity=1.0, ddr_activity=0.3)
        triple = simulate_pathway(gas_dna=0.5, trex1_activity=0.5, samhd1_activity=1.0,
                                  ddr_activity=0.3, mito_activity=0.3)
        assert triple.ifn_output > trex1_only.ifn_output
        assert triple.ifn_output > ddr_only.ifn_output

    def test_backward_compatible_defaults(self):
        """Default ddr/mito=1.0 should not change existing scenario results."""
        old = simulate_pathway(gas_dna=0.5, trex1_activity=0.5, samhd1_activity=0.5)
        new = simulate_pathway(gas_dna=0.5, trex1_activity=0.5, samhd1_activity=0.5,
                               ddr_activity=1.0, mito_activity=1.0)
        assert old.ifn_output == new.ifn_output

    def test_output_bounded_with_all_impaired(self):
        """All outputs should stay in [0, 1] even with maximum impairment."""
        state = simulate_pathway(gas_dna=1.0, trex1_activity=0.0, samhd1_activity=0.0,
                                 ddr_activity=0.0, mito_activity=0.0,
                                 hif1a_activity=0.0, pep_level=0.0)
        for val in [state.cytosolic_dna, state.cgas_activity, state.cgamp_level,
                    state.sting_activity, state.tbk1_activity, state.irf3_activity,
                    state.ifn_output]:
            assert 0.0 <= val <= 1.0

    def test_low_pep_increases_ifn(self):
        """Low PEP (age-related) should increase IFN output by lowering cGAS threshold."""
        normal_pep = simulate_pathway(gas_dna=0.5, trex1_activity=0.5, samhd1_activity=1.0, pep_level=1.0)
        low_pep = simulate_pathway(gas_dna=0.5, trex1_activity=0.5, samhd1_activity=1.0, pep_level=0.3)
        assert low_pep.ifn_output > normal_pep.ifn_output

    def test_low_hif1a_increases_ifn(self):
        """Low HIF-1a should increase IFN output by lowering STING threshold."""
        normal_hif = simulate_pathway(gas_dna=0.5, trex1_activity=0.5, samhd1_activity=1.0, hif1a_activity=1.0)
        low_hif = simulate_pathway(gas_dna=0.5, trex1_activity=0.5, samhd1_activity=1.0, hif1a_activity=0.3)
        assert low_hif.ifn_output > normal_hif.ifn_output

    def test_metabolic_modifiers_backward_compatible(self):
        """Default hif1a=1.0 and pep=1.0 should not change existing results."""
        old = simulate_pathway(gas_dna=0.5, trex1_activity=0.5, samhd1_activity=0.5)
        new = simulate_pathway(gas_dna=0.5, trex1_activity=0.5, samhd1_activity=0.5,
                               hif1a_activity=1.0, pep_level=1.0)
        assert old.ifn_output == new.ifn_output

    def test_complement_proportional_to_ifn(self):
        """Complement activation should be proportional to IFN output."""
        state = simulate_pathway(gas_dna=0.5, trex1_activity=0.1, samhd1_activity=0.1)
        assert state.complement_activation == state.ifn_output * 0.6

    def test_complement_zero_when_no_ifn(self):
        """No IFN → no complement activation."""
        state = simulate_pathway(gas_dna=0.0, trex1_activity=1.0, samhd1_activity=1.0)
        assert state.complement_activation == 0.0

    def test_complement_bounded(self):
        """Complement activation should be in [0, 0.6] (max = IFN=1.0 * 0.6)."""
        state = simulate_pathway(gas_dna=1.0, trex1_activity=0.0, samhd1_activity=0.0)
        assert 0.0 <= state.complement_activation <= 0.6

    def test_mg_like_samhd1_complement(self):
        """MG-like SAMHD1 variant should produce complement activation at moderate infection."""
        state = simulate_pathway(gas_dna=0.5, trex1_activity=1.0, samhd1_activity=0.3)
        assert state.complement_activation > 0


class TestRunGenotypSimulations:
    def test_returns_all_scenarios(self):
        results = run_genotype_simulations()
        assert "scenarios" in results
        assert len(results["scenarios"]) == len(DEFAULT_SCENARIOS)
        assert len(DEFAULT_SCENARIOS) == 13

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


class TestPepAgeSweep:
    def test_sweep_returns_expected_structure(self):
        result = run_pep_age_sweep()
        assert "sweep" in result
        assert len(result["sweep"]) == 8
        for entry in result["sweep"]:
            assert "pep_level" in entry
            assert "ifn_output" in entry

    def test_ifn_increases_as_pep_decreases(self):
        """IFN should increase as PEP declines (aging)."""
        result = run_pep_age_sweep()
        sweep = result["sweep"]
        assert sweep[-1]["ifn_output"] > sweep[0]["ifn_output"]


class TestRunModel:
    def test_creates_output_files(self, tmp_path):
        results = run_cgas_sting_model(dest_dir=tmp_path)
        assert (tmp_path / "pathway_activation_scores.json").exists()
        assert (tmp_path / "ifn_response_curves.png").exists()
        assert (tmp_path / "pathway_heatmap.png").exists()
        assert (tmp_path / "pep_age_sweep.json").exists()
        assert (tmp_path / "pep_age_sweep.png").exists()

    def test_json_output_valid(self, tmp_path):
        run_cgas_sting_model(dest_dir=tmp_path)
        with open(tmp_path / "pathway_activation_scores.json") as f:
            data = json.load(f)
        assert "metadata" in data
        assert "scenarios" in data
        assert len(data["scenarios"]) > 0
