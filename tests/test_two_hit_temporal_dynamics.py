"""Tests for the episodic-to-chronic temporal dynamics model."""

import json

from bioagentics.models.two_hit_temporal_dynamics import (
    CHRONICITY_THRESHOLD,
    GenotypeScenario,
    simulate_episodic_trajectory,
    run_temporal_simulations,
    run_temporal_model,
)


class TestSimulateEpisodicTrajectory:
    def test_wt_never_chronic(self):
        """Wild-type should never reach chronicity in 12 episodes."""
        scenario = GenotypeScenario("wt", 1.0, 1.0)
        traj = simulate_episodic_trajectory(scenario, num_episodes=12)
        assert traj.episodes_to_chronicity is None

    def test_compound_lof_reaches_chronicity(self):
        """Compound LOF should reach chronicity quickly."""
        scenario = GenotypeScenario("lof", 0.1, 0.1)
        traj = simulate_episodic_trajectory(scenario, num_episodes=12)
        assert traj.episodes_to_chronicity is not None
        assert traj.episodes_to_chronicity <= 12

    def test_lof_faster_than_het(self):
        """LOF should reach chronicity faster than het."""
        lof = simulate_episodic_trajectory(
            GenotypeScenario("lof", 0.1, 0.1), num_episodes=12)
        het = simulate_episodic_trajectory(
            GenotypeScenario("het", 0.5, 0.5), num_episodes=12)
        if lof.episodes_to_chronicity and het.episodes_to_chronicity:
            assert lof.episodes_to_chronicity <= het.episodes_to_chronicity

    def test_episode_count_matches(self):
        traj = simulate_episodic_trajectory(
            GenotypeScenario("test", 0.5, 1.0), num_episodes=8)
        assert len(traj.episodes) == 8

    def test_cumulative_baseline_monotonic(self):
        """Cumulative baseline should never decrease."""
        traj = simulate_episodic_trajectory(
            GenotypeScenario("test", 0.5, 1.0), num_episodes=12)
        baselines = [ep.cumulative_baseline for ep in traj.episodes]
        for i in range(len(baselines) - 1):
            assert baselines[i] <= baselines[i + 1]

    def test_bounded_output(self):
        """All values should be in [0, 1]."""
        traj = simulate_episodic_trajectory(
            GenotypeScenario("extreme", 0.0, 0.0, ddr_activity=0.0, mito_activity=0.0),
            num_episodes=12)
        for ep in traj.episodes:
            assert 0.0 <= ep.cumulative_baseline <= 1.0
            assert 0.0 <= ep.peak_sting <= 1.0


class TestRunTemporalSimulations:
    def test_returns_all_default_scenarios(self):
        results = run_temporal_simulations()
        assert "trajectories" in results
        assert len(results["trajectories"]) == 13

    def test_metadata_present(self):
        results = run_temporal_simulations()
        assert results["metadata"]["chronicity_threshold"] == CHRONICITY_THRESHOLD


class TestRunModel:
    def test_creates_output_files(self, tmp_path):
        run_temporal_model(dest_dir=tmp_path)
        assert (tmp_path / "temporal_dynamics.json").exists()
        assert (tmp_path / "temporal_dynamics.png").exists()

    def test_json_output_valid(self, tmp_path):
        run_temporal_model(dest_dir=tmp_path)
        with open(tmp_path / "temporal_dynamics.json") as f:
            data = json.load(f)
        assert "metadata" in data
        assert "trajectories" in data
