"""Episodic-to-chronic temporal dynamics model for PANDAS/PANS.

Models the transition from episodic infection-triggered flares to chronic
neuropsychiatric symptoms via cumulative cGAS-STING activation.

Based on Corona et al. (PMID 41816879): transient cGAS-STING activation is
neuroprotective (clears pathogen, resolves inflammation) but sustained activation
causes metabolic austerity suppressing synaptic plasticity → chronic damage.

Key concept: hypomorphic variants (unlike AGS null mutations) don't cause
constitutive activation. Instead, they retain residual STING activation after
each infection episode, meaning fewer episodes are needed to cross the
chronicity threshold.

Usage:
    uv run python -m bioagentics.models.two_hit_temporal_dynamics [--dest DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bioagentics.models.two_hit_cgas_sting_model import (
    DEFAULT_SCENARIOS,
    GenotypeScenario,
    simulate_pathway,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/pandas_pans/two-hit-interferonopathy-model")

# Chronicity threshold: cumulative baseline STING above this → chronic symptoms
CHRONICITY_THRESHOLD = 0.6


@dataclass
class EpisodeResult:
    """Result of a single infection episode."""
    episode: int
    gas_dna_pulse: float
    peak_sting: float
    peak_ifn: float
    residual_sting: float
    cumulative_baseline: float
    is_chronic: bool


@dataclass
class TemporalTrajectory:
    """Full trajectory of a genotype across multiple episodes."""
    scenario_name: str
    description: str
    episodes: list[EpisodeResult] = field(default_factory=list)
    episodes_to_chronicity: int | None = None


def simulate_episodic_trajectory(
    scenario: GenotypeScenario,
    num_episodes: int = 12,
    gas_dna_pulse: float = 0.6,
    resolution_rate: float | None = None,
    chronicity_threshold: float = CHRONICITY_THRESHOLD,
) -> TemporalTrajectory:
    """Simulate repeated infection episodes and track cumulative STING activation.

    After each episode, a fraction of STING activity persists as residual
    activation. The resolution rate depends on genotype: WT resolves fully,
    hypomorphic variants retain more residual activation.

    Args:
        scenario: Genotype to simulate.
        num_episodes: Number of GAS infection episodes.
        gas_dna_pulse: DNA exposure per episode (0-1).
        resolution_rate: Fraction of STING activity cleared between episodes.
            If None, computed from genotype (WT ~0.95, LOF ~0.5).
        chronicity_threshold: Cumulative baseline STING for chronic transition.

    Returns:
        TemporalTrajectory with episode-by-episode results.
    """
    if resolution_rate is None:
        # Resolution rate scales with clearance capacity
        clearance = scenario.trex1_activity * 0.7 + scenario.samhd1_activity * 0.3
        # DDR/mito impairment also reduces resolution (chronic DNA sources)
        axis_penalty = (1.0 - scenario.ddr_activity) * 0.1 + (1.0 - scenario.mito_activity) * 0.1
        resolution_rate = max(0.3, min(0.95, clearance * 0.95 - axis_penalty))

    trajectory = TemporalTrajectory(
        scenario_name=scenario.name,
        description=scenario.description,
    )

    cumulative_baseline = 0.0
    episodes_to_chronicity = None

    for ep in range(num_episodes):
        state = simulate_pathway(
            gas_dna=gas_dna_pulse,
            trex1_activity=scenario.trex1_activity,
            samhd1_activity=scenario.samhd1_activity,
            ddr_activity=scenario.ddr_activity,
            mito_activity=scenario.mito_activity,
            hif1a_activity=scenario.hif1a_activity,
            pep_level=scenario.pep_level,
        )

        peak_sting = max(state.sting_activity, cumulative_baseline)
        peak_ifn = state.ifn_output

        # Residual STING after resolution attempt
        residual = peak_sting * (1.0 - resolution_rate)
        cumulative_baseline = max(cumulative_baseline, residual) + residual * 0.5
        cumulative_baseline = min(1.0, cumulative_baseline)

        is_chronic = cumulative_baseline >= chronicity_threshold
        if is_chronic and episodes_to_chronicity is None:
            episodes_to_chronicity = ep + 1

        trajectory.episodes.append(EpisodeResult(
            episode=ep + 1,
            gas_dna_pulse=round(gas_dna_pulse, 3),
            peak_sting=round(peak_sting, 4),
            peak_ifn=round(peak_ifn, 4),
            residual_sting=round(residual, 4),
            cumulative_baseline=round(cumulative_baseline, 4),
            is_chronic=is_chronic,
        ))

    trajectory.episodes_to_chronicity = episodes_to_chronicity
    return trajectory


def run_temporal_simulations(
    scenarios: list[GenotypeScenario] | None = None,
    num_episodes: int = 12,
) -> dict:
    """Run temporal dynamics across all genotype scenarios.

    Returns dict with trajectories for each scenario.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS

    results = {
        "metadata": {
            "model_type": "episodic-to-chronic temporal dynamics",
            "chronicity_threshold": CHRONICITY_THRESHOLD,
            "num_episodes": num_episodes,
            "description": (
                "Models PANDAS episodic→chronic transition via cumulative "
                "cGAS-STING activation across repeated GAS infection episodes"
            ),
        },
        "trajectories": {},
    }

    for scenario in scenarios:
        trajectory = simulate_episodic_trajectory(scenario, num_episodes=num_episodes)
        logger.info("  %s: episodes_to_chronicity=%s",
                     scenario.name,
                     trajectory.episodes_to_chronicity or "never")

        results["trajectories"][scenario.name] = {
            "description": trajectory.description,
            "episodes_to_chronicity": trajectory.episodes_to_chronicity,
            "episodes": [
                {
                    "episode": ep.episode,
                    "peak_sting": ep.peak_sting,
                    "peak_ifn": ep.peak_ifn,
                    "residual_sting": ep.residual_sting,
                    "cumulative_baseline": ep.cumulative_baseline,
                    "is_chronic": ep.is_chronic,
                }
                for ep in trajectory.episodes
            ],
        }

    return results


def plot_temporal_trajectories(
    results: dict,
    dest: Path,
    title: str = "Cumulative STING Activation Over Infection Episodes",
) -> None:
    """Plot cumulative STING baseline over episodes for each genotype."""
    trajectories = results["trajectories"]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "wild_type": "#2ECC71",
        "trex1_het": "#F39C12",
        "samhd1_het": "#E67E22",
        "trex1_lof": "#E74C3C",
        "samhd1_lof": "#C0392B",
        "compound_het": "#8E44AD",
        "compound_lof": "#2C3E50",
        "ddr_impaired": "#3498DB",
        "mito_impaired": "#1ABC9C",
        "triple_hit": "#E91E63",
        "metabolic_vulnerable": "#FF6F00",
    }

    for name, data in trajectories.items():
        episodes = [ep["episode"] for ep in data["episodes"]]
        baselines = [ep["cumulative_baseline"] for ep in data["episodes"]]
        color = colors.get(name, "#95A5A6")
        ax.plot(episodes, baselines, "o-", color=color, label=name, linewidth=2, markersize=4)

    ax.axhline(y=CHRONICITY_THRESHOLD, color="red", linestyle="--", alpha=0.7,
               label=f"Chronicity threshold ({CHRONICITY_THRESHOLD})")
    ax.set_xlabel("Infection Episode", fontsize=12)
    ax.set_ylabel("Cumulative Baseline STING", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved temporal trajectories plot: %s", dest)


def run_temporal_model(dest_dir: Path | None = None) -> dict:
    """Run the full temporal dynamics model."""
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Temporal dynamics: episodic → chronic transition ===")
    results = run_temporal_simulations()

    # Save results
    out_path = dest_dir / "temporal_dynamics.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved temporal dynamics: %s", out_path)

    # Plot
    plot_temporal_trajectories(results, dest_dir / "temporal_dynamics.png")

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Episodic-to-chronic temporal dynamics model for PANDAS/PANS"
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_temporal_model(dest_dir=args.dest)

    print(f"\nSimulated {len(results['trajectories'])} genotype trajectories")
    print(f"Episodes: {results['metadata']['num_episodes']}")
    print(f"Chronicity threshold: {results['metadata']['chronicity_threshold']}")
    print("\nEpisodes to chronicity:")
    for name, data in results["trajectories"].items():
        etoc = data["episodes_to_chronicity"]
        print(f"  {name}: {etoc or 'never'}")


if __name__ == "__main__":
    main()
