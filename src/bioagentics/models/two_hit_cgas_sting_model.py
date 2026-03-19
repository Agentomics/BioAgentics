"""cGAS-STING pathway activation model for the two-hit interferonopathy hypothesis.

Models the cGAS-STING innate immune signaling cascade under different genotype
scenarios (wild-type vs loss-of-function TREX1/SAMHD1 variants) and GAS DNA
exposure levels. Predicts type I interferon output for each combination.

Pathway logic:
  GAS infection → pathogen DNA release → cytosolic DNA
  TREX1 (exonuclease) degrades cytosolic DNA
  SAMHD1 (dNTPase) restricts dNTP pools, limiting DNA accumulation
  Cytosolic DNA → cGAS detection → 2'3'-cGAMP → STING → TBK1 → IRF3 → IFN-β

Loss-of-function variants in TREX1/SAMHD1 reduce DNA clearance, lowering the
threshold for infection-triggered type I IFN storms.

Usage:
    uv run python -m bioagentics.models.two_hit_cgas_sting_model [--dest DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/pandas_pans/two-hit-interferonopathy-model")

# cGAS-STING pathway gene symbols and their roles
PATHWAY_GENES = {
    "TREX1": "3'-5' exonuclease; degrades cytosolic DNA",
    "SAMHD1": "dNTPase; restricts dNTP pools, limits DNA accumulation",
    "MB21D1": "cGAS (cyclic GMP-AMP synthase); detects cytosolic dsDNA",
    "TMEM173": "STING (stimulator of IFN genes); activated by 2'3'-cGAMP",
    "TBK1": "TANK-binding kinase 1; phosphorylates IRF3",
    "IRF3": "Interferon regulatory factor 3; transcription factor for IFN-β",
}

# Gene aliases for cross-referencing
GENE_ALIASES = {
    "cGAS": "MB21D1",
    "STING": "TMEM173",
    "STING1": "TMEM173",
}


@dataclass
class GenotypeScenario:
    """Represents a patient genotype for the cGAS-STING checkpoint genes."""
    name: str
    trex1_activity: float  # 0.0 (null) to 1.0 (wild-type)
    samhd1_activity: float  # 0.0 (null) to 1.0 (wild-type)
    description: str = ""


@dataclass
class PathwayState:
    """State of each node in the cGAS-STING signaling cascade."""
    gas_dna_input: float = 0.0
    cytosolic_dna: float = 0.0
    cgas_activity: float = 0.0
    cgamp_level: float = 0.0
    sting_activity: float = 0.0
    tbk1_activity: float = 0.0
    irf3_activity: float = 0.0
    ifn_output: float = 0.0


# Default genotype scenarios to simulate
DEFAULT_SCENARIOS: list[GenotypeScenario] = [
    GenotypeScenario(
        name="wild_type",
        trex1_activity=1.0,
        samhd1_activity=1.0,
        description="Normal TREX1 and SAMHD1 — full DNA clearance",
    ),
    GenotypeScenario(
        name="trex1_het",
        trex1_activity=0.5,
        samhd1_activity=1.0,
        description="Heterozygous TREX1 LOF — partial DNA clearance (carrier)",
    ),
    GenotypeScenario(
        name="samhd1_het",
        trex1_activity=1.0,
        samhd1_activity=0.5,
        description="Heterozygous SAMHD1 LOF — partial dNTP restriction",
    ),
    GenotypeScenario(
        name="trex1_lof",
        trex1_activity=0.1,
        samhd1_activity=1.0,
        description="TREX1 LOF (hypomorphic) — severely reduced DNA clearance",
    ),
    GenotypeScenario(
        name="samhd1_lof",
        trex1_activity=1.0,
        samhd1_activity=0.1,
        description="SAMHD1 LOF (hypomorphic) — severely reduced dNTP restriction",
    ),
    GenotypeScenario(
        name="compound_het",
        trex1_activity=0.5,
        samhd1_activity=0.5,
        description="Compound heterozygous — both partially impaired (PANS two-hit)",
    ),
    GenotypeScenario(
        name="compound_lof",
        trex1_activity=0.1,
        samhd1_activity=0.1,
        description="Compound LOF — both severely impaired (AGS-like)",
    ),
]

# GAS DNA exposure levels to simulate (normalized 0-1)
DEFAULT_DNA_EXPOSURES = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]


def hill(x: float, k: float = 0.5, n: float = 2.0) -> float:
    """Hill function for sigmoidal activation.

    Args:
        x: Input signal (0-1).
        k: Half-maximal activation threshold.
        n: Hill coefficient (cooperativity).

    Returns:
        Activated output (0-1).
    """
    if x <= 0:
        return 0.0
    return float(x**n / (k**n + x**n))


def simulate_pathway(
    gas_dna: float,
    trex1_activity: float,
    samhd1_activity: float,
    cgas_k: float = 0.3,
    sting_k: float = 0.4,
    tbk1_k: float = 0.4,
    irf3_k: float = 0.5,
) -> PathwayState:
    """Simulate steady-state cGAS-STING pathway activation.

    Model: GAS DNA → [TREX1/SAMHD1 clearance] → cytosolic DNA →
           cGAS → cGAMP → STING → TBK1 → IRF3 → IFN-β

    TREX1 and SAMHD1 act as negative regulators (DNA clearance).
    Their loss-of-function increases cytosolic DNA accumulation.

    Args:
        gas_dna: GAS pathogen DNA exposure level (0-1).
        trex1_activity: TREX1 enzymatic activity (0=null, 1=WT).
        samhd1_activity: SAMHD1 enzymatic activity (0=null, 1=WT).
        cgas_k: cGAS half-activation threshold.
        sting_k: STING half-activation threshold.
        tbk1_k: TBK1 half-activation threshold.
        irf3_k: IRF3 half-activation threshold.

    Returns:
        PathwayState with all intermediate and output values.
    """
    state = PathwayState(gas_dna_input=gas_dna)

    # Cytosolic DNA accumulation depends on input and clearance
    # TREX1 degrades DNA directly; SAMHD1 restricts dNTP pools
    # Clearance = TREX1 * 0.7 + SAMHD1 * 0.3 (TREX1 is primary)
    clearance = trex1_activity * 0.7 + samhd1_activity * 0.3
    state.cytosolic_dna = gas_dna * (1.0 - clearance * 0.9)
    state.cytosolic_dna = max(0.0, min(1.0, state.cytosolic_dna))

    # cGAS detects cytosolic dsDNA via sigmoidal activation
    state.cgas_activity = hill(state.cytosolic_dna, k=cgas_k)

    # cGAS produces 2'3'-cGAMP (proportional to cGAS activity)
    state.cgamp_level = state.cgas_activity * 0.95

    # STING activation by cGAMP
    state.sting_activity = hill(state.cgamp_level, k=sting_k)

    # TBK1 phosphorylation by active STING
    state.tbk1_activity = hill(state.sting_activity, k=tbk1_k)

    # IRF3 phosphorylation by active TBK1
    state.irf3_activity = hill(state.tbk1_activity, k=irf3_k)

    # Type I IFN output (IFN-β transcription driven by active IRF3)
    state.ifn_output = state.irf3_activity

    return state


def run_genotype_simulations(
    scenarios: list[GenotypeScenario] | None = None,
    dna_exposures: list[float] | None = None,
) -> dict:
    """Run pathway simulations across all genotype × exposure combinations.

    Returns:
        Dict with results for all scenarios keyed by scenario name,
        including pathway states at each DNA exposure level.
    """
    if scenarios is None:
        scenarios = DEFAULT_SCENARIOS
    if dna_exposures is None:
        dna_exposures = DEFAULT_DNA_EXPOSURES

    results = {
        "metadata": {
            "pathway_genes": PATHWAY_GENES,
            "model_type": "semi-quantitative steady-state (Hill functions)",
            "description": (
                "Simulates cGAS-STING pathway activation under different "
                "TREX1/SAMHD1 genotype scenarios and GAS DNA exposure levels"
            ),
        },
        "dna_exposure_levels": dna_exposures,
        "scenarios": {},
    }

    for scenario in scenarios:
        logger.info("Simulating: %s (TREX1=%.1f, SAMHD1=%.1f)",
                     scenario.name, scenario.trex1_activity, scenario.samhd1_activity)

        scenario_data = {
            "description": scenario.description,
            "trex1_activity": scenario.trex1_activity,
            "samhd1_activity": scenario.samhd1_activity,
            "exposure_results": [],
        }

        for dna in dna_exposures:
            state = simulate_pathway(
                gas_dna=dna,
                trex1_activity=scenario.trex1_activity,
                samhd1_activity=scenario.samhd1_activity,
            )
            scenario_data["exposure_results"].append({
                "gas_dna_input": round(dna, 3),
                "cytosolic_dna": round(state.cytosolic_dna, 4),
                "cgas_activity": round(state.cgas_activity, 4),
                "sting_activity": round(state.sting_activity, 4),
                "tbk1_activity": round(state.tbk1_activity, 4),
                "irf3_activity": round(state.irf3_activity, 4),
                "ifn_output": round(state.ifn_output, 4),
            })

        # Summary: IFN output at moderate infection (0.5 DNA exposure)
        moderate_state = simulate_pathway(
            gas_dna=0.5,
            trex1_activity=scenario.trex1_activity,
            samhd1_activity=scenario.samhd1_activity,
        )
        scenario_data["ifn_at_moderate_infection"] = round(moderate_state.ifn_output, 4)

        # Summary: threshold DNA exposure to trigger >50% IFN
        threshold = _find_ifn_threshold(scenario.trex1_activity, scenario.samhd1_activity)
        scenario_data["dna_threshold_50pct_ifn"] = round(threshold, 4) if threshold is not None else None

        results["scenarios"][scenario.name] = scenario_data

    return results


def _find_ifn_threshold(
    trex1_activity: float,
    samhd1_activity: float,
    target_ifn: float = 0.5,
    resolution: int = 100,
) -> float | None:
    """Find the minimum DNA exposure that produces >= target IFN output."""
    for i in range(resolution + 1):
        dna = i / resolution
        state = simulate_pathway(dna, trex1_activity, samhd1_activity)
        if state.ifn_output >= target_ifn:
            return dna
    return None


def plot_ifn_response_curves(
    results: dict,
    dest: Path,
    title: str = "cGAS-STING IFN Response by Genotype",
) -> None:
    """Plot IFN output vs GAS DNA exposure for each genotype scenario."""
    dna_levels = results["dna_exposure_levels"]
    scenarios = results["scenarios"]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        "wild_type": "#2ECC71",
        "trex1_het": "#F39C12",
        "samhd1_het": "#E67E22",
        "trex1_lof": "#E74C3C",
        "samhd1_lof": "#C0392B",
        "compound_het": "#8E44AD",
        "compound_lof": "#2C3E50",
    }

    for name, data in scenarios.items():
        ifn_values = [r["ifn_output"] for r in data["exposure_results"]]
        color = colors.get(name, "#95A5A6")
        label = f"{name} (TREX1={data['trex1_activity']}, SAMHD1={data['samhd1_activity']})"
        ax.plot(dna_levels, ifn_values, marker="o", color=color, label=label, linewidth=2)

    ax.set_xlabel("GAS DNA Exposure Level", fontsize=12)
    ax.set_ylabel("Type I IFN Output", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="50% IFN threshold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved IFN response curves: %s", dest)


def plot_pathway_heatmap(
    results: dict,
    dest: Path,
    dna_exposure: float = 0.5,
    title: str = "Pathway Node Activation at Moderate Infection",
) -> None:
    """Heatmap of pathway node activation across genotype scenarios."""
    scenarios = results["scenarios"]
    nodes = ["cytosolic_dna", "cgas_activity", "sting_activity",
             "tbk1_activity", "irf3_activity", "ifn_output"]
    node_labels = ["Cytosolic DNA", "cGAS", "STING", "TBK1", "IRF3", "IFN-β"]

    # Find the exposure result closest to dna_exposure
    dna_levels = results["dna_exposure_levels"]
    idx = min(range(len(dna_levels)), key=lambda i: abs(dna_levels[i] - dna_exposure))

    scenario_names = list(scenarios.keys())
    matrix = []
    for name in scenario_names:
        row = [scenarios[name]["exposure_results"][idx][n] for n in nodes]
        matrix.append(row)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(node_labels)))
    ax.set_xticklabels(node_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(scenario_names)))
    ax.set_yticklabels(scenario_names)

    for i in range(len(scenario_names)):
        for j in range(len(nodes)):
            val = matrix[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title(f"{title} (DNA={dna_levels[idx]:.1f})")
    plt.colorbar(im, label="Activation Level", shrink=0.8)
    plt.tight_layout()

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved pathway heatmap: %s", dest)


def run_cgas_sting_model(dest_dir: Path | None = None) -> dict:
    """Run the full cGAS-STING pathway activation model.

    Returns results dict with all simulation data.
    """
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Run simulations
    results = run_genotype_simulations()

    # Save results
    scores_path = dest_dir / "pathway_activation_scores.json"
    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved pathway activation scores: %s", scores_path)

    # Generate visualizations
    plot_ifn_response_curves(results, dest_dir / "ifn_response_curves.png")
    plot_pathway_heatmap(results, dest_dir / "pathway_heatmap.png")

    # Log key findings
    logger.info("=== Key findings ===")
    for name, data in results["scenarios"].items():
        threshold = data.get("dna_threshold_50pct_ifn")
        ifn_mod = data["ifn_at_moderate_infection"]
        threshold_str = f"{threshold:.2f}" if threshold is not None else ">1.0"
        logger.info("  %s: IFN@moderate=%.3f, 50%%_threshold=%s", name, ifn_mod, threshold_str)

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="cGAS-STING pathway activation model for PANS two-hit hypothesis"
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_cgas_sting_model(dest_dir=args.dest)

    print(f"\nSimulated {len(results['scenarios'])} genotype scenarios")
    print(f"DNA exposure levels: {results['dna_exposure_levels']}")
    print("\nIFN output at moderate infection (DNA=0.5):")
    for name, data in results["scenarios"].items():
        threshold = data.get("dna_threshold_50pct_ifn")
        t_str = f"{threshold:.2f}" if threshold is not None else ">1.0"
        print(f"  {name}: IFN={data['ifn_at_moderate_infection']:.3f}, "
              f"50% threshold={t_str}")


if __name__ == "__main__":
    main()
