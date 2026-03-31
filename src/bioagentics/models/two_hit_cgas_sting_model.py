"""cGAS-STING pathway activation model for the two-hit interferonopathy hypothesis.

Models the cGAS-STING innate immune signaling cascade under different genotype
scenarios (wild-type vs loss-of-function TREX1/SAMHD1 variants) and GAS DNA
exposure levels. Predicts type I interferon output for each combination.

Pathway logic:
  GAS infection → pathogen DNA release → cytosolic DNA
  TREX1 (exonuclease) degrades cytosolic DNA
  SAMHD1 (dNTPase) restricts dNTP pools, limiting DNA accumulation
  DDR failure (CUX1/USP45/PARP14) → unrepaired nuclear DNA → cytosolic leakage
  Mitochondrial impairment (PRKN/POLG) → mtDNA leakage → cytosolic DNA
  Cytosolic DNA → cGAS detection → 2'3'-cGAMP → STING → TBK1 → IRF3 → IFN-β

Loss-of-function variants in TREX1/SAMHD1 reduce DNA clearance, while DDR and
mitochondrial variants add additional cytosolic DNA sources, lowering the
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
    "CUX1": "Transcription factor; DDR regulation; haploinsufficiency",
    "USP45": "Deubiquitinase; base excision repair",
    "PARP14": "ADP-ribosyltransferase; immune-DDR crosstalk",
    "PRKN": "E3 ubiquitin ligase; mitophagy; mitochondrial DAMPs regulation",
    "POLG": "Mitochondrial DNA polymerase gamma; mtDNA maintenance",
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
    ddr_activity: float = 1.0  # 0.0 (null) to 1.0 (wild-type); CUX1/USP45/PARP14
    mito_activity: float = 1.0  # 0.0 (null) to 1.0 (wild-type); PRKN/POLG
    hif1a_activity: float = 1.0  # 0.0 (deficient) to 1.0 (normal); modulates STING threshold
    pep_level: float = 1.0  # 0.0 (depleted) to 1.0 (normal); age-dependent cGAS inhibitor
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
    GenotypeScenario(
        name="ddr_impaired",
        trex1_activity=1.0,
        samhd1_activity=1.0,
        ddr_activity=0.3,
        description="DDR impaired (CUX1/USP45/PARP14) — nuclear DNA leakage into cytosol",
    ),
    GenotypeScenario(
        name="mito_impaired",
        trex1_activity=1.0,
        samhd1_activity=1.0,
        mito_activity=0.3,
        description="Mitochondrial impaired (PRKN/POLG) — mtDNA leakage into cytosol",
    ),
    GenotypeScenario(
        name="triple_hit",
        trex1_activity=0.5,
        samhd1_activity=1.0,
        ddr_activity=0.3,
        mito_activity=0.3,
        description="Triple hit — TREX1 het + DDR + mito impaired (multi-axis vulnerability)",
    ),
    GenotypeScenario(
        name="metabolic_vulnerable",
        trex1_activity=0.5,
        samhd1_activity=1.0,
        hif1a_activity=0.4,
        pep_level=0.5,
        description="Metabolic vulnerable — TREX1 het + low HIF-1a + low PEP (onset-prone)",
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
    ddr_activity: float = 1.0,
    mito_activity: float = 1.0,
    hif1a_activity: float = 1.0,
    pep_level: float = 1.0,
    cgas_k: float = 0.3,
    sting_k: float = 0.4,
    tbk1_k: float = 0.4,
    irf3_k: float = 0.5,
) -> PathwayState:
    """Simulate steady-state cGAS-STING pathway activation.

    Model: GAS DNA → [TREX1/SAMHD1 clearance] → cytosolic DNA →
           cGAS → cGAMP → STING → TBK1 → IRF3 → IFN-β

    Additional cytosolic DNA sources:
      DDR failure (CUX1/USP45/PARP14) → unrepaired nuclear DNA leakage
      Mitochondrial impairment (PRKN/POLG) → mtDNA leakage

    Metabolic modifiers:
      HIF-1a deficiency → metabolic shift → lower STING activation threshold
      PEP depletion (age-dependent) → reduced cGAS inhibition → lower cGAS threshold

    Args:
        gas_dna: GAS pathogen DNA exposure level (0-1).
        trex1_activity: TREX1 enzymatic activity (0=null, 1=WT).
        samhd1_activity: SAMHD1 enzymatic activity (0=null, 1=WT).
        ddr_activity: DDR gene activity (0=null, 1=WT). CUX1/USP45/PARP14.
        mito_activity: Mitochondrial gene activity (0=null, 1=WT). PRKN/POLG.
        hif1a_activity: HIF-1a activity (0=deficient, 1=normal). Low → STING sensitized.
        pep_level: Phosphoenolpyruvate level (0=depleted, 1=normal). Low → cGAS sensitized.
        cgas_k: cGAS half-activation threshold (before PEP modulation).
        sting_k: STING half-activation threshold (before HIF-1a modulation).
        tbk1_k: TBK1 half-activation threshold.
        irf3_k: IRF3 half-activation threshold.

    Returns:
        PathwayState with all intermediate and output values.
    """
    state = PathwayState(gas_dna_input=gas_dna)

    # Metabolic modifier: PEP inhibits cGAS; low PEP lowers cGAS threshold
    # At pep=1.0: cgas_k unchanged; at pep=0.0: cgas_k drops to 30% of normal
    effective_cgas_k = cgas_k * (0.3 + 0.7 * pep_level)

    # Metabolic modifier: HIF-1a loss shifts glycolysis→OXPHOS, sensitizing STING
    # At hif1a=1.0: sting_k unchanged; at hif1a=0.0: sting_k drops to 30% of normal
    effective_sting_k = sting_k * (0.3 + 0.7 * hif1a_activity)

    # Cytosolic DNA accumulation depends on input and clearance
    # TREX1 degrades DNA directly; SAMHD1 restricts dNTP pools
    # Clearance = TREX1 * 0.7 + SAMHD1 * 0.3 (TREX1 is primary)
    clearance = trex1_activity * 0.7 + samhd1_activity * 0.3

    # DDR failure → unrepaired nuclear DNA → cytosolic leakage
    ddr_leakage = (1.0 - ddr_activity) * 0.3
    # Mitochondrial impairment → mtDNA leakage
    mito_leakage = (1.0 - mito_activity) * 0.25

    state.cytosolic_dna = gas_dna * (1.0 - clearance * 0.9) + ddr_leakage + mito_leakage
    state.cytosolic_dna = max(0.0, min(1.0, state.cytosolic_dna))

    # cGAS detects cytosolic dsDNA via sigmoidal activation
    state.cgas_activity = hill(state.cytosolic_dna, k=effective_cgas_k)

    # cGAS produces 2'3'-cGAMP (proportional to cGAS activity)
    state.cgamp_level = state.cgas_activity * 0.95

    # STING activation by cGAMP
    state.sting_activity = hill(state.cgamp_level, k=effective_sting_k)

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
                "TREX1/SAMHD1/DDR/mitochondrial genotype scenarios and GAS DNA exposure levels"
            ),
        },
        "dna_exposure_levels": dna_exposures,
        "scenarios": {},
    }

    for scenario in scenarios:
        logger.info("Simulating: %s (TREX1=%.1f, SAMHD1=%.1f, DDR=%.1f, MITO=%.1f, HIF1A=%.1f, PEP=%.1f)",
                     scenario.name, scenario.trex1_activity, scenario.samhd1_activity,
                     scenario.ddr_activity, scenario.mito_activity,
                     scenario.hif1a_activity, scenario.pep_level)

        scenario_data = {
            "description": scenario.description,
            "trex1_activity": scenario.trex1_activity,
            "samhd1_activity": scenario.samhd1_activity,
            "ddr_activity": scenario.ddr_activity,
            "mito_activity": scenario.mito_activity,
            "hif1a_activity": scenario.hif1a_activity,
            "pep_level": scenario.pep_level,
            "exposure_results": [],
        }

        for dna in dna_exposures:
            state = simulate_pathway(
                gas_dna=dna,
                trex1_activity=scenario.trex1_activity,
                samhd1_activity=scenario.samhd1_activity,
                ddr_activity=scenario.ddr_activity,
                mito_activity=scenario.mito_activity,
                hif1a_activity=scenario.hif1a_activity,
                pep_level=scenario.pep_level,
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
            ddr_activity=scenario.ddr_activity,
            mito_activity=scenario.mito_activity,
            hif1a_activity=scenario.hif1a_activity,
            pep_level=scenario.pep_level,
        )
        scenario_data["ifn_at_moderate_infection"] = round(moderate_state.ifn_output, 4)

        # Summary: threshold DNA exposure to trigger >50% IFN
        threshold = _find_ifn_threshold(
            scenario.trex1_activity, scenario.samhd1_activity,
            scenario.ddr_activity, scenario.mito_activity,
            scenario.hif1a_activity, scenario.pep_level,
        )
        scenario_data["dna_threshold_50pct_ifn"] = round(threshold, 4) if threshold is not None else None

        results["scenarios"][scenario.name] = scenario_data

    return results


def _find_ifn_threshold(
    trex1_activity: float,
    samhd1_activity: float,
    ddr_activity: float = 1.0,
    mito_activity: float = 1.0,
    hif1a_activity: float = 1.0,
    pep_level: float = 1.0,
    target_ifn: float = 0.5,
    resolution: int = 100,
) -> float | None:
    """Find the minimum DNA exposure that produces >= target IFN output."""
    for i in range(resolution + 1):
        dna = i / resolution
        state = simulate_pathway(
            dna, trex1_activity, samhd1_activity,
            ddr_activity=ddr_activity, mito_activity=mito_activity,
            hif1a_activity=hif1a_activity, pep_level=pep_level,
        )
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
        "ddr_impaired": "#3498DB",
        "mito_impaired": "#1ABC9C",
        "triple_hit": "#E91E63",
        "metabolic_vulnerable": "#FF6F00",
    }

    for name, data in scenarios.items():
        ifn_values = [r["ifn_output"] for r in data["exposure_results"]]
        color = colors.get(name, "#95A5A6")
        parts = [f"TREX1={data['trex1_activity']}", f"SAMHD1={data['samhd1_activity']}"]
        if data.get("ddr_activity", 1.0) != 1.0:
            parts.append(f"DDR={data['ddr_activity']}")
        if data.get("mito_activity", 1.0) != 1.0:
            parts.append(f"MITO={data['mito_activity']}")
        if data.get("hif1a_activity", 1.0) != 1.0:
            parts.append(f"HIF1A={data['hif1a_activity']}")
        if data.get("pep_level", 1.0) != 1.0:
            parts.append(f"PEP={data['pep_level']}")
        label = f"{name} ({', '.join(parts)})"
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


def run_pep_age_sweep(
    base_scenario: GenotypeScenario | None = None,
    pep_levels: list[float] | None = None,
    gas_dna: float = 0.5,
) -> dict:
    """Sweep PEP levels to model age-dependent onset window.

    PEP (phosphoenolpyruvate) directly inhibits cGAS. Age-related PEP decline
    explains the PANDAS developmental timing window (typically ages 3-12).

    Returns dict with PEP sweep results showing IFN threshold shift.
    """
    if base_scenario is None:
        base_scenario = GenotypeScenario(
            name="trex1_het_pep_sweep",
            trex1_activity=0.5,
            samhd1_activity=1.0,
            description="TREX1 het with PEP age sweep",
        )
    if pep_levels is None:
        pep_levels = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]

    sweep_results = []
    for pep in pep_levels:
        state = simulate_pathway(
            gas_dna=gas_dna,
            trex1_activity=base_scenario.trex1_activity,
            samhd1_activity=base_scenario.samhd1_activity,
            ddr_activity=base_scenario.ddr_activity,
            mito_activity=base_scenario.mito_activity,
            hif1a_activity=base_scenario.hif1a_activity,
            pep_level=pep,
        )
        threshold = _find_ifn_threshold(
            base_scenario.trex1_activity, base_scenario.samhd1_activity,
            base_scenario.ddr_activity, base_scenario.mito_activity,
            base_scenario.hif1a_activity, pep,
        )
        sweep_results.append({
            "pep_level": round(pep, 2),
            "ifn_output": round(state.ifn_output, 4),
            "dna_threshold_50pct_ifn": round(threshold, 4) if threshold is not None else None,
        })

    return {
        "description": "Age-dependent PEP sweep: PEP decline lowers cGAS threshold, modeling PANDAS onset window",
        "base_genotype": base_scenario.name,
        "gas_dna": gas_dna,
        "sweep": sweep_results,
    }


def plot_pep_age_sweep(sweep_data: dict, dest: Path) -> None:
    """Plot IFN output and threshold vs PEP level."""
    pep_levels = [r["pep_level"] for r in sweep_data["sweep"]]
    ifn_values = [r["ifn_output"] for r in sweep_data["sweep"]]
    thresholds = [r["dna_threshold_50pct_ifn"] for r in sweep_data["sweep"]]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(pep_levels, ifn_values, "o-", color="#E74C3C", linewidth=2, label="IFN output (DNA=0.5)")
    ax1.set_xlabel("PEP Level (1.0=young, 0.3=age-depleted)", fontsize=11)
    ax1.set_ylabel("IFN Output", fontsize=11, color="#E74C3C")
    ax1.tick_params(axis="y", labelcolor="#E74C3C")
    ax1.set_xlim(1.05, 0.25)

    ax2 = ax1.twinx()
    t_vals = [t if t is not None else 0.0 for t in thresholds]
    ax2.plot(pep_levels, t_vals, "s--", color="#3498DB", linewidth=2, label="50% IFN threshold")
    ax2.set_ylabel("DNA Threshold for 50% IFN", fontsize=11, color="#3498DB")
    ax2.tick_params(axis="y", labelcolor="#3498DB")

    ax1.set_title("Age-Dependent PEP Decline → PANDAS Onset Window", fontsize=13)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax1.grid(alpha=0.3)
    plt.tight_layout()

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved PEP age sweep plot: %s", dest)


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

    # Run age-dependent PEP sweep
    pep_sweep = run_pep_age_sweep()
    pep_path = dest_dir / "pep_age_sweep.json"
    with open(pep_path, "w") as f:
        json.dump(pep_sweep, f, indent=2)
    logger.info("Saved PEP age sweep: %s", pep_path)

    # Generate visualizations
    plot_ifn_response_curves(results, dest_dir / "ifn_response_curves.png")
    plot_pathway_heatmap(results, dest_dir / "pathway_heatmap.png")
    plot_pep_age_sweep(pep_sweep, dest_dir / "pep_age_sweep.png")

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
