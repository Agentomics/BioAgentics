"""Therapeutic model for cGAS-STING inhibitor prediction in PANS.

Models the effect of cGAS-STING pathway inhibitors on type I IFN output
across different genotype scenarios. Predicts which patients (by genotype
compound hit profile) would benefit most from:
  - STING inhibition (H-151)
  - cGAS inhibition (RU.521)
  - Complement replacement (for lectin complement deficiency)
  - Combined therapy

Uses the cGAS-STING pathway model from two_hit_cgas_sting_model.py as
the simulation engine.

Usage:
    uv run python -m bioagentics.models.two_hit_therapeutic_model [--dest DIR]
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

from bioagentics.models.two_hit_cgas_sting_model import simulate_pathway

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/pandas_pans/two-hit-interferonopathy-model")


@dataclass
class Inhibitor:
    """A cGAS-STING pathway inhibitor."""
    name: str
    target: str  # pathway node targeted
    efficacy: float  # fraction of target activity blocked (0-1)
    description: str = ""
    clinical_stage: str = ""


# Known cGAS-STING inhibitors
INHIBITORS = [
    Inhibitor(
        name="H-151",
        target="sting",
        efficacy=0.85,
        description="Small-molecule STING antagonist; palmitoylation inhibitor",
        clinical_stage="Preclinical (Haag et al. 2018, Nature)",
    ),
    Inhibitor(
        name="RU.521",
        target="cgas",
        efficacy=0.80,
        description="cGAS catalytic site inhibitor; blocks cGAMP synthesis",
        clinical_stage="Preclinical tool compound",
    ),
    Inhibitor(
        name="C-178",
        target="sting",
        efficacy=0.70,
        description="Covalent STING inhibitor; targets Cys91",
        clinical_stage="Preclinical",
    ),
]

# Compound genotype profiles for therapeutic prediction
COMPOUND_PROFILES = [
    {
        "name": "lectin_only",
        "description": "Lectin complement deficiency only (MBL2/MASP variants)",
        "trex1_activity": 1.0,
        "samhd1_activity": 1.0,
        "lectin_complement_deficient": True,
        "recommended_therapy": "complement_replacement",
    },
    {
        "name": "cgas_sting_only",
        "description": "cGAS-STING dysregulation only (TREX1/SAMHD1 variants)",
        "trex1_activity": 0.3,
        "samhd1_activity": 0.5,
        "lectin_complement_deficient": False,
        "recommended_therapy": "sting_inhibition",
    },
    {
        "name": "two_hit_mild",
        "description": "Two-hit: mild lectin + mild cGAS-STING (heterozygous carriers)",
        "trex1_activity": 0.5,
        "samhd1_activity": 0.7,
        "lectin_complement_deficient": True,
        "recommended_therapy": "combined",
    },
    {
        "name": "two_hit_severe",
        "description": "Two-hit: lectin complement + severe cGAS-STING",
        "trex1_activity": 0.1,
        "samhd1_activity": 0.3,
        "lectin_complement_deficient": True,
        "recommended_therapy": "combined_aggressive",
    },
]


def simulate_inhibitor_effect(
    trex1_activity: float,
    samhd1_activity: float,
    gas_dna: float,
    inhibitor: Inhibitor,
) -> dict:
    """Simulate pathway activation with and without an inhibitor.

    For STING inhibitors: reduce STING activation by efficacy fraction.
    For cGAS inhibitors: reduce cGAS → cGAMP production by efficacy fraction.

    Returns:
        Dict with baseline and inhibited IFN output plus reduction metrics.
    """
    # Baseline (no inhibitor)
    baseline = simulate_pathway(gas_dna, trex1_activity, samhd1_activity)

    # With inhibitor — modify relevant kinetic parameters
    if inhibitor.target == "sting":
        # STING inhibitor: increase the STING half-activation threshold
        # Higher k = harder to activate = less STING signaling
        inhibited_sting_k = 0.4 + (inhibitor.efficacy * 2.0)
        inhibited = simulate_pathway(
            gas_dna, trex1_activity, samhd1_activity,
            sting_k=inhibited_sting_k,
        )
    elif inhibitor.target == "cgas":
        # cGAS inhibitor: increase cGAS half-activation threshold
        inhibited_cgas_k = 0.3 + (inhibitor.efficacy * 2.0)
        inhibited = simulate_pathway(
            gas_dna, trex1_activity, samhd1_activity,
            cgas_k=inhibited_cgas_k,
        )
    else:
        inhibited = baseline

    baseline_ifn = baseline.ifn_output
    inhibited_ifn = inhibited.ifn_output
    reduction = baseline_ifn - inhibited_ifn
    pct_reduction = (reduction / baseline_ifn * 100) if baseline_ifn > 0.001 else 0.0

    return {
        "inhibitor": inhibitor.name,
        "target": inhibitor.target,
        "baseline_ifn": round(baseline_ifn, 4),
        "inhibited_ifn": round(inhibited_ifn, 4),
        "ifn_reduction": round(reduction, 4),
        "pct_reduction": round(pct_reduction, 1),
    }


def predict_therapy(
    profile: dict,
    inhibitors: list[Inhibitor] | None = None,
    gas_dna: float = 0.5,
) -> dict:
    """Predict optimal therapy for a compound genotype profile.

    Tests each inhibitor and determines which provides the greatest
    IFN reduction for this specific genotype combination.

    Returns:
        Dict with profile info, inhibitor results, and therapy recommendation.
    """
    if inhibitors is None:
        inhibitors = INHIBITORS

    trex1 = profile["trex1_activity"]
    samhd1 = profile["samhd1_activity"]
    lectin_def = profile.get("lectin_complement_deficient", False)

    # Test each inhibitor
    inhibitor_results = []
    for inh in inhibitors:
        result = simulate_inhibitor_effect(trex1, samhd1, gas_dna, inh)
        inhibitor_results.append(result)

    # Find best single inhibitor
    best_inh = max(inhibitor_results, key=lambda r: r["ifn_reduction"])

    # Determine therapy recommendation
    baseline = simulate_pathway(gas_dna, trex1, samhd1)
    needs_ifn_control = baseline.ifn_output > 0.3

    if lectin_def and needs_ifn_control:
        therapy = "combined: complement replacement + " + best_inh["inhibitor"]
        rationale = (
            f"Dual targeting: lectin complement replacement for pathogen clearance, "
            f"{best_inh['inhibitor']} ({best_inh['target']} inhibitor) for IFN control "
            f"({best_inh['pct_reduction']:.0f}% reduction)"
        )
    elif lectin_def and not needs_ifn_control:
        therapy = "complement_replacement"
        rationale = "Lectin complement deficiency primary driver; IFN pathway near normal"
    elif needs_ifn_control:
        therapy = best_inh["inhibitor"]
        rationale = (
            f"{best_inh['inhibitor']} ({best_inh['target']} inhibitor) reduces IFN by "
            f"{best_inh['pct_reduction']:.0f}%"
        )
    else:
        therapy = "monitoring"
        rationale = "Pathway activation within normal range at moderate infection"

    return {
        "profile": profile["name"],
        "description": profile["description"],
        "genotype": {
            "trex1_activity": trex1,
            "samhd1_activity": samhd1,
            "lectin_complement_deficient": lectin_def,
        },
        "baseline_ifn": round(baseline.ifn_output, 4),
        "inhibitor_results": inhibitor_results,
        "best_single_inhibitor": best_inh["inhibitor"],
        "recommended_therapy": therapy,
        "rationale": rationale,
    }


def run_therapeutic_predictions(
    profiles: list[dict] | None = None,
    inhibitors: list[Inhibitor] | None = None,
    dest_dir: Path | None = None,
) -> dict:
    """Run therapeutic predictions for all compound genotype profiles.

    Returns:
        Dict with all predictions and inhibitor metadata.
    """
    if profiles is None:
        profiles = COMPOUND_PROFILES
    if inhibitors is None:
        inhibitors = INHIBITORS
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "metadata": {
            "description": (
                "Therapeutic predictions for PANS patients based on compound "
                "genotype profiles. Models cGAS-STING inhibitor effects on "
                "type I IFN output."
            ),
            "inhibitors": [
                {
                    "name": inh.name,
                    "target": inh.target,
                    "efficacy": inh.efficacy,
                    "description": inh.description,
                    "clinical_stage": inh.clinical_stage,
                }
                for inh in inhibitors
            ],
        },
        "predictions": [],
    }

    for profile in profiles:
        logger.info("Predicting therapy for: %s", profile["name"])
        prediction = predict_therapy(profile, inhibitors)
        results["predictions"].append(prediction)
        logger.info("  → %s (baseline IFN=%.3f)",
                     prediction["recommended_therapy"],
                     prediction["baseline_ifn"])

    # Save results
    out_path = dest_dir / "therapeutic_predictions.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved therapeutic predictions: %s", out_path)

    # Generate visualization
    plot_therapeutic_comparison(results, dest_dir / "therapeutic_comparison.png")

    return results


def plot_therapeutic_comparison(
    results: dict,
    dest: Path,
    title: str = "cGAS-STING Inhibitor Effects by Genotype Profile",
) -> None:
    """Bar chart comparing inhibitor effects across genotype profiles."""
    predictions = results["predictions"]
    profiles = [p["profile"] for p in predictions]
    inhibitor_names = [inh["name"] for inh in results["metadata"]["inhibitors"]]

    n_profiles = len(profiles)
    n_inhibitors = len(inhibitor_names)
    x = np.arange(n_profiles)
    width = 0.8 / max(n_inhibitors, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#E74C3C", "#3498DB", "#F39C12", "#2ECC71"]

    # Left: IFN reduction by inhibitor
    ax1 = axes[0]
    for i, inh_name in enumerate(inhibitor_names):
        reductions = []
        for pred in predictions:
            inh_result = next(
                (r for r in pred["inhibitor_results"] if r["inhibitor"] == inh_name),
                None,
            )
            reductions.append(inh_result["pct_reduction"] if inh_result else 0)
        ax1.bar(x + i * width - width * (n_inhibitors - 1) / 2,
                reductions, width, label=inh_name,
                color=colors[i % len(colors)], alpha=0.8)

    ax1.set_xlabel("Genotype Profile")
    ax1.set_ylabel("IFN Reduction (%)")
    ax1.set_title("Inhibitor Efficacy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(profiles, rotation=30, ha="right")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Right: Baseline vs best-inhibited IFN
    ax2 = axes[1]
    baseline_ifn = [p["baseline_ifn"] for p in predictions]
    best_inhibited = []
    for pred in predictions:
        best_inh = pred["best_single_inhibitor"]
        best_result = next(
            r for r in pred["inhibitor_results"] if r["inhibitor"] == best_inh
        )
        best_inhibited.append(best_result["inhibited_ifn"])

    ax2.bar(x - 0.2, baseline_ifn, 0.35, label="Baseline", color="#E74C3C", alpha=0.8)
    ax2.bar(x + 0.2, best_inhibited, 0.35, label="Best Inhibitor", color="#2ECC71", alpha=0.8)
    ax2.axhline(y=0.3, color="gray", linestyle="--", alpha=0.5, label="Treatment threshold")
    ax2.set_xlabel("Genotype Profile")
    ax2.set_ylabel("IFN Output")
    ax2.set_title("IFN: Baseline vs Treated")
    ax2.set_xticks(x)
    ax2.set_xticklabels(profiles, rotation=30, ha="right")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved therapeutic comparison: %s", dest)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Therapeutic model for cGAS-STING inhibitor prediction in PANS"
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_therapeutic_predictions(dest_dir=args.dest)

    print(f"\nTherapeutic predictions for {len(results['predictions'])} profiles:")
    for pred in results["predictions"]:
        print(f"\n  {pred['profile']}: {pred['description']}")
        print(f"    Baseline IFN: {pred['baseline_ifn']:.3f}")
        print(f"    Recommended: {pred['recommended_therapy']}")
        print(f"    Rationale: {pred['rationale']}")


if __name__ == "__main__":
    main()
