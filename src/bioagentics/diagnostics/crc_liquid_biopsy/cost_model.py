"""Cost-effectiveness model for the CRC liquid biopsy panel.

Estimates per-test cost for the optimized panel and compares cost-effectiveness
against colonoscopy, Cologuard, GRAIL Galleri, and CEA alone using a QALY
framework for population screening (adults 45-75).

Output:
    output/diagnostics/crc-liquid-biopsy-panel/cost_model_results.json

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.cost_model [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "crc-liquid-biopsy-panel"


# ---------------------------------------------------------------------------
# Cost assumptions (USD)
# ---------------------------------------------------------------------------

# Per-marker assay costs
METHYLATION_BISULFITE_PCR = 75.0  # per target CpG
PROTEIN_IMMUNOASSAY = 15.0  # per target protein (ELISA/Luminex)
SAMPLE_COLLECTION = 20.0  # blood draw
LAB_OVERHEAD = 30.0  # processing, reporting, logistics

# Comparator test costs
COMPARATORS = {
    "Colonoscopy": {
        "cost": 2500.0,  # median of $2,000-$3,000 range
        "sensitivity": 0.95,  # for polyps >= 6mm
        "specificity": 0.99,
        "compliance": 0.60,  # adherence rate
        "frequency_years": 10,
        "description": "Optical colonoscopy with polypectomy",
    },
    "Cologuard (FIT-DNA)": {
        "cost": 650.0,
        "sensitivity": 0.92,  # for CRC
        "specificity": 0.87,
        "compliance": 0.72,
        "frequency_years": 3,
        "description": "Multi-target stool DNA test",
    },
    "GRAIL Galleri": {
        "cost": 949.0,
        "sensitivity": 0.67,  # for CRC (multi-cancer sensitivity varies)
        "specificity": 0.995,  # <0.5% FPR
        "compliance": 0.80,  # estimated (blood-based convenience)
        "frequency_years": 1,
        "description": "Multi-cancer early detection cfDNA methylation test",
    },
    "CEA alone": {
        "cost": 25.0,
        "sensitivity": 0.40,  # poor for early-stage CRC
        "specificity": 0.90,
        "compliance": 0.85,
        "frequency_years": 1,
        "description": "Carcinoembryonic antigen blood test",
    },
}

# QALY model parameters
CRC_INCIDENCE_45_75 = 0.005  # ~5/1000 annual incidence in screening-age pop
CRC_EARLY_STAGE_5YR_SURVIVAL = 0.91  # Stage I-II 5-year survival
CRC_LATE_STAGE_5YR_SURVIVAL = 0.15  # Stage IV 5-year survival
EARLY_DETECT_FRACTION_IF_SCREENED = 0.60  # fraction caught at stage I-II
LATE_DETECT_FRACTION_IF_NOT_SCREENED = 0.75  # fraction caught at stage III-IV
LIFE_EXPECTANCY_YEARS = 20.0  # average remaining life at ~60yr if cancer-free
QALY_DISCOUNT_RATE = 0.03  # standard 3% annual discount


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

def estimate_panel_cost(panel_config: dict) -> dict:
    """Estimate per-test cost for the optimized panel."""
    features = panel_config["optimal_panel"]["features"]
    n_meth = sum(1 for f in features if f.startswith("meth_"))
    n_prot = sum(1 for f in features if f.startswith("prot_"))

    marker_cost = n_meth * METHYLATION_BISULFITE_PCR + n_prot * PROTEIN_IMMUNOASSAY
    total_cost = marker_cost + SAMPLE_COLLECTION + LAB_OVERHEAD

    return {
        "n_methylation_markers": n_meth,
        "n_protein_markers": n_prot,
        "methylation_assay_cost": n_meth * METHYLATION_BISULFITE_PCR,
        "protein_assay_cost": n_prot * PROTEIN_IMMUNOASSAY,
        "sample_collection": SAMPLE_COLLECTION,
        "lab_overhead": LAB_OVERHEAD,
        "total_per_test_usd": total_cost,
        "meets_target": total_cost < 200.0,
    }


# ---------------------------------------------------------------------------
# QALY framework
# ---------------------------------------------------------------------------

def _discount(value: float, years: float, rate: float = QALY_DISCOUNT_RATE) -> float:
    """Discount a future value to present at given rate."""
    return value / ((1 + rate) ** years)


def compute_qalys_gained(
    sensitivity: float,
    specificity: float,
    compliance: float,
    cost_per_test: float,
    frequency_years: float,
    population: int = 100_000,
    time_horizon: int = 10,
) -> dict:
    """Estimate QALYs gained per $1000 spent over time horizon.

    Models a screening program for adults 45-75 with given test parameters.
    Compares to no-screening baseline.
    """
    total_cost = 0.0
    total_qalys_gained = 0.0

    for year in range(time_horizon):
        # Screen if this year aligns with frequency
        if year % frequency_years != 0:
            continue

        screened = population * compliance
        new_cancers = screened * CRC_INCIDENCE_45_75

        # True positives detected early
        tp = new_cancers * sensitivity * EARLY_DETECT_FRACTION_IF_SCREENED
        # False negatives (missed, detected late)
        fn = new_cancers * (1 - sensitivity) * LATE_DETECT_FRACTION_IF_NOT_SCREENED
        # False positives (unnecessary follow-up colonoscopy)
        fp = (screened - new_cancers) * (1 - specificity)

        # QALYs gained from early detection vs late detection
        # Early detection: better survival -> more QALYs
        qalys_early = tp * CRC_EARLY_STAGE_5YR_SURVIVAL * _discount(LIFE_EXPECTANCY_YEARS * 0.8, year)
        qalys_late = fn * CRC_LATE_STAGE_5YR_SURVIVAL * _discount(LIFE_EXPECTANCY_YEARS * 0.3, year)

        # Without screening, all would be detected late
        no_screen_cancers = new_cancers
        qalys_no_screen = (
            no_screen_cancers * LATE_DETECT_FRACTION_IF_NOT_SCREENED
            * CRC_LATE_STAGE_5YR_SURVIVAL * _discount(LIFE_EXPECTANCY_YEARS * 0.3, year)
        )

        qalys_gained = (qalys_early + qalys_late) - qalys_no_screen
        total_qalys_gained += qalys_gained

        # Costs: screening + follow-up colonoscopies for positives
        screen_cost = screened * cost_per_test
        followup_cost = (tp + fp) * 2500.0  # colonoscopy for all positives
        year_cost = _discount(screen_cost + followup_cost, year)
        total_cost += year_cost

    qalys_per_1000 = (total_qalys_gained / total_cost * 1000) if total_cost > 0 else 0.0
    icer = (total_cost / total_qalys_gained) if total_qalys_gained > 0 else float("inf")

    return {
        "total_cost_usd": round(total_cost, 2),
        "total_qalys_gained": round(total_qalys_gained, 2),
        "qalys_per_1000_usd": round(qalys_per_1000, 4),
        "icer_usd_per_qaly": round(icer, 2),
        "cost_effective": icer < 50_000,  # standard WTP threshold
        "population": population,
        "time_horizon_years": time_horizon,
    }


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    base_cost: float,
    base_sensitivity: float,
    base_specificity: float,
) -> list[dict]:
    """Run one-way sensitivity analysis on key cost assumptions."""
    results = []

    # Vary marker cost (+/- 50%)
    for cost_mult in [0.5, 1.0, 1.5, 2.0]:
        qaly = compute_qalys_gained(
            sensitivity=base_sensitivity,
            specificity=base_specificity,
            compliance=0.80,
            cost_per_test=base_cost * cost_mult,
            frequency_years=1,
        )
        results.append({
            "scenario": f"Test cost: ${base_cost * cost_mult:.0f}",
            "variable": "test_cost",
            "multiplier": cost_mult,
            "icer": qaly["icer_usd_per_qaly"],
            "qalys_per_1000": qaly["qalys_per_1000_usd"],
            "cost_effective": qaly["cost_effective"],
        })

    # Vary sensitivity
    for sens in [0.50, 0.65, 0.80, 0.90]:
        qaly = compute_qalys_gained(
            sensitivity=sens,
            specificity=base_specificity,
            compliance=0.80,
            cost_per_test=base_cost,
            frequency_years=1,
        )
        results.append({
            "scenario": f"Sensitivity: {sens:.0%}",
            "variable": "sensitivity",
            "multiplier": sens,
            "icer": qaly["icer_usd_per_qaly"],
            "qalys_per_1000": qaly["qalys_per_1000_usd"],
            "cost_effective": qaly["cost_effective"],
        })

    # Vary compliance
    for comp in [0.50, 0.70, 0.85, 0.95]:
        qaly = compute_qalys_gained(
            sensitivity=base_sensitivity,
            specificity=base_specificity,
            compliance=comp,
            cost_per_test=base_cost,
            frequency_years=1,
        )
        results.append({
            "scenario": f"Compliance: {comp:.0%}",
            "variable": "compliance",
            "multiplier": comp,
            "icer": qaly["icer_usd_per_qaly"],
            "qalys_per_1000": qaly["qalys_per_1000_usd"],
            "cost_effective": qaly["cost_effective"],
        })

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_cost_model(
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict:
    """Run cost-effectiveness model."""
    output_path = output_dir / "cost_model_results.json"
    if output_path.exists() and not force:
        logger.info("Loading cached cost model from %s", output_path)
        with open(output_path) as f:
            return json.load(f)

    # Load panel config
    panel_path = output_dir / "optimized_panel.json"
    with open(panel_path) as f:
        panel_config = json.load(f)

    # Load classifier results for performance metrics
    classifier_path = output_dir / "classifier_results.json"
    with open(classifier_path) as f:
        classifier_results = json.load(f)

    # 1. Panel cost estimate
    cost_estimate = estimate_panel_cost(panel_config)
    panel_cost = cost_estimate["total_per_test_usd"]
    panel_sensitivity = classifier_results["mean_sensitivity_at_95spec"]
    panel_specificity = 0.95  # by definition (sensitivity at 95% spec)

    logger.info("Panel cost: $%.0f (target < $200): %s",
                panel_cost, "PASS" if cost_estimate["meets_target"] else "FAIL")

    # 2. Cost-effectiveness comparison
    comparisons = {}
    for name, params in COMPARATORS.items():
        qaly = compute_qalys_gained(
            sensitivity=params["sensitivity"],
            specificity=params["specificity"],
            compliance=params["compliance"],
            cost_per_test=params["cost"],
            frequency_years=params["frequency_years"],
        )
        comparisons[name] = {
            **params,
            **qaly,
        }

    # Our panel
    our_qaly = compute_qalys_gained(
        sensitivity=panel_sensitivity,
        specificity=panel_specificity,
        compliance=0.80,  # estimated: blood-based = high compliance
        cost_per_test=panel_cost,
        frequency_years=1,  # annual screening
    )
    comparisons["CRC Liquid Biopsy Panel (ours)"] = {
        "cost": panel_cost,
        "sensitivity": panel_sensitivity,
        "specificity": panel_specificity,
        "compliance": 0.80,
        "frequency_years": 1,
        "description": f"Multi-analyte panel ({cost_estimate['n_protein_markers']}P + {cost_estimate['n_methylation_markers']}M)",
        **our_qaly,
    }

    # 3. Sensitivity analysis
    sa_results = sensitivity_analysis(panel_cost, panel_sensitivity, panel_specificity)

    result = {
        "cost_estimate": cost_estimate,
        "panel_performance": {
            "sensitivity_at_95spec": panel_sensitivity,
            "specificity": panel_specificity,
            "mean_auc": classifier_results["mean_auc"],
        },
        "comparisons": comparisons,
        "sensitivity_analysis": sa_results,
        "assumptions": {
            "methylation_cost_per_marker": METHYLATION_BISULFITE_PCR,
            "protein_cost_per_marker": PROTEIN_IMMUNOASSAY,
            "sample_collection": SAMPLE_COLLECTION,
            "lab_overhead": LAB_OVERHEAD,
            "crc_incidence_45_75": CRC_INCIDENCE_45_75,
            "early_stage_5yr_survival": CRC_EARLY_STAGE_5YR_SURVIVAL,
            "late_stage_5yr_survival": CRC_LATE_STAGE_5YR_SURVIVAL,
            "qaly_discount_rate": QALY_DISCOUNT_RATE,
            "wtp_threshold_usd": 50_000,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved cost model results to %s", output_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Cost-effectiveness model")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_cost_model(args.output_dir, force=args.force)

    ce = result["cost_estimate"]
    print(f"\n=== Panel Cost Estimate ===")
    print(f"Markers: {ce['n_protein_markers']} protein + {ce['n_methylation_markers']} methylation")
    print(f"Assay cost: ${ce['protein_assay_cost'] + ce['methylation_assay_cost']:.0f}")
    print(f"Total per-test: ${ce['total_per_test_usd']:.0f}")
    print(f"Target < $200: {'PASS' if ce['meets_target'] else 'FAIL'}")

    print(f"\n=== Cost-Effectiveness Comparison ===")
    print(f"{'Test':<35} {'Cost':>8} {'Sens':>7} {'ICER':>12} {'CE?':>5}")
    print("-" * 70)
    for name, c in result["comparisons"].items():
        icer_str = f"${c['icer_usd_per_qaly']:,.0f}" if c["icer_usd_per_qaly"] < 1e9 else "N/A"
        print(f"{name:<35} ${c['cost']:>6.0f} {c['sensitivity']:>6.1%} {icer_str:>12} {'Yes' if c['cost_effective'] else 'No':>5}")

    print(f"\n=== Sensitivity Analysis ===")
    for sa in result["sensitivity_analysis"]:
        print(f"  {sa['scenario']:<30} ICER: ${sa['icer']:,.0f}  CE: {'Yes' if sa['cost_effective'] else 'No'}")


if __name__ == "__main__":
    main()
