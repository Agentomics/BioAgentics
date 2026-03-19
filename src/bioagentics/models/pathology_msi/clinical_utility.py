"""Operating point analysis and cost-effectiveness modeling for MSI pre-screening.

Analyzes sensitivity/specificity trade-offs, identifies optimal operating
points, and models cost-effectiveness of H&E-based pre-screening versus
universal molecular testing.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_curve

logger = logging.getLogger(__name__)


@dataclass
class OperatingPoint:
    """A classification operating point on the ROC curve."""

    threshold: float
    sensitivity: float
    specificity: float
    fpr: float
    ppv: float  # positive predictive value
    npv: float  # negative predictive value
    n_molecular_tests_per_1000: float  # expected tests needed per 1000 patients


@dataclass
class CostComparison:
    """Cost comparison between screening strategies."""

    prevalence: float
    universal_cost_per_1000: float
    prescreening_cost_per_1000: float
    savings_per_1000: float
    savings_pct: float
    missed_cases_per_1000: float
    unnecessary_molecular_per_1000: float


def find_operating_points(
    probs: np.ndarray,
    labels: np.ndarray,
    prevalence: float | None = None,
) -> list[OperatingPoint]:
    """Find operating points along the ROC curve.

    Args:
        probs: Predicted probabilities for positive class, shape (N,).
        labels: True binary labels, shape (N,).
        prevalence: Disease prevalence for PPV/NPV calculation.
            If None, uses the sample prevalence.

    Returns:
        List of OperatingPoint objects at each unique threshold,
        sorted by threshold descending.
    """
    prev = float(labels.mean()) if prevalence is None else prevalence

    fpr_arr, tpr_arr, thresholds = roc_curve(labels, probs)

    points = []
    for i, thresh in enumerate(thresholds):
        sens = tpr_arr[i]
        spec = 1.0 - fpr_arr[i]

        # PPV and NPV using Bayes' theorem
        ppv = _ppv(sens, spec, prev)
        npv = _npv(sens, spec, prev)

        # Expected molecular tests per 1000 patients
        # Screen positives (TP + FP) need molecular confirmation
        n_pos_per_1000 = sens * prev * 1000 + fpr_arr[i] * (1 - prev) * 1000

        points.append(OperatingPoint(
            threshold=float(thresh),
            sensitivity=float(sens),
            specificity=float(spec),
            fpr=float(fpr_arr[i]),
            ppv=float(ppv),
            npv=float(npv),
            n_molecular_tests_per_1000=float(n_pos_per_1000),
        ))

    # Sort by threshold descending (high specificity first)
    points.sort(key=lambda p: p.threshold, reverse=True)
    return points


def find_optimal_point(
    points: list[OperatingPoint],
    min_sensitivity: float = 0.95,
) -> OperatingPoint | None:
    """Find the optimal operating point that minimizes molecular testing
    while maintaining at least min_sensitivity.

    Args:
        points: List of operating points.
        min_sensitivity: Minimum required sensitivity.

    Returns:
        The operating point that minimizes molecular tests while meeting
        the sensitivity constraint, or None if no point meets it.
    """
    eligible = [p for p in points if p.sensitivity >= min_sensitivity]
    if not eligible:
        logger.warning(
            f"No operating point achieves sensitivity >= {min_sensitivity}"
        )
        return None

    # Among eligible points, pick the one with fewest molecular tests
    return min(eligible, key=lambda p: p.n_molecular_tests_per_1000)


def cost_effectiveness_analysis(
    operating_point: OperatingPoint,
    prevalence_rates: list[float] | None = None,
    cost_molecular: float = 400.0,
    cost_prescreening: float = 10.0,
) -> list[CostComparison]:
    """Model cost-effectiveness of H&E pre-screening versus universal molecular testing.

    Args:
        operating_point: The chosen operating point (threshold, sensitivity, specificity).
        prevalence_rates: List of MSI prevalence rates to model.
            Defaults to [0.05, 0.10, 0.15, 0.20, 0.25, 0.30].
        cost_molecular: Cost per molecular MSI test in USD.
        cost_prescreening: Cost per H&E-based AI pre-screening in USD.

    Returns:
        List of CostComparison objects, one per prevalence rate.
    """
    if prevalence_rates is None:
        prevalence_rates = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    sens = operating_point.sensitivity
    spec = operating_point.specificity

    results = []
    for prev in prevalence_rates:
        # Universal strategy: test everyone
        universal_cost = cost_molecular * 1000

        # Pre-screening strategy: screen all with AI, confirm positives
        # Screen positives = TP + FP
        tp_per_1000 = sens * prev * 1000
        fp_per_1000 = (1 - spec) * (1 - prev) * 1000
        screen_positives = tp_per_1000 + fp_per_1000

        prescreen_cost = (
            cost_prescreening * 1000  # screen everyone
            + cost_molecular * screen_positives  # confirm positives
        )

        # Missed MSI-H cases (false negatives)
        fn_per_1000 = (1 - sens) * prev * 1000

        savings = universal_cost - prescreen_cost

        results.append(CostComparison(
            prevalence=prev,
            universal_cost_per_1000=round(universal_cost, 2),
            prescreening_cost_per_1000=round(prescreen_cost, 2),
            savings_per_1000=round(savings, 2),
            savings_pct=round(savings / universal_cost * 100, 1),
            missed_cases_per_1000=round(fn_per_1000, 2),
            unnecessary_molecular_per_1000=round(fp_per_1000, 2),
        ))

    return results


def run_clinical_utility_analysis(
    probs: np.ndarray,
    labels: np.ndarray,
    min_sensitivity: float = 0.95,
    cost_molecular: float = 400.0,
    cost_prescreening: float = 10.0,
    output_dir: str | Path | None = None,
) -> dict:
    """Run full clinical utility analysis.

    Args:
        probs: Predicted probabilities, shape (N,).
        labels: True binary labels, shape (N,).
        min_sensitivity: Minimum sensitivity for optimal point.
        cost_molecular: Cost per molecular test (USD).
        cost_prescreening: Cost per AI pre-screening (USD).
        output_dir: Where to save results.

    Returns:
        Dict with operating points, optimal point, and cost comparisons.
    """
    # Find all operating points
    points = find_operating_points(probs, labels)

    # Find optimal operating point
    optimal = find_optimal_point(points, min_sensitivity)

    results = {
        "n_samples": len(labels),
        "n_positive": int(labels.sum()),
        "sample_prevalence": float(labels.mean()),
        "min_sensitivity_constraint": min_sensitivity,
    }

    if optimal:
        results["optimal_point"] = {
            "threshold": optimal.threshold,
            "sensitivity": optimal.sensitivity,
            "specificity": optimal.specificity,
            "ppv_at_sample_prev": optimal.ppv,
            "npv_at_sample_prev": optimal.npv,
            "molecular_tests_per_1000": optimal.n_molecular_tests_per_1000,
        }

        # Cost-effectiveness at various prevalence rates
        cost_results = cost_effectiveness_analysis(
            optimal,
            cost_molecular=cost_molecular,
            cost_prescreening=cost_prescreening,
        )
        results["cost_comparison"] = [
            {
                "prevalence": c.prevalence,
                "universal_cost_per_1000": c.universal_cost_per_1000,
                "prescreening_cost_per_1000": c.prescreening_cost_per_1000,
                "savings_per_1000": c.savings_per_1000,
                "savings_pct": c.savings_pct,
                "missed_cases_per_1000": c.missed_cases_per_1000,
                "unnecessary_molecular_per_1000": c.unnecessary_molecular_per_1000,
            }
            for c in cost_results
        ]

        logger.info(
            f"Optimal point: threshold={optimal.threshold:.4f}, "
            f"sens={optimal.sensitivity:.3f}, spec={optimal.specificity:.3f}, "
            f"molecular tests/1000={optimal.n_molecular_tests_per_1000:.0f}"
        )
    else:
        results["optimal_point"] = None
        results["cost_comparison"] = []
        logger.warning("No operating point met the sensitivity constraint")

    # ROC data for plotting
    results["roc_data"] = [
        {
            "threshold": p.threshold,
            "sensitivity": p.sensitivity,
            "specificity": p.specificity,
            "fpr": p.fpr,
        }
        for p in points
    ]

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "clinical_utility.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Clinical utility results saved to {out}")

    return results


def _ppv(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Positive predictive value via Bayes' theorem."""
    num = sensitivity * prevalence
    denom = num + (1 - specificity) * (1 - prevalence)
    return num / denom if denom > 0 else 0.0


def _npv(sensitivity: float, specificity: float, prevalence: float) -> float:
    """Negative predictive value via Bayes' theorem."""
    num = specificity * (1 - prevalence)
    denom = num + (1 - sensitivity) * prevalence
    return num / denom if denom > 0 else 0.0
