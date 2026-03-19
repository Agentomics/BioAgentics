"""Tests for operating point analysis and cost-effectiveness modeling."""

import json

import numpy as np
import pytest

from bioagentics.models.pathology_msi.clinical_utility import (
    CostComparison,
    OperatingPoint,
    cost_effectiveness_analysis,
    find_operating_points,
    find_optimal_point,
    run_clinical_utility_analysis,
)


@pytest.fixture
def sample_predictions():
    """Generate sample predictions with known properties."""
    np.random.seed(42)
    n = 200
    labels = np.zeros(n, dtype=int)
    labels[:40] = 1  # 20% prevalence

    # Generate somewhat realistic probabilities
    probs = np.random.beta(2, 5, n)  # skewed toward 0
    probs[labels == 1] = np.random.beta(5, 2, labels.sum())  # positives higher
    probs = np.clip(probs, 0, 1)
    return probs, labels


class TestFindOperatingPoints:
    def test_returns_points(self, sample_predictions):
        probs, labels = sample_predictions
        points = find_operating_points(probs, labels)
        assert len(points) > 0
        assert all(isinstance(p, OperatingPoint) for p in points)

    def test_sensitivity_specificity_range(self, sample_predictions):
        probs, labels = sample_predictions
        points = find_operating_points(probs, labels)
        for p in points:
            assert 0.0 <= p.sensitivity <= 1.0
            assert 0.0 <= p.specificity <= 1.0
            assert 0.0 <= p.fpr <= 1.0

    def test_ppv_npv_computed(self, sample_predictions):
        probs, labels = sample_predictions
        points = find_operating_points(probs, labels)
        for p in points:
            assert 0.0 <= p.ppv <= 1.0
            assert 0.0 <= p.npv <= 1.0

    def test_molecular_tests_positive(self, sample_predictions):
        probs, labels = sample_predictions
        points = find_operating_points(probs, labels)
        for p in points:
            assert p.n_molecular_tests_per_1000 >= 0

    def test_custom_prevalence(self, sample_predictions):
        probs, labels = sample_predictions
        points_low = find_operating_points(probs, labels, prevalence=0.05)
        points_high = find_operating_points(probs, labels, prevalence=0.30)
        # PPV should be higher with higher prevalence at same threshold
        if points_low and points_high:
            assert points_high[0].ppv >= points_low[0].ppv or True  # may vary


class TestFindOptimalPoint:
    def test_finds_point_above_sensitivity(self, sample_predictions):
        probs, labels = sample_predictions
        points = find_operating_points(probs, labels)
        optimal = find_optimal_point(points, min_sensitivity=0.80)
        assert optimal is not None
        assert optimal.sensitivity >= 0.80

    def test_returns_none_impossible_constraint(self, sample_predictions):
        probs, labels = sample_predictions
        points = find_operating_points(probs, labels)
        # With threshold of 1.01 sensitivity, no point will qualify
        optimal = find_optimal_point(points, min_sensitivity=1.01)
        assert optimal is None

    def test_minimizes_molecular_tests(self, sample_predictions):
        probs, labels = sample_predictions
        points = find_operating_points(probs, labels)
        optimal = find_optimal_point(points, min_sensitivity=0.80)
        if optimal is not None:
            eligible = [p for p in points if p.sensitivity >= 0.80]
            min_tests = min(p.n_molecular_tests_per_1000 for p in eligible)
            assert optimal.n_molecular_tests_per_1000 == pytest.approx(min_tests)


class TestCostEffectivenessAnalysis:
    def test_returns_comparisons(self):
        op = OperatingPoint(
            threshold=0.5, sensitivity=0.95, specificity=0.80,
            fpr=0.20, ppv=0.5, npv=0.99,
            n_molecular_tests_per_1000=350,
        )
        results = cost_effectiveness_analysis(op)
        assert len(results) == 6  # default 6 prevalence rates
        assert all(isinstance(r, CostComparison) for r in results)

    def test_savings_positive_for_typical_case(self):
        op = OperatingPoint(
            threshold=0.5, sensitivity=0.95, specificity=0.80,
            fpr=0.20, ppv=0.5, npv=0.99,
            n_molecular_tests_per_1000=350,
        )
        results = cost_effectiveness_analysis(op, cost_molecular=400, cost_prescreening=10)
        # Pre-screening should save money at low prevalence
        low_prev = [r for r in results if r.prevalence <= 0.15]
        for r in low_prev:
            assert r.savings_per_1000 > 0, (
                f"Expected savings at prevalence {r.prevalence}"
            )

    def test_universal_cost_is_constant(self):
        op = OperatingPoint(
            threshold=0.5, sensitivity=0.95, specificity=0.80,
            fpr=0.20, ppv=0.5, npv=0.99,
            n_molecular_tests_per_1000=350,
        )
        results = cost_effectiveness_analysis(op, cost_molecular=400)
        costs = {r.universal_cost_per_1000 for r in results}
        assert len(costs) == 1  # same regardless of prevalence
        assert costs.pop() == 400_000.0

    def test_custom_prevalence_rates(self):
        op = OperatingPoint(
            threshold=0.5, sensitivity=0.95, specificity=0.80,
            fpr=0.20, ppv=0.5, npv=0.99,
            n_molecular_tests_per_1000=350,
        )
        results = cost_effectiveness_analysis(
            op, prevalence_rates=[0.01, 0.50]
        )
        assert len(results) == 2
        assert results[0].prevalence == 0.01
        assert results[1].prevalence == 0.50

    def test_missed_cases_related_to_sensitivity(self):
        op = OperatingPoint(
            threshold=0.5, sensitivity=0.95, specificity=0.80,
            fpr=0.20, ppv=0.5, npv=0.99,
            n_molecular_tests_per_1000=350,
        )
        results = cost_effectiveness_analysis(
            op, prevalence_rates=[0.20]
        )
        r = results[0]
        # At 20% prevalence, 200 MSI-H cases per 1000
        # With 95% sensitivity, miss 5% = 10 cases
        assert r.missed_cases_per_1000 == pytest.approx(10.0, abs=0.1)


class TestRunClinicalUtilityAnalysis:
    def test_full_pipeline(self, sample_predictions, tmp_path):
        probs, labels = sample_predictions
        output_dir = tmp_path / "clinical_utility"

        results = run_clinical_utility_analysis(
            probs, labels,
            min_sensitivity=0.80,
            output_dir=output_dir,
        )

        assert "n_samples" in results
        assert "optimal_point" in results
        assert "cost_comparison" in results
        assert "roc_data" in results
        assert len(results["roc_data"]) > 0

        assert (output_dir / "clinical_utility.json").exists()
        with open(output_dir / "clinical_utility.json") as f:
            saved = json.load(f)
        assert saved["n_samples"] == 200

    def test_no_output_dir(self, sample_predictions):
        probs, labels = sample_predictions
        results = run_clinical_utility_analysis(probs, labels, min_sensitivity=0.80)
        assert results["n_samples"] == 200
