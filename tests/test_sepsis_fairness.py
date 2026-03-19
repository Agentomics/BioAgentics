"""Tests for Phase 3 fairness analysis module."""

from __future__ import annotations

import numpy as np
import pytest

from bioagentics.diagnostics.sepsis.calibration.fairness import (
    assign_age_groups,
    compute_fairness_disparity,
    compute_subgroup_metrics,
)


@pytest.fixture
def fairness_data():
    """Synthetic data with demographic groups."""
    rng = np.random.default_rng(42)
    n = 400
    y_true = np.concatenate([np.zeros(280), np.ones(120)]).astype(int)
    y_prob = np.clip(
        np.where(y_true == 1, rng.beta(4, 2, n), rng.beta(2, 4, n)),
        0.01,
        0.99,
    )
    # Create groups with enough samples each
    groups = np.array(["A"] * 150 + ["B"] * 130 + ["C"] * 120)
    rng.shuffle(groups)
    return y_true, y_prob, groups


def test_compute_subgroup_metrics_structure(fairness_data):
    """Subgroup metrics have expected fields for each group."""
    y_true, y_prob, groups = fairness_data
    result = compute_subgroup_metrics(y_true, y_prob, groups)
    assert len(result) > 0
    for _, metrics in result.items():
        assert "auroc" in metrics
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert "ppv" in metrics
        assert "n_samples" in metrics
        assert "n_positive" in metrics
        assert 0 <= metrics["auroc"] <= 1
        assert 0 <= metrics["sensitivity"] <= 1
        assert 0 <= metrics["specificity"] <= 1
        assert 0 <= metrics["ppv"] <= 1


def test_compute_subgroup_metrics_all_groups(fairness_data):
    """All groups with sufficient data appear in results."""
    y_true, y_prob, groups = fairness_data
    result = compute_subgroup_metrics(y_true, y_prob, groups)
    assert "A" in result
    assert "B" in result
    assert "C" in result


def test_compute_subgroup_metrics_skips_small_groups():
    """Groups with <10 samples are skipped."""
    y_true = np.array([0, 0, 1, 1, 0, 0, 1, 1] + [0] * 50 + [1] * 50)
    y_prob = np.random.default_rng(42).random(len(y_true))
    groups = np.array(["tiny"] * 8 + ["large"] * 100)
    result = compute_subgroup_metrics(y_true, y_prob, groups)
    assert "large" in result
    assert "tiny" not in result  # Too few samples


def test_compute_fairness_disparity(fairness_data):
    """Disparity computation returns valid structure."""
    y_true, y_prob, groups = fairness_data
    subgroup = compute_subgroup_metrics(y_true, y_prob, groups)
    disparity = compute_fairness_disparity(subgroup)
    assert "max_auroc_disparity" in disparity
    assert "meets_target" in disparity
    assert disparity["max_auroc_disparity"] >= 0.0
    assert isinstance(disparity["meets_target"], bool)


def test_compute_fairness_disparity_single_group():
    """Disparity with one group is 0."""
    subgroup = {"only_group": {"auroc": 0.85}}
    disparity = compute_fairness_disparity(subgroup)
    assert disparity["max_auroc_disparity"] == 0.0
    assert disparity["meets_target"] is True


def test_assign_age_groups():
    """Age groups are assigned correctly."""
    ages = np.array([25, 50, 70, 85, 30, 60])
    groups = assign_age_groups(ages)
    assert groups[0] == "18-44"
    assert groups[1] == "45-64"
    assert groups[2] == "65-79"
    assert groups[3] == "80+"
    assert groups[4] == "18-44"
    assert groups[5] == "45-64"


def test_assign_age_groups_edge_cases():
    """Boundary ages are assigned correctly."""
    ages = np.array([18, 45, 65, 80])
    groups = assign_age_groups(ages)
    assert groups[0] == "18-44"  # 18 >= 18 and < 45
    assert groups[1] == "45-64"  # 45 >= 45 and < 65
    assert groups[2] == "65-79"  # 65 >= 65 and < 80
    assert groups[3] == "80+"    # 80 >= 80


def test_assign_age_groups_unknown():
    """Ages outside bins are 'unknown'."""
    ages = np.array([5, 10])
    groups = assign_age_groups(ages)
    assert all(g == "unknown" for g in groups)
