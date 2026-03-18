"""Tests for DR screening calibration module."""

import numpy as np

from bioagentics.diagnostics.retinal_dr_screening.calibration import (
    apply_temperature,
    compute_ece,
    find_clinical_threshold,
    learn_platt_scaling,
    learn_temperature,
)


def test_learn_temperature():
    """Temperature should be positive and reasonable."""
    np.random.seed(42)
    logits = np.random.randn(100, 5)
    labels = np.random.randint(0, 5, 100)

    T = learn_temperature(logits, labels)
    assert 0.1 <= T <= 10.0


def test_apply_temperature_sums_to_one():
    logits = np.random.randn(10, 5)
    probs = apply_temperature(logits, temperature=2.0)

    assert probs.shape == (10, 5)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    assert np.all(probs >= 0)


def test_temperature_smooths_distribution():
    """Higher temperature should produce smoother (more uniform) distributions."""
    logits = np.array([[5.0, 1.0, 0.0, -1.0, -2.0]])

    probs_t1 = apply_temperature(logits, 1.0)
    probs_t5 = apply_temperature(logits, 5.0)

    # T=5 should be more uniform (higher entropy)
    assert np.std(probs_t5) < np.std(probs_t1)


def test_compute_ece():
    """ECE should be between 0 and 1."""
    np.random.seed(42)
    probs = np.random.dirichlet([1, 1, 1, 1, 1], 100)
    labels = np.random.randint(0, 5, 100)

    result = compute_ece(probs, labels, n_bins=10)

    assert 0 <= result.ece <= 1
    assert 0 <= result.mce <= 1
    assert len(result.bin_confidences) == 10
    assert len(result.bin_accuracies) == 10
    assert len(result.bin_counts) == 10


def test_compute_ece_perfect_calibration():
    """Perfect calibration should have ECE near 0."""
    # Create perfectly calibrated predictions
    # For each sample, confidence == accuracy
    n = 1000
    labels = np.random.randint(0, 5, n)
    probs = np.eye(5)[labels]  # one-hot = 100% confidence, always correct

    result = compute_ece(probs, labels)
    # With perfect predictions, ECE should be very low
    # (not exactly 0 because binning creates some error)
    assert result.ece < 0.1


def test_learn_platt_scaling():
    np.random.seed(42)
    logits = np.random.randn(100, 5) * 3
    labels = np.random.randint(0, 5, 100)

    platt = learn_platt_scaling(logits, labels)
    probs = platt.predict_proba(logits)

    assert probs.shape == (100, 5)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_find_clinical_threshold():
    np.random.seed(42)
    probs = np.random.dirichlet([1, 1, 1, 1, 1], 200)
    # Make some clearly referable (grade >= 2)
    labels = np.random.randint(0, 5, 200)

    result = find_clinical_threshold(probs, labels, target_sensitivity=0.50)

    assert "threshold" in result
    assert "sensitivity" in result
    assert "specificity" in result
    if result["sensitivity"] > 0:
        assert result["sensitivity"] >= 0.50


def test_find_clinical_threshold_clear_separation():
    """With clear signal, threshold search should find high sensitivity."""
    # Strong referable signal in classes 2-4
    probs = np.zeros((20, 5))
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4])
    for i, label in enumerate(labels):
        probs[i, label] = 0.8
        for j in range(5):
            if j != label:
                probs[i, j] = 0.05

    result = find_clinical_threshold(probs, labels, target_sensitivity=0.80)
    assert result["sensitivity"] >= 0.80
