"""Tests for positive control validation (mock data)."""

import json

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.validation import (
    ValidationReport,
    save_validation_report,
    validate_positive_controls,
)


@pytest.fixture
def mock_validation_data():
    """Create mock data where KRAS dependency correlates with KRAS mutation."""
    rng = np.random.default_rng(42)
    n_patients = 100
    patient_ids = [f"TCGA-XX-{i:04d}" for i in range(n_patients)]

    # 30 KRAS-mutant, 70 WT
    kras_mut = np.array([True] * 30 + [False] * 70)
    # Dependency: KRAS-mutant patients more dependent (more negative)
    kras_dep = np.where(kras_mut, rng.normal(-1.5, 0.3, n_patients), rng.normal(-0.2, 0.3, n_patients))
    noise_dep = rng.normal(-0.5, 0.5, n_patients)

    dep_matrix = pd.DataFrame({
        "KRAS": kras_dep,
        "NOISE_GENE": noise_dep,
    }, index=patient_ids)

    subtypes = pd.Series(
        ["KP"] * 15 + ["KL"] * 10 + ["KOnly"] * 5 + ["KRAS-WT"] * 70,
        index=patient_ids,
    )

    mutations = pd.DataFrame({
        "KRAS_mutated": kras_mut,
        "EGFR_mutated": [False] * 100,  # no EGFR mutants for this test
    }, index=patient_ids)

    return dep_matrix, subtypes, mutations


def test_validate_basic(mock_validation_data):
    """Test basic validation runs and returns correct structure."""
    dep_matrix, subtypes, mutations = mock_validation_data
    report = validate_positive_controls(dep_matrix, subtypes, mutations)

    assert isinstance(report, ValidationReport)
    assert report.n_controls_tested >= 1


def test_kras_control_passes(mock_validation_data):
    """Test that KRAS positive control passes with strong signal."""
    dep_matrix, subtypes, mutations = mock_validation_data
    report = validate_positive_controls(dep_matrix, subtypes, mutations)

    assert "KRAS" in report.control_aucs
    assert report.control_aucs["KRAS"] > 0.75
    assert "KRAS" in report.controls_passed


def test_missing_control_skipped(mock_validation_data):
    """Test that missing control genes are skipped."""
    dep_matrix, subtypes, mutations = mock_validation_data
    report = validate_positive_controls(dep_matrix, subtypes, mutations)

    # ALK, ROS1, MET not in dep_matrix → should be skipped
    assert "ALK" in report.controls_skipped


def test_egfr_skipped_no_mutants(mock_validation_data):
    """Test that controls with no mutant patients are skipped."""
    dep_matrix, subtypes, mutations = mock_validation_data
    dep_matrix["EGFR"] = np.random.default_rng(1).normal(0, 1, len(dep_matrix))
    report = validate_positive_controls(dep_matrix, subtypes, mutations)

    # EGFR has no mutant patients
    assert "EGFR" in report.controls_skipped


def test_kl_pathway_flags(mock_validation_data):
    """Test that KL pathway genes are flagged when present."""
    dep_matrix, subtypes, mutations = mock_validation_data
    # Add some KL-relevant genes
    rng = np.random.default_rng(77)
    dep_matrix["CTLA4"] = rng.normal(-0.5, 0.3, len(dep_matrix))
    dep_matrix["MTOR"] = rng.normal(-0.4, 0.3, len(dep_matrix))

    report = validate_positive_controls(dep_matrix, subtypes, mutations)

    pathway_genes = {f["gene"] for f in report.kl_pathway_flags}
    assert "CTLA4" in pathway_genes or "MTOR" in pathway_genes


def test_save_report(mock_validation_data, tmp_path):
    """Test saving validation report to JSON."""
    dep_matrix, subtypes, mutations = mock_validation_data
    report = validate_positive_controls(dep_matrix, subtypes, mutations)
    out_path = save_validation_report(report, tmp_path)

    assert out_path.exists()
    with open(out_path) as f:
        data = json.load(f)
    assert "control_aucs" in data
    assert "kl_pathway_flags" in data


def test_custom_auc_threshold(mock_validation_data):
    """Test custom AUC threshold."""
    dep_matrix, subtypes, mutations = mock_validation_data
    report = validate_positive_controls(dep_matrix, subtypes, mutations, auc_threshold=0.99)
    assert report.auc_threshold == 0.99
    # With a very high threshold, KRAS might fail
    assert report.n_controls_passed <= report.n_controls_tested
