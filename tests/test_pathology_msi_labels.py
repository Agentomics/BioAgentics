"""Tests for MSI label curation module."""

import pandas as pd
import pytest

from bioagentics.models.pathology_msi.msi_labels import (
    _normalize_msi_call,
    classify_msi_from_mantis,
    curate_msi_labels,
    resolve_msi_status,
    save_labels,
)


class TestNormalizeMsiCall:
    def test_msi_h_variants(self):
        for call in ["MSI-H", "MSIH", "MSI_H", "msi-high", "High"]:
            assert _normalize_msi_call(call) == "MSI-H"

    def test_msi_l_variants(self):
        for call in ["MSI-L", "MSIL", "MSI_L", "msi-low"]:
            assert _normalize_msi_call(call) == "MSI-L"

    def test_mss_variants(self):
        for call in ["MSS", "msi-stable", "Stable"]:
            assert _normalize_msi_call(call) == "MSS"

    def test_unknown_variants(self):
        for call in [None, "", "NA", "NaN", "[Not Available]", "[Not Evaluated]"]:
            assert _normalize_msi_call(call) == "unknown"

    def test_nan_float(self):
        assert _normalize_msi_call(float("nan")) == "unknown"


class TestClassifyMsiFromMantis:
    def test_msi_h(self):
        assert classify_msi_from_mantis(0.5) == "MSI-H"
        assert classify_msi_from_mantis(0.4) == "MSI-H"

    def test_msi_l(self):
        assert classify_msi_from_mantis(0.3) == "MSI-L"
        assert classify_msi_from_mantis(0.2) == "MSI-L"

    def test_mss(self):
        assert classify_msi_from_mantis(0.1) == "MSS"
        assert classify_msi_from_mantis(0.0) == "MSS"

    def test_none(self):
        assert classify_msi_from_mantis(None) == "unknown"


class TestResolveMsiStatus:
    def test_concordant(self):
        status, source = resolve_msi_status("MSI-H", "MSI-H")
        assert status == "MSI-H"
        assert source == "mantis_pcr_concordant"

    def test_pcr_only(self):
        status, source = resolve_msi_status(None, "MSS")
        assert status == "MSS"
        assert source == "pcr_only"

    def test_mantis_only(self):
        status, source = resolve_msi_status("MSI-H", None)
        assert status == "MSI-H"
        assert source == "mantis_only"

    def test_conflict_prefers_pcr(self):
        status, source = resolve_msi_status("MSI-H", "MSS")
        assert status == "MSS"
        assert "pcr_preferred" in source

    def test_both_unknown_uses_mantis_score(self):
        status, source = resolve_msi_status(None, None, mantis_score=0.5)
        assert status == "MSI-H"
        assert source == "mantis_score"

    def test_both_unknown_no_score(self):
        status, source = resolve_msi_status(None, None)
        assert status == "unknown"
        assert source == "no_data"


class TestCurateMsiLabels:
    def test_basic_curation(self):
        cases = pd.DataFrame(
            {
                "case_id": ["c1", "c2", "c3"],
                "submitter_id": ["TCGA-AA-0001", "TCGA-AG-0002", "TCGA-AX-0003"],
                "project_id": ["TCGA-COAD", "TCGA-READ", "TCGA-UCEC"],
                "cancer_type": ["COAD", "READ", "UCEC"],
            }
        )
        annotations = pd.DataFrame(
            {
                "submitter_id": ["TCGA-AA-0001", "TCGA-AG-0002", "TCGA-AX-0003"],
                "mantis_score": [0.5, 0.1, None],
                "mantis_call": ["MSI-H", "MSS", None],
                "pcr_call": ["MSI-H", "MSS", "MSI-H"],
            }
        )

        result = curate_msi_labels(cases, annotations)
        assert len(result) == 3
        assert list(result["msi_status"]) == ["MSI-H", "MSS", "MSI-H"]

    def test_curation_without_annotations(self):
        cases = pd.DataFrame(
            {
                "case_id": ["c1"],
                "submitter_id": ["TCGA-AA-0001"],
                "project_id": ["TCGA-COAD"],
                "cancer_type": ["COAD"],
            }
        )
        result = curate_msi_labels(cases, None)
        assert len(result) == 1
        assert result["msi_status"].iloc[0] == "unknown"


def test_save_labels(tmp_path):
    df = pd.DataFrame(
        {
            "case_id": ["c1", "c2"],
            "msi_status": ["MSI-H", "MSS"],
            "cancer_type": ["COAD", "COAD"],
        }
    )
    path = save_labels(df, tmp_path)
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 2
