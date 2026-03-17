"""Tests for GDC API client for TCGA diagnostic WSI metadata."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bioagentics.models.pathology_msi.gdc_client import (
    GDC_DATA_ENDPOINT,
    MSI_CANCER_TYPES,
    generate_gdc_download_manifest,
    query_diagnostic_wsi_metadata,
    save_manifest,
    save_per_cancer_manifests,
)


def _make_gdc_hit(file_id: str, case_id: str, submitter_id: str, project_id: str) -> dict:
    """Create a mock GDC API hit."""
    return {
        "file_id": file_id,
        "file_name": f"{file_id}.svs",
        "file_size": 500_000_000,
        "md5sum": "abc123",
        "data_format": "SVS",
        "cases": [
            {
                "case_id": case_id,
                "submitter_id": submitter_id,
                "project": {"project_id": project_id},
            }
        ],
    }


def _mock_gdc_response(hits: list[dict]) -> dict:
    return {
        "data": {
            "hits": hits,
            "pagination": {"total": len(hits)},
        }
    }


@patch("bioagentics.models.pathology_msi.gdc_client.requests.get")
def test_query_diagnostic_wsi_metadata(mock_get):
    """Test that query parses GDC response into correct DataFrame."""
    hits = [
        _make_gdc_hit("f1", "c1", "TCGA-AA-0001", "TCGA-COAD"),
        _make_gdc_hit("f2", "c2", "TCGA-AG-0002", "TCGA-READ"),
        _make_gdc_hit("f3", "c3", "TCGA-AX-0003", "TCGA-UCEC"),
    ]
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_gdc_response(hits)
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    df = query_diagnostic_wsi_metadata()

    assert len(df) == 3
    assert set(df.columns) == {
        "file_id",
        "file_name",
        "case_id",
        "case_submitter_id",
        "project_id",
        "cancer_type",
        "data_format",
        "file_size",
        "md5sum",
        "download_url",
    }
    assert set(df["cancer_type"]) == {"COAD", "READ", "UCEC"}
    assert all(url.startswith(GDC_DATA_ENDPOINT) for url in df["download_url"])


@patch("bioagentics.models.pathology_msi.gdc_client.requests.get")
def test_query_handles_empty_response(mock_get):
    """Test that empty GDC response returns empty DataFrame."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_gdc_response([])
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    df = query_diagnostic_wsi_metadata()
    assert df.empty


@patch("bioagentics.models.pathology_msi.gdc_client.requests.get")
def test_query_skips_hits_without_cases(mock_get):
    """Test that hits without case information are skipped."""
    hits = [
        _make_gdc_hit("f1", "c1", "TCGA-AA-0001", "TCGA-COAD"),
        {"file_id": "f2", "file_name": "orphan.svs", "cases": []},
    ]
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_gdc_response(hits)
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    df = query_diagnostic_wsi_metadata()
    assert len(df) == 1


def test_save_manifest(tmp_path):
    """Test manifest CSV saving."""
    df = pd.DataFrame(
        {
            "file_id": ["f1", "f2"],
            "file_name": ["a.svs", "b.svs"],
            "case_id": ["c1", "c2"],
            "case_submitter_id": ["TCGA-AA-0001", "TCGA-AG-0002"],
            "project_id": ["TCGA-COAD", "TCGA-READ"],
            "cancer_type": ["COAD", "READ"],
            "data_format": ["SVS", "SVS"],
            "file_size": [500_000_000, 600_000_000],
            "md5sum": ["abc", "def"],
            "download_url": ["url1", "url2"],
        }
    )
    path = save_manifest(df, tmp_path)
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 2


def test_save_per_cancer_manifests(tmp_path):
    """Test per-cancer-type manifest saving."""
    df = pd.DataFrame(
        {
            "file_id": ["f1", "f2", "f3"],
            "cancer_type": ["COAD", "COAD", "READ"],
            "case_id": ["c1", "c2", "c3"],
        }
    )
    paths = save_per_cancer_manifests(df, tmp_path)
    assert len(paths) == 2
    assert all(p.exists() for p in paths)


def test_generate_gdc_download_manifest(tmp_path):
    """Test GDC download manifest generation."""
    df = pd.DataFrame(
        {
            "file_id": ["f1", "f2"],
            "file_name": ["a.svs", "b.svs"],
            "md5sum": ["abc", "def"],
            "file_size": [500, 600],
        }
    )
    path = generate_gdc_download_manifest(df, tmp_path)
    assert path.exists()
    loaded = pd.read_csv(path, sep="\t")
    assert list(loaded.columns) == ["id", "filename", "md5", "size"]
    assert len(loaded) == 2
