"""Tests for CPTAC external validation data preparation pipeline."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bioagentics.models.pathology_msi.cptac_data import (
    GDC_DATA_ENDPOINT,
    classify_msi_from_msisensor2,
    curate_cptac_labels,
    generate_gdc_download_manifest,
    query_cptac_wsi_metadata,
    save_cptac_labels,
    save_cptac_manifest,
    save_per_cancer_manifests,
    _extract_cancer_type,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _make_cptac_hit(
    file_id: str,
    case_id: str,
    submitter_id: str,
    project_id: str,
    primary_site: str,
) -> dict:
    """Create a mock GDC API hit for a CPTAC WSI."""
    return {
        "file_id": file_id,
        "file_name": f"{file_id}.svs",
        "file_size": 400_000_000,
        "md5sum": "abc123",
        "data_format": "SVS",
        "experimental_strategy": "Tissue Slide",
        "cases": [
            {
                "case_id": case_id,
                "submitter_id": submitter_id,
                "project": {"project_id": project_id},
                "diagnoses": [
                    {"tissue_or_organ_of_origin": primary_site},
                ],
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


def _sample_wsi_metadata() -> pd.DataFrame:
    """Create sample CPTAC WSI metadata DataFrame."""
    return pd.DataFrame(
        {
            "file_id": ["f1", "f2", "f3", "f4"],
            "file_name": ["a.svs", "b.svs", "c.svs", "d.svs"],
            "case_id": ["c1", "c2", "c3", "c4"],
            "case_submitter_id": ["CPT-0001", "CPT-0002", "CPT-0003", "CPT-0004"],
            "project_id": ["CPTAC-2", "CPTAC-2", "CPTAC-3", "CPTAC-3"],
            "cancer_type": ["COAD", "UCEC", "LSCC", "LUAD"],
            "primary_site": ["Colon", "Corpus uteri", "Lung, squamous", "Lung"],
            "data_format": ["SVS", "SVS", "SVS", "SVS"],
            "file_size": [400_000_000, 500_000_000, 300_000_000, 450_000_000],
            "md5sum": ["abc", "def", "ghi", "jkl"],
            "download_url": ["url1", "url2", "url3", "url4"],
        }
    )


# ─── Cancer type extraction ──────────────────────────────────────────────────


class TestExtractCancerType:
    def test_colon(self):
        assert _extract_cancer_type("CPTAC-2", "Colon") == "COAD"

    def test_rectum(self):
        assert _extract_cancer_type("CPTAC-2", "Rectum") == "COAD"

    def test_colorectal(self):
        assert _extract_cancer_type("CPTAC-2", "Colorectal") == "COAD"

    def test_endometrium(self):
        assert _extract_cancer_type("CPTAC-2", "Corpus uteri") == "UCEC"

    def test_uterus(self):
        assert _extract_cancer_type("CPTAC-2", "Uterus, NOS") == "UCEC"

    def test_lung_squamous(self):
        assert _extract_cancer_type("CPTAC-3", "Lung, squamous cell") == "LSCC"

    def test_lung_default(self):
        assert _extract_cancer_type("CPTAC-3", "Lung") == "LUAD"

    def test_bronchus(self):
        assert _extract_cancer_type("CPTAC-3", "Bronchus") == "LUAD"

    def test_unknown_site(self):
        result = _extract_cancer_type("CPTAC-3", "Kidney")
        assert result == "KIDNEY"

    def test_empty_site(self):
        assert _extract_cancer_type("CPTAC-3", "") == "UNKNOWN"


# ─── MSIsensor2 classification ───────────────────────────────────────────────


class TestClassifyMsiFromMsisensor2:
    def test_msi_h(self):
        assert classify_msi_from_msisensor2(25.0) == "MSI-H"

    def test_msi_h_at_threshold(self):
        assert classify_msi_from_msisensor2(20.0) == "MSI-H"

    def test_msi_l(self):
        assert classify_msi_from_msisensor2(15.0) == "MSI-L"

    def test_msi_l_at_lower_threshold(self):
        assert classify_msi_from_msisensor2(10.0) == "MSI-L"

    def test_mss(self):
        assert classify_msi_from_msisensor2(5.0) == "MSS"

    def test_mss_zero(self):
        assert classify_msi_from_msisensor2(0.0) == "MSS"

    def test_none(self):
        assert classify_msi_from_msisensor2(None) == "unknown"

    def test_nan(self):
        assert classify_msi_from_msisensor2(float("nan")) == "unknown"


# ─── GDC query ────────────────────────────────────────────────────────────────


@patch("bioagentics.models.pathology_msi.cptac_data.requests.get")
def test_query_cptac_wsi_metadata(mock_get):
    """Test that query parses GDC response into correct DataFrame."""
    hits = [
        _make_cptac_hit("f1", "c1", "CPT-0001", "CPTAC-2", "Colon"),
        _make_cptac_hit("f2", "c2", "CPT-0002", "CPTAC-2", "Corpus uteri"),
        _make_cptac_hit("f3", "c3", "CPT-0003", "CPTAC-3", "Lung"),
    ]
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_gdc_response(hits)
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    df = query_cptac_wsi_metadata()

    assert len(df) == 3
    assert "file_id" in df.columns
    assert "cancer_type" in df.columns
    assert "primary_site" in df.columns
    assert set(df["cancer_type"]) == {"COAD", "UCEC", "LUAD"}
    assert all(url.startswith(GDC_DATA_ENDPOINT) for url in df["download_url"])


@patch("bioagentics.models.pathology_msi.cptac_data.requests.get")
def test_query_filters_cancer_types(mock_get):
    """Test that non-requested cancer types are excluded."""
    hits = [
        _make_cptac_hit("f1", "c1", "CPT-0001", "CPTAC-2", "Colon"),
        _make_cptac_hit("f2", "c2", "CPT-0002", "CPTAC-3", "Kidney"),
    ]
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_gdc_response(hits)
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    df = query_cptac_wsi_metadata()

    assert len(df) == 1
    assert df.iloc[0]["cancer_type"] == "COAD"


@patch("bioagentics.models.pathology_msi.cptac_data.requests.get")
def test_query_handles_empty_response(mock_get):
    """Test that empty GDC response returns empty DataFrame."""
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_gdc_response([])
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    df = query_cptac_wsi_metadata()
    assert df.empty


@patch("bioagentics.models.pathology_msi.cptac_data.requests.get")
def test_query_skips_hits_without_cases(mock_get):
    """Test that hits without case information are skipped."""
    hits = [
        _make_cptac_hit("f1", "c1", "CPT-0001", "CPTAC-2", "Colon"),
        {"file_id": "f2", "file_name": "orphan.svs", "cases": []},
    ]
    mock_resp = MagicMock()
    mock_resp.json.return_value = _mock_gdc_response(hits)
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    df = query_cptac_wsi_metadata()
    assert len(df) == 1


# ─── Label curation ──────────────────────────────────────────────────────────


def test_curate_cptac_labels_without_annotations():
    """Test label curation with no annotation data sets unknown status."""
    wsi = _sample_wsi_metadata()
    result = curate_cptac_labels(wsi)

    assert len(result) == 4
    assert all(result["msi_status"] == "unknown")


def test_curate_cptac_labels_with_annotations():
    """Test label curation merges annotations correctly."""
    wsi = _sample_wsi_metadata()
    ann = pd.DataFrame(
        {
            "case_id": ["CPT-0001", "CPT-0002", "CPT-0003"],
            "msisensor2_score": [25.0, 5.0, 15.0],
            "msi_status": ["MSI-H", "MSS", "MSI-L"],
        }
    )

    result = curate_cptac_labels(wsi, ann)

    assert len(result) == 4
    # CPT-0001 matched via case_submitter_id
    row1 = result[result["case_submitter_id"] == "CPT-0001"].iloc[0]
    assert row1["msi_status"] == "MSI-H"
    assert row1["msisensor2_score"] == 25.0

    # CPT-0004 has no annotation
    row4 = result[result["case_submitter_id"] == "CPT-0004"].iloc[0]
    assert row4["msi_status"] == "unknown"


# ─── Annotations loading ─────────────────────────────────────────────────────


def test_load_cptac_msi_annotations_csv(tmp_path):
    """Test loading MSI annotations from CSV."""
    from bioagentics.models.pathology_msi.cptac_data import load_cptac_msi_annotations

    csv_path = tmp_path / "msi_scores.csv"
    pd.DataFrame(
        {
            "case_id": ["CPT-0001", "CPT-0002", "CPT-0003"],
            "msisensor2_score": [25.0, 5.0, 15.0],
        }
    ).to_csv(csv_path, index=False)

    result = load_cptac_msi_annotations(csv_path)
    assert len(result) == 3
    assert "msi_status" in result.columns
    assert result.loc[result["case_id"] == "CPT-0001", "msi_status"].iloc[0] == "MSI-H"
    assert result.loc[result["case_id"] == "CPT-0002", "msi_status"].iloc[0] == "MSS"
    assert result.loc[result["case_id"] == "CPT-0003", "msi_status"].iloc[0] == "MSI-L"


def test_load_cptac_msi_annotations_tsv(tmp_path):
    """Test loading MSI annotations from TSV."""
    from bioagentics.models.pathology_msi.cptac_data import load_cptac_msi_annotations

    tsv_path = tmp_path / "msi_scores.tsv"
    pd.DataFrame(
        {
            "sample_id": ["CPT-0001", "CPT-0002"],
            "MSIsensor_Score": [30.0, 2.0],
        }
    ).to_csv(tsv_path, sep="\t", index=False)

    result = load_cptac_msi_annotations(tsv_path)
    assert len(result) == 2
    assert result.iloc[0]["msi_status"] == "MSI-H"


def test_load_cptac_msi_annotations_missing_id(tmp_path):
    """Test that missing case identifier raises ValueError."""
    from bioagentics.models.pathology_msi.cptac_data import load_cptac_msi_annotations

    csv_path = tmp_path / "bad.csv"
    pd.DataFrame({"score": [25.0]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="Missing case identifier"):
        load_cptac_msi_annotations(csv_path)


# ─── File saving ──────────────────────────────────────────────────────────────


def test_save_cptac_manifest(tmp_path):
    """Test manifest CSV saving."""
    df = _sample_wsi_metadata()
    path = save_cptac_manifest(df, tmp_path)
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 4


def test_save_cptac_labels(tmp_path):
    """Test labels CSV saving."""
    df = _sample_wsi_metadata()
    df["msi_status"] = ["MSI-H", "MSS", "MSI-L", "unknown"]
    path = save_cptac_labels(df, tmp_path)
    assert path.exists()
    loaded = pd.read_csv(path)
    assert len(loaded) == 4
    assert "msi_status" in loaded.columns


def test_save_per_cancer_manifests(tmp_path):
    """Test per-cancer-type manifest saving."""
    df = _sample_wsi_metadata()
    paths = save_per_cancer_manifests(df, tmp_path)
    assert len(paths) == 4  # COAD, UCEC, LSCC, LUAD
    assert all(p.exists() for p in paths)


def test_generate_gdc_download_manifest(tmp_path):
    """Test GDC download manifest generation."""
    df = _sample_wsi_metadata()
    path = generate_gdc_download_manifest(df, tmp_path)
    assert path.exists()
    loaded = pd.read_csv(path, sep="\t")
    assert list(loaded.columns) == ["id", "filename", "md5", "size"]
    assert len(loaded) == 4
