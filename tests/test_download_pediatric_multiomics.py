"""Tests for pediatric CD multi-omics download script."""

from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock, patch

_mod = importlib.import_module(
    "crohns.microbiome_metabolome_subtyping.02_download_pediatric_multiomics"
)

ENA_BIOPROJECT = _mod.ENA_BIOPROJECT
MTBLS_ACCESSION = _mod.MTBLS_ACCESSION
OUTPUT_DIR = _mod.OUTPUT_DIR
PXD_ACCESSION = _mod.PXD_ACCESSION


def test_accession_formats():
    """Accessions are valid for their respective repositories."""
    assert MTBLS_ACCESSION.startswith("MTBLS")
    assert MTBLS_ACCESSION == "MTBLS9877"
    assert ENA_BIOPROJECT.startswith("PRJEB")
    assert ENA_BIOPROJECT == "PRJEB74164"
    assert PXD_ACCESSION.startswith("PXD")
    assert PXD_ACCESSION == "PXD062519"


def test_output_dir_path():
    """Output directory is under data/crohns/microbiome-metabolome-subtyping/."""
    parts = OUTPUT_DIR.parts
    assert "data" in parts
    assert "crohns" in parts
    assert "microbiome-metabolome-subtyping" in parts
    assert parts[-1] == "pediatric_multiomics"


def test_save_download_manifest(tmp_path):
    """Manifest file has correct content."""
    _mod._save_download_manifest(tmp_path)
    manifest = tmp_path / "MANIFEST.md"
    assert manifest.exists()
    content = manifest.read_text()
    assert "Pediatric CD" in content
    assert MTBLS_ACCESSION in content
    assert ENA_BIOPROJECT in content
    assert PXD_ACCESSION in content
    assert "58 pediatric" in content
    assert "27 remission" in content
    assert "31 active" in content


def test_download_with_retry_success(tmp_path):
    """Successful download writes file."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [b"data"]
    mock_resp.raise_for_status.return_value = None

    with patch.object(_mod.requests, "get", return_value=mock_resp):
        dest = tmp_path / "file.json"
        assert _mod._download_with_retry("https://example.com/f", dest) is True
        assert dest.exists()


def test_download_with_retry_404(tmp_path):
    """404 returns False."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch.object(_mod.requests, "get", return_value=mock_resp) as mock_get:
        dest = tmp_path / "missing.json"
        assert _mod._download_with_retry("https://example.com/x", dest) is False
        assert mock_get.call_count == 1


def test_fetch_metabolights_metadata(tmp_path):
    """MetaboLights metadata fetch creates directory and files."""
    mock_json_data = {"study": {"identifier": "MTBLS9877"}}

    with (
        patch.object(
            _mod, "_fetch_json_with_retry", return_value=mock_json_data
        ),
        patch.object(_mod, "_download_with_retry", return_value=True),
    ):
        result = _mod.fetch_metabolights_metadata(tmp_path)
        assert result is True
        mtbls_dir = tmp_path / "metabolomics"
        assert mtbls_dir.is_dir()
        assert (mtbls_dir / "study_descriptor.json").exists()
        descriptor = json.loads(
            (mtbls_dir / "study_descriptor.json").read_text()
        )
        assert descriptor["study"]["identifier"] == "MTBLS9877"


def test_fetch_ena_run_info(tmp_path):
    """ENA run info fetched and saved."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.text = (
        "run_accession\texperiment_accession\n"
        "ERR1234567\tERX1234567\n"
        "ERR1234568\tERX1234568\n"
    )
    mock_resp.raise_for_status.return_value = None

    with patch.object(_mod.requests, "get", return_value=mock_resp):
        result = _mod.fetch_ena_run_info(tmp_path)
        assert result is True
        runinfo = tmp_path / "amplicon" / "ena_runinfo.tsv"
        assert runinfo.exists()
        content = runinfo.read_text()
        assert "ERR1234567" in content


def test_fetch_ena_run_info_existing(tmp_path):
    """Skip if file already exists."""
    amp_dir = tmp_path / "amplicon"
    amp_dir.mkdir()
    (amp_dir / "ena_runinfo.tsv").write_text("existing")

    with patch.object(_mod.requests, "get") as mock_get:
        result = _mod.fetch_ena_run_info(tmp_path)
        assert result is True
        assert mock_get.call_count == 0


def test_fetch_pride_metadata(tmp_path):
    """PRIDE metadata fetch creates directory and files."""
    responses = [
        {"accession": "PXD062519", "title": "Pediatric CD proteomics"},
        [{"fileName": "sample1.raw"}, {"fileName": "sample2.raw"}],
    ]

    with patch.object(
        _mod, "_fetch_json_with_retry", side_effect=responses
    ):
        result = _mod.fetch_pride_metadata(tmp_path)
        assert result is True
        prot_dir = tmp_path / "proteomics"
        assert prot_dir.is_dir()
        assert (prot_dir / "pride_project.json").exists()
        assert (prot_dir / "pride_files.json").exists()
