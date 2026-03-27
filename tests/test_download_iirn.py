"""Tests for IIRN metabolomics download script."""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

_mod = importlib.import_module(
    "crohns.microbiome_metabolome_subtyping.01_download_iirn"
)

BIOPROJECT_16S = _mod.BIOPROJECT_16S
BIOPROJECT_SHOTGUN = _mod.BIOPROJECT_SHOTGUN
OUTPUT_DIR = _mod.OUTPUT_DIR


def test_bioproject_accessions():
    """BioProject accessions are valid NCBI format."""
    assert BIOPROJECT_16S.startswith("PRJNA")
    assert BIOPROJECT_SHOTGUN.startswith("PRJNA")
    assert BIOPROJECT_16S == "PRJNA1053872"
    assert BIOPROJECT_SHOTGUN == "PRJNA1057679"


def test_output_dir_path():
    """Output directory is under data/crohns/microbiome-metabolome-subtyping/."""
    parts = OUTPUT_DIR.parts
    assert "data" in parts
    assert "crohns" in parts
    assert "microbiome-metabolome-subtyping" in parts
    assert parts[-1] == "iirn_metabolomics"


def test_save_download_manifest(tmp_path):
    """Manifest file is created with correct content."""
    _mod._save_download_manifest(tmp_path)
    manifest = tmp_path / "MANIFEST.md"
    assert manifest.exists()
    content = manifest.read_text()
    assert "IIRN" in content
    assert BIOPROJECT_16S in content
    assert BIOPROJECT_SHOTGUN in content
    assert "80 Crohn" in content
    assert "43 healthy" in content


def test_download_with_retry_success(tmp_path):
    """Successful download writes file."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_content.return_value = [b"test data"]
    mock_resp.raise_for_status.return_value = None

    with patch.object(_mod.requests, "get", return_value=mock_resp) as mock_get:
        dest = tmp_path / "test_file.tsv"
        result = _mod._download_with_retry("https://example.com/file", dest)
        assert result is True
        assert dest.exists()
        assert dest.read_bytes() == b"test data"


def test_download_with_retry_404(tmp_path):
    """404 response returns False without retrying."""
    mock_resp = MagicMock()
    mock_resp.status_code = 404

    with patch.object(_mod.requests, "get", return_value=mock_resp) as mock_get:
        dest = tmp_path / "missing.tsv"
        result = _mod._download_with_retry("https://example.com/missing", dest)
        assert result is False
        assert not dest.exists()
        assert mock_get.call_count == 1


def test_fetch_sra_runinfo_existing(tmp_path):
    """Skip download if file already exists."""
    dest = tmp_path / "runinfo.tsv"
    dest.write_text("existing data")

    with patch.object(_mod.requests, "get") as mock_get:
        result = _mod._fetch_sra_runinfo("PRJNA000000", dest)
        assert result is True
        assert mock_get.call_count == 0


def test_fetch_sra_runinfo_via_ena(tmp_path):
    """Fetches run info from ENA API."""
    entrez_resp = MagicMock()
    entrez_resp.status_code = 200
    entrez_resp.json.return_value = {
        "esearchresult": {"count": "5", "idlist": ["1", "2", "3", "4", "5"]}
    }
    entrez_resp.raise_for_status.return_value = None

    ena_resp = MagicMock()
    ena_resp.status_code = 200
    ena_resp.text = (
        "run_accession\texperiment_accession\tsample_accession\n"
        "SRR1234567\tSRX1234567\tSRS1234567\n"
        "SRR1234568\tSRX1234568\tSRS1234568\n"
    )
    ena_resp.raise_for_status.return_value = None

    with patch.object(_mod.requests, "get", side_effect=[entrez_resp, ena_resp]):
        dest = tmp_path / "runinfo.tsv"
        result = _mod._fetch_sra_runinfo("PRJNA1053872", dest)
        assert result is True
        assert dest.exists()
        content = dest.read_text()
        assert "SRR1234567" in content
        assert "SRR1234568" in content
