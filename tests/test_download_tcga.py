"""Tests for the GDC/TCGA download script."""

from pathlib import Path
from unittest.mock import MagicMock, patch

from bioagentics.data.download_tcga import (
    DATA_TYPES,
    _build_filter,
    _format_size,
    md5_file,
    save_manifest,
)


def test_build_filter_basic():
    filt = _build_filter("TCGA-BRCA", "Transcriptome Profiling", "Gene Expression Quantification")
    assert filt["op"] == "and"
    assert len(filt["content"]) == 4  # project, category, type, access


def test_build_filter_with_workflow():
    filt = _build_filter(
        "TCGA-BRCA",
        "Transcriptome Profiling",
        "Gene Expression Quantification",
        workflow_type="STAR - Counts",
    )
    assert len(filt["content"]) == 5


def test_data_types_configured():
    assert "expression" in DATA_TYPES
    assert "mutations" in DATA_TYPES
    assert "copy_number" in DATA_TYPES


def test_md5_file(tmp_path: Path):
    p = tmp_path / "test.txt"
    p.write_text("hello")
    digest = md5_file(p)
    assert len(digest) == 32
    # Known MD5 of "hello"
    assert digest == "5d41402abc4b2a76b9719d911017c592"


def test_save_manifest(tmp_path: Path):
    files = [
        {"file_id": "abc-123", "file_name": "data.tsv", "md5sum": "deadbeef", "file_size": 1024},
        {"file_id": "def-456", "file_name": "data2.tsv", "md5sum": "cafebabe", "file_size": 2048},
    ]
    manifest_path = tmp_path / "manifest.tsv"
    save_manifest(files, manifest_path)

    lines = manifest_path.read_text().strip().split("\n")
    assert lines[0] == "id\tfilename\tmd5\tsize\tstate"
    assert len(lines) == 3
    assert "abc-123" in lines[1]
    assert "def-456" in lines[2]


def test_format_size():
    assert "B" in _format_size(500)
    assert "KB" in _format_size(5_000)
    assert "MB" in _format_size(5_000_000)
    assert "GB" in _format_size(5_000_000_000)


@patch("bioagentics.data.download_tcga.query_files")
@patch("bioagentics.data.download_tcga.query_clinical_cases")
def test_main_manifest_only(mock_clinical: MagicMock, mock_query: MagicMock, tmp_path: Path):
    mock_query.return_value = [
        {"file_id": "a", "file_name": "f.tsv", "md5sum": "m", "file_size": 100},
    ]
    mock_clinical.return_value = [{"case_id": "c1"}]

    from bioagentics.data.download_tcga import main

    main(["--manifest-only", "--dest", str(tmp_path)])

    # Should create manifests but not download files
    brca_dir = tmp_path / "brca"
    assert brca_dir.exists()
