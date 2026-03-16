"""Tests for the DepMap download script."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from bioagentics.data.download_depmap import (
    DEFAULT_FILES,
    PRISM_FILES,
    download_from_url,
    main,
    sha256_file,
)


def test_default_files_not_empty():
    assert len(DEFAULT_FILES) >= 6


def test_prism_files_not_empty():
    assert len(PRISM_FILES) >= 1


def test_sha256_file(tmp_path: Path):
    p = tmp_path / "test.txt"
    p.write_text("hello")
    digest = sha256_file(p)
    assert len(digest) == 64
    # Known SHA-256 of "hello"
    assert digest == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_download_from_url_skips_existing(tmp_path: Path):
    existing = tmp_path / "Model.csv"
    existing.write_text("data")

    result = download_from_url("https://example.com/Model.csv", "Model.csv", tmp_path)
    assert result == existing
    # File content unchanged (download was skipped)
    assert existing.read_text() == "data"


@patch("bioagentics.data.download_depmap.requests.get")
def test_download_from_url_downloads(mock_get: MagicMock, tmp_path: Path):
    mock_resp = MagicMock()
    mock_resp.headers = {"content-length": "5"}
    mock_resp.iter_content.return_value = [b"hello"]
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_get.return_value = mock_resp

    dest = download_from_url("https://example.com/Model.csv", "Model.csv", tmp_path, force=True)
    assert dest.read_bytes() == b"hello"


def test_main_list_flag(capsys):
    """--list should call the API and exit without downloading."""
    with patch("bioagentics.data.download_depmap.list_release_files") as mock_list:
        mock_list.return_value = [
            {"fileName": "Model.csv", "fileSize": "1MB"},
        ]
        main(["--list"])
    out = capsys.readouterr().out
    assert "Model.csv" in out
