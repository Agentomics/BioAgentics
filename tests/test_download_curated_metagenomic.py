"""Tests for curatedMetagenomicData download wrapper."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_mod = importlib.import_module(
    "crohns.microbiome_metabolome_subtyping.03_download_curated_metagenomic"
)

DEFAULT_STUDIES = _mod.DEFAULT_STUDIES
R_SCRIPT = _mod.R_SCRIPT


def test_default_studies():
    """Default study list contains the 3 target cohorts."""
    assert len(DEFAULT_STUDIES) == 3
    assert "NielsenHB_2014" in DEFAULT_STUDIES
    assert "HallAB_2017" in DEFAULT_STUDIES
    assert "SchirmerM_2016" in DEFAULT_STUDIES


def test_r_script_path():
    """R script path points to export_curated_metagenomic.R in same directory."""
    assert R_SCRIPT.name == "export_curated_metagenomic.R"
    assert R_SCRIPT.parent == Path(_mod.__file__).parent


def test_r_script_exists():
    """The R export script file exists on disk."""
    assert R_SCRIPT.exists(), f"R script not found at {R_SCRIPT}"


def test_check_r_available_success():
    """check_r_available returns 'Rscript' when R is found."""
    mock_result = MagicMock()
    mock_result.stderr = "Rscript (R) version 4.5.3"
    mock_result.stdout = ""

    with patch.object(_mod.subprocess, "run", return_value=mock_result):
        result = _mod.check_r_available()
        assert result == "Rscript"


def test_check_r_available_missing():
    """check_r_available exits when Rscript is not found."""
    with patch.object(
        _mod.subprocess, "run", side_effect=FileNotFoundError
    ):
        with pytest.raises(SystemExit) as exc_info:
            _mod.check_r_available()
        assert exc_info.value.code == 1


def test_export_study_success(tmp_path):
    """Successful R script run creates expected output files."""
    study = "NielsenHB_2014"
    # Pre-create the output files that the R script would produce
    species_file = tmp_path / f"{study}_species.tsv"
    metadata_file = tmp_path / f"{study}_metadata.tsv"
    species_file.write_text("sample_id\ts__Species_A\ts__Species_B\nS1\t0.5\t0.3\n")
    metadata_file.write_text("sample_id\tstudy_condition\nS1\tCD\n")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Fetching NielsenHB_2014...\nDone.\n"
    mock_result.stderr = ""

    with patch.object(_mod.subprocess, "run", return_value=mock_result):
        result = _mod.export_study("Rscript", study, tmp_path)
        assert result is True


def test_export_study_failure(tmp_path):
    """Failed R script returns False."""
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stdout = ""
    mock_result.stderr = "Error in library(curatedMetagenomicData)"

    with patch.object(_mod.subprocess, "run", return_value=mock_result):
        result = _mod.export_study("Rscript", "BadStudy_2099", tmp_path)
        assert result is False


def test_export_study_timeout(tmp_path):
    """Timeout returns False."""
    import subprocess

    with patch.object(
        _mod.subprocess, "run", side_effect=subprocess.TimeoutExpired("Rscript", 600)
    ):
        result = _mod.export_study("Rscript", "SlowStudy_2020", tmp_path)
        assert result is False


def test_export_study_missing_output(tmp_path):
    """Successful exit code but missing output files returns False."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = "Done.\n"
    mock_result.stderr = ""

    with patch.object(_mod.subprocess, "run", return_value=mock_result):
        result = _mod.export_study("Rscript", "NoOutput_2020", tmp_path)
        assert result is False
