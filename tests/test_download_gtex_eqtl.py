"""Tests for GTEx v8 brain eQTL download script."""

from __future__ import annotations

import gzip
import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from bioagentics.tourettes.ts_gwas_functional_annotation.download_gtex_eqtl import (
    BRAIN_REGIONS,
    FILE_SUFFIX,
    _extract_brain_eqtls,
    _write_checksums,
    _write_readme,
    compute_md5,
    download_gtex_v8_eqtl,
    verify_checksums,
)


# --- Fixtures ---


@pytest.fixture
def sample_eqtl_content() -> bytes:
    """Minimal tab-separated eQTL data (gzipped)."""
    tsv = (
        "variant_id\tgene_id\ttss_distance\tma_samples\tma_count\tmaf\t"
        "pval_nominal\tslope\tslope_se\tpval_nominal_threshold\t"
        "min_pval_nominal\tpval_beta\n"
        "chr1_100_A_G_b38\tENSG00000001.1\t500\t10\t15\t0.05\t"
        "1e-6\t0.3\t0.05\t1e-4\t1e-7\t1e-5\n"
    )
    buf = io.BytesIO()
    with gzip.open(buf, "wb") as gz:
        gz.write(tsv.encode())
    return buf.getvalue()


@pytest.fixture
def fake_tar(tmp_path: Path, sample_eqtl_content: bytes) -> Path:
    """Create a fake GTEx eQTL tar with brain region files."""
    tar_path = tmp_path / "GTEx_Analysis_v8_eQTL.tar"

    with tarfile.open(tar_path, "w") as tar:
        for region in BRAIN_REGIONS:
            filename = f"GTEx_Analysis_v8_eQTL/{region}{FILE_SUFFIX}"
            info = tarfile.TarInfo(name=filename)
            info.size = len(sample_eqtl_content)
            tar.addfile(info, io.BytesIO(sample_eqtl_content))

        # Add a non-brain tissue file that should be ignored
        extra = f"GTEx_Analysis_v8_eQTL/Liver{FILE_SUFFIX}"
        info = tarfile.TarInfo(name=extra)
        info.size = len(sample_eqtl_content)
        tar.addfile(info, io.BytesIO(sample_eqtl_content))

    return tar_path


# --- compute_md5 ---


def test_compute_md5(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_bytes(b"hello world")
    expected = hashlib.md5(b"hello world").hexdigest()
    assert compute_md5(f) == expected


def test_compute_md5_empty(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_bytes(b"")
    expected = hashlib.md5(b"").hexdigest()
    assert compute_md5(f) == expected


# --- _extract_brain_eqtls ---


def test_extract_brain_eqtls(fake_tar: Path, tmp_path: Path) -> None:
    dest = tmp_path / "output"
    extracted = _extract_brain_eqtls(fake_tar, dest)

    assert len(extracted) == len(BRAIN_REGIONS)
    for region in BRAIN_REGIONS:
        expected_file = dest / f"{region}{FILE_SUFFIX}"
        assert expected_file.exists()
        assert expected_file.stat().st_size > 0

    # Liver file should NOT be extracted
    assert not (dest / f"Liver{FILE_SUFFIX}").exists()


def test_extract_brain_eqtls_flat_tar(
    tmp_path: Path, sample_eqtl_content: bytes,
) -> None:
    """Files directly in tar root (no subdirectory) are also extracted."""
    tar_path = tmp_path / "flat.tar"
    with tarfile.open(tar_path, "w") as tar:
        filename = f"{BRAIN_REGIONS[0]}{FILE_SUFFIX}"
        info = tarfile.TarInfo(name=filename)
        info.size = len(sample_eqtl_content)
        tar.addfile(info, io.BytesIO(sample_eqtl_content))

    dest = tmp_path / "output"
    extracted = _extract_brain_eqtls(tar_path, dest)
    assert len(extracted) == 1
    assert extracted[0].name == f"{BRAIN_REGIONS[0]}{FILE_SUFFIX}"


def test_extract_brain_eqtls_missing_regions(
    tmp_path: Path, sample_eqtl_content: bytes,
) -> None:
    """Only available regions are extracted; missing ones logged as warning."""
    tar_path = tmp_path / "partial.tar"
    with tarfile.open(tar_path, "w") as tar:
        filename = f"GTEx_Analysis_v8_eQTL/{BRAIN_REGIONS[0]}{FILE_SUFFIX}"
        info = tarfile.TarInfo(name=filename)
        info.size = len(sample_eqtl_content)
        tar.addfile(info, io.BytesIO(sample_eqtl_content))

    dest = tmp_path / "output"
    extracted = _extract_brain_eqtls(tar_path, dest)
    assert len(extracted) == 1


# --- _write_checksums ---


def test_write_checksums(tmp_path: Path) -> None:
    f1 = tmp_path / "file1.txt.gz"
    f2 = tmp_path / "file2.txt.gz"
    f1.write_bytes(b"data1")
    f2.write_bytes(b"data2")

    checksums = _write_checksums([f1, f2], tmp_path)

    assert len(checksums) == 2
    assert checksums["file1.txt.gz"] == hashlib.md5(b"data1").hexdigest()
    assert checksums["file2.txt.gz"] == hashlib.md5(b"data2").hexdigest()

    md5_file = tmp_path / "md5sums.txt"
    assert md5_file.exists()
    lines = md5_file.read_text().strip().split("\n")
    assert len(lines) == 2


# --- verify_checksums ---


def test_verify_checksums_pass(tmp_path: Path) -> None:
    f = tmp_path / "test.txt.gz"
    f.write_bytes(b"test data")
    md5 = hashlib.md5(b"test data").hexdigest()

    checksum_file = tmp_path / "md5sums.txt"
    checksum_file.write_text(f"{md5}  test.txt.gz\n")

    assert verify_checksums(tmp_path) is True


def test_verify_checksums_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "test.txt.gz"
    f.write_bytes(b"test data")

    checksum_file = tmp_path / "md5sums.txt"
    checksum_file.write_text("0000000000000000  test.txt.gz\n")

    assert verify_checksums(tmp_path) is False


def test_verify_checksums_missing_file(tmp_path: Path) -> None:
    checksum_file = tmp_path / "md5sums.txt"
    checksum_file.write_text("abc123  missing.txt.gz\n")

    assert verify_checksums(tmp_path) is False


def test_verify_checksums_no_checksum_file(tmp_path: Path) -> None:
    assert verify_checksums(tmp_path) is False


# --- _write_readme ---


def test_write_readme(tmp_path: Path) -> None:
    checksums = {
        f"{BRAIN_REGIONS[0]}{FILE_SUFFIX}": "abc123",
        f"{BRAIN_REGIONS[1]}{FILE_SUFFIX}": "def456",
    }
    _write_readme(tmp_path, checksums)

    readme = tmp_path / "README.md"
    assert readme.exists()
    content = readme.read_text()
    assert "GTEx v8" in content
    assert "Brain Caudate basal ganglia" in content
    assert "abc123" in content
    assert "10.1126/science.aaz1776" in content


# --- download_gtex_v8_eqtl (verify_only mode) ---


def test_download_verify_only_no_files(tmp_path: Path) -> None:
    result = download_gtex_v8_eqtl(dest_dir=tmp_path, verify_only=True)
    assert result == []


def test_download_verify_only_with_files(tmp_path: Path) -> None:
    f = tmp_path / f"{BRAIN_REGIONS[0]}{FILE_SUFFIX}"
    f.write_bytes(b"data")
    md5 = hashlib.md5(b"data").hexdigest()
    (tmp_path / "md5sums.txt").write_text(f"{md5}  {f.name}\n")

    result = download_gtex_v8_eqtl(dest_dir=tmp_path, verify_only=True)
    assert len(result) == 1


# --- download_gtex_v8_eqtl (skip existing) ---


def test_download_skips_existing(tmp_path: Path) -> None:
    for region in BRAIN_REGIONS:
        (tmp_path / f"{region}{FILE_SUFFIX}").write_bytes(b"data")

    result = download_gtex_v8_eqtl(dest_dir=tmp_path)
    assert len(result) == len(BRAIN_REGIONS)


# --- Constants ---


def test_brain_regions_count() -> None:
    assert len(BRAIN_REGIONS) == 5


def test_file_suffix() -> None:
    assert FILE_SUFFIX == ".v8.signif_variant_gene_pairs.txt.gz"


def test_brain_region_names() -> None:
    expected = {
        "Brain_Caudate_basal_ganglia",
        "Brain_Putamen_basal_ganglia",
        "Brain_Nucleus_accumbens_basal_ganglia",
        "Brain_Frontal_Cortex_BA9",
        "Brain_Cerebellum",
    }
    assert set(BRAIN_REGIONS) == expected
