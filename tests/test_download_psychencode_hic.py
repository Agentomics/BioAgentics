"""Tests for PsychENCODE brain Hi-C download script."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from bioagentics.tourettes.ts_gwas_functional_annotation.download_psychencode_hic import (
    HIC_DATA_DIR,
    PROMOTER_WINDOW_BP,
    RESOURCE_URLS,
    _convert_cross_assembly_to_bedpe,
    _convert_simple_to_bedpe,
    _write_checksums,
    _write_readme,
    compute_md5,
    download_psychencode_hic,
    verify_checksums,
)


# --- Fixtures ---


@pytest.fixture
def cross_assembly_csv(tmp_path: Path) -> Path:
    """Create a fake INT-16 cross-assembly CSV."""
    data = (
        "Enhancer_Chromosome_hg19,Enhancer_Start_hg19,Enhancer_End_hg19,"
        "PEC_Enhancer_ID,Enhancer_Chromosome_hg38,Enhancer_Start_hg38,"
        "Enhancer_End_hg38,Transcription_Start_Site_hg19,"
        "Target_Gene_Name,Target_Ensembl_Name\n"
        "chr1,1000,2000,PEC_E001,chr1,1100,2100,50000,BRCA1,ENSG00000001\n"
        "chr2,3000,4000,PEC_E002,chr2,3100,4100,60000,TP53,ENSG00000002\n"
        "chr3,5000,6000,PEC_E003,chr3,5100,6100,500,,ENSG00000003\n"
        "chrX,7000,8000,PEC_E004,chrX,7100,8100,90000,XIST,ENSG00000004\n"
    )
    csv_path = tmp_path / "INT-16_HiC_EP_linkages_cross_assembly.csv"
    csv_path.write_text(data)
    return csv_path


@pytest.fixture
def simple_csv(tmp_path: Path) -> Path:
    """Create a fake INT-16 simple CSV."""
    data = (
        "Chromosome,Transcription_Start_Site,Target_Gene_Name,"
        "Target_Ensembl_Name,Enhancer_Start,Enhancer_End\n"
        "chr1,50000,BRCA1,ENSG00000001,1000,2000\n"
        "chr2,60000,TP53,ENSG00000002,3000,4000\n"
        "chr22,70000,BCR,ENSG00000003,5000,6000\n"
    )
    csv_path = tmp_path / "INT-16_HiC_EP_linkages.csv"
    csv_path.write_text(data)
    return csv_path


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


# --- _convert_cross_assembly_to_bedpe ---


def test_convert_cross_assembly_basic(cross_assembly_csv: Path, tmp_path: Path) -> None:
    out = _convert_cross_assembly_to_bedpe(cross_assembly_csv, tmp_path)
    assert out.exists()
    assert out.name == "psychencode_hic_ep_linkages_hg19.tsv"

    df = pd.read_csv(out, sep="\t")
    assert len(df) == 3  # chrX is excluded (non-autosomal)
    assert set(df.columns) == {
        "CHR1", "START1", "END1", "CHR2", "START2", "END2", "TISSUE", "GENE",
    }


def test_convert_cross_assembly_coordinates(cross_assembly_csv: Path, tmp_path: Path) -> None:
    out = _convert_cross_assembly_to_bedpe(cross_assembly_csv, tmp_path)
    df = pd.read_csv(out, sep="\t")

    # First row: chr1, enhancer 1000-2000, TSS at 50000
    row = df[df["CHR1"] == 1].iloc[0]
    assert row["START1"] == 1000
    assert row["END1"] == 2000
    assert row["START2"] == 50000 - PROMOTER_WINDOW_BP
    assert row["END2"] == 50000 + PROMOTER_WINDOW_BP
    assert row["TISSUE"] == "DLPFC"
    assert row["GENE"] == "BRCA1"


def test_convert_cross_assembly_tss_near_zero(tmp_path: Path) -> None:
    """TSS - PROMOTER_WINDOW should not go below 0."""
    data = (
        "Enhancer_Chromosome_hg19,Enhancer_Start_hg19,Enhancer_End_hg19,"
        "PEC_Enhancer_ID,Enhancer_Chromosome_hg38,Enhancer_Start_hg38,"
        "Enhancer_End_hg38,Transcription_Start_Site_hg19,"
        "Target_Gene_Name,Target_Ensembl_Name\n"
        "chr1,100,200,PEC_E001,chr1,100,200,500,GENE1,ENSG00000001\n"
    )
    csv_path = tmp_path / "cross.csv"
    csv_path.write_text(data)

    out = _convert_cross_assembly_to_bedpe(csv_path, tmp_path)
    df = pd.read_csv(out, sep="\t")
    assert df.iloc[0]["START2"] == 0  # 500 - 2000 = -1500, clamped to 0


def test_convert_cross_assembly_filters_sex_chromosomes(
    cross_assembly_csv: Path, tmp_path: Path,
) -> None:
    out = _convert_cross_assembly_to_bedpe(cross_assembly_csv, tmp_path)
    df = pd.read_csv(out, sep="\t")
    # chrX row should be excluded
    assert (df["CHR1"] <= 22).all()
    assert len(df) == 3


def test_convert_cross_assembly_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("col1,col2\n1,2\n")
    with pytest.raises(ValueError, match="missing columns"):
        _convert_cross_assembly_to_bedpe(csv_path, tmp_path)


def test_convert_cross_assembly_nan_gene(tmp_path: Path) -> None:
    """Rows with NaN gene names should get empty string."""
    data = (
        "Enhancer_Chromosome_hg19,Enhancer_Start_hg19,Enhancer_End_hg19,"
        "PEC_Enhancer_ID,Enhancer_Chromosome_hg38,Enhancer_Start_hg38,"
        "Enhancer_End_hg38,Transcription_Start_Site_hg19,"
        "Target_Gene_Name,Target_Ensembl_Name\n"
        "chr1,1000,2000,PEC_E001,chr1,1100,2100,50000,,ENSG00000001\n"
    )
    csv_path = tmp_path / "nan_gene.csv"
    csv_path.write_text(data)

    out = _convert_cross_assembly_to_bedpe(csv_path, tmp_path)
    df = pd.read_csv(out, sep="\t")
    # Empty gene names are written as empty strings, read back as NaN by pandas
    assert pd.isna(df.iloc[0]["GENE"]) or df.iloc[0]["GENE"] == ""


# --- _convert_simple_to_bedpe ---


def test_convert_simple_basic(simple_csv: Path, tmp_path: Path) -> None:
    out = _convert_simple_to_bedpe(simple_csv, tmp_path)
    assert out.exists()
    assert out.name == "psychencode_hic_ep_linkages.tsv"

    df = pd.read_csv(out, sep="\t")
    assert len(df) == 3
    assert set(df.columns) == {
        "CHR1", "START1", "END1", "CHR2", "START2", "END2", "TISSUE", "GENE",
    }


def test_convert_simple_coordinates(simple_csv: Path, tmp_path: Path) -> None:
    out = _convert_simple_to_bedpe(simple_csv, tmp_path)
    df = pd.read_csv(out, sep="\t")

    row = df[df["CHR1"] == 1].iloc[0]
    assert row["START1"] == 1000
    assert row["END1"] == 2000
    assert row["START2"] == 50000 - PROMOTER_WINDOW_BP
    assert row["END2"] == 50000 + PROMOTER_WINDOW_BP
    assert row["GENE"] == "BRCA1"


def test_convert_simple_missing_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("col1,col2\n1,2\n")
    with pytest.raises(ValueError, match="missing columns"):
        _convert_simple_to_bedpe(csv_path, tmp_path)


# --- _write_checksums ---


def test_write_checksums(tmp_path: Path) -> None:
    f1 = tmp_path / "file1.tsv"
    f2 = tmp_path / "file2.tsv"
    f1.write_bytes(b"data1")
    f2.write_bytes(b"data2")

    checksums = _write_checksums([f1, f2], tmp_path)

    assert len(checksums) == 2
    assert checksums["file1.tsv"] == hashlib.md5(b"data1").hexdigest()
    assert checksums["file2.tsv"] == hashlib.md5(b"data2").hexdigest()

    md5_file = tmp_path / "md5sums.txt"
    assert md5_file.exists()
    lines = md5_file.read_text().strip().split("\n")
    assert len(lines) == 2


# --- verify_checksums ---


def test_verify_checksums_pass(tmp_path: Path) -> None:
    f = tmp_path / "test.tsv"
    f.write_bytes(b"test data")
    md5 = hashlib.md5(b"test data").hexdigest()
    (tmp_path / "md5sums.txt").write_text(f"{md5}  test.tsv\n")
    assert verify_checksums(tmp_path) is True


def test_verify_checksums_mismatch(tmp_path: Path) -> None:
    f = tmp_path / "test.tsv"
    f.write_bytes(b"test data")
    (tmp_path / "md5sums.txt").write_text("0000000000000000  test.tsv\n")
    assert verify_checksums(tmp_path) is False


def test_verify_checksums_missing_file(tmp_path: Path) -> None:
    (tmp_path / "md5sums.txt").write_text("abc123  missing.tsv\n")
    assert verify_checksums(tmp_path) is False


def test_verify_checksums_no_checksum_file(tmp_path: Path) -> None:
    assert verify_checksums(tmp_path) is False


# --- _write_readme ---


def test_write_readme(tmp_path: Path) -> None:
    f = tmp_path / "psychencode_hic_ep_linkages_hg19.tsv"
    f.write_text("CHR1\tSTART1\n")
    checksums = {"psychencode_hic_ep_linkages_hg19.tsv": "abc123"}
    _write_readme(tmp_path, checksums, [f])

    readme = tmp_path / "README.md"
    assert readme.exists()
    content = readme.read_text()
    assert "PsychENCODE" in content
    assert "DLPFC" in content
    assert "abc123" in content
    assert "10.1126/science.aat8464" in content


# --- download_psychencode_hic ---


def test_download_verify_only_no_files(tmp_path: Path) -> None:
    result = download_psychencode_hic(dest_dir=tmp_path, verify_only=True)
    assert result == []


def test_download_verify_only_with_files(tmp_path: Path) -> None:
    f = tmp_path / "psychencode_hic_ep_linkages_hg19.tsv"
    f.write_bytes(b"data")
    md5 = hashlib.md5(b"data").hexdigest()
    (tmp_path / "md5sums.txt").write_text(f"{md5}  {f.name}\n")

    result = download_psychencode_hic(dest_dir=tmp_path, verify_only=True)
    assert len(result) == 1


def test_download_skips_existing(tmp_path: Path) -> None:
    f = tmp_path / "psychencode_hic_ep_linkages_hg19.tsv"
    f.write_text("CHR1\tSTART1\tEND1\n")

    result = download_psychencode_hic(dest_dir=tmp_path)
    assert len(result) == 1
    assert result[0] == f


# --- Constants ---


def test_hic_data_dir() -> None:
    assert HIC_DATA_DIR.name == "hic_data"
    assert "ts-gwas-functional-annotation" in str(HIC_DATA_DIR)


def test_promoter_window() -> None:
    assert PROMOTER_WINDOW_BP == 2000


def test_resource_urls() -> None:
    assert "INT-16_HiC_EP_linkages_cross_assembly.csv" in RESOURCE_URLS
    assert "INT-16_HiC_EP_linkages.csv" in RESOURCE_URLS
    for urls in RESOURCE_URLS.values():
        assert len(urls) >= 2  # At least 2 mirror URLs
