"""Tests for LDSC reference data download module."""

from __future__ import annotations

import tarfile
from unittest.mock import patch

import pytest

from tourettes.ts_comorbidity_genetic_architecture.download_ld_reference import (
    BASE_URL,
    DATA_DIR,
    DOWNLOADS,
    _extract_archive,
    _md5sum,
    _write_manifest,
    download_all,
)


class TestDownloadConfig:
    """Validate download configuration."""

    def test_base_url_points_to_broad(self):
        assert "broadinstitute.org" in BASE_URL

    def test_five_datasets_defined(self):
        assert len(DOWNLOADS) == 5

    def test_all_entries_have_required_keys(self):
        required = {"name", "url", "filename", "extract_dir", "description"}
        for item in DOWNLOADS:
            assert required.issubset(item.keys()), f"Missing keys in {item['name']}"

    def test_all_urls_start_with_base(self):
        for item in DOWNLOADS:
            assert item["url"].startswith(BASE_URL), f"Bad URL for {item['name']}"

    def test_filenames_are_archives(self):
        for item in DOWNLOADS:
            assert item["filename"].endswith((".tar.bz2", ".tgz", ".tar.gz")), (
                f"Not an archive: {item['filename']}"
            )

    def test_data_dir_is_correct(self):
        assert "ts-comorbidity-genetic-architecture" in str(DATA_DIR)
        assert "reference" in str(DATA_DIR)

    def test_dataset_names(self):
        names = {d["name"] for d in DOWNLOADS}
        assert "EUR LD scores" in names
        assert "1000G Phase 3 plink files" in names
        assert "Baseline-LD annotations" in names
        assert "HM3 regression weights (no HLA)" in names
        assert "HapMap3 SNPs" in names


class TestMd5Sum:
    """Test MD5 checksum computation."""

    def test_md5_of_known_content(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world\n")
        md5 = _md5sum(f)
        assert len(md5) == 32
        assert md5 == "6f5902ac237024bdd0c176cb93063dc4"

    def test_md5_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        md5 = _md5sum(f)
        assert len(md5) == 32
        assert md5 == "d41d8cd98f00b204e9800998ecf8427e"


class TestExtractArchive:
    """Test archive extraction."""

    def test_extract_tgz(self, tmp_path):
        # Create a test .tgz archive
        archive = tmp_path / "test.tgz"
        content_file = tmp_path / "content.txt"
        content_file.write_text("test data")
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(content_file, arcname="test_dir/content.txt")
        content_file.unlink()

        dest = tmp_path / "extracted"
        dest.mkdir()
        _extract_archive(archive, dest)
        assert (dest / "test_dir" / "content.txt").exists()
        assert (dest / "test_dir" / "content.txt").read_text() == "test data"

    def test_extract_tar_bz2(self, tmp_path):
        archive = tmp_path / "test.tar.bz2"
        content_file = tmp_path / "data.csv"
        content_file.write_text("col1,col2\n1,2\n")
        with tarfile.open(archive, "w:bz2") as tf:
            tf.add(content_file, arcname="bz2_dir/data.csv")
        content_file.unlink()

        dest = tmp_path / "extracted"
        dest.mkdir()
        _extract_archive(archive, dest)
        assert (dest / "bz2_dir" / "data.csv").exists()

    def test_unknown_format_raises(self, tmp_path):
        archive = tmp_path / "test.zip"
        archive.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unknown archive format"):
            _extract_archive(archive, tmp_path)


class TestWriteManifest:
    """Test manifest generation."""

    def test_manifest_created(self, tmp_path):
        results = {"EUR LD scores": tmp_path / "eur_w_ld_chr"}
        _write_manifest(tmp_path, results)
        manifest = tmp_path / "README.txt"
        assert manifest.exists()
        content = manifest.read_text()
        assert "hg19" in content
        assert "GRCh37" in content
        assert "Broad Institute" in content

    def test_manifest_lists_all_datasets(self, tmp_path):
        results = {d["name"]: tmp_path / d["extract_dir"] for d in DOWNLOADS}
        _write_manifest(tmp_path, results)
        content = (tmp_path / "README.txt").read_text()
        for d in DOWNLOADS:
            assert d["name"] in content
            assert d["filename"] in content


class TestDownloadAll:
    """Test download_all with mocked network."""

    @patch("tourettes.ts_comorbidity_genetic_architecture.download_ld_reference._download_file")
    def test_skip_existing(self, mock_dl, tmp_path):
        # Pre-create all extracted dirs so everything is skipped
        for d in DOWNLOADS:
            (tmp_path / d["extract_dir"]).mkdir()

        results = download_all(data_dir=tmp_path, skip_existing=True)
        mock_dl.assert_not_called()
        assert len(results) == 5

    @patch("tourettes.ts_comorbidity_genetic_architecture.download_ld_reference._download_file")
    def test_downloads_missing(self, mock_dl, tmp_path):
        # Create archives that _download_file would produce
        def fake_download(url, dest):
            # Create a minimal tgz with the expected extract dir
            matching = [d for d in DOWNLOADS if d["url"] == url]
            if not matching:
                return
            item = matching[0]
            with tarfile.open(dest, "w:bz2" if dest.name.endswith(".bz2") else "w:gz") as tf:
                # Add a dummy file inside expected dir
                import io
                data = b"dummy"
                info = tarfile.TarInfo(name=f"{item['extract_dir']}/dummy.txt")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

        mock_dl.side_effect = fake_download

        results = download_all(data_dir=tmp_path, skip_existing=True)
        assert mock_dl.call_count == 5
        assert len(results) == 5
        # Check manifest was written
        assert (tmp_path / "README.txt").exists()

    @patch("tourettes.ts_comorbidity_genetic_architecture.download_ld_reference._download_file")
    def test_archives_cleaned_up(self, mock_dl, tmp_path):
        def fake_download(url, dest):
            matching = [d for d in DOWNLOADS if d["url"] == url]
            item = matching[0]
            import io
            with tarfile.open(dest, "w:bz2" if dest.name.endswith(".bz2") else "w:gz") as tf:
                data = b"x"
                info = tarfile.TarInfo(name=f"{item['extract_dir']}/f.txt")
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))

        mock_dl.side_effect = fake_download
        download_all(data_dir=tmp_path, skip_existing=False)

        # Archives should be removed after extraction
        for d in DOWNLOADS:
            assert not (tmp_path / d["filename"]).exists(), (
                f"Archive {d['filename']} should have been removed"
            )
