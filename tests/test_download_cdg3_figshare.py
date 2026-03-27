"""Tests for CDG3 figshare GWAS download module."""

from __future__ import annotations

import gzip
import json
from unittest.mock import patch

from tourettes.ts_comorbidity_genetic_architecture.download_cdg3_figshare import (
    DATA_DIR,
    DOWNLOADS,
    EXPECTED_FACTORS,
    FIGSHARE_ARTICLE_ID,
    _md5sum,
    _verify_gwas_header,
    _write_manifest,
    download_all,
)


class TestDownloadConfig:
    """Validate download configuration."""

    def test_figshare_article_id(self):
        assert FIGSHARE_ARTICLE_ID == "30359017"

    def test_eight_files_defined(self):
        assert len(DOWNLOADS) == 8

    def test_all_entries_have_required_keys(self):
        required = {"name", "url", "filename", "factor", "description"}
        for item in DOWNLOADS:
            assert required.issubset(item.keys()), f"Missing keys in {item['name']}"

    def test_all_urls_are_figshare(self):
        for item in DOWNLOADS:
            assert "figshare.com" in item["url"], f"Non-figshare URL for {item['name']}"

    def test_five_factor_gwas_files(self):
        factor_files = [d for d in DOWNLOADS if d["factor"] in EXPECTED_FACTORS]
        assert len(factor_files) == 5

    def test_pfactor_included(self):
        pfactor = [d for d in DOWNLOADS if d["factor"] == "pfactor"]
        assert len(pfactor) == 1

    def test_readme_included(self):
        readme = [d for d in DOWNLOADS if "README" in d["name"]]
        assert len(readme) == 1

    def test_hits_included(self):
        hits = [d for d in DOWNLOADS if "Hits" in d["name"]]
        assert len(hits) == 1

    def test_data_dir_is_correct(self):
        assert "ts-comorbidity-genetic-architecture" in str(DATA_DIR)
        assert "grotzinger_cdg3_factor_gwas" in str(DATA_DIR)

    def test_expected_factors(self):
        assert set(EXPECTED_FACTORS) == {
            "compulsive", "sb", "neurodevelopmental", "internalizing", "substance_use"
        }

    def test_gwas_files_are_gzipped_tsv(self):
        gwas_files = [d for d in DOWNLOADS if d["factor"] is not None]
        for item in gwas_files:
            assert item["filename"].endswith(".tsv.gz"), (
                f"GWAS file not gzipped TSV: {item['filename']}"
            )

    def test_unique_filenames(self):
        filenames = [d["filename"] for d in DOWNLOADS]
        assert len(filenames) == len(set(filenames))

    def test_unique_names(self):
        names = [d["name"] for d in DOWNLOADS]
        assert len(names) == len(set(names))


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
        assert md5 == "d41d8cd98f00b204e9800998ecf8427e"


class TestVerifyGwasHeader:
    """Test GWAS header verification."""

    def test_valid_gzipped_tsv(self, tmp_path):
        f = tmp_path / "test.tsv.gz"
        content = "SNP\tA1\tA2\tBETA\tSE\tP\nrs1\tA\tG\t0.01\t0.005\t0.05\n"
        with gzip.open(f, "wt") as gz:
            gz.write(content)
        result = _verify_gwas_header(f)
        assert result["valid"]
        assert "SNP" in result["columns"]
        assert result["n_preview_rows"] == 1

    def test_multiple_rows(self, tmp_path):
        f = tmp_path / "multi.tsv.gz"
        lines = ["CHR\tBP\tSNP\tA1\tA2\tBETA\tSE\tP\n"]
        for i in range(10):
            lines.append(f"1\t{1000+i}\trs{i}\tA\tG\t0.01\t0.005\t0.05\n")
        with gzip.open(f, "wt") as gz:
            gz.write("".join(lines))
        result = _verify_gwas_header(f)
        assert result["valid"]
        assert len(result["columns"]) == 8
        assert result["n_preview_rows"] == 5  # limited to 5 preview rows

    def test_non_gz_file_returns_valid(self, tmp_path):
        f = tmp_path / "readme.pdf"
        f.write_bytes(b"fake pdf")
        result = _verify_gwas_header(f)
        assert result["valid"]
        assert result["columns"] == []

    def test_empty_gz_returns_invalid(self, tmp_path):
        f = tmp_path / "empty.tsv.gz"
        with gzip.open(f, "wt") as gz:
            gz.write("")
        result = _verify_gwas_header(f)
        assert not result["valid"]


class TestWriteManifest:
    """Test manifest JSON generation."""

    def test_manifest_created(self, tmp_path):
        results = {
            "files": {"F1": {"path": str(tmp_path / "f1.tsv.gz"), "md5": "abc"}},
            "verification": {},
            "all_factors_present": True,
        }
        _write_manifest(tmp_path, results)
        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert "Grotzinger" in data["source"]
        assert data["doi_paper"] == "10.1038/s41586-025-09820-3"
        assert data["doi_figshare"] == "10.6084/m9.figshare.30359017"

    def test_manifest_has_factor_descriptions(self, tmp_path):
        results = {"files": {}, "verification": {}, "all_factors_present": True}
        _write_manifest(tmp_path, results)
        data = json.loads((tmp_path / "manifest.json").read_text())
        assert "Compulsive" in data["factors"]["F1"]
        assert "Schizophrenia" in data["factors"]["F2"]
        assert "Neurodevelopmental" in data["factors"]["F3"]
        assert "Internalizing" in data["factors"]["F4"]
        assert "Substance Use" in data["factors"]["F5"]
        assert "PFactor" in data["factors"]


class TestDownloadAll:
    """Test download_all with mocked network."""

    def _make_fake_gwas_gz(self, path, factor_name="test"):
        """Create a minimal gzipped TSV with GWAS-like headers."""
        content = "SNP\tA1\tA2\tBETA\tSE\tP\tN\nrs1\tA\tG\t0.01\t0.005\t0.05\t50000\n"
        with gzip.open(path, "wt") as gz:
            gz.write(content)

    @patch(
        "tourettes.ts_comorbidity_genetic_architecture.download_cdg3_figshare._download_file"
    )
    def test_skip_existing(self, mock_dl, tmp_path):
        # Pre-create all files
        for d in DOWNLOADS:
            dest = tmp_path / d["filename"]
            if d["filename"].endswith(".tsv.gz"):
                self._make_fake_gwas_gz(dest)
            else:
                dest.write_bytes(b"fake content")

        results = download_all(data_dir=tmp_path, skip_existing=True)
        mock_dl.assert_not_called()
        assert len(results["files"]) == 8

    @patch(
        "tourettes.ts_comorbidity_genetic_architecture.download_cdg3_figshare._download_file"
    )
    def test_downloads_missing(self, mock_dl, tmp_path):
        def fake_download(url, dest):
            matching = [d for d in DOWNLOADS if d["url"] == url]
            if not matching:
                return
            item = matching[0]
            if item["filename"].endswith(".tsv.gz"):
                self._make_fake_gwas_gz(dest)
            else:
                dest.write_bytes(b"fake content")

        mock_dl.side_effect = fake_download
        results = download_all(data_dir=tmp_path, skip_existing=True)
        assert mock_dl.call_count == 8
        assert len(results["files"]) == 8
        assert (tmp_path / "manifest.json").exists()

    @patch(
        "tourettes.ts_comorbidity_genetic_architecture.download_cdg3_figshare._download_file"
    )
    def test_all_factors_found(self, mock_dl, tmp_path):
        def fake_download(url, dest):
            matching = [d for d in DOWNLOADS if d["url"] == url]
            if not matching:
                return
            item = matching[0]
            if item["filename"].endswith(".tsv.gz"):
                self._make_fake_gwas_gz(dest)
            else:
                dest.write_bytes(b"fake content")

        mock_dl.side_effect = fake_download
        results = download_all(data_dir=tmp_path)
        assert results["all_factors_present"]
        assert len(results["missing_factors"]) == 0
        assert set(EXPECTED_FACTORS).issubset(set(results["factors_found"]))

    @patch(
        "tourettes.ts_comorbidity_genetic_architecture.download_cdg3_figshare._download_file"
    )
    def test_verification_results(self, mock_dl, tmp_path):
        def fake_download(url, dest):
            matching = [d for d in DOWNLOADS if d["url"] == url]
            if not matching:
                return
            item = matching[0]
            if item["filename"].endswith(".tsv.gz"):
                self._make_fake_gwas_gz(dest)
            else:
                dest.write_bytes(b"fake content")

        mock_dl.side_effect = fake_download
        results = download_all(data_dir=tmp_path)

        # All GWAS files should be verified
        gwas_count = sum(1 for d in DOWNLOADS if d["filename"].endswith(".tsv.gz"))
        assert len(results["verification"]) == gwas_count

        for name, v in results["verification"].items():
            assert v["valid"], f"Verification failed for {name}"
            assert "SNP" in v["columns"]

    @patch(
        "tourettes.ts_comorbidity_genetic_architecture.download_cdg3_figshare._download_file"
    )
    def test_file_metadata_recorded(self, mock_dl, tmp_path):
        def fake_download(url, dest):
            matching = [d for d in DOWNLOADS if d["url"] == url]
            if not matching:
                return
            item = matching[0]
            if item["filename"].endswith(".tsv.gz"):
                self._make_fake_gwas_gz(dest)
            else:
                dest.write_bytes(b"fake content")

        mock_dl.side_effect = fake_download
        results = download_all(data_dir=tmp_path)

        for name, info in results["files"].items():
            assert "path" in info
            assert "md5" in info
            assert "size_mb" in info
            assert len(info["md5"]) == 32
