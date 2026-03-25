"""Tests for HMP2/IBDMDB download & transform script (cd-flare-longitudinal-prediction)."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from crohns.cd_flare_longitudinal_prediction.download_data import (
    RAW_FILES,
    _build_sample_map,
    _map_samples,
    _read_tsv_gz,
    _strip_sample_suffix,
    _transform_hbi,
    _transform_metadata,
    _transform_pathways,
    _transform_species,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_metadata(tmp_path: Path) -> Path:
    """Write a minimal HMP2-style metadata CSV and return its path."""
    rows = []
    for pid_i in range(5):
        pid = f"C300{pid_i+1}"
        dx = "CD" if pid_i < 3 else "nonIBD"
        for v in range(1, 4):
            ext_id = f"CSM{pid_i}{v}ZZZ"
            rows.append({
                "Project": "HMP2",
                "External ID": ext_id,
                "Participant ID": pid,
                "data_type": "metagenomics",
                "visit_num": v,
                "week_num": v * 2,
                "date_of_receipt": f"2015-0{v}-15",
                "diagnosis": dx,
                "consent_age": 30 + pid_i,
                "sex": "Male",
                "hbi": np.random.default_rng(pid_i + v).integers(0, 15)
                if dx == "CD"
                else np.nan,
            })
    meta = pd.DataFrame(rows)
    path = tmp_path / "hmp2_metadata.csv"
    meta.to_csv(path, index=False)
    return path


def _make_tsv_gz(tmp_path: Path, filename: str, features: list[str], sample_ids: list[str]) -> Path:
    """Write a gzipped MetaPhlAn/HUMAnN-style TSV (features × samples)."""
    rng = np.random.default_rng(99)
    data = rng.random((len(features), len(sample_ids)))
    df = pd.DataFrame(data, index=features, columns=sample_ids)
    df.index.name = "clade_name"

    path = tmp_path / filename
    with gzip.open(path, "wt") as fh:
        fh.write(f"#clade_name\t" + "\t".join(sample_ids) + "\n")
        for feat in features:
            vals = "\t".join(str(v) for v in df.loc[feat])
            fh.write(f"{feat}\t{vals}\n")
    return path


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    """Create a minimal set of raw HMP2-like files for transform testing."""
    meta_path = _make_metadata(tmp_path)

    # Read metadata to get External IDs for sample columns
    meta = pd.read_csv(meta_path)
    ext_ids = meta["External ID"].unique().tolist()

    # Taxonomic profiles (species-level)
    species_features = [
        "k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Faecalibacterium|s__Faecalibacterium_prausnitzii",
        "k__Bacteria|p__Bacteroidetes|c__Bacteroidia|o__Bacteroidales|f__Bacteroidaceae|g__Bacteroides|s__Bacteroides_vulgatus",
        "k__Bacteria|p__Firmicutes|c__Clostridia|o__Clostridiales|f__Ruminococcaceae|g__Ruminococcus|s__Ruminococcus_bromii|t__SGB123",
    ]
    _make_tsv_gz(tmp_path, "taxonomic_profiles.tsv.gz", species_features, ext_ids)

    # Pathway abundances
    pathway_features = [
        "PWY-5100: pyruvate fermentation to acetate and lactate II",
        "PWY-6305: putrescine biosynthesis IV",
        "UNMAPPED",
        "UNINTEGRATED",
        "PWY-5100|g__Faecalibacterium.s__Faecalibacterium_prausnitzii",
    ]
    _make_tsv_gz(tmp_path, "pathabundances_3.tsv.gz", pathway_features, ext_ids)

    return tmp_path


# ---------------------------------------------------------------------------
# Tests: RAW_FILES manifest
# ---------------------------------------------------------------------------


def test_raw_files_has_all_layers():
    expected = {"metadata", "taxonomic", "pathways", "metabolomics",
                "transcriptomics", "serology", "dysbiosis"}
    assert set(RAW_FILES.keys()) == expected


def test_raw_files_urls_are_https():
    for key, info in RAW_FILES.items():
        assert info["url"].startswith("https://"), f"{key} URL not HTTPS"


def test_raw_files_have_filenames():
    for key, info in RAW_FILES.items():
        assert info["filename"], f"{key} missing filename"


# ---------------------------------------------------------------------------
# Tests: sample ID helpers
# ---------------------------------------------------------------------------


class TestStripSampleSuffix:
    def test_strips_P(self):
        assert _strip_sample_suffix("CSM5FZ4P_P") == "CSM5FZ4P"

    def test_strips_TR(self):
        assert _strip_sample_suffix("HSM6XRS9_TR") == "HSM6XRS9"

    def test_strips_M(self):
        assert _strip_sample_suffix("CSM123_M") == "CSM123"

    def test_strips_MBX(self):
        assert _strip_sample_suffix("CSM123_MBX") == "CSM123"

    def test_no_suffix_unchanged(self):
        assert _strip_sample_suffix("CSM5FZ4P") == "CSM5FZ4P"

    def test_empty(self):
        assert _strip_sample_suffix("") == ""


class TestBuildSampleMap:
    def test_basic_mapping(self, raw_dir: Path):
        meta = pd.read_csv(raw_dir / "hmp2_metadata.csv")
        smap = _build_sample_map(meta)
        assert len(smap) > 0
        # Check a known entry
        first_ext = meta["External ID"].iloc[0]
        first_pid = meta["Participant ID"].iloc[0]
        first_vn = int(meta["visit_num"].iloc[0])
        assert smap[first_ext] == (first_pid, first_vn)

    def test_skips_nan_ids(self):
        df = pd.DataFrame({
            "External ID": ["ABC", np.nan, "DEF"],
            "Participant ID": ["P1", "P2", np.nan],
            "visit_num": [1, 2, 3],
        })
        smap = _build_sample_map(df)
        assert "ABC" in smap
        assert len(smap) == 1  # nan entries skipped


class TestMapSamples:
    def test_adds_subject_visit_columns(self):
        smap = {"S1": ("P1", 1), "S2": ("P1", 2), "S3": ("P2", 1)}
        df = pd.DataFrame(
            {"feat_a": [0.1, 0.2, 0.3], "feat_b": [0.4, 0.5, 0.6]},
            index=["S1", "S2", "S3"],
        )
        result = _map_samples(df, smap)
        assert "subject_id" in result.columns
        assert "visit_num" in result.columns
        assert len(result) == 3
        assert list(result["subject_id"]) == ["P1", "P1", "P2"]

    def test_drops_unmapped_samples(self):
        smap = {"S1": ("P1", 1)}
        df = pd.DataFrame({"x": [1, 2]}, index=["S1", "UNKNOWN"])
        result = _map_samples(df, smap)
        assert len(result) == 1

    def test_strips_suffix_for_matching(self):
        smap = {"CSM123": ("P1", 1)}
        df = pd.DataFrame({"x": [10]}, index=["CSM123_P"])
        result = _map_samples(df, smap)
        assert len(result) == 1
        assert result["subject_id"].iloc[0] == "P1"


# ---------------------------------------------------------------------------
# Tests: read helpers
# ---------------------------------------------------------------------------


def test_read_tsv_gz(raw_dir: Path):
    df = _read_tsv_gz(raw_dir / "taxonomic_profiles.tsv.gz")
    # Should be samples-as-rows after transpose
    assert df.shape[0] == 15  # 5 subjects × 3 visits
    assert df.shape[1] == 3  # 3 features


# ---------------------------------------------------------------------------
# Tests: transforms
# ---------------------------------------------------------------------------


class TestTransformMetadata:
    def test_produces_csv(self, raw_dir: Path, tmp_path: Path):
        out = _transform_metadata(raw_dir / "hmp2_metadata.csv", tmp_path)
        assert out.exists()
        df = pd.read_csv(out)
        assert "subject_id" in df.columns
        assert "visit_num" in df.columns
        assert "diagnosis" in df.columns
        # Should be deduplicated: 5 subjects × 3 visits = 15 rows
        assert len(df) == 15

    def test_sorted_output(self, raw_dir: Path, tmp_path: Path):
        _transform_metadata(raw_dir / "hmp2_metadata.csv", tmp_path)
        df = pd.read_csv(tmp_path / "hmp2_metadata.csv")
        # Verify sorted by subject_id then visit_num
        assert list(df["subject_id"]) == sorted(df["subject_id"])


class TestTransformHBI:
    def test_produces_csv(self, raw_dir: Path, tmp_path: Path):
        out = _transform_hbi(raw_dir / "hmp2_metadata.csv", tmp_path)
        assert out.exists()
        df = pd.read_csv(out)
        assert "subject_id" in df.columns
        assert "hbi_score" in df.columns
        # Only CD subjects (3) have HBI, × 3 visits = 9
        assert len(df) == 9

    def test_no_nan_scores(self, raw_dir: Path, tmp_path: Path):
        _transform_hbi(raw_dir / "hmp2_metadata.csv", tmp_path)
        df = pd.read_csv(tmp_path / "hbi_scores.csv")
        assert df["hbi_score"].notna().all()


class TestTransformSpecies:
    def test_produces_csv(self, raw_dir: Path, tmp_path: Path):
        meta = pd.read_csv(raw_dir / "hmp2_metadata.csv")
        smap = _build_sample_map(meta)
        out = _transform_species(
            raw_dir / "taxonomic_profiles.tsv.gz", tmp_path, smap
        )
        assert out.exists()
        df = pd.read_csv(out)
        assert "subject_id" in df.columns
        assert "visit_num" in df.columns
        # Should filter to species-level only (2 species, the t__ one excluded)
        feature_cols = [c for c in df.columns if c not in ("subject_id", "visit_num")]
        assert len(feature_cols) == 2

    def test_simplified_names(self, raw_dir: Path, tmp_path: Path):
        meta = pd.read_csv(raw_dir / "hmp2_metadata.csv")
        smap = _build_sample_map(meta)
        _transform_species(raw_dir / "taxonomic_profiles.tsv.gz", tmp_path, smap)
        df = pd.read_csv(tmp_path / "metaphlan_species.csv")
        feature_cols = [c for c in df.columns if c not in ("subject_id", "visit_num")]
        # Check simplified naming (genus_species format)
        assert any("Faecalibacterium" in c for c in feature_cols)
        assert all("|" not in c for c in feature_cols)


class TestTransformPathways:
    def test_produces_csv(self, raw_dir: Path, tmp_path: Path):
        meta = pd.read_csv(raw_dir / "hmp2_metadata.csv")
        smap = _build_sample_map(meta)
        out = _transform_pathways(
            raw_dir / "pathabundances_3.tsv.gz", tmp_path, smap
        )
        assert out.exists()
        df = pd.read_csv(out)
        assert "subject_id" in df.columns

    def test_excludes_unmapped_and_stratified(self, raw_dir: Path, tmp_path: Path):
        meta = pd.read_csv(raw_dir / "hmp2_metadata.csv")
        smap = _build_sample_map(meta)
        _transform_pathways(raw_dir / "pathabundances_3.tsv.gz", tmp_path, smap)
        df = pd.read_csv(tmp_path / "humann_pathways.csv")
        feature_cols = [c for c in df.columns if c not in ("subject_id", "visit_num")]
        # UNMAPPED, UNINTEGRATED, and species-stratified should be excluded
        assert "UNMAPPED" not in feature_cols
        assert "UNINTEGRATED" not in feature_cols
        assert all("|" not in c for c in feature_cols)
        # 2 community-level pathways remain
        assert len(feature_cols) == 2
