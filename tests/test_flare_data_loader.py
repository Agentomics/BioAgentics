"""Tests for HMP2/IBDMDB data loader (cd-flare-longitudinal-prediction)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.crohns.flare_prediction.data_loader import (
    HMP2DataLoader,
    _filter_cd,
    summarize_data,
)


@pytest.fixture()
def synth_data_dir(tmp_path):
    """Create synthetic HMP2-like CSV files in a temp directory."""
    rng = np.random.default_rng(42)

    subjects = [f"S{i:03d}" for i in range(10)]
    diagnoses = ["CD"] * 6 + ["UC"] * 2 + ["nonIBD"] * 2
    visits_per_subject = 8

    rows = []
    for sid, dx in zip(subjects, diagnoses):
        for v in range(1, visits_per_subject + 1):
            rows.append({
                "subject_id": sid,
                "visit_num": v,
                "date": f"2015-01-{v:02d}",
                "diagnosis": dx,
            })
    meta = pd.DataFrame(rows)
    meta.to_csv(tmp_path / "hmp2_metadata.csv", index=False)

    # HBI scores
    hbi_rows = []
    for sid, dx in zip(subjects, diagnoses):
        for v in range(1, visits_per_subject + 1):
            score = rng.integers(0, 15) if dx == "CD" else rng.integers(0, 4)
            hbi_rows.append({
                "subject_id": sid,
                "visit_num": v,
                "date": f"2015-01-{v:02d}",
                "hbi_score": int(score),
            })
    pd.DataFrame(hbi_rows).to_csv(tmp_path / "hbi_scores.csv", index=False)

    # MetaPhlAn species (5 species for testing)
    species_names = [f"species_{i}" for i in range(5)]
    species_rows = []
    for sid in subjects:
        for v in range(1, visits_per_subject + 1):
            row = {"subject_id": sid, "visit_num": v}
            abundances = rng.dirichlet(np.ones(5))
            for sn, ab in zip(species_names, abundances):
                row[sn] = float(ab)
            species_rows.append(row)
    pd.DataFrame(species_rows).to_csv(tmp_path / "metaphlan_species.csv", index=False)

    # HUMAnN pathways (4 pathways)
    pw_names = ["PWY-butyrate", "PWY-sulfur", "PWY-oxidative", "PWY-tryptophan"]
    pw_rows = []
    for sid in subjects:
        for v in range(1, visits_per_subject + 1):
            row = {"subject_id": sid, "visit_num": v}
            for pw in pw_names:
                row[pw] = float(rng.exponential(10))
            pw_rows.append(row)
    pd.DataFrame(pw_rows).to_csv(tmp_path / "humann_pathways.csv", index=False)

    # Metabolomics (20 metabolites for testing, real data has ~8000)
    met_names = [f"metabolite_{i}" for i in range(20)]
    met_rows = []
    for sid in subjects:
        for v in range(1, visits_per_subject + 1):
            row = {"subject_id": sid, "visit_num": v}
            for mn in met_names:
                row[mn] = float(rng.normal(0, 1))
            met_rows.append(row)
    pd.DataFrame(met_rows).to_csv(tmp_path / "metabolomics.csv", index=False)

    # Serology
    sero_rows = []
    for sid in subjects:
        for v in range(1, visits_per_subject + 1):
            sero_rows.append({
                "subject_id": sid,
                "visit_num": v,
                "ASCA_IgA": float(rng.exponential(5)),
                "ASCA_IgG": float(rng.exponential(5)),
                "anti_CBir1": float(rng.exponential(3)),
                "anti_OmpC": float(rng.exponential(3)),
            })
    pd.DataFrame(sero_rows).to_csv(tmp_path / "serology.csv", index=False)

    # No transcriptomics file — tests optional loading

    return tmp_path


class TestHMP2DataLoader:
    def test_load_metadata(self, synth_data_dir):
        loader = HMP2DataLoader(synth_data_dir)
        meta = loader.load_metadata()
        assert len(meta) == 80  # 10 subjects × 8 visits
        assert meta.index.names == ["subject_id", "visit_num"]
        assert "diagnosis" in meta.columns

    def test_load_hbi(self, synth_data_dir):
        loader = HMP2DataLoader(synth_data_dir)
        hbi = loader.load_hbi()
        assert "hbi_score" in hbi.columns
        assert hbi["hbi_score"].dtype in (np.int64, np.float64)

    def test_load_species(self, synth_data_dir):
        loader = HMP2DataLoader(synth_data_dir)
        species = loader.load_species()
        assert species.shape[1] == 5  # 5 species columns
        # Abundances should be non-negative
        assert (species.values >= 0).all()

    def test_load_pathways(self, synth_data_dir):
        loader = HMP2DataLoader(synth_data_dir)
        pw = loader.load_pathways()
        assert pw.shape[1] == 4

    def test_load_metabolomics(self, synth_data_dir):
        loader = HMP2DataLoader(synth_data_dir)
        met = loader.load_metabolomics()
        assert met.shape[1] == 20

    def test_load_serology(self, synth_data_dir):
        loader = HMP2DataLoader(synth_data_dir)
        sero = loader.load_serology()
        assert sero is not None
        assert "ASCA_IgA" in sero.columns

    def test_load_transcriptomics_missing(self, synth_data_dir):
        loader = HMP2DataLoader(synth_data_dir)
        tx = loader.load_transcriptomics()
        assert tx is None  # file not present

    def test_load_all_cd_only(self, synth_data_dir):
        loader = HMP2DataLoader(synth_data_dir)
        data = loader.load_all(cd_only=True)
        assert data["transcriptomics"] is None
        # Check CD filtering: 6 CD subjects × 8 visits = 48
        meta = data["metadata"]
        assert len(meta) == 48
        assert set(meta.reset_index()["subject_id"].unique()) == {
            f"S{i:03d}" for i in range(6)
        }

    def test_load_all_no_filter(self, synth_data_dir):
        loader = HMP2DataLoader(synth_data_dir)
        data = loader.load_all(cd_only=False)
        assert len(data["metadata"]) == 80

    def test_missing_file_raises(self, tmp_path):
        loader = HMP2DataLoader(tmp_path)
        with pytest.raises(FileNotFoundError):
            loader.load_metadata()

    def test_missing_hbi_column_raises(self, tmp_path):
        # Write a CSV without hbi_score column
        pd.DataFrame({
            "subject_id": ["S001"],
            "visit_num": [1],
            "wrong_column": [5],
        }).to_csv(tmp_path / "hbi_scores.csv", index=False)
        loader = HMP2DataLoader(tmp_path)
        with pytest.raises(ValueError, match="hbi_score"):
            loader.load_hbi()


class TestSummarizeData:
    def test_summary_shape(self, synth_data_dir):
        loader = HMP2DataLoader(synth_data_dir)
        data = loader.load_all(cd_only=False)
        summary = summarize_data(data)
        assert summary["metadata"]["available"] is True
        assert summary["transcriptomics"]["available"] is False
        assert summary["species"]["n_features"] == 5
        assert summary["metabolomics"]["n_subjects"] == 10
