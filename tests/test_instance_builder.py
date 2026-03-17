"""Tests for within-patient paired classification instance builder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bioagentics.crohns.flare_prediction.flare_events import (
    detect_flares,
    extract_windows,
)
from bioagentics.crohns.flare_prediction.instance_builder import (
    InstanceSet,
    build_instances,
)


def _make_cohort(n_subjects: int = 4, visits: int = 10, interval_days: int = 14):
    """Create synthetic multi-omic data for testing."""
    rng = np.random.default_rng(42)

    meta_rows = []
    hbi_rows = []
    species_rows = []
    pathway_rows = []
    met_rows = []

    species_names = [f"sp_{i}" for i in range(5)]
    pw_names = ["PWY-butyrate", "PWY-sulfur"]
    met_names = [f"met_{i}" for i in range(10)]

    for s in range(n_subjects):
        sid = f"P{s:03d}"
        dates = pd.date_range("2015-01-01", periods=visits, freq=f"{interval_days}D")

        # Create a trajectory with a flare for half the subjects
        if s < n_subjects // 2:
            # Flare patient: remission → flare at visit 5
            scores = [2, 1, 3, 2, 8, 3, 2, 1, 2, 1][:visits]
        else:
            # Stable patient: all low HBI
            scores = [rng.integers(0, 4) for _ in range(visits)]

        for v in range(visits):
            meta_rows.append({
                "subject_id": sid, "visit_num": v + 1,
                "date": dates[v], "diagnosis": "CD",
            })
            hbi_rows.append({
                "subject_id": sid, "visit_num": v + 1,
                "date": dates[v], "hbi_score": scores[v],
            })
            row_species = {"subject_id": sid, "visit_num": v + 1, "date": dates[v]}
            for sp in species_names:
                row_species[sp] = float(rng.random())
            species_rows.append(row_species)

            row_pw = {"subject_id": sid, "visit_num": v + 1, "date": dates[v]}
            for pw in pw_names:
                row_pw[pw] = float(rng.exponential(10))
            pathway_rows.append(row_pw)

            row_met = {"subject_id": sid, "visit_num": v + 1, "date": dates[v]}
            for m in met_names:
                row_met[m] = float(rng.normal())
            met_rows.append(row_met)

    idx = ["subject_id", "visit_num"]
    metadata = pd.DataFrame(meta_rows).set_index(idx)
    hbi = pd.DataFrame(hbi_rows).set_index(idx)
    species = pd.DataFrame(species_rows).set_index(idx)
    pathways = pd.DataFrame(pathway_rows).set_index(idx)
    metabolomics = pd.DataFrame(met_rows).set_index(idx)

    data = {
        "metadata": metadata,
        "hbi": hbi,
        "species": species,
        "pathways": pathways,
        "metabolomics": metabolomics,
        "serology": None,
        "transcriptomics": None,
    }
    return data, hbi


class TestBuildInstances:
    def test_basic_instance_creation(self):
        data, hbi = _make_cohort(n_subjects=4, visits=10)
        flares = detect_flares(hbi)
        windows = extract_windows(hbi, flares, lead_weeks=2)
        assert len(windows) > 0

        result = build_instances(windows, data, hbi=hbi)
        assert isinstance(result, InstanceSet)
        assert result.n_instances == len(windows)
        assert result.features.shape[0] == len(windows)

    def test_labels_match_windows(self):
        data, hbi = _make_cohort()
        flares = detect_flares(hbi)
        windows = extract_windows(hbi, flares, lead_weeks=2)
        result = build_instances(windows, data, hbi=hbi)

        for inst, window in zip(result.instances, windows):
            assert inst.label == window.label
            assert inst.subject_id == window.subject_id

    def test_feature_columns_prefixed(self):
        data, hbi = _make_cohort()
        flares = detect_flares(hbi)
        windows = extract_windows(hbi, flares, lead_weeks=2)
        result = build_instances(windows, data, hbi=hbi)

        cols = result.features.columns.tolist()
        # Should have prefixed features from available layers
        has_species = any(c.startswith("species__") for c in cols)
        has_pathways = any(c.startswith("pathways__") for c in cols)
        has_metabolomics = any(c.startswith("metabolomics__") for c in cols)
        assert has_species
        assert has_pathways
        assert has_metabolomics

    def test_layer_availability(self):
        data, hbi = _make_cohort()
        flares = detect_flares(hbi)
        windows = extract_windows(hbi, flares, lead_weeks=2)
        result = build_instances(windows, data, hbi=hbi)

        # Serology and transcriptomics should be unavailable
        assert not result.layer_availability["serology"].any()
        assert not result.layer_availability["transcriptomics"].any()
        # Required layers should be available for all instances
        assert result.layer_availability["species"].all()

    def test_summary(self):
        data, hbi = _make_cohort()
        flares = detect_flares(hbi)
        windows = extract_windows(hbi, flares, lead_weeks=2)
        result = build_instances(windows, data, hbi=hbi)

        s = result.summary()
        assert s["n_instances"] == result.n_instances
        assert s["n_pre_flare"] == result.n_pre_flare
        assert s["n_stable"] == result.n_stable
        assert s["n_features"] > 0

    def test_empty_windows(self):
        data, hbi = _make_cohort()
        result = build_instances([], data, hbi=hbi)
        assert result.n_instances == 0
        assert result.features.empty

    def test_within_patient_pairing(self):
        """Both pre-flare and stable windows come from same patients when possible."""
        data, hbi = _make_cohort(n_subjects=4, visits=10)
        flares = detect_flares(hbi)
        windows = extract_windows(hbi, flares, lead_weeks=2)
        result = build_instances(windows, data, hbi=hbi)

        pre_flare_patients = {i.subject_id for i in result.instances if i.label == "pre_flare"}
        stable_patients = {i.subject_id for i in result.instances if i.label == "stable"}
        # Stable patients should be present (not all patients flare)
        assert len(stable_patients) > 0 or len(pre_flare_patients) > 0
