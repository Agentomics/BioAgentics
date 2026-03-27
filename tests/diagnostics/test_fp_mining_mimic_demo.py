"""Tests for MIMIC-IV demo sepsis adapter and pipeline integration."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bioagentics.diagnostics.fp_mining.adapters.sepsis_adapter import (
    MIMIC_DEMO_DIR,
    MimicDemoSepsisAdapter,
    _compute_heuristic_risk_score,
)
from bioagentics.diagnostics.fp_mining.extract import extract_at_operating_points

# Skip all tests if MIMIC demo data is not present
pytestmark = pytest.mark.skipif(
    not MIMIC_DEMO_DIR.exists(),
    reason="MIMIC-IV demo data not available",
)


class TestMimicDemoSepsisAdapter:
    def test_protocol_compliance(self, tmp_path: Path) -> None:
        adapter = MimicDemoSepsisAdapter(output_dir=tmp_path)
        df = adapter.load_predictions()
        assert {"sample_id", "y_true", "y_score"}.issubset(df.columns)
        assert len(df) > 0

    def test_sample_id_format(self, tmp_path: Path) -> None:
        adapter = MimicDemoSepsisAdapter(output_dir=tmp_path)
        df = adapter.load_predictions()
        # sample_id should be {subject_id}_{hadm_id}
        parts = df["sample_id"].str.split("_", n=1)
        assert all(len(p) == 2 for p in parts)
        # Both parts should be numeric
        assert all(p[0].isdigit() and p[1].isdigit() for p in parts)

    def test_labels_binary(self, tmp_path: Path) -> None:
        adapter = MimicDemoSepsisAdapter(output_dir=tmp_path)
        df = adapter.load_predictions()
        assert set(df["y_true"].unique()).issubset({0, 1})
        # Should have both classes
        assert df["y_true"].sum() > 0
        assert (df["y_true"] == 0).sum() > 0

    def test_scores_valid_range(self, tmp_path: Path) -> None:
        adapter = MimicDemoSepsisAdapter(output_dir=tmp_path)
        df = adapter.load_predictions()
        assert df["y_score"].min() >= 0.0
        assert df["y_score"].max() <= 1.0
        assert not df["y_score"].isna().any()

    def test_has_clinical_features(self, tmp_path: Path) -> None:
        adapter = MimicDemoSepsisAdapter(output_dir=tmp_path)
        df = adapter.load_predictions()
        feature_cols = [
            c for c in df.columns if c not in ("sample_id", "y_true", "y_score")
        ]
        assert len(feature_cols) >= 10
        # Should include key vitals
        assert "heart_rate" in df.columns
        assert "creatinine" in df.columns

    def test_per_stay_export(self, tmp_path: Path) -> None:
        adapter = MimicDemoSepsisAdapter(output_dir=tmp_path)
        adapter.load_predictions()
        # per_stay_predictions.parquet should be created
        ps_path = tmp_path / "per_stay_predictions.parquet"
        assert ps_path.exists()
        ps = pd.read_parquet(ps_path)
        assert "subject_id" in ps.columns
        assert "hadm_id" in ps.columns
        assert "stay_id" in ps.columns
        assert "y_true" in ps.columns
        assert "y_score" in ps.columns
        assert len(ps) > 0

    def test_multi_admission_subjects(self, tmp_path: Path) -> None:
        adapter = MimicDemoSepsisAdapter(output_dir=tmp_path)
        adapter.load_predictions()
        ps = pd.read_parquet(tmp_path / "per_stay_predictions.parquet")
        multi = ps.groupby("subject_id")["hadm_id"].nunique()
        # MIMIC demo has 48 patients with >1 admission overall;
        # at least some should have >1 ICU admission
        assert (multi > 1).sum() > 0

    def test_works_with_extract(self, tmp_path: Path) -> None:
        adapter = MimicDemoSepsisAdapter(output_dir=tmp_path)
        results = extract_at_operating_points(adapter, specificities=[0.90])
        assert len(results) == 1
        r = results[0]
        total = (
            len(r.false_positives)
            + len(r.true_negatives)
            + len(r.true_positives)
            + len(r.false_negatives)
        )
        assert total > 0
        assert r.summary["specificity"] >= 0.85  # Allow some tolerance


class TestHeuristicRiskScore:
    def test_returns_valid_range(self) -> None:
        df = pd.DataFrame({
            "heart_rate": [80, 120, 60],
            "resp_rate": [16, 30, 12],
            "temperature": [37.0, 39.5, 35.5],
            "creatinine": [0.8, 3.0, 1.2],
        })
        scores = _compute_heuristic_risk_score(df)
        assert (scores >= 0).all()
        assert (scores <= 1).all()
        assert len(scores) == 3

    def test_sicker_patients_score_higher(self) -> None:
        healthy = pd.DataFrame({
            "heart_rate": [75],
            "resp_rate": [14],
            "temperature": [36.8],
            "sbp": [120],
            "creatinine": [0.9],
            "lactate": [0.8],
        })
        sick = pd.DataFrame({
            "heart_rate": [130],
            "resp_rate": [35],
            "temperature": [39.5],
            "sbp": [75],
            "creatinine": [3.5],
            "lactate": [6.0],
        })
        h_score = _compute_heuristic_risk_score(healthy).iloc[0]
        s_score = _compute_heuristic_risk_score(sick).iloc[0]
        assert s_score > h_score

    def test_handles_all_nan(self) -> None:
        df = pd.DataFrame({
            "heart_rate": [np.nan],
            "creatinine": [np.nan],
        })
        scores = _compute_heuristic_risk_score(df)
        assert len(scores) == 1
        assert scores.iloc[0] == 0.5  # Default for no data

    def test_handles_partial_nan(self) -> None:
        df = pd.DataFrame({
            "heart_rate": [100, np.nan],
            "creatinine": [np.nan, 2.0],
        })
        scores = _compute_heuristic_risk_score(df)
        assert len(scores) == 2
        assert not scores.isna().any()
