"""Tests for FP mining data adapters."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.diagnostics.fp_mining.adapters.crc_adapter import (
    MockCrcAdapter,
    create_mock_crc_data,
)
from bioagentics.diagnostics.fp_mining.adapters.sepsis_adapter import (
    MockSepsisAdapter,
    create_mock_sepsis_data,
)
from bioagentics.diagnostics.fp_mining.extract import extract_at_operating_points


class TestMockSepsisData:
    def test_schema(self) -> None:
        df = create_mock_sepsis_data(n_admissions=100)
        assert "sample_id" in df.columns
        assert "y_true" in df.columns
        assert "y_score" in df.columns
        assert len(df) == 100

    def test_class_balance(self) -> None:
        df = create_mock_sepsis_data(n_admissions=1000, sepsis_rate=0.15)
        pos_rate = df["y_true"].mean()
        assert 0.10 <= pos_rate <= 0.20

    def test_scores_in_range(self) -> None:
        df = create_mock_sepsis_data()
        assert df["y_score"].min() >= 0.0
        assert df["y_score"].max() <= 1.0

    def test_reproducible(self) -> None:
        df1 = create_mock_sepsis_data(seed=99)
        df2 = create_mock_sepsis_data(seed=99)
        pd.testing.assert_frame_equal(df1, df2)


class TestMockSepsisAdapter:
    def test_protocol_compliance(self) -> None:
        adapter = MockSepsisAdapter()
        assert adapter.domain == "sepsis"
        df = adapter.load_predictions()
        assert {"sample_id", "y_true", "y_score"}.issubset(df.columns)

    def test_works_with_extract(self) -> None:
        adapter = MockSepsisAdapter(n_admissions=200)
        results = extract_at_operating_points(adapter, specificities=[0.95])
        assert len(results) == 1
        r = results[0]
        total = len(r.false_positives) + len(r.true_negatives) + len(r.true_positives) + len(r.false_negatives)
        assert total == 200


class TestMockCrcData:
    def test_schema(self) -> None:
        df = create_mock_crc_data(n_samples=100)
        assert "sample_id" in df.columns
        assert "y_true" in df.columns
        assert "y_score" in df.columns
        assert "stage_numeric" in df.columns
        assert len(df) == 100

    def test_has_protein_features(self) -> None:
        df = create_mock_crc_data()
        prot_cols = [c for c in df.columns if c.startswith("prot_")]
        assert len(prot_cols) == 7


class TestMockCrcAdapter:
    def test_protocol_compliance(self) -> None:
        adapter = MockCrcAdapter()
        assert adapter.domain == "crc"
        df = adapter.load_predictions()
        assert {"sample_id", "y_true", "y_score"}.issubset(df.columns)

    def test_works_with_extract(self) -> None:
        adapter = MockCrcAdapter(n_samples=200)
        results = extract_at_operating_points(adapter, specificities=[0.90])
        assert len(results) == 1
        assert results[0].summary["n_total"] == 200
