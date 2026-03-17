"""Tests for flare event detection and window extraction."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from bioagentics.crohns.flare_prediction.flare_events import (
    FlareEvent,
    Window,
    detect_flares,
    extract_windows,
    summarize_flares,
    summarize_windows,
)


def _make_hbi(subject_id: str, scores: list[float], start: str = "2015-01-01", interval_days: int = 14) -> pd.DataFrame:
    """Build a synthetic HBI DataFrame for one patient."""
    dates = pd.date_range(start, periods=len(scores), freq=f"{interval_days}D")
    return pd.DataFrame({
        "subject_id": subject_id,
        "visit_num": range(1, len(scores) + 1),
        "date": dates,
        "hbi_score": scores,
    }).set_index(["subject_id", "visit_num"])


def _concat(*dfs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(dfs)


class TestDetectFlares:
    def test_transition_flare(self):
        """HBI goes from remission (<5) to active (>=5)."""
        hbi = _make_hbi("P001", [2, 3, 7, 4, 1])
        flares = detect_flares(hbi)
        assert len(flares) == 1
        assert flares[0].trigger == "transition"
        assert flares[0].flare_visit == 3
        assert flares[0].prior_hbi == 3
        assert flares[0].flare_hbi == 7

    def test_increase_flare(self):
        """HBI increases by >=3 points while already active."""
        hbi = _make_hbi("P001", [6, 9, 8])
        flares = detect_flares(hbi)
        assert len(flares) == 1
        assert flares[0].trigger == "increase"
        assert flares[0].flare_visit == 2

    def test_no_flare_stable(self):
        """All HBI below remission — no flares."""
        hbi = _make_hbi("P001", [1, 2, 1, 3, 2])
        flares = detect_flares(hbi)
        assert len(flares) == 0

    def test_multiple_flares_same_patient(self):
        """Patient flares, recovers, then flares again."""
        hbi = _make_hbi("P001", [2, 8, 3, 2, 7])
        flares = detect_flares(hbi)
        assert len(flares) == 2
        assert flares[0].flare_visit == 2
        assert flares[1].flare_visit == 5

    def test_borderline_no_flare(self):
        """HBI increase of exactly 2 — should NOT trigger (threshold is 3)."""
        hbi = _make_hbi("P001", [5, 7, 5])
        flares = detect_flares(hbi)
        assert len(flares) == 0

    def test_multi_patient(self):
        """Flare detection across multiple patients."""
        hbi = _concat(
            _make_hbi("P001", [2, 8, 3]),
            _make_hbi("P002", [1, 2, 1]),
            _make_hbi("P003", [3, 6, 2]),
        )
        flares = detect_flares(hbi)
        assert len(flares) == 2  # P001 and P003 have flares
        subjects = {f.subject_id for f in flares}
        assert subjects == {"P001", "P003"}

    def test_both_triggers_same_event(self):
        """Transition from <5 to >=5 that's also a >=3 increase picks transition."""
        hbi = _make_hbi("P001", [2, 8])
        flares = detect_flares(hbi)
        assert len(flares) == 1
        assert flares[0].trigger == "transition"

    def test_sparse_sampling(self):
        """Works with wide visit spacing (30-day intervals)."""
        hbi = _make_hbi("P001", [2, 3, 9, 4], interval_days=30)
        flares = detect_flares(hbi)
        assert len(flares) == 1


class TestExtractWindows:
    def test_pre_flare_window(self):
        """Pre-flare window extracted before a flare event."""
        hbi = _make_hbi("P001", [2, 3, 7, 4, 1])
        flares = detect_flares(hbi)
        windows = extract_windows(hbi, flares, lead_weeks=2)
        pre_flare = [w for w in windows if w.label == "pre_flare"]
        assert len(pre_flare) == 1
        assert pre_flare[0].subject_id == "P001"
        # Window end should be the flare date
        assert pre_flare[0].window_end == flares[0].flare_date
        # Window start should be 2 weeks before
        expected_start = flares[0].flare_date - timedelta(weeks=2)
        assert pre_flare[0].window_start == expected_start

    def test_stable_window(self):
        """Stable window extracted from long remission period."""
        # 10 visits, all in remission, 14-day intervals = ~18 weeks total
        hbi = _make_hbi("P001", [2, 1, 3, 2, 1, 3, 2, 1, 2, 1])
        flares = detect_flares(hbi)
        assert len(flares) == 0
        windows = extract_windows(hbi, flares, lead_weeks=2)
        stable = [w for w in windows if w.label == "stable"]
        assert len(stable) == 1
        assert stable[0].label == "stable"

    def test_no_stable_if_too_short(self):
        """Short remission periods don't yield stable windows."""
        hbi = _make_hbi("P001", [2, 3, 7])  # only 2 visits in remission
        flares = detect_flares(hbi)
        windows = extract_windows(hbi, flares, lead_weeks=2)
        stable = [w for w in windows if w.label == "stable"]
        assert len(stable) == 0

    def test_no_stable_if_flare_in_buffer(self):
        """Stable window rejected if a flare occurs within the buffer period."""
        # P001: stable for 6 visits then flares at visit 7
        scores = [2, 1, 2, 1, 2, 1, 8]
        hbi = _make_hbi("P001", scores, interval_days=7)  # ~6 weeks total
        flares = detect_flares(hbi)
        windows = extract_windows(hbi, flares, lead_weeks=2, stable_buffer_weeks=4)
        stable = [w for w in windows if w.label == "stable"]
        # The stable run ends at visit 6, but flare at visit 7 is only 1 week later
        assert len(stable) == 0

    def test_4_week_lead(self):
        """Pre-flare windows adjust to 4-week lead time."""
        hbi = _make_hbi("P001", [2, 3, 7, 4])
        flares = detect_flares(hbi)
        windows = extract_windows(hbi, flares, lead_weeks=4)
        pre_flare = [w for w in windows if w.label == "pre_flare"]
        assert len(pre_flare) == 1
        expected_start = flares[0].flare_date - timedelta(weeks=4)
        assert pre_flare[0].window_start == expected_start


class TestSummaries:
    def test_summarize_flares_empty(self):
        s = summarize_flares([])
        assert s["n_flares"] == 0

    def test_summarize_flares(self):
        hbi = _concat(
            _make_hbi("P001", [2, 8, 3, 7]),
            _make_hbi("P002", [3, 6]),
        )
        flares = detect_flares(hbi)
        s = summarize_flares(flares)
        assert s["n_flares"] == 3
        assert s["n_patients"] == 2
        assert s["flares_per_patient_max"] == 2

    def test_summarize_windows(self):
        hbi = _make_hbi("P001", [2, 3, 7, 2, 1, 2, 1, 2, 1, 2])
        flares = detect_flares(hbi)
        windows = extract_windows(hbi, flares, lead_weeks=2)
        s = summarize_windows(windows)
        assert s["n_windows"] >= 1
        assert s["n_patients"] == 1

    def test_summarize_windows_empty(self):
        s = summarize_windows([])
        assert s["n_windows"] == 0
