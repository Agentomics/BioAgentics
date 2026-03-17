"""Detect CD flare events from HBI trajectories and extract classification windows.

Flare definition (from research plan):
- Sustained HBI increase >= 3 points, OR
- Transition from HBI < 5 (remission) to HBI >= 5 (active disease)

Window types:
- **Pre-flare**: 2-week or 4-week window immediately before a flare event
- **Stable**: >= 4 weeks of HBI < 5 with no subsequent flare within 4 weeks

Usage::

    from bioagentics.crohns.flare_prediction.flare_events import (
        detect_flares, extract_windows, summarize_windows,
    )

    flares = detect_flares(hbi_df)
    windows = extract_windows(hbi_df, flares, lead_weeks=2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

logger = logging.getLogger(__name__)

# Thresholds from research plan
HBI_REMISSION = 5  # HBI < 5 is remission
HBI_INCREASE_THRESHOLD = 3  # sustained increase >= 3 points = flare


@dataclass
class FlareEvent:
    """A single detected flare event."""

    subject_id: str
    flare_date: pd.Timestamp
    flare_visit: int
    prior_hbi: float
    flare_hbi: float
    trigger: str  # "increase" or "transition"


@dataclass
class Window:
    """A classification window (pre-flare or stable)."""

    subject_id: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    label: str  # "pre_flare" or "stable"
    anchor_visit: int  # visit that defines the window endpoint


def detect_flares(hbi: pd.DataFrame) -> list[FlareEvent]:
    """Detect flare events from HBI time-series.

    Parameters
    ----------
    hbi:
        DataFrame with index (subject_id, visit_num) and columns
        ``hbi_score`` and ``date``.

    Returns
    -------
    List of FlareEvent instances sorted by (subject_id, date).
    """
    flares: list[FlareEvent] = []
    hbi_reset = hbi.reset_index()

    for subject_id, group in hbi_reset.groupby("subject_id"):
        group = group.sort_values("visit_num").reset_index(drop=True)
        scores = group["hbi_score"].values
        dates = pd.to_datetime(group["date"]).values
        visits = group["visit_num"].values

        for i in range(1, len(scores)):
            prior = float(scores[i - 1])
            current = float(scores[i])
            triggered = False
            trigger = ""

            # Criterion 1: transition from remission to active
            if prior < HBI_REMISSION and current >= HBI_REMISSION:
                triggered = True
                trigger = "transition"
            # Criterion 2: sustained increase >= 3 points
            elif current - prior >= HBI_INCREASE_THRESHOLD:
                triggered = True
                trigger = "increase"

            if triggered:
                flares.append(
                    FlareEvent(
                        subject_id=str(subject_id),
                        flare_date=pd.Timestamp(dates[i]),
                        flare_visit=int(visits[i]),
                        prior_hbi=prior,
                        flare_hbi=current,
                        trigger=trigger,
                    )
                )

    flares.sort(key=lambda f: (f.subject_id, f.flare_date))
    logger.info("Detected %d flare events across %d patients",
                len(flares), len({f.subject_id for f in flares}))
    return flares


def extract_windows(
    hbi: pd.DataFrame,
    flares: list[FlareEvent],
    lead_weeks: int = 2,
    stable_min_weeks: int = 4,
    stable_buffer_weeks: int = 4,
) -> list[Window]:
    """Extract pre-flare and stable windows from HBI trajectories.

    Parameters
    ----------
    hbi:
        HBI DataFrame (same format as detect_flares input).
    flares:
        List of detected flare events.
    lead_weeks:
        Length of pre-flare window in weeks (default 2).
    stable_min_weeks:
        Minimum duration of stable period in weeks (default 4).
    stable_buffer_weeks:
        Required flare-free buffer after stable window in weeks (default 4).

    Returns
    -------
    List of Window instances.
    """
    lead_delta = timedelta(weeks=lead_weeks)
    stable_min_delta = timedelta(weeks=stable_min_weeks)
    stable_buffer_delta = timedelta(weeks=stable_buffer_weeks)

    hbi_reset = hbi.reset_index()
    hbi_reset["date"] = pd.to_datetime(hbi_reset["date"])

    # Build per-subject flare date sets for quick lookup
    flare_dates_by_subject: dict[str, list[pd.Timestamp]] = {}
    for f in flares:
        flare_dates_by_subject.setdefault(f.subject_id, []).append(f.flare_date)

    windows: list[Window] = []

    # --- Pre-flare windows ---
    for f in flares:
        window_end = f.flare_date
        window_start = window_end - lead_delta
        windows.append(
            Window(
                subject_id=f.subject_id,
                window_start=window_start,
                window_end=window_end,
                label="pre_flare",
                anchor_visit=f.flare_visit,
            )
        )

    # --- Stable windows ---
    for subject_id, group in hbi_reset.groupby("subject_id"):
        subject_id = str(subject_id)
        group = group.sort_values("date").reset_index(drop=True)
        scores = group["hbi_score"].values
        dates = pd.to_datetime(group["date"]).values
        visits = group["visit_num"].values

        subject_flare_dates = flare_dates_by_subject.get(subject_id, [])

        # Find contiguous runs of HBI < 5
        i = 0
        while i < len(scores):
            if scores[i] >= HBI_REMISSION:
                i += 1
                continue

            # Start of a potential stable run
            run_start = i
            while i < len(scores) and scores[i] < HBI_REMISSION:
                i += 1
            run_end = i - 1  # inclusive

            run_start_date = pd.Timestamp(dates[run_start])
            run_end_date = pd.Timestamp(dates[run_end])
            run_duration = run_end_date - run_start_date

            if run_duration < stable_min_delta:
                continue

            # Check that no flare occurs within buffer after the window
            buffer_end = run_end_date + stable_buffer_delta
            flare_in_buffer = any(
                run_end_date < fd <= buffer_end for fd in subject_flare_dates
            )
            if flare_in_buffer:
                continue

            # Use the end of the stable run as the window endpoint
            windows.append(
                Window(
                    subject_id=subject_id,
                    window_start=run_end_date - lead_delta,
                    window_end=run_end_date,
                    label="stable",
                    anchor_visit=int(visits[run_end]),
                )
            )

    windows.sort(key=lambda w: (w.subject_id, w.window_start))
    n_pre = sum(1 for w in windows if w.label == "pre_flare")
    n_stable = sum(1 for w in windows if w.label == "stable")
    logger.info("Extracted %d windows: %d pre-flare, %d stable", len(windows), n_pre, n_stable)
    return windows


def summarize_flares(flares: list[FlareEvent]) -> dict:
    """Return summary statistics for detected flare events."""
    if not flares:
        return {"n_flares": 0, "n_patients": 0}

    patients = {}
    for f in flares:
        patients.setdefault(f.subject_id, []).append(f)

    flares_per_patient = [len(v) for v in patients.values()]
    triggers = [f.trigger for f in flares]

    return {
        "n_flares": len(flares),
        "n_patients": len(patients),
        "flares_per_patient_mean": sum(flares_per_patient) / len(flares_per_patient),
        "flares_per_patient_max": max(flares_per_patient),
        "trigger_counts": {
            "transition": triggers.count("transition"),
            "increase": triggers.count("increase"),
        },
    }


def summarize_windows(windows: list[Window]) -> dict:
    """Return summary statistics for extracted windows."""
    if not windows:
        return {"n_windows": 0}

    pre_flare = [w for w in windows if w.label == "pre_flare"]
    stable = [w for w in windows if w.label == "stable"]
    patients = {w.subject_id for w in windows}

    return {
        "n_windows": len(windows),
        "n_pre_flare": len(pre_flare),
        "n_stable": len(stable),
        "n_patients": len(patients),
        "class_balance": len(pre_flare) / len(windows) if windows else 0,
    }
