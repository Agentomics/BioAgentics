"""Build within-patient paired classification instances from windows and omic data.

Pairs pre-flare and stable windows per patient and extracts multi-omic
samples falling within each window to create a structured classification
dataset.

Usage::

    from bioagentics.crohns.flare_prediction.instance_builder import (
        build_instances, InstanceSet,
    )

    instances = build_instances(windows, data, lead_weeks=2)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)

# Omic layers that are required vs optional
REQUIRED_LAYERS = ("species", "pathways", "metabolomics")
OPTIONAL_LAYERS = ("serology", "transcriptomics")
ALL_LAYERS = REQUIRED_LAYERS + OPTIONAL_LAYERS


@dataclass
class Instance:
    """A single classification instance (one window)."""

    instance_id: int
    subject_id: str
    label: str  # "pre_flare" or "stable"
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    n_samples_in_window: int
    available_layers: list[str] = field(default_factory=list)


@dataclass
class InstanceSet:
    """Collection of instances with their feature matrices."""

    instances: list[Instance]
    features: pd.DataFrame  # rows = instance_id, cols = features
    layer_availability: pd.DataFrame  # rows = instance_id, cols = layer names, values = bool

    @property
    def n_instances(self) -> int:
        return len(self.instances)

    @property
    def n_pre_flare(self) -> int:
        return sum(1 for i in self.instances if i.label == "pre_flare")

    @property
    def n_stable(self) -> int:
        return sum(1 for i in self.instances if i.label == "stable")

    @property
    def labels(self) -> pd.Series:
        return pd.Series(
            {i.instance_id: i.label for i in self.instances}, name="label"
        )

    def summary(self) -> dict:
        patients = {i.subject_id for i in self.instances}
        per_patient = {}
        for i in self.instances:
            per_patient.setdefault(i.subject_id, []).append(i)
        return {
            "n_instances": self.n_instances,
            "n_pre_flare": self.n_pre_flare,
            "n_stable": self.n_stable,
            "n_patients": len(patients),
            "instances_per_patient_mean": (
                sum(len(v) for v in per_patient.values()) / len(per_patient)
                if per_patient else 0
            ),
            "n_features": self.features.shape[1],
            "layer_availability": {
                col: float(self.layer_availability[col].mean())
                for col in self.layer_availability.columns
            },
        }


def _extract_window_samples(
    layer_df: pd.DataFrame, window: Window
) -> pd.DataFrame:
    """Get samples from a layer that fall within a time window.

    Requires a ``date`` column (or the layer indexed by date-sortable visit_num).
    """
    df = layer_df.reset_index()
    if "subject_id" not in df.columns:
        return pd.DataFrame()

    mask = df["subject_id"] == window.subject_id

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        mask = mask & (df["date"] >= window.window_start) & (df["date"] <= window.window_end)
    # Fallback: if no date, just use all samples for the subject
    # (downstream feature engineering handles temporal alignment)

    return df.loc[mask]


def _aggregate_samples(samples: pd.DataFrame, prefix: str) -> pd.Series:
    """Aggregate multiple samples within a window into a single feature vector.

    Uses mean of numeric columns. Prefixes column names with the layer name.
    """
    meta_cols = {"subject_id", "visit_num", "date", "diagnosis"}
    numeric = samples.select_dtypes(include="number")
    numeric = numeric.drop(columns=[c for c in meta_cols if c in numeric.columns], errors="ignore")

    if numeric.empty:
        return pd.Series(dtype=float)

    means = numeric.mean()
    means.index = [f"{prefix}__{col}" for col in means.index]
    return means


def build_instances(
    windows: list[Window],
    data: dict[str, pd.DataFrame | None],
    hbi: pd.DataFrame | None = None,
) -> InstanceSet:
    """Build classification instances from windows and multi-omic data.

    Parameters
    ----------
    windows:
        List of Window instances from ``extract_windows``.
    data:
        Dict of omic DataFrames keyed by layer name (from ``HMP2DataLoader.load_all``).
        Must include metadata with ``date`` column for temporal alignment.
    hbi:
        Optional HBI DataFrame. If provided and ``data`` does not already
        contain a ``date`` column per layer, HBI dates are used to map
        visit_num → date for temporal alignment.

    Returns
    -------
    InstanceSet with feature matrices and layer availability.
    """
    # Ensure date info is available in omic layers
    date_map = _build_date_map(data, hbi)

    instances: list[Instance] = []
    feature_rows: list[pd.Series] = []
    availability_rows: list[dict[str, bool]] = []

    for idx, window in enumerate(windows):
        layer_avail: dict[str, bool] = {}
        layer_aggs: list[pd.Series] = []

        n_samples = 0
        available: list[str] = []

        for layer_name in ALL_LAYERS:
            layer_df = data.get(layer_name)
            if layer_df is None:
                layer_avail[layer_name] = False
                continue

            # Inject dates if not present
            layer_with_dates = _inject_dates(layer_df, date_map)
            samples = _extract_window_samples(layer_with_dates, window)

            if len(samples) == 0:
                layer_avail[layer_name] = False
                continue

            layer_avail[layer_name] = True
            available.append(layer_name)
            n_samples = max(n_samples, len(samples))

            agg = _aggregate_samples(samples, prefix=layer_name)
            if not agg.empty:
                layer_aggs.append(agg)

        instance_features = pd.concat(layer_aggs) if layer_aggs else pd.Series(dtype=float, name=idx)

        instances.append(
            Instance(
                instance_id=idx,
                subject_id=window.subject_id,
                label=window.label,
                window_start=window.window_start,
                window_end=window.window_end,
                n_samples_in_window=n_samples,
                available_layers=available,
            )
        )
        feature_rows.append(instance_features)
        availability_rows.append(layer_avail)

    features = pd.DataFrame(feature_rows, index=range(len(windows)))
    features.index.name = "instance_id"

    layer_availability = pd.DataFrame(availability_rows, index=range(len(windows)))
    layer_availability.index.name = "instance_id"

    logger.info(
        "Built %d instances (%d pre-flare, %d stable) with %d features",
        len(instances),
        sum(1 for i in instances if i.label == "pre_flare"),
        sum(1 for i in instances if i.label == "stable"),
        features.shape[1],
    )
    return InstanceSet(
        instances=instances, features=features, layer_availability=layer_availability
    )


def _build_date_map(
    data: dict[str, pd.DataFrame | None],
    hbi: pd.DataFrame | None,
) -> dict[tuple[str, int], pd.Timestamp]:
    """Build a mapping from (subject_id, visit_num) → date."""
    date_map: dict[tuple[str, int], pd.Timestamp] = {}

    # Try metadata first
    meta = data.get("metadata")
    if meta is not None:
        df = meta.reset_index()
        if "date" in df.columns and "subject_id" in df.columns and "visit_num" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            for _, row in df.iterrows():
                date_map[(str(row["subject_id"]), int(row["visit_num"]))] = row["date"]

    # Fall back to HBI if metadata lacks dates
    if not date_map and hbi is not None:
        df = hbi.reset_index()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            for _, row in df.iterrows():
                date_map[(str(row["subject_id"]), int(row["visit_num"]))] = row["date"]

    return date_map


def _inject_dates(
    layer_df: pd.DataFrame,
    date_map: dict[tuple[str, int], pd.Timestamp],
) -> pd.DataFrame:
    """Add a ``date`` column to a layer if it doesn't have one."""
    df = layer_df.reset_index()
    if "date" in df.columns:
        return df

    if "subject_id" in df.columns and "visit_num" in df.columns and date_map:
        df["date"] = df.apply(
            lambda r: date_map.get((str(r["subject_id"]), int(r["visit_num"]))),
            axis=1,
        )
    return df
