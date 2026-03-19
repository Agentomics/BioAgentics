"""Unified classical feature extraction pipeline.

Combines all feature extractors (acoustic, pitch, mfcc, temporal) into a
single extract_all_features() function. Processes recordings from a manifest
and outputs features to a CSV.
"""

import csv
import logging
from pathlib import Path

from bioagentics.voice_pd.config import FEATURES_DIR
from bioagentics.voice_pd.features.acoustic import extract_acoustic_features
from bioagentics.voice_pd.features.mfcc import extract_mfcc_features
from bioagentics.voice_pd.features.pitch import extract_pitch_features
from bioagentics.voice_pd.features.temporal import extract_temporal_features
from bioagentics.voice_pd.utils import read_manifest

log = logging.getLogger(__name__)


def extract_all_features(audio_path: str | Path) -> dict[str, float | None]:
    """Extract all classical features from a single audio file.

    Combines acoustic (jitter/shimmer/HNR), pitch (F0/formants),
    MFCC (78-dim), and temporal (speech rate/pauses) features.

    Returns a single flat dict of all features.
    """
    features: dict[str, float | None] = {}

    # Acoustic biomarkers (jitter, shimmer, HNR, NHR)
    try:
        features.update(extract_acoustic_features(audio_path))
    except Exception:
        log.exception("Acoustic extraction failed: %s", audio_path)

    # F0 and formant frequencies
    try:
        features.update(extract_pitch_features(audio_path))
    except Exception:
        log.exception("Pitch extraction failed: %s", audio_path)

    # MFCCs + deltas
    try:
        features.update(extract_mfcc_features(audio_path))
    except Exception:
        log.exception("MFCC extraction failed: %s", audio_path)

    # Speech rate and pause patterns
    try:
        features.update(extract_temporal_features(audio_path))
    except Exception:
        log.exception("Temporal extraction failed: %s", audio_path)

    return features


def process_manifest(
    manifest_path: str | Path,
    output_path: str | Path | None = None,
    audio_col: str = "audio_path",
    id_col: str = "recording_id",
    label_col: str = "pd_status",
) -> Path:
    """Process all recordings in a manifest CSV and output features.

    Args:
        manifest_path: Path to CSV with audio file paths and metadata.
        output_path: Output CSV path. Defaults to FEATURES_DIR/classical_features.csv.
        audio_col: Column name containing audio file paths.
        id_col: Column name for recording identifier.
        label_col: Column name for PD status label.

    Returns:
        Path to the output CSV file.
    """
    if output_path is None:
        output_path = FEATURES_DIR / "classical_features.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(manifest_path)
    log.info("Processing %d recordings from %s", len(manifest), manifest_path)

    rows: list[dict] = []
    failed = 0

    for i, row in enumerate(manifest):
        recording_id = row.get(id_col, f"recording_{i}")
        audio_path = row.get(audio_col, "")
        label = row.get(label_col, "")

        if not audio_path or not Path(audio_path).exists():
            log.warning("Skipping %s: audio file not found (%s)", recording_id, audio_path)
            failed += 1
            continue

        try:
            features = extract_all_features(audio_path)
            features["recording_id"] = recording_id
            features["pd_status"] = label
            features["audio_path"] = audio_path
            rows.append(features)
        except Exception:
            log.exception("Failed to extract features for %s", recording_id)
            failed += 1

        if (i + 1) % 100 == 0:
            log.info("Processed %d/%d recordings", i + 1, len(manifest))

    if rows:
        fieldnames = list(rows[0].keys())
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info(
            "Wrote %d rows to %s (%d failed)", len(rows), output_path, failed
        )
    else:
        log.warning("No features extracted — output file not created")

    return output_path
