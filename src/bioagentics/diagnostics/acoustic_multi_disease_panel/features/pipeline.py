"""Unified multi-disease feature extraction pipeline.

Combines classical voice features (from voice_pd) with cough-specific features
to produce a comprehensive feature vector for multi-disease screening.

Recording-type-aware: applies appropriate feature extractors based on
whether the recording is a sustained vowel, cough, connected speech,
or reading passage.
"""

import csv
import logging
from pathlib import Path

from bioagentics.diagnostics.acoustic_multi_disease_panel.config import (
    FEATURES_DIR,
    RECORDING_TYPES,
)
from bioagentics.diagnostics.acoustic_multi_disease_panel.features.cough import (
    extract_cough_features,
)
from bioagentics.diagnostics.acoustic_multi_disease_panel.features.spectral import (
    extract_spectral_features,
)
from bioagentics.voice_pd.features.acoustic import extract_acoustic_features
from bioagentics.voice_pd.features.mfcc import extract_mfcc_features
from bioagentics.voice_pd.features.pitch import extract_pitch_features
from bioagentics.voice_pd.features.temporal import extract_temporal_features

log = logging.getLogger(__name__)

# Which extractors apply to which recording types
_EXTRACTOR_MAP: dict[str, list[str]] = {
    "sustained_vowel": ["acoustic", "pitch", "mfcc", "spectral"],
    "cough": ["cough", "mfcc", "spectral"],
    "counting": ["acoustic", "pitch", "mfcc", "temporal", "spectral"],
    "reading_passage": ["acoustic", "pitch", "mfcc", "temporal", "spectral"],
}

_EXTRACTORS = {
    "acoustic": extract_acoustic_features,
    "pitch": extract_pitch_features,
    "mfcc": extract_mfcc_features,
    "temporal": extract_temporal_features,
    "cough": extract_cough_features,
    "spectral": extract_spectral_features,
}


def extract_features(
    audio_path: str | Path,
    recording_type: str = "reading_passage",
) -> dict[str, float | None]:
    """Extract features from a single audio file, adapting to recording type.

    Args:
        audio_path: Path to a WAV file (16kHz mono expected).
        recording_type: One of 'sustained_vowel', 'cough', 'counting',
                        'reading_passage'. Determines which extractors run.

    Returns:
        Flat dict of all extracted features. Features from non-applicable
        extractors are omitted (not set to None).
    """
    if recording_type not in RECORDING_TYPES:
        log.warning(
            "Unknown recording type '%s', using all extractors", recording_type
        )
        extractor_names = list(_EXTRACTORS.keys())
    else:
        extractor_names = _EXTRACTOR_MAP[recording_type]

    features: dict[str, float | None] = {}

    for name in extractor_names:
        extractor = _EXTRACTORS[name]
        try:
            features.update(extractor(audio_path))
        except Exception:
            log.exception(
                "%s extraction failed for %s", name, audio_path
            )

    return features


def extract_all_features(audio_path: str | Path) -> dict[str, float | None]:
    """Extract ALL feature types regardless of recording type.

    Useful for exploratory analysis to see which features have signal
    across different recording types.
    """
    features: dict[str, float | None] = {}

    for name, extractor in _EXTRACTORS.items():
        try:
            features.update(extractor(audio_path))
        except Exception:
            log.exception("%s extraction failed for %s", name, audio_path)

    return features


def process_manifest(
    manifest_path: str | Path,
    output_path: str | Path | None = None,
    audio_col: str = "audio_path",
    id_col: str = "recording_id",
    label_col: str = "condition",
    recording_type_col: str = "recording_type",
) -> Path:
    """Process all recordings in a manifest CSV and output features.

    The manifest must have columns for audio path, recording ID, condition
    label, and recording type. Features are extracted per-recording using
    the appropriate extractors for each recording type.

    Args:
        manifest_path: Path to CSV with audio file paths and metadata.
        output_path: Output CSV path. Defaults to FEATURES_DIR/multi_disease_features.csv.
        audio_col: Column name containing audio file paths.
        id_col: Column name for recording identifier.
        label_col: Column name for condition label (parkinsons/respiratory/mci/healthy).
        recording_type_col: Column name for recording type.

    Returns:
        Path to the output CSV file.
    """
    if output_path is None:
        output_path = FEATURES_DIR / "multi_disease_features.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest(manifest_path)
    log.info("Processing %d recordings from %s", len(manifest), manifest_path)

    rows: list[dict] = []
    failed = 0

    for i, row in enumerate(manifest):
        recording_id = row.get(id_col, f"recording_{i}")
        audio_path = row.get(audio_col, "")
        condition = row.get(label_col, "")
        recording_type = row.get(recording_type_col, "reading_passage")

        if not audio_path or not Path(audio_path).exists():
            log.warning(
                "Skipping %s: audio file not found (%s)", recording_id, audio_path
            )
            failed += 1
            continue

        try:
            feat = extract_features(audio_path, recording_type=recording_type)
            row_out: dict = dict(feat)
            row_out["recording_id"] = recording_id
            row_out["condition"] = condition
            row_out["recording_type"] = recording_type
            row_out["audio_path"] = audio_path
            rows.append(row_out)
        except Exception:
            log.exception("Failed to extract features for %s", recording_id)
            failed += 1

        if (i + 1) % 100 == 0:
            log.info("Processed %d/%d recordings", i + 1, len(manifest))

    if rows:
        # Collect all feature keys across all rows (different recording types
        # produce different feature sets)
        all_keys: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_keys, restval=None)
            writer.writeheader()
            writer.writerows(rows)
        log.info(
            "Wrote %d rows to %s (%d failed)", len(rows), output_path, failed
        )
    else:
        log.warning("No features extracted — output file not created")

    return output_path


def _read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV manifest file."""
    path = Path(path)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)
