"""Audio standardization pipeline — resample all audio to 16kHz mono WAV.

Handles various input formats (mp3, m4a, wav at different sample rates).
Saves processed files to data/voice-biomarkers-parkinsons/processed/.
Generates a unified manifest combining all datasets.
"""

import csv
import logging
from pathlib import Path

from bioagentics.voice_pd.config import (
    AUDIO_FORMAT,
    DATA_DIR,
    PROCESSED_DIR,
    SAMPLE_RATE,
)

log = logging.getLogger(__name__)

# Supported input audio formats
SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def standardize_audio(
    input_path: str | Path,
    output_path: str | Path | None = None,
    sr: int = SAMPLE_RATE,
) -> Path | None:
    """Resample audio to 16kHz mono WAV.

    Args:
        input_path: Path to input audio file (any supported format).
        output_path: Output WAV path. Auto-generated if None.
        sr: Target sample rate.

    Returns:
        Path to standardized WAV file, or None on failure.
    """
    import librosa
    import soundfile as sf

    input_path = Path(input_path)
    if not input_path.exists():
        log.warning("Input file not found: %s", input_path)
        return None

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        log.warning("Unsupported format: %s", input_path.suffix)
        return None

    if output_path is None:
        output_path = PROCESSED_DIR / f"{input_path.stem}.{AUDIO_FORMAT}"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        y, loaded_sr = librosa.load(str(input_path), sr=sr, mono=True)
        sf.write(str(output_path), y, sr)
        log.debug("Standardized %s -> %s (%d samples)", input_path, output_path, len(y))
        return output_path
    except Exception:
        log.exception("Failed to standardize: %s", input_path)
        return None


def standardize_dataset(
    raw_dir: str | Path,
    output_dir: str | Path | None = None,
    dataset_name: str = "",
) -> list[dict]:
    """Standardize all audio files in a directory.

    Args:
        raw_dir: Directory containing raw audio files.
        output_dir: Output directory. Defaults to PROCESSED_DIR/dataset_name/.
        dataset_name: Name for this dataset (used in manifest).

    Returns:
        List of manifest rows with original_path, processed_path, dataset.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        log.warning("Raw directory not found: %s", raw_dir)
        return []

    if output_dir is None:
        output_dir = PROCESSED_DIR / (dataset_name or raw_dir.name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    audio_files = [
        f for f in sorted(raw_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    log.info("Standardizing %d files from %s", len(audio_files), raw_dir)

    for i, audio_file in enumerate(audio_files):
        # Preserve relative path structure
        rel = audio_file.relative_to(raw_dir)
        out_path = output_dir / rel.with_suffix(f".{AUDIO_FORMAT}")

        result = standardize_audio(audio_file, out_path)
        if result is not None:
            manifest_rows.append({
                "recording_id": audio_file.stem,
                "original_path": str(audio_file),
                "audio_path": str(result),
                "dataset": dataset_name,
            })

        if (i + 1) % 100 == 0:
            log.info("Processed %d/%d files", i + 1, len(audio_files))

    log.info("Standardized %d/%d files from %s", len(manifest_rows), len(audio_files), raw_dir)
    return manifest_rows


def build_unified_manifest(
    dataset_manifests: dict[str, list[dict]],
    output_path: str | Path | None = None,
) -> Path:
    """Combine per-dataset manifests into a single unified manifest.

    Args:
        dataset_manifests: {dataset_name: list of manifest rows}.
        output_path: Output CSV path. Defaults to DATA_DIR/unified_manifest.csv.

    Returns:
        Path to unified manifest CSV.
    """
    if output_path is None:
        output_path = DATA_DIR / "unified_manifest.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for name, rows in dataset_manifests.items():
        for row in rows:
            row.setdefault("dataset", name)
            all_rows.append(row)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        # Ensure common columns come first
        priority = ["recording_id", "audio_path", "dataset", "pd_status"]
        ordered = [c for c in priority if c in fieldnames]
        ordered += [c for c in fieldnames if c not in ordered]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ordered, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(all_rows)
        log.info("Wrote unified manifest: %d rows to %s", len(all_rows), output_path)
    else:
        log.warning("No rows for unified manifest")

    return output_path
