"""Audio preprocessing for multi-disease acoustic screening.

Extends voice_pd preprocessing with multi-dataset, multi-recording-type
manifest generation. Standardizes audio from PD, respiratory, and MCI
datasets into a unified format with condition and recording type labels.
"""

import csv
import logging
from pathlib import Path

from bioagentics.diagnostics.acoustic_multi_disease_panel.config import (
    AUDIO_FORMAT,
    DATA_DIR,
    DATASETS,
    PROCESSED_DIR,
    SAMPLE_RATE,
)
from bioagentics.voice_pd.preprocessing import (
    SUPPORTED_EXTENSIONS,
    standardize_audio,
)

log = logging.getLogger(__name__)


def standardize_dataset(
    raw_dir: str | Path,
    output_dir: str | Path | None = None,
    dataset_name: str = "",
    condition: str = "",
    recording_type: str = "reading_passage",
) -> list[dict[str, str]]:
    """Standardize all audio files in a dataset directory.

    Args:
        raw_dir: Directory with raw audio files.
        output_dir: Output directory. Defaults to PROCESSED_DIR/dataset_name/.
        dataset_name: Dataset identifier.
        condition: Disease condition label (parkinsons/respiratory/mci/healthy).
        recording_type: Type of recording (sustained_vowel/cough/counting/reading_passage).

    Returns:
        List of manifest row dicts.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        log.warning("Raw directory not found: %s", raw_dir)
        return []

    if output_dir is None:
        output_dir = PROCESSED_DIR / (dataset_name or raw_dir.name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_files = sorted(
        f for f in raw_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    log.info("Standardizing %d files from %s", len(audio_files), raw_dir)

    manifest_rows: list[dict[str, str]] = []
    for i, audio_file in enumerate(audio_files):
        rel = audio_file.relative_to(raw_dir)
        out_path = output_dir / rel.with_suffix(f".{AUDIO_FORMAT}")

        result = standardize_audio(audio_file, out_path, sr=SAMPLE_RATE)
        if result is not None:
            manifest_rows.append({
                "recording_id": audio_file.stem,
                "audio_path": str(result),
                "original_path": str(audio_file),
                "dataset": dataset_name,
                "condition": condition,
                "recording_type": recording_type,
            })

        if (i + 1) % 100 == 0:
            log.info("Processed %d/%d files", i + 1, len(audio_files))

    log.info(
        "Standardized %d/%d files from %s",
        len(manifest_rows), len(audio_files), raw_dir,
    )
    return manifest_rows


def build_multi_disease_manifest(
    output_path: str | Path | None = None,
) -> Path:
    """Build unified manifest from all configured datasets.

    Iterates over DATASETS config, standardizes each dataset's audio,
    and produces a single manifest CSV with condition and recording type
    labels for every recording.

    Args:
        output_path: Output CSV path. Defaults to DATA_DIR/multi_disease_manifest.csv.

    Returns:
        Path to the manifest CSV.
    """
    if output_path is None:
        output_path = DATA_DIR / "multi_disease_manifest.csv"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, str]] = []

    for ds_key, ds_info in DATASETS.items():
        raw_dir = ds_info["raw_dir"]
        condition = ds_info["condition"]
        rec_types = ds_info["recording_types"]

        if not Path(raw_dir).is_dir():
            log.info("Skipping %s: raw directory not found (%s)", ds_key, raw_dir)
            continue

        for rec_type in rec_types:
            rows = standardize_dataset(
                raw_dir=raw_dir,
                dataset_name=ds_key,
                condition=condition,
                recording_type=rec_type,
            )
            all_rows.extend(rows)

    if all_rows:
        fieldnames = [
            "recording_id", "audio_path", "dataset", "condition",
            "recording_type", "original_path",
        ]
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        log.info("Wrote manifest: %d rows to %s", len(all_rows), output_path)
    else:
        log.warning("No datasets found — manifest not created")

    return output_path


def load_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load a multi-disease manifest CSV."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
