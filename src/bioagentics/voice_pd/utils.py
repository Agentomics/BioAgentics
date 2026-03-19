"""Shared utilities for voice-biomarkers-parkinsons project."""

import csv
import logging
from pathlib import Path

import numpy as np

from bioagentics.voice_pd.config import SAMPLE_RATE

log = logging.getLogger(__name__)


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio file as a numpy array at the target sample rate.

    Returns (audio_array, sample_rate).
    """
    import librosa

    y, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    return y, loaded_sr


def save_audio(y: np.ndarray, path: str | Path, sr: int = SAMPLE_RATE) -> None:
    """Save audio array to WAV file."""
    import soundfile as sf

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), y, sr)


def read_manifest(path: str | Path) -> list[dict]:
    """Read a CSV manifest file and return list of row dicts."""
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_manifest(rows: list[dict], path: str | Path) -> None:
    """Write a list of dicts to a CSV manifest file."""
    if not rows:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def duration_seconds(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Return duration of audio in seconds."""
    return len(y) / sr


def pad_or_trim(y: np.ndarray, target_length: int) -> np.ndarray:
    """Pad with zeros or trim audio to exact target length in samples."""
    if len(y) >= target_length:
        return y[:target_length]
    return np.pad(y, (0, target_length - len(y)))
