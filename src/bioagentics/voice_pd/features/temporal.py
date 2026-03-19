"""Speech rate and pause pattern analysis for connected speech.

PD patients show increased pausing and reduced speech rate.
Uses energy-based segmentation to identify speech/silence regions.
"""

import logging
from pathlib import Path

import numpy as np

from bioagentics.voice_pd.config import SAMPLE_RATE

log = logging.getLogger(__name__)

# Energy-based VAD parameters
FRAME_MS = 25  # frame length in ms
HOP_MS = 10  # hop length in ms
SILENCE_THRESHOLD_DB = -40.0  # below this is silence
MIN_SPEECH_MS = 80  # minimum speech segment duration
MIN_PAUSE_MS = 150  # minimum pause duration to count


def _frame_energy_db(y: np.ndarray, sr: int, frame_ms: int, hop_ms: int) -> np.ndarray:
    """Compute per-frame energy in dB."""
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)

    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        frame = y[start : start + frame_len]
        rms = np.sqrt(np.mean(frame**2) + 1e-10)
        energy[i] = 20.0 * np.log10(rms + 1e-10)
    return energy


def _segment_speech_silence(
    energy_db: np.ndarray, threshold_db: float, hop_ms: int
) -> list[tuple[str, float, float]]:
    """Segment into speech and silence regions.

    Returns list of (label, start_ms, end_ms) where label is 'speech' or 'silence'.
    """
    is_speech = energy_db > threshold_db
    segments: list[tuple[str, float, float]] = []

    if len(is_speech) == 0:
        return segments

    current_label = "speech" if is_speech[0] else "silence"
    start_idx = 0

    for i in range(1, len(is_speech)):
        label = "speech" if is_speech[i] else "silence"
        if label != current_label:
            segments.append((current_label, start_idx * hop_ms, i * hop_ms))
            current_label = label
            start_idx = i

    segments.append((current_label, start_idx * hop_ms, len(is_speech) * hop_ms))
    return segments


def extract_temporal_features(audio_path: str | Path) -> dict[str, float | None]:
    """Extract speech rate and pause features from an audio file."""
    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        log.exception("Failed to load audio: %s", audio_path)
        return _empty_features()

    return _extract_from_array(y, sr)


def extract_temporal_features_from_array(
    y: np.ndarray, sr: int = SAMPLE_RATE
) -> dict[str, float | None]:
    """Extract speech rate and pause features from a numpy audio array."""
    return _extract_from_array(y, sr)


def _extract_from_array(y: np.ndarray, sr: int) -> dict[str, float | None]:
    """Core temporal feature extraction."""
    if len(y) < sr * 0.1:  # less than 100ms
        return _empty_features()

    total_duration_ms = len(y) / sr * 1000.0

    # Compute frame energy
    energy_db = _frame_energy_db(y, sr, FRAME_MS, HOP_MS)

    # Adaptive threshold: use median energy - 20dB if it's higher than fixed threshold
    median_energy = float(np.median(energy_db))
    threshold = max(SILENCE_THRESHOLD_DB, median_energy - 20.0)

    # Segment into speech/silence
    segments = _segment_speech_silence(energy_db, threshold, HOP_MS)

    # Filter by minimum duration
    speech_segs = [
        (s, e) for label, s, e in segments
        if label == "speech" and (e - s) >= MIN_SPEECH_MS
    ]
    pause_segs = [
        (s, e) for label, s, e in segments
        if label == "silence" and (e - s) >= MIN_PAUSE_MS
    ]

    speech_durations = [e - s for s, e in speech_segs]
    pause_durations = [e - s for s, e in pause_segs]

    total_speech_ms = sum(speech_durations) if speech_durations else 0.0

    features: dict[str, float | None] = {}

    # Phonation time ratio
    features["phonation_ratio"] = (
        total_speech_ms / total_duration_ms if total_duration_ms > 0 else None
    )

    # Articulation rate (speech segments only, excluding pauses)
    # Approximate syllable count from energy peaks in speech segments
    features["n_speech_segments"] = float(len(speech_segs))

    # Pause features
    features["n_pauses"] = float(len(pause_segs))
    features["pause_frequency"] = (
        len(pause_segs) / (total_duration_ms / 1000.0) if total_duration_ms > 0 else None
    )
    features["mean_pause_ms"] = (
        float(np.mean(pause_durations)) if pause_durations else None
    )
    features["std_pause_ms"] = (
        float(np.std(pause_durations)) if len(pause_durations) > 1 else None
    )
    features["max_pause_ms"] = (
        float(np.max(pause_durations)) if pause_durations else None
    )

    # Speech rate approximation (speech segments per second of total audio)
    features["speech_rate"] = (
        len(speech_segs) / (total_duration_ms / 1000.0) if total_duration_ms > 0 else None
    )

    # Mean speech segment duration
    features["mean_speech_ms"] = (
        float(np.mean(speech_durations)) if speech_durations else None
    )

    return features


def _empty_features() -> dict[str, float | None]:
    """Return feature dict with all None values."""
    keys = [
        "phonation_ratio", "n_speech_segments", "n_pauses",
        "pause_frequency", "mean_pause_ms", "std_pause_ms",
        "max_pause_ms", "speech_rate", "mean_speech_ms",
    ]
    return {k: None for k in keys}
