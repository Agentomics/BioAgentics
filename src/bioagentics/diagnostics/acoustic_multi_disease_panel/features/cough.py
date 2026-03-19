"""Cough-specific acoustic feature extraction for respiratory screening.

Extracts features relevant to TB and COPD detection from cough recordings:
spectral shape, energy envelope, and cough event segmentation.
"""

import logging
from pathlib import Path

import numpy as np

from bioagentics.diagnostics.acoustic_multi_disease_panel.config import (
    HOP_LENGTH,
    N_FFT,
    SAMPLE_RATE,
)

log = logging.getLogger(__name__)

# Cough segmentation parameters
COUGH_ENERGY_THRESHOLD_DB = -30.0
MIN_COUGH_DURATION_MS = 50
MAX_COUGH_DURATION_MS = 2000
MIN_INTER_COUGH_MS = 100


def _segment_cough_events(
    y: np.ndarray, sr: int
) -> list[tuple[int, int]]:
    """Segment audio into individual cough events using energy envelope.

    Returns list of (start_sample, end_sample) tuples.
    """
    frame_len = int(sr * 0.025)
    hop_len = int(sr * 0.010)
    n_frames = max(1, (len(y) - frame_len) // hop_len + 1)

    energy = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_len
        frame = y[start : start + frame_len]
        rms = np.sqrt(np.mean(frame**2) + 1e-10)
        energy[i] = 20.0 * np.log10(rms + 1e-10)

    # Adaptive threshold
    threshold = max(COUGH_ENERGY_THRESHOLD_DB, float(np.median(energy)) + 6.0)
    is_active = energy > threshold

    # Find contiguous active regions
    events: list[tuple[int, int]] = []
    in_event = False
    start_frame = 0

    for i in range(len(is_active)):
        if is_active[i] and not in_event:
            in_event = True
            start_frame = i
        elif not is_active[i] and in_event:
            in_event = False
            start_sample = start_frame * hop_len
            end_sample = min(i * hop_len, len(y))
            dur_ms = (end_sample - start_sample) / sr * 1000
            if MIN_COUGH_DURATION_MS <= dur_ms <= MAX_COUGH_DURATION_MS:
                events.append((start_sample, end_sample))

    if in_event:
        start_sample = start_frame * hop_len
        end_sample = len(y)
        dur_ms = (end_sample - start_sample) / sr * 1000
        if MIN_COUGH_DURATION_MS <= dur_ms <= MAX_COUGH_DURATION_MS:
            events.append((start_sample, end_sample))

    # Merge events closer than MIN_INTER_COUGH_MS
    if len(events) > 1:
        merged: list[tuple[int, int]] = [events[0]]
        for start, end in events[1:]:
            prev_end = merged[-1][1]
            gap_ms = (start - prev_end) / sr * 1000
            if gap_ms < MIN_INTER_COUGH_MS:
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        events = merged

    return events


def extract_cough_features(audio_path: str | Path) -> dict[str, float | None]:
    """Extract cough-specific features from an audio file.

    Features include spectral shape descriptors, cough event statistics,
    and energy envelope characteristics relevant to respiratory screening.
    """
    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        log.exception("Failed to load audio: %s", audio_path)
        return _empty_features()

    return _extract_from_array(y, int(sr))


def extract_cough_features_from_array(
    y: np.ndarray, sr: int = SAMPLE_RATE
) -> dict[str, float | None]:
    """Extract cough features from a numpy audio array."""
    return _extract_from_array(y, sr)


def _extract_from_array(y: np.ndarray, sr: int) -> dict[str, float | None]:
    """Core cough feature extraction."""
    import librosa

    if len(y) < sr * 0.05:
        return _empty_features()

    features: dict[str, float | None] = {}

    # -- Spectral shape features (computed on full recording) --
    try:
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        features["spectral_centroid_mean"] = float(np.mean(spectral_centroid))
        features["spectral_centroid_std"] = float(np.std(spectral_centroid))
    except Exception:
        log.warning("spectral_centroid extraction failed", exc_info=True)
        features["spectral_centroid_mean"] = None
        features["spectral_centroid_std"] = None

    try:
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloff))
        features["spectral_rolloff_std"] = float(np.std(spectral_rolloff))
    except Exception:
        log.warning("spectral_rolloff extraction failed", exc_info=True)
        features["spectral_rolloff_mean"] = None
        features["spectral_rolloff_std"] = None

    try:
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidth))
        features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidth))
    except Exception:
        log.warning("spectral_bandwidth extraction failed", exc_info=True)
        features["spectral_bandwidth_mean"] = None
        features["spectral_bandwidth_std"] = None

    try:
        spectral_flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=N_FFT, hop_length=HOP_LENGTH
        )[0]
        features["spectral_flatness_mean"] = float(np.mean(spectral_flatness))
        features["spectral_flatness_std"] = float(np.std(spectral_flatness))
    except Exception:
        log.warning("spectral_flatness extraction failed", exc_info=True)
        features["spectral_flatness_mean"] = None
        features["spectral_flatness_std"] = None

    # -- Zero-crossing rate --
    try:
        zcr = librosa.feature.zero_crossing_rate(
            y=y, frame_length=N_FFT, hop_length=HOP_LENGTH
        )[0]
        features["zcr_mean"] = float(np.mean(zcr))
        features["zcr_std"] = float(np.std(zcr))
    except Exception:
        log.warning("zero_crossing_rate extraction failed", exc_info=True)
        features["zcr_mean"] = None
        features["zcr_std"] = None

    # -- Cough event segmentation and statistics --
    try:
        events = _segment_cough_events(y, sr)
        features["n_cough_events"] = float(len(events))

        if events:
            durations_ms = [(e - s) / sr * 1000 for s, e in events]
            features["cough_duration_mean_ms"] = float(np.mean(durations_ms))
            features["cough_duration_std_ms"] = (
                float(np.std(durations_ms)) if len(durations_ms) > 1 else 0.0
            )

            if len(events) > 1:
                gaps_ms = [
                    (events[i + 1][0] - events[i][1]) / sr * 1000
                    for i in range(len(events) - 1)
                ]
                features["inter_cough_mean_ms"] = float(np.mean(gaps_ms))
                features["inter_cough_std_ms"] = (
                    float(np.std(gaps_ms)) if len(gaps_ms) > 1 else 0.0
                )
            else:
                features["inter_cough_mean_ms"] = None
                features["inter_cough_std_ms"] = None

            # Per-event peak frequency (dominant frequency in each cough)
            peak_freqs = []
            for start, end in events:
                cough_segment = y[start:end]
                if len(cough_segment) > N_FFT:
                    spectrum = np.abs(np.fft.rfft(cough_segment, n=N_FFT))
                    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / sr)
                    peak_freqs.append(float(freqs[np.argmax(spectrum)]))
            if peak_freqs:
                features["cough_peak_freq_mean"] = float(np.mean(peak_freqs))
                features["cough_peak_freq_std"] = (
                    float(np.std(peak_freqs)) if len(peak_freqs) > 1 else 0.0
                )
            else:
                features["cough_peak_freq_mean"] = None
                features["cough_peak_freq_std"] = None
        else:
            features["cough_duration_mean_ms"] = None
            features["cough_duration_std_ms"] = None
            features["inter_cough_mean_ms"] = None
            features["inter_cough_std_ms"] = None
            features["cough_peak_freq_mean"] = None
            features["cough_peak_freq_std"] = None
    except Exception:
        log.exception("Cough event extraction failed")
        features["n_cough_events"] = None
        features["cough_duration_mean_ms"] = None
        features["cough_duration_std_ms"] = None
        features["inter_cough_mean_ms"] = None
        features["inter_cough_std_ms"] = None
        features["cough_peak_freq_mean"] = None
        features["cough_peak_freq_std"] = None

    # -- RMS energy envelope --
    try:
        rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
        features["rms_energy_mean"] = float(np.mean(rms))
        features["rms_energy_std"] = float(np.std(rms))
        features["rms_energy_max"] = float(np.max(rms))
    except Exception:
        log.warning("rms_energy extraction failed", exc_info=True)
        features["rms_energy_mean"] = None
        features["rms_energy_std"] = None
        features["rms_energy_max"] = None

    return features


def _empty_features() -> dict[str, float | None]:
    """Return feature dict with all None values."""
    keys = [
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_rolloff_mean", "spectral_rolloff_std",
        "spectral_bandwidth_mean", "spectral_bandwidth_std",
        "spectral_flatness_mean", "spectral_flatness_std",
        "zcr_mean", "zcr_std",
        "n_cough_events",
        "cough_duration_mean_ms", "cough_duration_std_ms",
        "inter_cough_mean_ms", "inter_cough_std_ms",
        "cough_peak_freq_mean", "cough_peak_freq_std",
        "rms_energy_mean", "rms_energy_std", "rms_energy_max",
    ]
    return {k: None for k in keys}
