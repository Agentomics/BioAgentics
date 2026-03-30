"""Composite spectrogram generation for multi-vowel PD detection.

Vertically stacks log-mel spectrograms from 5 sustained vowels (/a/, /e/,
/i/, /o/, /u/) per patient into a single composite image.  Missing vowels
are zero-padded so every composite has identical dimensions.

Reference: RD journal #2242 — literature reports AUROC 0.928 for composite
spectrogram approaches vs 0.85-0.89 for single-vowel models.
"""

import logging
from pathlib import Path

import numpy as np

from bioagentics.voice_pd.config import (
    FIXED_DURATION_SEC,
    HOP_LENGTH,
    N_FFT,
    N_MELS,
)
from bioagentics.voice_pd.deep.spectrogram import audio_to_mel_spectrogram

log = logging.getLogger(__name__)

VOWELS = ("a", "e", "i", "o", "u")


def generate_composite_spectrogram(
    vowel_paths: dict[str, str | Path | None],
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    duration_sec: float = FIXED_DURATION_SEC,
) -> np.ndarray:
    """Build a composite spectrogram by vertically stacking per-vowel spectrograms.

    Args:
        vowel_paths: Mapping from vowel label to audio file path.
            Keys should be from VOWELS ('a', 'e', 'i', 'o', 'u').
            A value of None or a missing key means the vowel is unavailable
            and will be zero-padded.
        n_mels: Number of mel frequency bins per vowel strip.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        duration_sec: Fixed duration each vowel clip is padded/trimmed to.

    Returns:
        2D numpy array of shape (n_mels * 5, time_frames) with values in [0, 1].
        Vowels are stacked top-to-bottom in VOWELS order (a, e, i, o, u).
    """
    target_samples = int(duration_sec * 16_000)
    # Compute expected time frames from spectrogram parameters
    expected_frames = 1 + target_samples // hop_length

    strips: list[np.ndarray] = []
    for vowel in VOWELS:
        path = vowel_paths.get(vowel)
        if path is not None:
            path = Path(path)
            if path.exists():
                mel = audio_to_mel_spectrogram(
                    path,
                    n_mels=n_mels,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    duration_sec=duration_sec,
                )
                # Ensure consistent width
                if mel.shape[1] != expected_frames:
                    padded = np.zeros((n_mels, expected_frames), dtype=np.float32)
                    w = min(mel.shape[1], expected_frames)
                    padded[:, :w] = mel[:, :w]
                    mel = padded
                strips.append(mel)
                continue

        # Zero-pad missing vowel
        log.debug("Vowel '%s' missing — zero-padding", vowel)
        strips.append(np.zeros((n_mels, expected_frames), dtype=np.float32))

    composite = np.vstack(strips)  # (n_mels * 5, time_frames)
    return composite


def composite_to_rgb_tensor(composite: np.ndarray) -> np.ndarray:
    """Convert a composite spectrogram to a 3-channel tensor for CNN input.

    Args:
        composite: 2D array of shape (n_mels * 5, time_frames).

    Returns:
        3D array of shape (3, n_mels * 5, time_frames).
    """
    return np.stack([composite, composite, composite], axis=0).astype(np.float32)


def patient_audio_to_composite(
    vowel_paths: dict[str, str | Path | None],
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    duration_sec: float = FIXED_DURATION_SEC,
) -> np.ndarray:
    """Full pipeline: patient vowel audio files -> 3-channel composite tensor.

    Returns:
        3D numpy array of shape (3, n_mels * 5, time_frames), values in [0, 1].
    """
    composite = generate_composite_spectrogram(
        vowel_paths,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        duration_sec=duration_sec,
    )
    return composite_to_rgb_tensor(composite)


def batch_generate_composites(
    patients: list[dict[str, str | Path | None]],
    labels: np.ndarray,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    duration_sec: float = FIXED_DURATION_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate composite spectrograms for a batch of patients.

    Processes patients one at a time to stay within 8GB RAM.

    Args:
        patients: List of dicts, each mapping vowel label -> audio path (or None).
        labels: Array of shape (N,) with binary labels (1=PD, 0=healthy).
        n_mels: Number of mel frequency bins.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        duration_sec: Fixed vowel duration.

    Returns:
        Tuple of (composites, labels) where composites has shape
        (N, 3, n_mels * 5, time_frames).
    """
    composites = []
    for i, vowel_paths in enumerate(patients):
        tensor = patient_audio_to_composite(
            vowel_paths,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            duration_sec=duration_sec,
        )
        composites.append(tensor)
        if (i + 1) % 50 == 0:
            log.info("Generated %d/%d composite spectrograms", i + 1, len(patients))

    return np.array(composites, dtype=np.float32), labels
