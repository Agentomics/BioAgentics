"""Mel-spectrogram generation for deep learning pipeline.

Converts raw audio files to mel-spectrogram tensors suitable for CNN input.
Uses librosa for spectrogram computation and normalizes to [0, 1] range.
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
from bioagentics.voice_pd.utils import load_audio, pad_or_trim

log = logging.getLogger(__name__)


def audio_to_mel_spectrogram(
    audio_path: str | Path,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    duration_sec: float = FIXED_DURATION_SEC,
) -> np.ndarray:
    """Convert audio file to a log-mel spectrogram.

    Args:
        audio_path: Path to WAV audio file.
        n_mels: Number of mel frequency bins.
        n_fft: FFT window size.
        hop_length: Hop length for STFT.
        duration_sec: Fixed duration to pad/trim audio to.

    Returns:
        2D numpy array of shape (n_mels, time_frames) with values in [0, 1].
    """
    import librosa

    y, sr = load_audio(audio_path)
    target_len = int(duration_sec * sr)
    y = pad_or_trim(y, target_len)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_min = log_mel.min()
    mel_max = log_mel.max()
    if mel_max - mel_min > 0:
        log_mel = (log_mel - mel_min) / (mel_max - mel_min)
    else:
        log_mel = np.zeros_like(log_mel)

    return log_mel.astype(np.float32)


def mel_to_rgb_tensor(mel: np.ndarray) -> np.ndarray:
    """Convert single-channel mel-spectrogram to 3-channel RGB array.

    MobileNetV2 expects 3-channel input. We replicate the single mel
    channel across R, G, B.

    Args:
        mel: 2D array of shape (n_mels, time_frames).

    Returns:
        3D array of shape (3, n_mels, time_frames).
    """
    return np.stack([mel, mel, mel], axis=0)


def audio_to_cnn_input(
    audio_path: str | Path,
    n_mels: int = N_MELS,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    duration_sec: float = FIXED_DURATION_SEC,
) -> np.ndarray:
    """Full pipeline: audio file -> 3-channel mel-spectrogram tensor.

    Returns:
        3D numpy array of shape (3, n_mels, time_frames), values in [0, 1].
    """
    mel = audio_to_mel_spectrogram(
        audio_path, n_mels=n_mels, n_fft=n_fft,
        hop_length=hop_length, duration_sec=duration_sec,
    )
    return mel_to_rgb_tensor(mel)
