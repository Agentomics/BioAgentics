"""Audio augmentation for robustness training.

Simulates real-world degradation: background noise, microphone frequency
response limitations, and lossy compression artifacts.
"""

import numpy as np

from bioagentics.voice_pd.config import SAMPLE_RATE


def add_background_noise(
    y: np.ndarray,
    snr_db: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Gaussian background noise at a specified SNR.

    Args:
        y: Audio signal (mono float32).
        snr_db: Desired signal-to-noise ratio in dB.
        rng: NumPy random generator (for reproducibility).

    Returns:
        Noisy signal with same shape and dtype as *y*.
    """
    if rng is None:
        rng = np.random.default_rng()

    signal_power = np.mean(y ** 2)
    if signal_power < 1e-12:
        # Silent signal — noise would dominate; return unchanged.
        return y.copy()

    noise = rng.normal(0, 1, size=y.shape).astype(y.dtype)
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = noise * np.sqrt(noise_power / np.mean(noise ** 2))

    return (y + noise).astype(y.dtype)


def add_microphone_artifacts(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    low_cutoff_hz: float = 200.0,
    high_cutoff_hz: float = 4000.0,
) -> np.ndarray:
    """Simulate cheap-microphone frequency response via bandpass filtering.

    Uses a simple FFT-domain brick-wall filter to attenuate energy outside
    the passband, mimicking the limited bandwidth of phone microphones.

    Args:
        y: Audio signal (mono float32).
        sr: Sample rate.
        low_cutoff_hz: Lower cutoff frequency.
        high_cutoff_hz: Upper cutoff frequency.

    Returns:
        Filtered signal with same shape and dtype.
    """
    n = len(y)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    spectrum = np.fft.rfft(y)

    # Brick-wall bandpass
    mask = (freqs >= low_cutoff_hz) & (freqs <= high_cutoff_hz)
    spectrum[~mask] = 0.0

    filtered = np.fft.irfft(spectrum, n=n)
    return filtered.astype(y.dtype)


def add_compression_artifacts(
    y: np.ndarray,
    bit_depth: int = 8,
) -> np.ndarray:
    """Simulate lossy compression by quantizing to a reduced bit depth.

    Args:
        y: Audio signal (mono float32), expected range [-1, 1].
        bit_depth: Target quantization depth (e.g. 8 for 256 levels).

    Returns:
        Quantized signal with same shape and dtype.
    """
    levels = 2 ** bit_depth
    quantized = np.round(y * (levels / 2)) / (levels / 2)
    return quantized.astype(y.dtype)


def augment_audio(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    noise_snr_db: float | None = 20.0,
    mic_low_hz: float | None = 200.0,
    mic_high_hz: float | None = 4000.0,
    compression_bits: int | None = 8,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a chain of augmentations to simulate real-world recording conditions.

    Each augmentation step can be disabled by passing ``None`` for its parameter.

    Args:
        y: Audio signal (mono float32).
        sr: Sample rate.
        noise_snr_db: SNR for background noise (None to skip).
        mic_low_hz: Low cutoff for mic simulation (None to skip).
        mic_high_hz: High cutoff for mic simulation (None to skip).
        compression_bits: Bit depth for compression artifacts (None to skip).
        rng: NumPy random generator.

    Returns:
        Augmented signal with same shape and dtype.
    """
    out = y.copy()

    if noise_snr_db is not None:
        out = add_background_noise(out, snr_db=noise_snr_db, rng=rng)

    if mic_low_hz is not None and mic_high_hz is not None:
        out = add_microphone_artifacts(out, sr=sr, low_cutoff_hz=mic_low_hz, high_cutoff_hz=mic_high_hz)

    if compression_bits is not None:
        out = add_compression_artifacts(out, bit_depth=compression_bits)

    return out
