"""Voice task protocol for mobile PD screening.

Defines the 30-second recording protocol: sustained vowel phonation
followed by reading a standard sentence. Includes audio segmentation
and validation utilities.
"""

import logging

import numpy as np

from bioagentics.voice_pd.config import FIXED_DURATION_SEC, SAMPLE_RATE

log = logging.getLogger(__name__)

# ── Protocol constants ──

PROTOCOL_DURATION_SEC = 30.0
VOWEL_DURATION_SEC = 10.0
SENTENCE_DURATION_SEC = 15.0
TRANSITION_SEC = 5.0  # pause between tasks

STANDARD_SENTENCE = (
    "The quick brown fox jumps over the lazy dog near the sunny riverbank."
)

PROTOCOL_STEPS = [
    {
        "task": "sustained_vowel",
        "instruction": 'Say "ahhh" at a comfortable pitch and volume for 10 seconds.',
        "start_sec": 0.0,
        "duration_sec": VOWEL_DURATION_SEC,
    },
    {
        "task": "transition",
        "instruction": "Pause briefly.",
        "start_sec": VOWEL_DURATION_SEC,
        "duration_sec": TRANSITION_SEC,
    },
    {
        "task": "sentence_reading",
        "instruction": f'Read aloud: "{STANDARD_SENTENCE}"',
        "start_sec": VOWEL_DURATION_SEC + TRANSITION_SEC,
        "duration_sec": SENTENCE_DURATION_SEC,
    },
]


def validate_recording(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    min_duration_sec: float = 20.0,
    max_clipping_ratio: float = 0.05,
    min_rms: float = 0.005,
) -> dict:
    """Validate a recording meets minimum quality for analysis.

    Args:
        audio: 1D float32 audio array.
        sr: Sample rate.
        min_duration_sec: Minimum acceptable recording length.
        max_clipping_ratio: Max fraction of clipped samples allowed.
        min_rms: Minimum RMS energy (rejects silence).

    Returns:
        Dict with is_valid, duration_sec, rms, clipping_ratio, issues list.
    """
    issues: list[str] = []

    duration_sec = len(audio) / sr
    if duration_sec < min_duration_sec:
        issues.append(f"Recording too short: {duration_sec:.1f}s < {min_duration_sec:.1f}s")

    rms = float(np.sqrt(np.mean(audio**2)))
    if rms < min_rms:
        issues.append(f"Recording too quiet (RMS={rms:.4f})")

    clipping_ratio = float(np.mean(np.abs(audio) > 0.99))
    if clipping_ratio > max_clipping_ratio:
        issues.append(f"Excessive clipping: {clipping_ratio:.1%} of samples")

    return {
        "is_valid": len(issues) == 0,
        "duration_sec": duration_sec,
        "rms": rms,
        "clipping_ratio": clipping_ratio,
        "issues": issues,
    }


def segment_recording(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> dict[str, np.ndarray]:
    """Segment a 30-second protocol recording into vowel and sentence parts.

    Uses fixed time boundaries defined by the protocol. Falls back
    gracefully if the recording is shorter than expected.

    Args:
        audio: 1D float32 audio array from the full protocol recording.
        sr: Sample rate.

    Returns:
        Dict with keys 'vowel' and 'sentence', each an audio segment.
    """
    total_samples = len(audio)

    vowel_start = 0
    vowel_end = min(int(VOWEL_DURATION_SEC * sr), total_samples)

    sentence_start = min(int((VOWEL_DURATION_SEC + TRANSITION_SEC) * sr), total_samples)
    sentence_end = total_samples

    return {
        "vowel": audio[vowel_start:vowel_end],
        "sentence": audio[sentence_start:sentence_end],
    }


def extract_vowel_clip(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    clip_duration_sec: float = FIXED_DURATION_SEC,
) -> np.ndarray:
    """Extract a fixed-length clip from the vowel segment for model input.

    Takes the center portion of the vowel to avoid onset/offset artifacts.
    Pads with zeros if the vowel segment is shorter than clip_duration.

    Args:
        audio: Vowel segment audio (1D float32).
        sr: Sample rate.
        clip_duration_sec: Target clip length in seconds.

    Returns:
        Audio clip of exactly clip_duration_sec * sr samples.
    """
    target_samples = int(clip_duration_sec * sr)
    n = len(audio)

    if n >= target_samples:
        # Center crop
        start = (n - target_samples) // 2
        return audio[start : start + target_samples]
    else:
        # Zero-pad
        clip = np.zeros(target_samples, dtype=audio.dtype)
        start = (target_samples - n) // 2
        clip[start : start + n] = audio
        return clip


def get_protocol_description() -> dict:
    """Return the full protocol specification as a serializable dict.

    Useful for embedding in mobile app configuration or documentation.
    """
    return {
        "name": "PD Voice Screening Protocol v1",
        "total_duration_sec": PROTOCOL_DURATION_SEC,
        "sample_rate": SAMPLE_RATE,
        "audio_format": "wav",
        "channels": 1,
        "steps": PROTOCOL_STEPS,
        "standard_sentence": STANDARD_SENTENCE,
        "analysis_clip_sec": FIXED_DURATION_SEC,
        "notes": [
            "Record in a quiet environment",
            "Hold phone 15-20 cm from mouth",
            "Use consistent volume throughout",
        ],
    }
