"""End-to-end mobile inference benchmarking.

Measures the full pipeline time: audio preprocessing → spectrogram
generation → model inference → probability output. Validates against
the <2 second mobile inference target.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np

from bioagentics.voice_pd.config import (
    EVAL_DIR,
    MOBILE_INFERENCE_SEC,
    N_MELS,
    SAMPLE_RATE,
)

log = logging.getLogger(__name__)


def _generate_spectrogram(audio_clip: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a 3-channel mel-spectrogram from an audio clip.

    Produces the same format expected by SpectrogramCNN:
    (3, n_mels, time_frames) in [0, 1].

    Uses numpy-only approximation for benchmarking when librosa unavailable.
    """
    try:
        import librosa

        S = librosa.feature.melspectrogram(
            y=audio_clip, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512,
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        # Normalize to [0, 1]
        S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
        # Stack to 3 channels (RGB-like for MobileNetV2)
        return np.stack([S_norm, S_norm, S_norm], axis=0).astype(np.float32)
    except ImportError:
        # Fallback: numpy-only STFT approximation
        n_fft = 2048
        hop = 512
        n_frames = 1 + (len(audio_clip) - n_fft) // hop
        if n_frames <= 0:
            spec = np.zeros((N_MELS, 1), dtype=np.float32)
        else:
            frames = np.lib.stride_tricks.sliding_window_view(audio_clip, n_fft)[::hop]
            window = np.hanning(n_fft).astype(np.float32)
            frames = frames * window
            fft = np.fft.rfft(frames, axis=1)
            power = np.abs(fft) ** 2
            # Simple mel-like binning via linear interpolation
            n_fft_bins = power.shape[1]
            mel_matrix = np.zeros((N_MELS, n_fft_bins), dtype=np.float32)
            bin_edges = np.linspace(0, n_fft_bins, N_MELS + 2, dtype=int)
            for i in range(N_MELS):
                mel_matrix[i, bin_edges[i]:bin_edges[i + 2]] = 1.0 / max(bin_edges[i + 2] - bin_edges[i], 1)
            spec = mel_matrix @ power.T
            spec = np.log1p(spec)
            spec = spec / (spec.max() + 1e-8)

        return np.stack([spec, spec, spec], axis=0).astype(np.float32)


def benchmark_end_to_end(
    n_runs: int = 10,
    warmup: int = 2,
    output_dir: str | Path | None = None,
) -> dict:
    """Benchmark the full mobile inference pipeline.

    Simulates the complete flow: generate synthetic voice audio →
    extract vowel clip → compute spectrogram → run model → get probability.

    Args:
        n_runs: Number of timed iterations.
        warmup: Warmup iterations (not timed).
        output_dir: Directory to save benchmark results JSON.

    Returns:
        Dict with timing breakdown and overall pass/fail.
    """
    from bioagentics.voice_pd.deep.cnn_model import build_model
    from bioagentics.voice_pd.mobile.converter import run_pytorch_inference
    from bioagentics.voice_pd.mobile.protocol import extract_vowel_clip

    if output_dir is None:
        output_dir = EVAL_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup: synthetic 30-sec recording and model
    rng = np.random.default_rng(42)
    full_audio = rng.standard_normal(int(30.0 * SAMPLE_RATE)).astype(np.float32) * 0.3

    model = build_model(pretrained=False)
    model.eval()

    def _single_run() -> dict[str, float]:
        """Run one complete pipeline pass, returning per-stage times."""
        t0 = time.perf_counter()
        clip = extract_vowel_clip(full_audio)
        t1 = time.perf_counter()
        spec = _generate_spectrogram(clip)
        t2 = time.perf_counter()
        _ = run_pytorch_inference(model, spec)
        t3 = time.perf_counter()
        return {
            "clip_sec": t1 - t0,
            "spectrogram_sec": t2 - t1,
            "inference_sec": t3 - t2,
            "total_sec": t3 - t0,
        }

    # Warmup
    for _ in range(warmup):
        _single_run()

    # Timed runs
    runs = [_single_run() for _ in range(n_runs)]

    # Aggregate
    totals = np.array([r["total_sec"] for r in runs])
    results = {
        "pipeline": {
            "clip_sec": float(np.median([r["clip_sec"] for r in runs])),
            "spectrogram_sec": float(np.median([r["spectrogram_sec"] for r in runs])),
            "inference_sec": float(np.median([r["inference_sec"] for r in runs])),
        },
        "total": {
            "mean_sec": float(totals.mean()),
            "std_sec": float(totals.std()),
            "median_sec": float(np.median(totals)),
            "min_sec": float(totals.min()),
            "max_sec": float(totals.max()),
        },
        "target_sec": MOBILE_INFERENCE_SEC,
        "meets_target": float(np.median(totals)) < MOBILE_INFERENCE_SEC,
        "n_runs": n_runs,
    }

    results_path = output_dir / "mobile_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(
        "E2E benchmark: median=%.3fs (clip=%.3fs, spec=%.3fs, infer=%.3fs) — %s",
        results["total"]["median_sec"],
        results["pipeline"]["clip_sec"],
        results["pipeline"]["spectrogram_sec"],
        results["pipeline"]["inference_sec"],
        "PASS" if results["meets_target"] else "FAIL",
    )
    return results
