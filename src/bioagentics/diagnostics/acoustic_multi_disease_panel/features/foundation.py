"""Audio foundation model embedding extraction (Wav2Vec 2.0 / HuBERT).

Extracts 768-dim embeddings from pretrained audio foundation models.
Designed for 8GB RAM constraint: uses float16, batch size 1, and
disk caching to avoid recomputation.

Supported models:
    - facebook/wav2vec2-base (95M params, ~380MB)
    - facebook/hubert-base-ls960 (95M params, ~380MB)
"""

import hashlib
import logging
from pathlib import Path

import numpy as np

from bioagentics.diagnostics.acoustic_multi_disease_panel.config import (
    OUTPUT_DIR,
    SAMPLE_RATE,
)

log = logging.getLogger(__name__)

DEFAULT_MODEL = "facebook/wav2vec2-base"
EMBEDDING_DIM = 768
EMBEDDING_CACHE_DIR = OUTPUT_DIR / "embedding_cache"

# Maximum audio length in seconds to process (prevents OOM)
MAX_AUDIO_SECONDS = 30


def _audio_hash(audio_path: str | Path) -> str:
    """Compute hash of audio file for cache key."""
    h = hashlib.md5()
    with open(audio_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_cache_path(audio_path: str | Path, model_name: str) -> Path:
    """Get cache file path for an embedding."""
    model_slug = model_name.replace("/", "_")
    file_hash = _audio_hash(audio_path)
    return EMBEDDING_CACHE_DIR / model_slug / f"{file_hash}.npy"


def _load_cached(cache_path: Path) -> np.ndarray | None:
    """Load cached embedding if it exists."""
    if cache_path.exists():
        try:
            return np.load(cache_path)
        except Exception:
            log.warning("Corrupted cache file, recomputing: %s", cache_path)
    return None


def _save_cache(cache_path: Path, embedding: np.ndarray) -> None:
    """Save embedding to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, embedding)


def extract_foundation_embedding(
    audio_path: str | Path,
    model_name: str = DEFAULT_MODEL,
    use_cache: bool = True,
) -> dict[str, float | None]:
    """Extract foundation model embedding from an audio file.

    Loads audio, runs through pretrained Wav2Vec 2.0 or HuBERT,
    mean-pools frame-level embeddings to a fixed 768-dim vector.

    Args:
        audio_path: Path to a WAV file (16kHz mono expected).
        model_name: HuggingFace model identifier.
        use_cache: If True, cache embeddings to disk.

    Returns:
        Dict with keys fm_emb_0 through fm_emb_767 (768-dim embedding).
    """
    audio_path = Path(audio_path)

    # Check cache first
    if use_cache:
        cache_path = _get_cache_path(audio_path, model_name)
        cached = _load_cached(cache_path)
        if cached is not None:
            return _embedding_to_dict(cached)

    # Load audio
    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        log.exception("Failed to load audio: %s", audio_path)
        return _empty_features()

    embedding = _extract_from_array(y, int(sr), model_name)

    # Cache result
    if use_cache and embedding is not None:
        cache_path = _get_cache_path(audio_path, model_name)
        _save_cache(cache_path, embedding)

    return _embedding_to_dict(embedding)


def extract_foundation_embedding_from_array(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    model_name: str = DEFAULT_MODEL,
) -> dict[str, float | None]:
    """Extract foundation model embedding from a numpy audio array."""
    embedding = _extract_from_array(y, sr, model_name)
    return _embedding_to_dict(embedding)


# Lazy-loaded model cache (load once, reuse)
_model_cache: dict[str, tuple] = {}


def _get_model(model_name: str):
    """Lazy-load and cache the model and processor."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    try:
        import torch
        from transformers import AutoModel, Wav2Vec2Processor
    except ImportError:
        log.error(
            "transformers and torch required for foundation model embeddings. "
            "Install with: uv add --optional research transformers torch"
        )
        raise

    log.info("Loading foundation model: %s (this may take a moment)...", model_name)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    model.eval()

    # Use CPU — GPU would exceed 8GB RAM constraint
    device = torch.device("cpu")
    model = model.to(device)

    _model_cache[model_name] = (processor, model, device)
    log.info("Model loaded: %s (%.0fMB)", model_name, _model_memory_mb(model))
    return processor, model, device


def _model_memory_mb(model) -> float:
    """Estimate model memory usage in MB."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    return param_bytes / (1024 * 1024)


def _extract_from_array(
    y: np.ndarray,
    sr: int,
    model_name: str,
) -> np.ndarray | None:
    """Core embedding extraction from audio array."""
    import torch

    if len(y) < sr * 0.1:  # less than 100ms
        log.warning("Audio too short for foundation model (%d samples)", len(y))
        return None

    # Truncate to MAX_AUDIO_SECONDS to prevent OOM
    max_samples = int(sr * MAX_AUDIO_SECONDS)
    if len(y) > max_samples:
        log.info("Truncating audio from %.1fs to %ds", len(y) / sr, MAX_AUDIO_SECONDS)
        y = y[:max_samples]

    try:
        processor, model, device = _get_model(model_name)

        # Process audio
        inputs = processor(
            y, sampling_rate=sr, return_tensors="pt", padding=True
        )
        input_values = inputs.input_values.to(device=device, dtype=torch.float16)

        # Extract embeddings (no gradient computation)
        with torch.no_grad():
            outputs = model(input_values)
            # Use last hidden state: (1, n_frames, 768)
            hidden_states = outputs.last_hidden_state

        # Mean pooling across frames -> (768,)
        embedding = hidden_states.squeeze(0).mean(dim=0).float().cpu().numpy()

        return embedding

    except Exception:
        log.exception("Foundation model embedding extraction failed")
        return None


def _embedding_to_dict(embedding: np.ndarray | None) -> dict[str, float | None]:
    """Convert embedding array to feature dict."""
    if embedding is None:
        return _empty_features()
    return {f"fm_emb_{i}": float(v) for i, v in enumerate(embedding)}


def _empty_features() -> dict[str, float | None]:
    """Return feature dict with all None values."""
    return {f"fm_emb_{i}": None for i in range(EMBEDDING_DIM)}


def clear_model_cache() -> None:
    """Release loaded models from memory."""
    import gc

    _model_cache.clear()
    gc.collect()
    log.info("Foundation model cache cleared")
