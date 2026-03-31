"""Wav2Vec 2.0 embedding extraction for voice PD detection.

Uses a pre-trained wav2vec 2.0 model (facebook/wav2vec2-base) to extract
fixed-size embeddings from raw audio. Average pooling across time frames
produces one 768-d vector per recording. These pre-trained speech
representations may capture PD-relevant patterns missed by handcrafted
features.
"""

import logging
from pathlib import Path

import numpy as np
import torch

from bioagentics.voice_pd.config import FEATURES_DIR, SAMPLE_RATE
from bioagentics.voice_pd.utils import load_audio, pad_or_trim

log = logging.getLogger(__name__)

# Memory-safe defaults (8GB machine)
MAX_AUDIO_SEC = 10.0  # cap per-recording length to limit memory
DEFAULT_MODEL_NAME = "facebook/wav2vec2-base"
EMBEDDING_DIM = 768  # wav2vec2-base hidden size


def load_wav2vec_model(
    model_name: str = DEFAULT_MODEL_NAME,
    device: str | None = None,
) -> tuple:
    """Load pre-trained Wav2Vec 2.0 model and processor.

    Args:
        model_name: HuggingFace model identifier.
        device: torch device string. Auto-selects CPU/MPS/CUDA if None.

    Returns:
        (model, processor, device_str) tuple.
    """
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    log.info("Loading %s on %s", model_name, device)
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2Model.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    return model, processor, device


def extract_embedding(
    audio_path: str | Path,
    model: torch.nn.Module,
    processor,
    device: str = "cpu",
    max_sec: float = MAX_AUDIO_SEC,
) -> np.ndarray:
    """Extract a fixed-size embedding from a single audio file.

    Loads audio at 16kHz, passes through wav2vec 2.0, and average-pools
    hidden states across time to produce a single vector.

    Args:
        audio_path: Path to WAV file.
        model: Loaded Wav2Vec2Model.
        processor: Loaded Wav2Vec2Processor.
        device: Torch device string.
        max_sec: Maximum audio duration in seconds (truncates longer files).

    Returns:
        1D numpy array of shape (768,).
    """
    y, sr = load_audio(audio_path)
    max_samples = int(max_sec * sr)
    y = pad_or_trim(y, min(len(y), max_samples))

    inputs = processor(
        y, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True,
    )
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        outputs = model(input_values)
        # outputs.last_hidden_state: (1, time_frames, 768)
        hidden = outputs.last_hidden_state.squeeze(0)  # (T, 768)
        embedding = hidden.mean(dim=0)  # (768,)

    return embedding.cpu().numpy().astype(np.float32)


def batch_extract_embeddings(
    audio_paths: list[str | Path],
    model_name: str = DEFAULT_MODEL_NAME,
    output_path: str | Path | None = None,
    max_sec: float = MAX_AUDIO_SEC,
) -> np.ndarray:
    """Extract embeddings for a batch of audio files.

    Processes files one at a time to stay within 8GB RAM limit.

    Args:
        audio_paths: List of paths to WAV files.
        model_name: HuggingFace model identifier.
        output_path: If provided, saves embeddings as .npy file.
        max_sec: Maximum audio duration per file.

    Returns:
        2D numpy array of shape (n_files, 768).
    """
    model, processor, device = load_wav2vec_model(model_name)
    n = len(audio_paths)
    embeddings = np.zeros((n, EMBEDDING_DIM), dtype=np.float32)

    for i, path in enumerate(audio_paths):
        try:
            embeddings[i] = extract_embedding(
                path, model, processor, device, max_sec,
            )
        except Exception:
            log.warning("Failed to extract embedding for %s", path, exc_info=True)
            # Leave as zeros for failed files

        if (i + 1) % 100 == 0:
            log.info("Extracted %d / %d embeddings", i + 1, n)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)
        log.info("Saved embeddings to %s", output_path)

    return embeddings


def extract_and_save_all(
    manifest_path: str | Path,
    audio_col: str = "audio_path",
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str | Path | None = None,
) -> Path:
    """Extract wav2vec embeddings for all recordings in a manifest.

    Args:
        manifest_path: Path to CSV manifest with audio file paths.
        audio_col: Column name containing audio paths.
        model_name: HuggingFace model identifier.
        output_dir: Output directory (defaults to FEATURES_DIR).

    Returns:
        Path to saved .npy file.
    """
    from bioagentics.voice_pd.utils import read_manifest

    if output_dir is None:
        output_dir = FEATURES_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_manifest(manifest_path)
    audio_paths = [row[audio_col] for row in manifest]
    output_path = output_dir / "wav2vec_embeddings.npy"

    log.info(
        "Extracting wav2vec embeddings for %d recordings from %s",
        len(audio_paths), manifest_path,
    )
    batch_extract_embeddings(audio_paths, model_name, output_path)
    return output_path
