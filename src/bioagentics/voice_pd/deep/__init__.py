"""Deep learning models for voice-based PD detection."""

from bioagentics.voice_pd.deep.cnn_model import SpectrogramCNN, build_model
from bioagentics.voice_pd.deep.composite_benchmark import (
    bootstrap_auroc_ci,
    run_benchmark,
)
from bioagentics.voice_pd.deep.composite_spectrogram import (
    VOWELS,
    batch_generate_composites,
    composite_to_rgb_tensor,
    generate_composite_spectrogram,
    patient_audio_to_composite,
)
from bioagentics.voice_pd.deep.composite_train import (
    CompositeCNN,
    CompositeSpectrogramDataset,
    build_composite_model,
    train_composite_model,
)
from bioagentics.voice_pd.deep.gradcam import (
    GradCAM,
    save_heatmap_overlay,
    vowel_contributions,
)
from bioagentics.voice_pd.deep.spectrogram import (
    audio_to_cnn_input,
    audio_to_mel_spectrogram,
    mel_to_rgb_tensor,
)
from bioagentics.voice_pd.deep.train import (
    SpectrogramDataset,
    train_deep_model,
)

__all__ = [
    "CompositeCNN",
    "CompositeSpectrogramDataset",
    "GradCAM",
    "SpectrogramCNN",
    "SpectrogramDataset",
    "VOWELS",
    "audio_to_cnn_input",
    "audio_to_mel_spectrogram",
    "batch_generate_composites",
    "bootstrap_auroc_ci",
    "build_composite_model",
    "build_model",
    "composite_to_rgb_tensor",
    "generate_composite_spectrogram",
    "mel_to_rgb_tensor",
    "patient_audio_to_composite",
    "run_benchmark",
    "save_heatmap_overlay",
    "train_composite_model",
    "train_deep_model",
    "vowel_contributions",
]
