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
from bioagentics.voice_pd.deep.wav2vec_contrastive import (
    SupConLoss,
    Wav2VecContrastiveModel,
    train_contrastive,
)
from bioagentics.voice_pd.deep.wav2vec_features import (
    batch_extract_embeddings,
    extract_and_save_all,
    extract_embedding,
)
from bioagentics.voice_pd.deep.wav2vec_fusion import (
    FusionMLP,
    compare_modalities,
    train_fusion_model,
)

__all__ = [
    "CompositeCNN",
    "CompositeSpectrogramDataset",
    "FusionMLP",
    "GradCAM",
    "SpectrogramCNN",
    "SpectrogramDataset",
    "SupConLoss",
    "VOWELS",
    "Wav2VecContrastiveModel",
    "audio_to_cnn_input",
    "audio_to_mel_spectrogram",
    "batch_extract_embeddings",
    "batch_generate_composites",
    "bootstrap_auroc_ci",
    "build_composite_model",
    "build_model",
    "compare_modalities",
    "composite_to_rgb_tensor",
    "extract_and_save_all",
    "extract_embedding",
    "generate_composite_spectrogram",
    "mel_to_rgb_tensor",
    "patient_audio_to_composite",
    "run_benchmark",
    "save_heatmap_overlay",
    "train_composite_model",
    "train_contrastive",
    "train_deep_model",
    "train_fusion_model",
    "vowel_contributions",
]
