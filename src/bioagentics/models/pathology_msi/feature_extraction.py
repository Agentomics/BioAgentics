"""Feature extraction pipelines for pathology foundation models.

Extracts patch-level feature vectors from tiled WSIs using frozen
foundation models. Supports UNI 2, CONCH, and ResNet-50 ImageNet baseline.
Outputs per-slide feature matrices as HDF5 files.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PatchDataset(Dataset):
    """Dataset that loads patches from a WSI given tile coordinates."""

    def __init__(
        self,
        slide_path: str | Path,
        tile_coords: pd.DataFrame,
        transform: transforms.Compose | None = None,
        tile_size: int = 256,
    ):
        self.slide_path = Path(slide_path)
        self.tile_coords = tile_coords
        self.transform = transform
        self.tile_size = tile_size
        self._slide = None

    def _get_slide(self):
        if self._slide is None:
            import openslide
            self._slide = openslide.OpenSlide(str(self.slide_path))
        return self._slide

    def __len__(self) -> int:
        return len(self.tile_coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.tile_coords.iloc[idx]
        slide = self._get_slide()

        x, y = int(row["x"]), int(row["y"])
        level = int(row["level"])
        downsample = float(row["downsample"])
        level_downsample = float(row["level_downsample"])

        # Read region
        read_size = int(self.tile_size * downsample / level_downsample)
        region = slide.read_region((x, y), level, (read_size, read_size))
        region = region.convert("RGB")

        if region.size[0] != self.tile_size or region.size[1] != self.tile_size:
            region = region.resize(
                (self.tile_size, self.tile_size), Image.Resampling.LANCZOS
            )

        if self.transform:
            region = self.transform(region)
        else:
            region = transforms.ToTensor()(region)

        return region

    def close(self):
        if self._slide is not None:
            self._slide.close()
            self._slide = None


class FeatureExtractor(ABC):
    """Base class for feature extractors."""

    def __init__(self, device: str = "cpu", batch_size: int = 64):
        self.device = device
        self.batch_size = batch_size

    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the frozen feature extraction model."""

    @abstractmethod
    def get_transform(self) -> transforms.Compose:
        """Return the preprocessing transform for input patches."""

    @abstractmethod
    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""

    @torch.no_grad()
    def extract_from_patches(self, patches: DataLoader) -> np.ndarray:
        """Extract features from a DataLoader of patches.

        Returns:
            Feature matrix (N_patches, D_features).
        """
        model = self.get_model().to(self.device).eval()
        all_features = []

        for batch in patches:
            batch = batch.to(self.device)
            features = model(batch)
            if isinstance(features, tuple):
                features = features[0]
            all_features.append(features.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def extract_from_slide(
        self,
        slide_path: str | Path,
        tile_coords: pd.DataFrame,
        output_path: str | Path,
    ) -> Path:
        """Extract features from a WSI and save as HDF5.

        Args:
            slide_path: Path to the WSI file.
            tile_coords: DataFrame with tile coordinates.
            output_path: Path for the output HDF5 file.

        Returns:
            Path to the saved HDF5 file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        dataset = PatchDataset(
            slide_path, tile_coords, transform=self.get_transform()
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0,
        )

        try:
            features = self.extract_from_patches(loader)
        finally:
            dataset.close()

        # Save as HDF5
        with h5py.File(output_path, "w") as f:
            f.create_dataset("features", data=features, compression="gzip")
            f.attrs["extractor"] = self.__class__.__name__
            f.attrs["feature_dim"] = self.get_feature_dim()
            f.attrs["n_patches"] = features.shape[0]
            f.attrs["slide_path"] = str(slide_path)

        logger.info(
            f"  Saved {features.shape[0]} x {features.shape[1]} features to {output_path}"
        )
        return output_path


# ─── UNI 2 ───────────────────────────────────────────────────────────────────


class UNI2Extractor(FeatureExtractor):
    """Feature extraction using UNI 2 (ViT-Large, Mahmood Lab).

    UNI 2 is a pathology foundation model trained on 200M+ histopathology images.
    Open-access via HuggingFace (MahmoodLab/UNI2).
    """

    MODEL_NAME = "MahmoodLab/UNI2"
    FEATURE_DIM = 1024

    def __init__(self, device: str = "cpu", batch_size: int = 64):
        super().__init__(device, batch_size)
        self._model = None

    def get_model(self) -> nn.Module:
        if self._model is None:
            try:
                import timm

                self._model = timm.create_model(
                    "vit_large_patch16_224",
                    init_values=1e-5,
                    num_classes=0,  # Remove classification head
                    dynamic_img_size=True,
                )
                # Try to load UNI 2 weights from HuggingFace
                try:
                    from huggingface_hub import hf_hub_download

                    weights_path = hf_hub_download(
                        repo_id=self.MODEL_NAME,
                        filename="pytorch_model.bin",
                    )
                    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                    self._model.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded UNI 2 weights from HuggingFace")
                except Exception as e:
                    logger.warning(
                        f"Could not load UNI 2 weights: {e}. "
                        "Using randomly initialized ViT-Large (results will be meaningless). "
                        "Request access at huggingface.co/MahmoodLab/UNI2"
                    )

                self._model.eval()
                for p in self._model.parameters():
                    p.requires_grad = False

            except ImportError:
                raise ImportError("timm is required for UNI 2: uv add timm")

        return self._model

    def get_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def get_feature_dim(self) -> int:
        return self.FEATURE_DIM


# ─── CONCH ───────────────────────────────────────────────────────────────────


class CONCHExtractor(FeatureExtractor):
    """Feature extraction using CONCH (visual-language pathology FM).

    CONCH is a multimodal vision-language foundation model for pathology.
    """

    MODEL_NAME = "MahmoodLab/CONCH"
    FEATURE_DIM = 512

    def __init__(self, device: str = "cpu", batch_size: int = 64):
        super().__init__(device, batch_size)
        self._model = None

    def get_model(self) -> nn.Module:
        if self._model is None:
            try:
                import timm

                # CONCH uses a ViT-B/16 backbone
                self._model = timm.create_model(
                    "vit_base_patch16_224",
                    num_classes=0,
                    dynamic_img_size=True,
                )
                try:
                    from huggingface_hub import hf_hub_download

                    weights_path = hf_hub_download(
                        repo_id=self.MODEL_NAME,
                        filename="pytorch_model.bin",
                    )
                    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
                    self._model.load_state_dict(state_dict, strict=False)
                    logger.info("Loaded CONCH weights from HuggingFace")
                except Exception as e:
                    logger.warning(
                        f"Could not load CONCH weights: {e}. "
                        "Using randomly initialized ViT-Base."
                    )

                self._model.eval()
                for p in self._model.parameters():
                    p.requires_grad = False

            except ImportError:
                raise ImportError("timm is required for CONCH: uv add timm")

        return self._model

    def get_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def get_feature_dim(self) -> int:
        return self.FEATURE_DIM


# ─── ResNet-50 Baseline ─────────────────────────────────────────────────────


class ResNet50Extractor(FeatureExtractor):
    """Feature extraction using ResNet-50 pretrained on ImageNet.

    Non-pathology control baseline. Extracts features from the penultimate
    layer (before the classification head).
    """

    FEATURE_DIM = 2048

    def __init__(self, device: str = "cpu", batch_size: int = 64):
        super().__init__(device, batch_size)
        self._model = None

    def get_model(self) -> nn.Module:
        if self._model is None:
            from torchvision.models import ResNet50_Weights, resnet50

            full_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # Remove classification head — use all layers except the last FC
            self._model = nn.Sequential(*list(full_model.children())[:-1], nn.Flatten())

            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad = False

        return self._model

    def get_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def get_feature_dim(self) -> int:
        return self.FEATURE_DIM


# ─── Factory ─────────────────────────────────────────────────────────────────

_EXTRACTOR_REGISTRY = {
    "uni2": UNI2Extractor,
    "conch": CONCHExtractor,
    "resnet50": ResNet50Extractor,
}


def create_feature_extractor(
    name: str,
    device: str = "cpu",
    batch_size: int = 64,
) -> FeatureExtractor:
    """Create a feature extractor by name.

    Args:
        name: One of 'uni2', 'conch', 'resnet50'.
        device: PyTorch device.
        batch_size: Batch size for inference.

    Returns:
        Initialized FeatureExtractor.
    """
    key = name.lower()
    if key not in _EXTRACTOR_REGISTRY:
        raise ValueError(
            f"Unknown extractor: {name}. Available: {list(_EXTRACTOR_REGISTRY.keys())}"
        )
    return _EXTRACTOR_REGISTRY[key](device=device, batch_size=batch_size)
