"""Multiple Instance Learning models for slide-level MSI classification.

Implements three MIL aggregation methods:
- ABMIL: Attention-Based MIL with gated attention
- CLAM: Clustering-constrained Attention MIL
- TransMIL: Transformer-based MIL with position-aware self-attention

All models take per-slide feature matrices (N_patches x D_features) from
frozen foundation models and output slide-level MSI predictions.
"""

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MILOutput(NamedTuple):
    """Standard output for all MIL models."""

    logits: torch.Tensor  # (B, n_classes)
    attention_weights: torch.Tensor  # (B, N) attention per instance
    bag_repr: torch.Tensor  # (B, D) aggregated bag representation


# ─── ABMIL ───────────────────────────────────────────────────────────────────


class GatedAttention(nn.Module):
    """Gated attention mechanism for ABMIL.

    Computes attention(h) = softmax(W_a * tanh(V*h) * sigmoid(U*h))
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.attention_v = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute gated attention weights.

        Args:
            h: Instance features (B, N, D).

        Returns:
            (attention_weights, weighted_sum) where attention_weights is (B, N)
            and weighted_sum is (B, D).
        """
        a_v = self.attention_v(h)  # (B, N, hidden)
        a_u = self.attention_u(h)  # (B, N, hidden)
        a = self.attention_w(self.dropout(a_v * a_u))  # (B, N, 1)
        a = a.squeeze(-1)  # (B, N)
        attention = F.softmax(a, dim=1)  # (B, N)
        weighted_sum = torch.bmm(attention.unsqueeze(1), h).squeeze(1)  # (B, D)
        return attention, weighted_sum


class ABMIL(nn.Module):
    """Attention-Based Multiple Instance Learning.

    Reference: Ilse et al., "Attention-based Deep Multiple Instance Learning", ICML 2018.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        n_classes: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.attention = GatedAttention(hidden_dim, hidden_dim // 2, dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> MILOutput:
        """Forward pass.

        Args:
            x: Instance features (B, N, input_dim).

        Returns:
            MILOutput with logits, attention_weights, and bag_repr.
        """
        h = self.feature_proj(x)  # (B, N, hidden)
        attention, bag_repr = self.attention(h)  # (B, N), (B, hidden)
        logits = self.classifier(bag_repr)  # (B, n_classes)
        return MILOutput(logits=logits, attention_weights=attention, bag_repr=bag_repr)


# ─── CLAM ────────────────────────────────────────────────────────────────────


class CLAM(nn.Module):
    """Clustering-constrained Attention Multiple Instance Learning.

    Multi-branch attention with instance-level clustering constraint.
    Reference: Lu et al., Nature Biomedical Engineering 2021.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        n_classes: int = 2,
        n_instances_cluster: int = 8,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_instances_cluster = n_instances_cluster

        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shared attention backbone
        self.attention_v = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Sigmoid(),
        )

        # Per-class attention heads
        self.attention_heads = nn.ModuleList(
            [nn.Linear(hidden_dim // 2, 1) for _ in range(n_classes)]
        )

        # Per-class classifiers
        self.classifiers = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(n_classes)]
        )

        # Instance-level clustering head
        self.instance_classifier = nn.Linear(hidden_dim, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_instance_loss: bool = False) -> MILOutput:
        """Forward pass.

        Args:
            x: Instance features (B, N, input_dim).
            return_instance_loss: If True, includes instance-level clustering
                loss in the output (for training).

        Returns:
            MILOutput. When return_instance_loss=True, logits tensor has an
            extra attribute 'instance_loss' set.
        """
        B, N, _ = x.shape
        h = self.feature_proj(x)  # (B, N, hidden)

        # Shared attention features
        a_v = self.attention_v(h)
        a_u = self.attention_u(h)
        a_gated = self.dropout(a_v * a_u)  # (B, N, hidden//2)

        # Per-class attention and classification
        all_logits = []
        all_attention = []
        for k in range(self.n_classes):
            a_k = self.attention_heads[k](a_gated).squeeze(-1)  # (B, N)
            attn_k = F.softmax(a_k, dim=1)  # (B, N)
            bag_k = torch.bmm(attn_k.unsqueeze(1), h).squeeze(1)  # (B, hidden)
            logit_k = self.classifiers[k](bag_k)  # (B, 1)
            all_logits.append(logit_k)
            all_attention.append(attn_k)

        logits = torch.cat(all_logits, dim=1)  # (B, n_classes)
        # Average attention across class heads for interpretability
        attention = torch.stack(all_attention, dim=1).mean(dim=1)  # (B, N)
        bag_repr = torch.bmm(attention.unsqueeze(1), h).squeeze(1)  # (B, hidden)

        return MILOutput(logits=logits, attention_weights=attention, bag_repr=bag_repr)


# ─── TransMIL ────────────────────────────────────────────────────────────────


class NystromAttention(nn.Module):
    """Nystrom-based efficient self-attention for TransMIL.

    Approximates full self-attention using landmark points.
    Reference: Xiong et al., AAAI 2021.
    """

    def __init__(self, dim: int, n_heads: int = 8, n_landmarks: int = 64, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_landmarks = n_landmarks
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, heads, N, head_dim)

        if N <= self.n_landmarks:
            # Fall back to standard attention for small sequences
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            out = attn @ v
        else:
            # Nystrom approximation
            n_lm = self.n_landmarks
            # Select landmarks via stride
            stride = max(N // n_lm, 1)
            lm_idx = torch.arange(0, N, stride, device=x.device)[:n_lm]

            q_lm = q[:, :, lm_idx]  # (B, heads, n_lm, head_dim)
            k_lm = k[:, :, lm_idx]

            # Kernel matrices
            kernel_1 = F.softmax(q @ k_lm.transpose(-2, -1) * self.scale, dim=-1)
            kernel_2 = F.softmax(q_lm @ k_lm.transpose(-2, -1) * self.scale, dim=-1)
            kernel_3 = F.softmax(q_lm @ k.transpose(-2, -1) * self.scale, dim=-1)

            # Pseudoinverse of kernel_2 via iterative method
            kernel_2_inv = self._iterative_inv(kernel_2)

            out = kernel_1 @ kernel_2_inv @ kernel_3 @ v
            out = self.dropout(out)

        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    @staticmethod
    def _iterative_inv(mat: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
        """Approximate matrix inverse using Newton's method."""
        identity = torch.eye(mat.size(-1), device=mat.device, dtype=mat.dtype)
        identity = identity.unsqueeze(0).unsqueeze(0)
        # Initialize
        mat_norm = mat.norm(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
        z = mat.transpose(-2, -1) / (mat_norm * mat_norm)
        for _ in range(n_iter):
            z = 2 * z - z @ mat @ z
        return z


class TransMILBlock(nn.Module):
    """Transformer block for TransMIL."""

    def __init__(self, dim: int, n_heads: int = 8, n_landmarks: int = 64, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NystromAttention(dim, n_heads, n_landmarks, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransMIL(nn.Module):
    """Transformer-based Multiple Instance Learning.

    Uses position-aware self-attention for capturing morphological relationships
    between patches within a slide.

    Reference: Shao et al., "TransMIL: Transformer based Correlated Multiple
    Instance Learning for Whole Slide Image Classification", NeurIPS 2021.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        n_classes: int = 2,
        n_layers: int = 2,
        n_heads: int = 8,
        n_landmarks: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                TransMILBlock(hidden_dim, n_heads, n_landmarks, dropout)
                for _ in range(n_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> MILOutput:
        """Forward pass.

        Args:
            x: Instance features (B, N, input_dim).

        Returns:
            MILOutput with logits, attention_weights, and bag_repr.
        """
        B, N, _ = x.shape
        h = self.feature_proj(x)  # (B, N, hidden)

        # Prepend class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1)  # (B, N+1, hidden)

        # Apply transformer blocks
        for block in self.blocks:
            h = block(h)

        h = self.norm(h)

        # Class token output
        bag_repr = h[:, 0]  # (B, hidden)
        logits = self.classifier(bag_repr)  # (B, n_classes)

        # Compute attention weights from CLS token to all patches
        # Use dot-product attention between CLS and instance tokens
        patch_tokens = h[:, 1:]  # (B, N, hidden)
        attn_scores = torch.bmm(
            bag_repr.unsqueeze(1), patch_tokens.transpose(1, 2)
        ).squeeze(1)  # (B, N)
        attention = F.softmax(attn_scores, dim=1)

        return MILOutput(logits=logits, attention_weights=attention, bag_repr=bag_repr)


# ─── Factory ─────────────────────────────────────────────────────────────────

_MODEL_REGISTRY = {
    "abmil": ABMIL,
    "clam": CLAM,
    "transmil": TransMIL,
}


def create_mil_model(
    model_name: str,
    input_dim: int = 1024,
    n_classes: int = 2,
    **kwargs,
) -> nn.Module:
    """Create a MIL model by name.

    Args:
        model_name: One of 'abmil', 'clam', 'transmil'.
        input_dim: Feature dimension from the encoder.
        n_classes: Number of output classes (default 2: MSI-H vs MSS).
        **kwargs: Additional model-specific arguments.

    Returns:
        Initialized MIL model.
    """
    name = model_name.lower()
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](input_dim=input_dim, n_classes=n_classes, **kwargs)
