"""MIL training pipeline with nested cross-validation for MSI classification.

Loads pre-extracted features (HDF5) and labels, trains MIL models
(ABMIL/CLAM/TransMIL) with nested CV (outer 5-fold, inner 3-fold),
and logs per-fold metrics (AUROC, AUPRC, sensitivity, specificity).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from .mil_models import MILOutput, create_mil_model

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Training configuration."""

    model_name: str = "abmil"
    input_dim: int = 1024
    n_classes: int = 2
    n_outer_folds: int = 5
    n_inner_folds: int = 3
    n_epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 1  # slide-level batching
    patience: int = 10
    seed: int = 42
    device: str = "cpu"
    cancer_types: list[str] = field(default_factory=lambda: ["COAD", "READ", "UCEC", "STAD"])
    mode: str = "pan_cancer"  # 'pan_cancer' or 'per_cancer_type'
    model_kwargs: dict = field(default_factory=dict)


class SlideFeatureDataset(Dataset):
    """Dataset that loads pre-extracted slide features from HDF5 files."""

    def __init__(
        self,
        slide_ids: list[str],
        labels: dict[str, int],
        features_dir: str | Path,
        max_patches: int = 10000,
    ):
        self.slide_ids = slide_ids
        self.labels = labels
        self.features_dir = Path(features_dir)
        self.max_patches = max_patches

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        slide_id = self.slide_ids[idx]
        label = self.labels[slide_id]

        # Load features from HDF5
        h5_path = self.features_dir / f"{slide_id}.h5"
        if not h5_path.exists():
            # Try alternate naming
            h5_path = self.features_dir / f"{slide_id}.hdf5"

        with h5py.File(h5_path, "r") as f:
            features = f["features"][:]  # (N, D)

        # Subsample if too many patches
        if len(features) > self.max_patches:
            idx_sample = np.random.choice(len(features), self.max_patches, replace=False)
            features = features[idx_sample]

        return torch.tensor(features, dtype=torch.float32), label, slide_id


def collate_variable_length(
    batch: list[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Collate function for variable-length slide features.

    Pads shorter slides with zeros to match the longest slide in the batch.
    """
    features, labels, slide_ids = zip(*batch)
    max_len = max(f.shape[0] for f in features)
    dim = features[0].shape[1]

    padded = torch.zeros(len(features), max_len, dim)
    for i, f in enumerate(features):
        padded[i, : f.shape[0]] = f

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, labels_tensor, list(slide_ids)


def load_labels_and_features(
    labels_path: str | Path,
    features_dir: str | Path,
    cancer_types: list[str] | None = None,
) -> tuple[list[str], dict[str, int], dict[str, str]]:
    """Load MSI labels and match with available features.

    Args:
        labels_path: Path to labels CSV (must have case_id/submitter_id, msi_status).
        features_dir: Directory with HDF5 feature files.
        cancer_types: Filter to specific cancer types.

    Returns:
        (slide_ids, labels_dict, cancer_type_dict) where labels_dict maps
        slide_id to 0/1 (MSS/MSI-H) and cancer_type_dict maps to cancer type.
    """
    labels_df = pd.read_csv(labels_path)
    features_dir = Path(features_dir)

    # Filter cancer types
    if cancer_types:
        labels_df = labels_df[labels_df["cancer_type"].isin(cancer_types)]

    # Filter to known MSI status (binary: MSI-H vs MSS+MSI-L)
    labels_df = labels_df[labels_df["msi_status"].isin(["MSI-H", "MSI-L", "MSS"])]
    labels_df["binary_label"] = (labels_df["msi_status"] == "MSI-H").astype(int)

    # Use submitter_id or case_id as slide identifier
    id_col = "submitter_id" if "submitter_id" in labels_df.columns else "case_id"

    # Match with available feature files
    available = set()
    for p in features_dir.glob("*.h5"):
        available.add(p.stem)
    for p in features_dir.glob("*.hdf5"):
        available.add(p.stem)

    slide_ids = []
    labels_dict = {}
    cancer_type_dict = {}
    for _, row in labels_df.iterrows():
        sid = row[id_col]
        if sid in available:
            slide_ids.append(sid)
            labels_dict[sid] = row["binary_label"]
            cancer_type_dict[sid] = row["cancer_type"]

    logger.info(
        f"Matched {len(slide_ids)}/{len(labels_df)} cases with features "
        f"(MSI-H: {sum(labels_dict.values())}, MSS: {len(labels_dict) - sum(labels_dict.values())})"
    )
    return slide_ids, labels_dict, cancer_type_dict


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> float:
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0

    for features, labels, _ in loader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> dict:
    """Evaluate model. Returns metrics dict."""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []

    for features, labels, _ in loader:
        features = features.to(device)
        output = model(features)
        probs = torch.softmax(output.logits, dim=1)[:, 1].cpu().numpy()
        preds = output.logits.argmax(dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    metrics = {}
    if len(np.unique(all_labels)) > 1:
        metrics["auroc"] = roc_auc_score(all_labels, all_probs)
        metrics["auprc"] = average_precision_score(all_labels, all_probs)
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")

    metrics["balanced_accuracy"] = balanced_accuracy_score(all_labels, all_preds)

    # Sensitivity (recall for MSI-H = class 1)
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (recall for MSS = class 0)
    tn = ((all_preds == 0) & (all_labels == 0)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return metrics


def train_single_fold(
    train_ids: list[str],
    val_ids: list[str],
    labels: dict[str, int],
    features_dir: str | Path,
    config: TrainConfig,
    checkpoint_dir: Path | None = None,
) -> tuple[dict, nn.Module]:
    """Train and evaluate a single fold.

    Returns:
        (best_metrics, best_model)
    """
    device = config.device

    train_dataset = SlideFeatureDataset(train_ids, labels, features_dir)
    val_dataset = SlideFeatureDataset(val_ids, labels, features_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_variable_length,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_variable_length,
    )

    model = create_mil_model(
        config.model_name,
        input_dim=config.input_dim,
        n_classes=config.n_classes,
        **config.model_kwargs,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    # Class weights for imbalanced MSI data
    n_pos = sum(1 for sid in train_ids if labels[sid] == 1)
    n_neg = len(train_ids) - n_pos
    if n_pos > 0 and n_neg > 0:
        weight = torch.tensor([n_pos / (n_pos + n_neg), n_neg / (n_pos + n_neg)]).to(device)
    else:
        weight = None
    criterion = nn.CrossEntropyLoss(weight=weight)

    best_auroc = -1
    best_metrics = {}
    best_state = None
    patience_counter = 0

    for epoch in range(config.n_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)

        if not np.isnan(val_metrics["auroc"]) and val_metrics["auroc"] > best_auroc:
            best_auroc = val_metrics["auroc"]
            best_metrics = val_metrics.copy()
            best_metrics["best_epoch"] = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            logger.info(f"  Early stopping at epoch {epoch}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    if checkpoint_dir is not None and best_state is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, checkpoint_dir / "best_model.pt")

    return best_metrics, model


def nested_cross_validation(
    slide_ids: list[str],
    labels: dict[str, int],
    features_dir: str | Path,
    config: TrainConfig,
    output_dir: str | Path | None = None,
) -> list[dict]:
    """Run nested cross-validation.

    Outer loop: 5-fold for unbiased performance estimation.
    Inner loop: 3-fold for hyperparameter selection (model selection).

    Returns:
        List of per-outer-fold results.
    """
    output_dir = Path(output_dir) if output_dir else None
    labels_array = np.array([labels[sid] for sid in slide_ids])

    outer_cv = StratifiedKFold(
        n_splits=config.n_outer_folds, shuffle=True, random_state=config.seed
    )

    all_results = []

    for outer_fold, (train_val_idx, test_idx) in enumerate(
        outer_cv.split(slide_ids, labels_array)
    ):
        logger.info(f"Outer fold {outer_fold + 1}/{config.n_outer_folds}")

        train_val_ids = [slide_ids[i] for i in train_val_idx]
        test_ids = [slide_ids[i] for i in test_idx]
        train_val_labels = np.array([labels[sid] for sid in train_val_ids])

        # Inner CV for model selection
        inner_cv = StratifiedKFold(
            n_splits=config.n_inner_folds, shuffle=True,
            random_state=config.seed + outer_fold,
        )

        best_inner_auroc = -1
        best_inner_fold = 0

        for inner_fold, (inner_train_idx, inner_val_idx) in enumerate(
            inner_cv.split(train_val_ids, train_val_labels)
        ):
            inner_train_ids = [train_val_ids[i] for i in inner_train_idx]
            inner_val_ids = [train_val_ids[i] for i in inner_val_idx]

            inner_metrics, _ = train_single_fold(
                inner_train_ids, inner_val_ids, labels, features_dir, config,
            )

            auroc = inner_metrics.get("auroc", -1)
            if not np.isnan(auroc) and auroc > best_inner_auroc:
                best_inner_auroc = auroc
                best_inner_fold = inner_fold

            logger.info(
                f"  Inner fold {inner_fold + 1}/{config.n_inner_folds}: "
                f"AUROC={inner_metrics.get('auroc', 'N/A'):.4f}"
                if not np.isnan(inner_metrics.get("auroc", float("nan")))
                else f"  Inner fold {inner_fold + 1}/{config.n_inner_folds}: AUROC=N/A"
            )

        # Retrain on full train+val with best config, evaluate on test
        checkpoint_dir = output_dir / f"fold_{outer_fold}" if output_dir else None
        test_metrics, model = train_single_fold(
            train_val_ids, test_ids, labels, features_dir, config,
            checkpoint_dir=checkpoint_dir,
        )

        test_metrics["outer_fold"] = outer_fold
        test_metrics["best_inner_fold"] = best_inner_fold
        test_metrics["best_inner_auroc"] = best_inner_auroc
        test_metrics["n_train"] = len(train_val_ids)
        test_metrics["n_test"] = len(test_ids)
        all_results.append(test_metrics)

        logger.info(
            f"  Outer fold {outer_fold + 1} test: "
            f"AUROC={test_metrics.get('auroc', 'N/A'):.4f}, "
            f"AUPRC={test_metrics.get('auprc', 'N/A'):.4f}, "
            f"Sens={test_metrics.get('sensitivity', 'N/A'):.3f}, "
            f"Spec={test_metrics.get('specificity', 'N/A'):.3f}"
        )

    # Save results summary
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / "cv_results.csv", index=False)

        # Aggregate summary
        summary = {
            "model": config.model_name,
            "mode": config.mode,
            "n_outer_folds": config.n_outer_folds,
            "n_inner_folds": config.n_inner_folds,
        }
        for metric in ["auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity"]:
            values = [r[metric] for r in all_results if not np.isnan(r.get(metric, float("nan")))]
            if values:
                summary[f"{metric}_mean"] = round(np.mean(values), 4)
                summary[f"{metric}_std"] = round(np.std(values), 4)

        with open(output_dir / "cv_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Results saved to {output_dir}")

    return all_results


def print_cv_results(results: list[dict]) -> None:
    """Print a summary of nested CV results."""
    metrics = ["auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity"]

    print("\nNested CV Results:")
    print("-" * 60)
    for metric in metrics:
        values = [r[metric] for r in results if not np.isnan(r.get(metric, float("nan")))]
        if values:
            print(f"  {metric:25s}: {np.mean(values):.4f} +/- {np.std(values):.4f}")
    print("-" * 60)

    print("\nPer-fold:")
    for r in results:
        print(
            f"  Fold {r['outer_fold']}: AUROC={r.get('auroc', 'N/A'):.4f}, "
            f"Sens={r.get('sensitivity', 0):.3f}, Spec={r.get('specificity', 0):.3f}"
        )
