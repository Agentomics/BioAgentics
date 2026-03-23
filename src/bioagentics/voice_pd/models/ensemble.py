"""Ensemble fusion of classical GBM and deep CNN for PD detection.

Implements three fusion strategies per the research plan:
- Late fusion: weighted averaging of probability predictions
- Early fusion: concatenated classical features + CNN embeddings
- Stacked generalization: meta-learner on out-of-fold predictions

Compares all strategies against individual model baselines to verify
the >=3% AUC improvement target.
"""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from bioagentics.voice_pd.config import ENSEMBLE_IMPROVEMENT, MODELS_DIR, TARGET_AUC
from bioagentics.voice_pd.deep.cnn_model import build_model
from bioagentics.voice_pd.deep.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    SpectrogramDataset,
    _evaluate,
    _train_one_epoch,
)

log = logging.getLogger(__name__)


def late_fusion(
    classical_probs: np.ndarray,
    deep_probs: np.ndarray,
    weight_classical: float = 0.5,
) -> np.ndarray:
    """Weighted average of probability predictions.

    Args:
        classical_probs: Probabilities from classical GBM (n_samples,).
        deep_probs: Probabilities from deep CNN (n_samples,).
        weight_classical: Weight for classical model (1 - weight for deep).

    Returns:
        Fused probabilities.
    """
    return weight_classical * classical_probs + (1.0 - weight_classical) * deep_probs


def optimize_fusion_weight(
    classical_probs: np.ndarray,
    deep_probs: np.ndarray,
    y_true: np.ndarray,
) -> tuple[float, float]:
    """Find optimal late-fusion weight via grid search.

    Returns:
        (best_weight_classical, best_auc)
    """
    from sklearn.metrics import roc_auc_score

    best_weight = 0.5
    best_auc = 0.0

    for w in np.arange(0.0, 1.05, 0.05):
        fused = late_fusion(classical_probs, deep_probs, weight_classical=w)
        auc = roc_auc_score(y_true, fused)
        if auc > best_auc:
            best_auc = auc
            best_weight = round(float(w), 2)

    return best_weight, best_auc


@torch.no_grad()
def extract_cnn_embeddings(
    model: torch.nn.Module,
    spectrograms: np.ndarray,
    batch_size: int = 8,
) -> np.ndarray:
    """Extract 1280-dim embeddings from CNN before classification head.

    Uses model.features + model.pool to get the penultimate representation.

    Returns:
        Embeddings array of shape (n_samples, 1280).
    """
    model.eval()
    device = next(model.parameters()).device
    dataset = SpectrogramDataset(spectrograms, np.zeros(len(spectrograms)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    for X_batch, _ in loader:
        X_batch = X_batch.to(device)
        x = model.features(X_batch)
        x = model.pool(x)
        x = torch.flatten(x, 1)
        embeddings.append(x.cpu().numpy())

    return np.concatenate(embeddings, axis=0)


def train_ensemble(
    X_classical: np.ndarray,
    spectrograms: np.ndarray,
    labels: np.ndarray,
    output_dir: str | Path | None = None,
    n_splits: int = 5,
    cnn_epochs: int = DEFAULT_EPOCHS,
    cnn_batch_size: int = DEFAULT_BATCH_SIZE,
    cnn_lr: float = DEFAULT_LR,
    cnn_pretrained: bool = True,
) -> dict:
    """Train and evaluate all ensemble fusion strategies.

    Runs stratified CV, training both classical GBM and deep CNN per fold,
    then evaluates late fusion, early fusion, and stacked generalization.

    Args:
        X_classical: Classical feature matrix (n_samples, n_classical_features).
            May contain NaN values (imputed internally).
        spectrograms: Spectrogram array (n_samples, 3, n_mels, time_frames).
        labels: Binary labels (1=PD, 0=healthy).
        output_dir: Directory to save results.
        n_splits: Number of CV folds.
        cnn_epochs: Training epochs per fold for CNN.
        cnn_batch_size: Mini-batch size for CNN (keep small for 8GB RAM).
        cnn_lr: Learning rate for CNN.
        cnn_pretrained: Use ImageNet-pretrained MobileNetV2 backbone.

    Returns:
        Dict with results for all strategies and comparison metrics.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if output_dir is None:
        output_dir = MODELS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")  # 8GB RAM — stay on CPU
    spec_dataset = SpectrogramDataset(spectrograms, labels)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    n = len(labels)
    oof_classical = np.zeros(n)
    oof_deep = np.zeros(n)
    oof_early = np.zeros(n)

    fold_classical_aucs: list[float] = []
    fold_deep_aucs: list[float] = []
    fold_early_aucs: list[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_classical, labels)):
        log.info("=== Fold %d/%d ===", fold + 1, n_splits)

        # ── Classical GBM ──
        classical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ])
        classical_pipeline.fit(X_classical[train_idx], labels[train_idx])
        classical_probs = classical_pipeline.predict_proba(X_classical[val_idx])[:, 1]
        oof_classical[val_idx] = classical_probs
        classical_auc = roc_auc_score(labels[val_idx], classical_probs)
        fold_classical_aucs.append(classical_auc)
        log.info("Classical AUC: %.4f", classical_auc)

        # ── Deep CNN ──
        train_loader = DataLoader(
            Subset(spec_dataset, train_idx.tolist()),
            batch_size=cnn_batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            Subset(spec_dataset, val_idx.tolist()),
            batch_size=cnn_batch_size, shuffle=False,
        )

        model = build_model(pretrained=cnn_pretrained).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cnn_lr)
        criterion = torch.nn.BCEWithLogitsLoss()

        for _ in range(cnn_epochs):
            _train_one_epoch(model, train_loader, optimizer, criterion, device)

        _, deep_probs = _evaluate(model, val_loader, device)
        oof_deep[val_idx] = deep_probs
        deep_auc = roc_auc_score(labels[val_idx], deep_probs)
        fold_deep_aucs.append(deep_auc)
        log.info("Deep AUC: %.4f", deep_auc)

        # ── Early fusion: classical features + CNN embeddings ──
        train_emb = extract_cnn_embeddings(
            model, spectrograms[train_idx], batch_size=cnn_batch_size,
        )
        val_emb = extract_cnn_embeddings(
            model, spectrograms[val_idx], batch_size=cnn_batch_size,
        )

        X_early_train = np.hstack([X_classical[train_idx], train_emb])
        X_early_val = np.hstack([X_classical[val_idx], val_emb])

        early_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ])
        early_pipeline.fit(X_early_train, labels[train_idx])
        early_probs = early_pipeline.predict_proba(X_early_val)[:, 1]
        oof_early[val_idx] = early_probs
        early_auc = roc_auc_score(labels[val_idx], early_probs)
        fold_early_aucs.append(early_auc)
        log.info("Early fusion AUC: %.4f", early_auc)

        # Free memory between folds
        del model, optimizer, train_emb, val_emb
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Late fusion (optimized weight on OOF predictions) ──
    best_weight, late_auc = optimize_fusion_weight(oof_classical, oof_deep, labels)
    log.info("Late fusion: AUC=%.4f (weight_classical=%.2f)", late_auc, best_weight)

    # ── Stacked generalization (meta-learner on OOF predictions) ──
    meta_features = np.column_stack([oof_classical, oof_deep])
    stacked_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(random_state=42, max_iter=1000)),
    ])
    stacked_fold_aucs: list[float] = []
    for train_idx, val_idx in skf.split(meta_features, labels):
        stacked_pipeline.fit(meta_features[train_idx], labels[train_idx])
        stacked_probs = stacked_pipeline.predict_proba(meta_features[val_idx])[:, 1]
        stacked_fold_aucs.append(roc_auc_score(labels[val_idx], stacked_probs))
    stacked_auc = float(np.mean(stacked_fold_aucs))
    log.info("Stacked generalization: AUC=%.4f", stacked_auc)

    # ── Comparison ──
    classical_mean = float(np.mean(fold_classical_aucs))
    deep_mean = float(np.mean(fold_deep_aucs))
    early_mean = float(np.mean(fold_early_aucs))
    best_single = max(classical_mean, deep_mean)

    strategies = {
        "late": late_auc,
        "early": early_mean,
        "stacked": stacked_auc,
    }
    best_strategy = max(strategies, key=lambda k: strategies[k])
    best_ensemble_auc = strategies[best_strategy]
    improvement = best_ensemble_auc - best_single

    results = {
        "classical": {
            "mean_auc": classical_mean,
            "std_auc": float(np.std(fold_classical_aucs)),
            "fold_aucs": fold_classical_aucs,
        },
        "deep": {
            "mean_auc": deep_mean,
            "std_auc": float(np.std(fold_deep_aucs)),
            "fold_aucs": fold_deep_aucs,
        },
        "late_fusion": {
            "auc": late_auc,
            "weight_classical": best_weight,
        },
        "early_fusion": {
            "mean_auc": early_mean,
            "std_auc": float(np.std(fold_early_aucs)),
            "fold_aucs": fold_early_aucs,
        },
        "stacked": {
            "mean_auc": stacked_auc,
            "std_auc": float(np.std(stacked_fold_aucs)),
            "fold_aucs": stacked_fold_aucs,
        },
        "best_strategy": best_strategy,
        "best_ensemble_auc": best_ensemble_auc,
        "best_single_auc": best_single,
        "improvement": improvement,
        "meets_improvement_target": improvement >= ENSEMBLE_IMPROVEMENT,
        "target_auc": TARGET_AUC,
        "improvement_target": ENSEMBLE_IMPROVEMENT,
        "n_samples": len(labels),
        "n_pd": int(labels.sum()),
        "n_healthy": int((1 - labels).sum()),
    }

    results_path = output_dir / "ensemble_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Best strategy: %s (AUC=%.4f, +%.4f over best single model)",
             best_strategy, best_ensemble_auc, improvement)

    return results
