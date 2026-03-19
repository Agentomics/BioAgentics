"""Gradient boosting classifier on classical acoustic features.

Trains on the 100+ dimensional classical feature vector (acoustic, pitch,
MFCC, temporal) to classify PD vs. healthy. This serves as the classical
arm of the ensemble fusion model.
"""

import csv
import json
import logging
from pathlib import Path

import numpy as np

from bioagentics.voice_pd.config import MODELS_DIR, TARGET_AUC

log = logging.getLogger(__name__)

# Feature groups for ablation and importance analysis
FEATURE_GROUPS = {
    "acoustic": ["local_jitter", "local_shimmer", "hnr_db", "nhr"],
    "pitch": [
        "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_cv",
        "f1_mean", "f1_bandwidth", "f2_mean", "f2_bandwidth",
        "f3_mean", "f3_bandwidth", "f4_mean", "f4_bandwidth",
    ],
    "temporal": [
        "phonation_ratio", "n_speech_segments", "n_pauses",
        "pause_frequency", "mean_pause_ms", "std_pause_ms",
        "max_pause_ms", "speech_rate", "mean_speech_ms",
    ],
}
# MFCC features are generated dynamically (78 features)
for i in range(1, 14):
    for prefix in ("mfcc", "delta", "delta2"):
        FEATURE_GROUPS.setdefault("mfcc", []).append(f"{prefix}_{i}_mean")
        FEATURE_GROUPS["mfcc"].append(f"{prefix}_{i}_std")


def load_feature_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load classical features CSV, returning (X, y, feature_names).

    Expects columns: recording_id, pd_status, audio_path, plus feature columns.
    pd_status must be 'pd' or 'healthy' — rows with other values are dropped.
    """
    rows: list[dict] = []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    # Filter to labeled rows only
    labeled = [r for r in rows if r.get("pd_status") in ("pd", "healthy")]
    if not labeled:
        raise ValueError(f"No labeled rows (pd/healthy) found in {path}")

    # Identify feature columns (exclude metadata)
    meta_cols = {"recording_id", "pd_status", "audio_path", "dataset", "subject_id"}
    feature_names = [k for k in labeled[0] if k not in meta_cols]

    X = np.zeros((len(labeled), len(feature_names)), dtype=np.float32)
    y = np.zeros(len(labeled), dtype=np.int32)

    for i, row in enumerate(labeled):
        y[i] = 1 if row["pd_status"] == "pd" else 0
        for j, fname in enumerate(feature_names):
            val = row.get(fname, "")
            try:
                X[i, j] = float(val) if val else np.nan
            except (ValueError, TypeError):
                X[i, j] = np.nan

    log.info("Loaded %d samples (%d PD, %d healthy), %d features",
             len(y), int(y.sum()), int((1 - y).sum()), len(feature_names))
    return X, y, feature_names


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    output_dir: str | Path | None = None,
    n_splits: int = 5,
) -> dict:
    """Train gradient boosting classifier with cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features). May contain NaN.
        y: Binary labels (1=PD, 0=healthy).
        feature_names: Feature column names.
        output_dir: Directory to save model and results.
        n_splits: Number of cross-validation folds.

    Returns:
        Dict with keys: mean_auc, std_auc, fold_aucs, feature_importance,
        best_model (fitted sklearn estimator).
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    if output_dir is None:
        output_dir = MODELS_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline: impute NaNs -> scale -> classify
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )),
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_aucs: list[float] = []
    importances_sum = np.zeros(len(feature_names))

    best_auc = 0.0
    best_model = None

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_prob)
        fold_aucs.append(auc)

        # Accumulate feature importances from the GBM
        importances_sum += pipeline.named_steps["clf"].feature_importances_

        if auc > best_auc:
            best_auc = auc
            best_model = pipeline

        log.info("Fold %d/%d: AUC=%.4f", fold + 1, n_splits, auc)

    mean_auc = float(np.mean(fold_aucs))
    std_auc = float(np.std(fold_aucs))
    avg_importances = importances_sum / n_splits

    # Rank features by importance
    importance_ranking = sorted(
        zip(feature_names, avg_importances.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    results = {
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "fold_aucs": fold_aucs,
        "target_auc": TARGET_AUC,
        "meets_target": mean_auc >= TARGET_AUC,
        "feature_importance": importance_ranking[:30],  # top 30
        "n_samples": len(y),
        "n_features": len(feature_names),
        "n_pd": int(y.sum()),
        "n_healthy": int((1 - y).sum()),
    }

    # Save results
    results_path = output_dir / "classical_gbm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Mean AUC: %.4f +/- %.4f (target: %.2f)", mean_auc, std_auc, TARGET_AUC)
    log.info("Results saved to %s", results_path)

    # Save model
    try:
        import joblib
        model_path = output_dir / "classical_gbm_model.joblib"
        joblib.dump(best_model, model_path)
        log.info("Model saved to %s", model_path)
    except ImportError:
        log.warning("joblib not available — model not saved to disk")

    return {**results, "best_model": best_model}


def feature_group_ablation(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Evaluate AUC using each feature group alone.

    Returns dict of {group_name: mean_auc}.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    group_aucs: dict[str, float] = {}

    for group_name, group_features in FEATURE_GROUPS.items():
        col_idx = [i for i, fn in enumerate(feature_names) if fn in group_features]
        if not col_idx:
            log.warning("No features found for group %s", group_name)
            group_aucs[group_name] = 0.0
            continue

        X_group = X[:, col_idx]
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=100, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ])

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs: list[float] = []

        for train_idx, val_idx in skf.split(X_group, y):
            pipeline.fit(X_group[train_idx], y[train_idx])
            y_prob = pipeline.predict_proba(X_group[val_idx])[:, 1]
            fold_aucs.append(roc_auc_score(y[val_idx], y_prob))

        group_aucs[group_name] = float(np.mean(fold_aucs))
        log.info("Group %s (%d features): AUC=%.4f",
                 group_name, len(col_idx), group_aucs[group_name])

    return group_aucs
