"""External validation on eICU for sepsis early warning models.

Evaluates MIMIC-IV-trained models on the eICU multi-center dataset
to assess cross-institution generalization. Includes per-center
performance analysis to identify site-specific effects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from bioagentics.diagnostics.sepsis.config import RANDOM_STATE, RESULTS_DIR

logger = logging.getLogger(__name__)

# eICU vital/lab column mapping to MIMIC-IV feature space
EICU_VITAL_MAP: dict[str, str] = {
    "heartrate": "heart_rate",
    "systemicsystolic": "sbp",
    "systemicmean": "map",
    "respiration": "resp_rate",
    "sao2": "spo2",
    "temperature": "temperature",
}

EICU_LAB_MAP: dict[str, str] = {
    "WBC x 1000": "wbc",
    "Hgb": "hemoglobin",
    "Hct": "hematocrit",
    "platelets x 1000": "platelets",
    "lactate": "lactate",
    "creatinine": "creatinine",
    "BUN": "bun",
    "glucose": "glucose",
    "sodium": "sodium",
    "potassium": "potassium",
    "chloride": "chloride",
    "bicarbonate": "bicarbonate",
    "total bilirubin": "bilirubin_total",
    "ALT (SGPT)": "alt",
    "AST (SGOT)": "ast",
    "albumin": "albumin",
    "PT - Loss of INR": "inr",
    "PTT": "ptt",
    "paO2": "pao2",
    "FiO2": "fio2",
}


def map_eicu_features(
    eicu_df_columns: list[str],
    target_features: list[str],
) -> dict[str, str | None]:
    """Map eICU column names to MIMIC-IV feature space.

    Parameters
    ----------
    eicu_df_columns : Column names from eICU data.
    target_features : Expected MIMIC-IV feature names.

    Returns
    -------
    Dictionary mapping target_feature -> eicu_column (or None if missing).
    """
    combined_map = {**EICU_VITAL_MAP, **EICU_LAB_MAP}
    reverse_map = {v: k for k, v in combined_map.items()}

    mapping: dict[str, str | None] = {}
    for feat in target_features:
        eicu_name = reverse_map.get(feat)
        if eicu_name and eicu_name in eicu_df_columns:
            mapping[feat] = eicu_name
        else:
            mapping[feat] = None

    return mapping


def align_features(
    X_external: np.ndarray,
    external_feature_names: list[str],
    target_feature_names: list[str],
) -> np.ndarray:
    """Align external dataset features to target feature order.

    Missing features are filled with NaN (to be imputed later).

    Parameters
    ----------
    X_external : External dataset feature matrix.
    external_feature_names : Column names of X_external.
    target_feature_names : Target feature ordering (from training data).

    Returns
    -------
    Aligned feature matrix with columns matching target_feature_names.
    """
    ext_name_to_idx = {name: i for i, name in enumerate(external_feature_names)}
    aligned = np.full((X_external.shape[0], len(target_feature_names)), np.nan)

    for j, feat in enumerate(target_feature_names):
        if feat in ext_name_to_idx:
            aligned[:, j] = X_external[:, ext_name_to_idx[feat]]

    return aligned


def _compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute classification metrics."""
    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auroc = 0.5
    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        auprc = 0.0

    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2 or len(np.unique(y_pred)) < 2:
        sensitivity = specificity = ppv = 0.0
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "n_samples": len(y_true),
        "n_positive": int(y_true.sum()),
    }


def train_on_source(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict:
    """Train models on source (MIMIC-IV) data.

    Returns fitted models and preprocessing objects.
    """
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_clean = scaler.fit_transform(imp.fit_transform(X_train))

    lr = LogisticRegression(
        C=0.1, l1_ratio=1.0, solver="saga",
        max_iter=2000, random_state=RANDOM_STATE,
    )
    lr.fit(X_clean, y_train)

    imp_gbm = SimpleImputer(strategy="median")
    X_imp = imp_gbm.fit_transform(X_train)

    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        use_label_encoder=False, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=1, verbosity=0,
    )
    xgb_model.fit(X_imp, y_train)

    import lightgbm as lgbm
    lgbm_model = lgbm.LGBMClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        random_state=RANDOM_STATE, n_jobs=1, verbose=-1,
    )
    lgbm_model.fit(X_imp, y_train)

    return {
        "lr": lr,
        "xgb": xgb_model,
        "lgbm": lgbm_model,
        "imputer_lr": imp,
        "scaler_lr": scaler,
        "imputer_gbm": imp_gbm,
    }


def evaluate_external(
    trained: dict,
    X_external: np.ndarray,
    y_external: np.ndarray,
) -> dict:
    """Evaluate trained models on external dataset.

    Parameters
    ----------
    trained : Output of train_on_source.
    X_external : External (eICU) features, aligned to training feature order.
    y_external : External labels.

    Returns
    -------
    Per-model metrics on external data.
    """
    results = {}

    # LR
    X_lr = trained["scaler_lr"].transform(
        trained["imputer_lr"].transform(X_external)
    )
    lr_probs = trained["lr"].predict_proba(X_lr)[:, 1]
    results["logistic_regression"] = _compute_metrics(y_external, lr_probs)

    # XGBoost
    X_gbm = trained["imputer_gbm"].transform(X_external)
    xgb_probs = trained["xgb"].predict_proba(X_gbm)[:, 1]
    results["xgboost"] = _compute_metrics(y_external, xgb_probs)

    # LightGBM
    lgbm_probs = trained["lgbm"].predict_proba(X_gbm)[:, 1]
    results["lightgbm"] = _compute_metrics(y_external, lgbm_probs)

    # Ensemble average
    ens_probs = (lr_probs + xgb_probs + lgbm_probs) / 3.0
    results["ensemble_avg"] = _compute_metrics(y_external, ens_probs)

    best_auroc = max(r["auroc"] for r in results.values())
    results["best_auroc"] = best_auroc
    results["meets_target_080"] = best_auroc >= 0.80

    return results


def per_center_analysis(
    trained: dict,
    X_external: np.ndarray,
    y_external: np.ndarray,
    center_ids: np.ndarray,
    min_samples: int = 30,
    min_positive: int = 5,
) -> dict:
    """Evaluate per-center performance on external data.

    Parameters
    ----------
    trained : Output of train_on_source.
    X_external : External features.
    y_external : External labels.
    center_ids : Hospital/center identifier per sample.
    min_samples : Minimum samples to include a center.
    min_positive : Minimum positive cases to include a center.

    Returns
    -------
    Per-center AUROC and summary statistics.
    """
    X_gbm = trained["imputer_gbm"].transform(X_external)
    xgb_probs = trained["xgb"].predict_proba(X_gbm)[:, 1]

    unique_centers = np.unique(center_ids)
    center_results = {}

    for cid in unique_centers:
        mask = center_ids == cid
        n = int(mask.sum())
        n_pos = int(y_external[mask].sum())

        if n < min_samples or n_pos < min_positive:
            continue

        try:
            auroc = float(roc_auc_score(y_external[mask], xgb_probs[mask]))
        except ValueError:
            auroc = 0.5

        center_results[str(cid)] = {
            "auroc": auroc,
            "n_samples": n,
            "n_positive": n_pos,
            "prevalence": float(n_pos / n),
        }

    if not center_results:
        return {"n_centers": 0, "centers": {}}

    aurocs = [c["auroc"] for c in center_results.values()]
    return {
        "n_centers": len(center_results),
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auroc_min": float(np.min(aurocs)),
        "auroc_max": float(np.max(aurocs)),
        "auroc_median": float(np.median(aurocs)),
        "centers": center_results,
    }


def run_external_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_external: np.ndarray,
    y_external: np.ndarray,
    center_ids: np.ndarray | None = None,
    feature_names: list[str] | None = None,
    results_dir: Path = RESULTS_DIR,
    label: str = "",
) -> dict:
    """Run full external validation pipeline.

    Parameters
    ----------
    X_train : MIMIC-IV training features.
    y_train : MIMIC-IV training labels.
    X_external : eICU features (aligned to training feature order).
    y_external : eICU labels.
    center_ids : Hospital IDs for per-center analysis (optional).
    feature_names : Feature names (optional).
    results_dir : Output directory.
    label : Optional filename label.

    Returns
    -------
    Dictionary with external validation results.
    """
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "External validation: train=%d, external=%d",
        len(y_train), len(y_external),
    )

    trained = train_on_source(X_train, y_train)
    results = evaluate_external(trained, X_external, y_external)

    if center_ids is not None:
        results["per_center"] = per_center_analysis(
            trained, X_external, y_external, center_ids,
        )

    results["source_info"] = {
        "n_train": len(y_train),
        "train_prevalence": float(y_train.mean()),
    }
    results["external_info"] = {
        "n_samples": len(y_external),
        "n_positive": int(y_external.sum()),
        "prevalence": float(y_external.mean()),
    }
    if feature_names:
        results["n_features"] = len(feature_names)

    # Remove non-serializable objects before saving
    suffix = f"_{label}" if label else ""
    out_path = results_dir / f"external_validation{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved %s", out_path)

    return results
