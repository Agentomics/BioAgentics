"""Ensemble stacking framework for sepsis early warning.

Meta-learner that combines predictions from LR baseline, GBM (XGBoost/LightGBM),
and temporal (LSTM/GRU) models. Uses logistic regression as the stacking
meta-learner with out-of-fold predictions to avoid leakage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from bioagentics.diagnostics.sepsis.config import (
    OUTER_CV_FOLDS,
    RANDOM_STATE,
    RESULTS_DIR,
)
from bioagentics.diagnostics.sepsis.models.lr_baseline import _build_pipeline as _lr_pipeline
from bioagentics.diagnostics.sepsis.models.gbm import _make_xgb_model, _make_lgbm_model, PARAM_GRID
from bioagentics.diagnostics.sepsis.models.temporal import (
    SepsisRNN,
    build_sequences,
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    TEMPORAL_PARAM_GRID,
    _predict_proba as _rnn_predict_proba,
    _train_epoch,
)

logger = logging.getLogger(__name__)


def _get_oof_predictions(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = OUTER_CV_FOLDS,
) -> np.ndarray:
    """Generate out-of-fold predictions from base models.

    Trains LR, XGBoost, LightGBM, LSTM, GRU on each fold and collects
    held-out predictions. Returns (n_samples, 5) array of probabilities.

    Parameters
    ----------
    X : Feature matrix (n_samples, n_features).
    y : Binary labels.
    n_folds : Number of CV folds.

    Returns
    -------
    oof_preds : (n_samples, 5) out-of-fold probability predictions.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    n_models = 5  # LR, XGB, LGBM, LSTM, GRU
    oof_preds = np.zeros((len(X), n_models))

    # Precompute sequences for temporal models
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_clean = scaler.fit_transform(imputer.fit_transform(X))
    X_seq, _ = build_sequences(X_clean, y)

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        logger.info("Ensemble fold %d/%d", fold_i + 1, n_folds)
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # 1. LR
        lr_pipe = _lr_pipeline(C=0.1)
        lr_pipe.fit(X_train, y_train)
        oof_preds[test_idx, 0] = lr_pipe.predict_proba(X_test)[:, 1]

        # 2. XGBoost
        imp = SimpleImputer(strategy="median")
        X_tr_imp = imp.fit_transform(X_train)
        X_te_imp = imp.transform(X_test)
        xgb = _make_xgb_model(PARAM_GRID[0])
        xgb.fit(X_tr_imp, y_train)
        oof_preds[test_idx, 1] = xgb.predict_proba(X_te_imp)[:, 1]

        # 3. LightGBM
        lgbm = _make_lgbm_model(PARAM_GRID[0])
        lgbm.fit(X_tr_imp, y_train)
        oof_preds[test_idx, 2] = lgbm.predict_proba(X_te_imp)[:, 1]

        # 4. LSTM
        X_seq_train = X_seq[train_idx]
        X_seq_test = X_seq[test_idx]
        oof_preds[test_idx, 3] = _train_rnn_and_predict(
            X_seq_train, y_train, X_seq_test, "lstm"
        )

        # 5. GRU
        oof_preds[test_idx, 4] = _train_rnn_and_predict(
            X_seq_train, y_train, X_seq_test, "gru"
        )

    return oof_preds


def _train_rnn_and_predict(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_test_seq: np.ndarray,
    rnn_type: str,
) -> np.ndarray:
    """Train RNN and return test predictions."""
    torch.manual_seed(RANDOM_STATE)
    params = TEMPORAL_PARAM_GRID[0]
    input_size = X_train_seq.shape[2]

    model = SepsisRNN(
        input_size=input_size,
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        rnn_type=rnn_type,
    )

    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_seq, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight = torch.tensor(
        [(len(y_train) - y_train.sum()) / max(y_train.sum(), 1)]
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_loss = float("inf")
    patience_counter = 0
    for _ in range(MAX_EPOCHS):
        loss = _train_epoch(model, X_train_t, y_train_t, optimizer, criterion)
        if loss < best_loss - 1e-4:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    return _rnn_predict_proba(model, X_test_t)


def train_stacking_ensemble(
    oof_preds: np.ndarray,
    y: np.ndarray,
) -> LogisticRegression:
    """Train the stacking meta-learner on out-of-fold predictions.

    Parameters
    ----------
    oof_preds : (n_samples, n_base_models) OOF predictions.
    y : True labels.

    Returns
    -------
    Fitted LogisticRegression meta-learner.
    """
    meta = LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    meta.fit(oof_preds, y)
    return meta


def evaluate_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = OUTER_CV_FOLDS,
) -> dict:
    """Evaluate ensemble via cross-validation.

    Uses a two-level CV: inner loop generates OOF base predictions,
    outer loop evaluates the meta-learner.

    Parameters
    ----------
    X : Feature matrix.
    y : Binary labels.
    n_folds : Number of CV folds.

    Returns
    -------
    Dictionary with ensemble and per-model metrics.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    fold_results = []
    model_names = ["lr", "xgboost", "lightgbm", "lstm", "gru"]

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Get OOF predictions on training set
        oof_train = _get_oof_predictions(X_train, y_train, n_folds=max(n_folds - 1, 2))

        # Train meta-learner
        meta = train_stacking_ensemble(oof_train, y_train)

        # Get base model predictions on test set (full retrain)
        test_preds = _get_test_predictions(X_train, y_train, X_test)

        # Ensemble prediction
        ensemble_prob = meta.predict_proba(test_preds)[:, 1]

        try:
            ens_auroc = float(roc_auc_score(y_test, ensemble_prob))
        except ValueError:
            ens_auroc = 0.5
        try:
            ens_auprc = float(average_precision_score(y_test, ensemble_prob))
        except ValueError:
            ens_auprc = 0.0

        # Individual model metrics for comparison
        individual = {}
        for j, name in enumerate(model_names):
            try:
                individual[name] = {
                    "auroc": float(roc_auc_score(y_test, test_preds[:, j])),
                }
            except ValueError:
                individual[name] = {"auroc": 0.5}

        fold_results.append({
            "fold": fold_i,
            "ensemble_auroc": ens_auroc,
            "ensemble_auprc": ens_auprc,
            "individual": individual,
        })
        logger.info(
            "Fold %d: ensemble AUROC=%.4f AUPRC=%.4f",
            fold_i, ens_auroc, ens_auprc,
        )

    aurocs = [f["ensemble_auroc"] for f in fold_results]
    auprcs = [f["ensemble_auprc"] for f in fold_results]

    # Average individual model AUROCs
    avg_individual = {}
    for name in model_names:
        model_aurocs = [f["individual"].get(name, {}).get("auroc", 0.5) for f in fold_results]
        avg_individual[name] = float(np.mean(model_aurocs))

    return {
        "model": "stacking_ensemble",
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "individual_aurocs": avg_individual,
        "fold_results": fold_results,
    }


def _get_test_predictions(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Train all base models on full training set and predict on test."""
    n_models = 5
    test_preds = np.zeros((len(X_test), n_models))

    # 1. LR
    lr_pipe = _lr_pipeline(C=0.1)
    lr_pipe.fit(X_train, y_train)
    test_preds[:, 0] = lr_pipe.predict_proba(X_test)[:, 1]

    # 2-3. XGB and LGBM
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_train)
    X_te_imp = imp.transform(X_test)

    xgb = _make_xgb_model(PARAM_GRID[0])
    xgb.fit(X_tr_imp, y_train)
    test_preds[:, 1] = xgb.predict_proba(X_te_imp)[:, 1]

    lgbm = _make_lgbm_model(PARAM_GRID[0])
    lgbm.fit(X_tr_imp, y_train)
    test_preds[:, 2] = lgbm.predict_proba(X_te_imp)[:, 1]

    # 4-5. LSTM and GRU
    scaler = StandardScaler()
    imp2 = SimpleImputer(strategy="median")
    X_train_clean = scaler.fit_transform(imp2.fit_transform(X_train))
    X_test_clean = scaler.transform(imp2.transform(X_test))

    X_seq_train, _ = build_sequences(X_train_clean, y_train)
    X_seq_test, _ = build_sequences(X_test_clean, np.zeros(len(X_test)))

    test_preds[:, 3] = _train_rnn_and_predict(
        X_seq_train, y_train, X_seq_test, "lstm"
    )
    test_preds[:, 4] = _train_rnn_and_predict(
        X_seq_train, y_train, X_seq_test, "gru"
    )

    return test_preds


def run_ensemble(
    datasets: dict[int, dict[str, np.ndarray]],
    results_dir: Path = RESULTS_DIR,
) -> dict[int, dict]:
    """Run ensemble stacking on all lookahead windows.

    Returns
    -------
    Dictionary keyed by lookahead with ensemble metrics.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[int, dict] = {}

    for lh in sorted(datasets.keys()):
        data = datasets[lh]
        X = np.vstack([data["X_train"], data["X_test"]])
        y = np.concatenate([data["y_train"], data["y_test"]])

        logger.info("=== Ensemble: %dh lookahead (%d samples) ===", lh, len(y))
        metrics = evaluate_ensemble(X, y)
        metrics["lookahead_hours"] = lh
        metrics["n_samples"] = len(y)
        metrics["n_positive"] = int(y.sum())
        all_results[lh] = metrics

        out_path = results_dir / f"ensemble_{lh}h.json"
        with open(out_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved %s", out_path)

    # Summary
    summary = {
        str(lh): {
            "ensemble_auroc": r["auroc_mean"],
            "individual": r["individual_aurocs"],
        }
        for lh, r in all_results.items()
    }
    summary_path = results_dir / "ensemble_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return all_results
