"""Bootstrap validation for methylation-only CRC classifier.

Addresses analyst-flagged overfitting risk: 20 features selected from 298
candidates on n=24 samples. Feature selection is placed INSIDE the LOO loop
to prevent information leakage.

Validation strategy:
  1. Corrected LOO-CV with feature selection inside each fold (primary estimate)
  2. Bootstrap CI of LOO-CV predictions (resample prediction-label pairs, 1000x)
  3. OOB bootstrap (train on bootstrap sample, test on held-out OOB samples, 1000x)
  4. Permutation null (500 shuffled-label runs) for p-values

Output:
    output/diagnostics/crc-liquid-biopsy-panel/methylation_bootstrap_validation.json

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.methylation_bootstrap_validation [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from bioagentics.config import REPO_ROOT
from bioagentics.diagnostics.crc_liquid_biopsy.methylation_classifier import (
    build_cfdna_features,
    extract_cmcd_cpgs,
)

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "crc-liquid-biopsy-panel"

N_BOOTSTRAP = 1000
N_PERMUTATIONS = 500
MAX_FEATURES = 20


def _select_features_on_train(
    X_train: np.ndarray, y_train: np.ndarray, max_features: int = MAX_FEATURES
) -> np.ndarray:
    """Select top CpGs by individual AUC on training data only."""
    aucs = []
    for i in range(X_train.shape[1]):
        col = X_train[:, i]
        if np.std(col) < 1e-10:
            aucs.append(0.5)
            continue
        try:
            auc = roc_auc_score(y_train, col)
            aucs.append(max(auc, 1.0 - auc))
        except ValueError:
            aucs.append(0.5)
    return np.argsort(aucs)[::-1][:max_features]


def _fit_predict(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    max_features: int = MAX_FEATURES, C: float = 1.0,
) -> np.ndarray:
    """Feature selection + scale + fit + predict in one step."""
    feat_idx = _select_features_on_train(X_train, y_train, max_features)
    X_tr = X_train[:, feat_idx]
    X_te = X_test[:, feat_idx]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    model = LogisticRegression(
        C=C, penalty="l2", solver="lbfgs", max_iter=5000, random_state=42,
    )
    model.fit(X_tr, y_train)
    return model.predict_proba(X_te)[:, 1]


def _compute_metrics(y: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Compute AUC and sensitivity at 95% specificity."""
    auc = float(roc_auc_score(y, probs))
    fpr, tpr, _ = roc_curve(y, probs)
    sens_95 = float(np.interp(0.05, fpr, tpr))
    return auc, sens_95


def corrected_loo_cv(
    X: np.ndarray, y: np.ndarray, max_features: int = MAX_FEATURES
) -> tuple[float, float, np.ndarray]:
    """LOO-CV with feature selection INSIDE each fold (LogisticRegressionCV).

    This is the primary, careful estimate using inner CV for C selection.
    """
    loo = LeaveOneOut()
    probs = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        feat_idx = _select_features_on_train(X_train, y_train, max_features)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train[:, feat_idx])
        X_test_s = scaler.transform(X_test[:, feat_idx])

        model = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 10).tolist(),
            penalty="l2",
            solver="lbfgs",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring="roc_auc",
            max_iter=5000,
            random_state=42,
        )
        model.fit(X_train_s, y_train)
        probs[test_idx] = model.predict_proba(X_test_s)[:, 1]

    auc, sens_95 = _compute_metrics(y, probs)
    return auc, sens_95, probs


def bootstrap_ci_from_predictions(
    y: np.ndarray, probs: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, seed: int = 42
) -> dict:
    """Bootstrap CI by resampling (prediction, label) pairs from LOO-CV output.

    This avoids the LOO-with-duplicates problem. The LOO-CV predictions are
    already computed; we just resample them to estimate metric variability.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    sens95s = []

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = y[idx]
        p_boot = probs[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        auc, sens95 = _compute_metrics(y_boot, p_boot)
        aucs.append(auc)
        sens95s.append(sens95)

    aucs_arr = np.array(aucs)
    sens_arr = np.array(sens95s)

    return {
        "method": "Bootstrap CI from LOO-CV prediction-label pairs",
        "n_valid_resamples": len(aucs),
        "auc": {
            "mean": float(np.mean(aucs_arr)),
            "median": float(np.median(aucs_arr)),
            "std": float(np.std(aucs_arr)),
            "ci_lower": float(np.percentile(aucs_arr, 2.5)),
            "ci_upper": float(np.percentile(aucs_arr, 97.5)),
        },
        "sensitivity_at_95spec": {
            "mean": float(np.mean(sens_arr)),
            "median": float(np.median(sens_arr)),
            "std": float(np.std(sens_arr)),
            "ci_lower": float(np.percentile(sens_arr, 2.5)),
            "ci_upper": float(np.percentile(sens_arr, 97.5)),
        },
    }


def oob_bootstrap(
    X: np.ndarray, y: np.ndarray, n_bootstrap: int = N_BOOTSTRAP, seed: int = 42
) -> dict:
    """Out-of-bag bootstrap: train on bootstrap sample, test on OOB samples.

    Avoids LOO-CV duplicate problem. Each iteration:
      1. Draw n samples with replacement -> training set
      2. Samples not drawn -> OOB test set (~37% of data)
      3. Feature selection + model on training, evaluate on OOB
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    aucs = []
    sens95s = []

    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        oob_mask = np.ones(n, dtype=bool)
        oob_mask[idx] = False
        oob_idx = np.where(oob_mask)[0]

        if len(oob_idx) < 4:
            continue
        if len(np.unique(y[idx])) < 2 or len(np.unique(y[oob_idx])) < 2:
            continue

        probs = _fit_predict(X[idx], y[idx], X[oob_idx])
        try:
            auc, sens95 = _compute_metrics(y[oob_idx], probs)
            aucs.append(auc)
            sens95s.append(sens95)
        except ValueError:
            continue

        if (i + 1) % 200 == 0:
            logger.info(
                "OOB bootstrap %d/%d: median AUC=%.3f",
                i + 1, n_bootstrap, np.median(aucs),
            )

    aucs_arr = np.array(aucs)
    sens_arr = np.array(sens95s)

    return {
        "method": "Out-of-bag bootstrap (train on resample, test on OOB)",
        "n_valid_resamples": len(aucs),
        "auc": {
            "mean": float(np.mean(aucs_arr)),
            "median": float(np.median(aucs_arr)),
            "std": float(np.std(aucs_arr)),
            "ci_lower": float(np.percentile(aucs_arr, 2.5)),
            "ci_upper": float(np.percentile(aucs_arr, 97.5)),
        },
        "sensitivity_at_95spec": {
            "mean": float(np.mean(sens_arr)),
            "median": float(np.median(sens_arr)),
            "std": float(np.std(sens_arr)),
            "ci_lower": float(np.percentile(sens_arr, 2.5)),
            "ci_upper": float(np.percentile(sens_arr, 97.5)),
        },
    }


def run_permutation_null(
    X: np.ndarray, y: np.ndarray, n_perms: int = N_PERMUTATIONS, seed: int = 123
) -> tuple[dict, list[float]]:
    """Permutation test on the original data (no bootstrap). Shuffles labels,
    runs LOO-CV with internal feature selection using fixed-C LR for speed.

    Returns (summary_dict, raw_null_aucs).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    null_aucs: list[float] = []
    null_sens95s: list[float] = []

    for i in range(n_perms):
        y_perm = rng.permutation(y)
        if len(np.unique(y_perm)) < 2:
            continue

        # Fast LOO-CV with fixed C on permuted labels
        loo = LeaveOneOut()
        probs = np.zeros(n)
        for train_idx, test_idx in loo.split(X, y_perm):
            p = _fit_predict(X[train_idx], y_perm[train_idx], X[test_idx])
            probs[test_idx] = p

        auc, sens95 = _compute_metrics(y_perm, probs)
        null_aucs.append(auc)
        null_sens95s.append(sens95)

        if (i + 1) % 100 == 0:
            logger.info(
                "Permutation %d/%d: median null AUC=%.3f",
                i + 1, n_perms, np.median(null_aucs),
            )

    summary = {
        "n_permutations": len(null_aucs),
        "null_auc": {
            "mean": float(np.mean(null_aucs)),
            "std": float(np.std(null_aucs)),
            "p95": float(np.percentile(null_aucs, 95)),
        },
        "null_sensitivity_at_95spec": {
            "mean": float(np.mean(null_sens95s)),
            "std": float(np.std(null_sens95s)),
            "p95": float(np.percentile(null_sens95s, 95)),
        },
    }
    return summary, null_aucs


def run_validation(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict:
    """Run full bootstrap validation pipeline."""
    output_path = output_dir / "methylation_bootstrap_validation.json"
    if output_path.exists() and not force:
        logger.info("Loading cached results from %s", output_path)
        with open(output_path) as f:
            return json.load(f)

    # Load data
    cpgs = extract_cmcd_cpgs(output_dir)
    X, y, _ = build_cfdna_features(cpgs, data_dir)
    logger.info("Feature matrix: %d samples x %d CpGs, %d CRC / %d control",
                X.shape[0], X.shape[1], y.sum(), len(y) - y.sum())

    # 1. Corrected LOO-CV (feature selection inside, LogisticRegressionCV)
    logger.info("=== Corrected LOO-CV (feature selection inside loop) ===")
    corrected_auc, corrected_sens95, loo_probs = corrected_loo_cv(X, y)
    logger.info("Corrected LOO-CV: AUC=%.3f, sens@95spec=%.3f", corrected_auc, corrected_sens95)

    # 2. Bootstrap CI from LOO-CV predictions (fast - no retraining)
    logger.info("=== Bootstrap CI from LOO predictions (%d resamples) ===", N_BOOTSTRAP)
    bootstrap_pred = bootstrap_ci_from_predictions(y, loo_probs)
    logger.info("Bootstrap CI AUC: [%.3f, %.3f]",
                bootstrap_pred["auc"]["ci_lower"], bootstrap_pred["auc"]["ci_upper"])

    # 3. OOB bootstrap (independent train/test, feature selection inside)
    logger.info("=== OOB bootstrap (%d resamples) ===", N_BOOTSTRAP)
    oob = oob_bootstrap(X, y)
    logger.info("OOB bootstrap: median AUC=%.3f", oob["auc"]["median"])

    # 4. Permutation null
    logger.info("=== Permutation null (%d permutations) ===", N_PERMUTATIONS)
    permutation, null_aucs_raw = run_permutation_null(X, y)

    # 5. Compute p-values
    p_value_auc = float(np.mean([a >= corrected_auc for a in null_aucs_raw]))

    results = {
        "corrected_loo_cv": {
            "auc": float(corrected_auc),
            "sensitivity_at_95spec": float(corrected_sens95),
            "method": "LOO-CV with feature selection INSIDE each fold (LogisticRegressionCV)",
            "max_features_per_fold": MAX_FEATURES,
            "n_candidate_features": X.shape[1],
            "n_samples": len(y),
        },
        "original_loo_cv": {
            "auc": 0.979,
            "sensitivity_at_95spec": 0.833,
            "method": "LOO-CV with feature selection OUTSIDE (leaked)",
            "note": "From methylation_classifier.py - feature selection on full dataset before CV",
        },
        "bootstrap_ci": bootstrap_pred,
        "oob_bootstrap": oob,
        "permutation_null": permutation,
        "p_value_auc": p_value_auc,
        "overfitting_assessment": "",
    }

    # Assess overfitting
    original_auc = 0.979
    drop = original_auc - corrected_auc
    if drop > 0.1:
        severity = "SEVERE"
    elif drop > 0.05:
        severity = "MODERATE"
    else:
        severity = "MINIMAL"

    results["overfitting_assessment"] = (
        f"{severity} overfitting detected. "
        f"Original AUC (leaked): {original_auc:.3f} -> Corrected AUC: {corrected_auc:.3f} "
        f"(drop: {drop:.3f}). "
        f"Bootstrap 95% CI: [{bootstrap_pred['auc']['ci_lower']:.3f}, {bootstrap_pred['auc']['ci_upper']:.3f}]. "
        f"OOB bootstrap median AUC: {oob['auc']['median']:.3f}. "
        f"Permutation p-value: {p_value_auc:.4f}."
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", output_path)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap validation for methylation CRC classifier"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_validation(args.data_dir, args.output_dir, force=args.force)

    print("\n=== Bootstrap Validation: Methylation Classifier ===")
    corrected = results["corrected_loo_cv"]
    print(f"Corrected LOO-CV (selection inside): AUC={corrected['auc']:.3f}, "
          f"sens@95spec={corrected['sensitivity_at_95spec']:.3f}")

    orig = results["original_loo_cv"]
    print(f"Original  LOO-CV (selection outside): AUC={orig['auc']:.3f}, "
          f"sens@95spec={orig['sensitivity_at_95spec']:.3f}")

    boot = results["bootstrap_ci"]
    print(f"\nBootstrap CI ({boot['n_valid_resamples']} resamples of LOO predictions):")
    print(f"  AUC: {boot['auc']['mean']:.3f} [{boot['auc']['ci_lower']:.3f}, {boot['auc']['ci_upper']:.3f}]")
    print(f"  Sens@95spec: {boot['sensitivity_at_95spec']['mean']:.3f} "
          f"[{boot['sensitivity_at_95spec']['ci_lower']:.3f}, "
          f"{boot['sensitivity_at_95spec']['ci_upper']:.3f}]")

    oob = results["oob_bootstrap"]
    print(f"\nOOB Bootstrap ({oob['n_valid_resamples']} valid resamples):")
    print(f"  AUC: {oob['auc']['median']:.3f} [{oob['auc']['ci_lower']:.3f}, {oob['auc']['ci_upper']:.3f}]")
    print(f"  Sens@95spec: {oob['sensitivity_at_95spec']['median']:.3f} "
          f"[{oob['sensitivity_at_95spec']['ci_lower']:.3f}, "
          f"{oob['sensitivity_at_95spec']['ci_upper']:.3f}]")

    perm = results["permutation_null"]
    print(f"\nPermutation null ({perm['n_permutations']} perms):")
    print(f"  Null AUC: {perm['null_auc']['mean']:.3f} +/- {perm['null_auc']['std']:.3f}")
    print(f"  p-value (AUC): {results['p_value_auc']:.4f}")

    print(f"\n{results['overfitting_assessment']}")


if __name__ == "__main__":
    main()
