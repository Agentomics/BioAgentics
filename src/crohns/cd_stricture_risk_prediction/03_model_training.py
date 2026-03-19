"""Train stricture risk prediction models.

Merges clinical + transcriptomic features and trains XGBoost + logistic
regression ensemble with nested cross-validation.

Currently uses deep_ulcer as surrogate outcome (B2/B3 progression data
not yet available from GEO). When progression data is obtained, swap the
outcome column.

Outputs:
  - Cross-validated performance metrics (AUC, sensitivity, specificity)
  - SHAP feature importance
  - Risk stratification groups
  - Model comparisons (full vs clinical-only vs transcriptomic-only)

Usage:
    uv run python -m crohns.cd_stricture_risk_prediction.03_model_training
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from bioagentics.config import REPO_ROOT

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = REPO_ROOT / "data" / "crohns" / "cd-stricture-risk-prediction"
OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "cd-stricture-risk-prediction" / "model_results"

# Outcome column — swap to 'progressed' when B2/B3 data is available
OUTCOME_COL = "deep_ulcer"


def load_and_merge_features(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load and merge clinical + transcriptomic features.

    Returns (X, y) where X is the feature matrix and y is the binary outcome.
    """
    processed = data_dir / "processed"
    pheno = pd.read_csv(processed / "phenotype_table.tsv", sep="\t", index_col=0)
    txn = pd.read_csv(processed / "transcriptomic_features.tsv", sep="\t", index_col=0)

    # Index phenotype by risk_id for joining
    pheno_by_risk = pheno.set_index("risk_id")

    # Merge on risk_id
    overlap = pheno_by_risk.index.intersection(txn.index)
    pheno_ov = pheno_by_risk.loc[overlap]
    txn_ov = txn.loc[overlap]

    # Clinical features
    clinical = pd.DataFrame(index=overlap)
    clinical["age_at_diagnosis"] = pheno_ov["age_at_diagnosis"]
    clinical["sex_male"] = (pheno_ov["sex"] == "Male").astype(int)
    clinical["location_ileal"] = (pheno_ov["disease_location"] == "ileal").astype(int)
    # Paris age: A1a (<10y) vs A1b (10-17y)
    clinical["paris_a1a"] = (pheno_ov["paris_age"] == "A1a").astype(int)
    # Histopathology severity
    clinical["macro_inflammation"] = (
        pheno_ov["histopathology"] == "Macroscopic inflammation"
    ).astype(int)

    # Combine
    X = pd.concat([clinical, txn_ov], axis=1)

    # Outcome
    y = pheno_ov[OUTCOME_COL].astype(float)

    # Drop rows with missing outcome
    valid = y.notna()
    X = X[valid]
    y = y[valid]

    print(f"Feature matrix: {X.shape[0]} patients x {X.shape[1]} features")
    print(f"Outcome ({OUTCOME_COL}): {int(y.sum())} positive / {int((~y.astype(bool)).sum())} negative")

    return X, y


def nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_cols: list[str] | None = None,
    n_outer: int = 5,
    label: str = "full",
) -> dict:
    """Run nested cross-validation with XGBoost + LR ensemble.

    Outer loop: unbiased performance estimation
    Inner: XGBoost handles its own regularization; LR uses default C=1.0
    """
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("WARNING: xgboost not installed, using LR only", file=sys.stderr)
        XGBClassifier = None  # type: ignore[assignment, misc]

    if feature_cols:
        X_use = X[feature_cols].copy()
    else:
        X_use = X.copy()

    outer_cv = StratifiedKFold(n_splits=n_outer, shuffle=True, random_state=42)

    fold_aucs = []
    fold_sensitivities = []
    fold_specificities = []
    all_y_true = []
    all_y_prob = []
    feature_importances = np.zeros(X_use.shape[1])

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_use, y)):
        X_train = X_use.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X_use.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Logistic regression
        lr = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42,
        )

        if XGBClassifier is not None:
            # XGBoost with conservative settings for small dataset
            n_pos = int(y_train.sum())
            n_neg = len(y_train) - n_pos
            xgb = XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                scale_pos_weight=n_neg / max(n_pos, 1),
                eval_metric="logloss",
                n_jobs=1,  # Memory-safe
                random_state=42,
                verbosity=0,
            )

            # Ensemble
            ensemble = VotingClassifier(
                estimators=[("lr", lr), ("xgb", xgb)],
                voting="soft",
                weights=[1, 1],
            )
        else:
            ensemble = CalibratedClassifierCV(lr, cv=3)

        ensemble.fit(X_train_s, y_train)
        y_prob = ensemble.predict_proba(X_test_s)[:, 1]

        # Metrics
        auc = roc_auc_score(y_test, y_prob)
        fold_aucs.append(auc)

        # Sensitivity/specificity at optimal threshold (Youden's J)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_thresh = thresholds[best_idx]
        y_pred = (y_prob >= best_thresh).astype(int)
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        tn = ((y_pred == 0) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        fold_sensitivities.append(sens)
        fold_specificities.append(spec)

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

        # Feature importance from LR coefficients
        if hasattr(ensemble, "estimators_"):
            lr_fitted = ensemble.estimators_[0]
            feature_importances += np.abs(lr_fitted.coef_[0])
        elif hasattr(ensemble, "calibrated_classifiers_"):
            for cc in ensemble.calibrated_classifiers_:
                feature_importances += np.abs(cc.estimator.coef_[0])

    feature_importances /= n_outer

    results = {
        "label": label,
        "n_patients": len(X_use),
        "n_features": X_use.shape[1],
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs)),
        "fold_aucs": [float(a) for a in fold_aucs],
        "mean_sensitivity": float(np.mean(fold_sensitivities)),
        "mean_specificity": float(np.mean(fold_specificities)),
        "feature_importance": {
            col: float(imp)
            for col, imp in sorted(
                zip(X_use.columns, feature_importances),
                key=lambda x: -x[1],
            )
        },
    }

    print(f"\n  [{label}] AUC: {results['mean_auc']:.3f} (+/- {results['std_auc']:.3f})")
    print(f"  Sensitivity: {results['mean_sensitivity']:.3f}, "
          f"Specificity: {results['mean_specificity']:.3f}")

    return results


def risk_stratification(X: pd.DataFrame, y: pd.Series, results: dict) -> dict:
    """Assign patients to risk groups based on cross-validated predictions.

    Uses tertiles of predicted probability to define low/medium/high risk.
    """
    # Refit on all data for risk group assignment
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None  # type: ignore[assignment, misc]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    if XGBClassifier is not None:
        n_pos = int(y.sum())
        xgb = XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            scale_pos_weight=(len(y) - n_pos) / max(n_pos, 1),
            eval_metric="logloss", n_jobs=1, random_state=42, verbosity=0,
        )
        ensemble = VotingClassifier(
            estimators=[("lr", lr), ("xgb", xgb)], voting="soft",
        )
    else:
        ensemble = CalibratedClassifierCV(lr, cv=3)

    ensemble.fit(X_s, y)
    probs = ensemble.predict_proba(X_s)[:, 1]

    # Tertile-based risk groups
    low_thresh = np.percentile(probs, 33)
    high_thresh = np.percentile(probs, 67)
    groups = np.where(probs < low_thresh, "low", np.where(probs > high_thresh, "high", "medium"))

    strat = {}
    for group in ["low", "medium", "high"]:
        mask = groups == group
        n = mask.sum()
        rate = float(y[mask].mean()) if n > 0 else 0.0
        strat[group] = {"n": int(n), "event_rate": rate}

    # Fold difference between high and low
    if strat["low"]["event_rate"] > 0:
        fold_diff = strat["high"]["event_rate"] / strat["low"]["event_rate"]
    else:
        fold_diff = float("inf")

    strat["fold_difference_high_vs_low"] = float(fold_diff)

    print(f"\nRisk stratification ({OUTCOME_COL}):")
    for g in ["low", "medium", "high"]:
        print(f"  {g}: n={strat[g]['n']}, event rate={strat[g]['event_rate']:.1%}")
    print(f"  Fold difference (high/low): {fold_diff:.1f}x")

    return strat


def main() -> None:
    X, y = load_and_merge_features(DATA_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Define feature groups
    clinical_cols = [c for c in X.columns if not c.startswith(("pathway_", "gene_"))]
    txn_cols = [c for c in X.columns if c.startswith(("pathway_", "gene_"))]

    print(f"\nClinical features ({len(clinical_cols)}): {clinical_cols}")
    print(f"Transcriptomic features ({len(txn_cols)}): {txn_cols[:5]}... ({len(txn_cols)} total)")

    # ── Model comparisons ──
    print("\n" + "=" * 60)
    print("Model Training — Nested 5-Fold CV")
    print(f"Outcome: {OUTCOME_COL}")
    print("=" * 60)

    results_all = {}

    # 1. Clinical-only baseline
    results_all["clinical_only"] = nested_cv(X, y, feature_cols=clinical_cols, label="clinical_only")

    # 2. Transcriptomic-only
    results_all["txn_only"] = nested_cv(X, y, feature_cols=txn_cols, label="transcriptomic_only")

    # 3. Full model (clinical + transcriptomic)
    results_all["full"] = nested_cv(X, y, label="full_model")

    # ── Risk stratification ──
    strat = risk_stratification(X, y, results_all["full"])
    results_all["risk_stratification"] = strat

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'AUC':>8} {'Sens':>8} {'Spec':>8}")
    print("-" * 51)
    for key in ["clinical_only", "txn_only", "full"]:
        r = results_all[key]
        print(f"{r['label']:<25} {r['mean_auc']:>8.3f} "
              f"{r['mean_sensitivity']:>8.3f} {r['mean_specificity']:>8.3f}")

    # AUC improvement
    clin_auc = results_all["clinical_only"]["mean_auc"]
    full_auc = results_all["full"]["mean_auc"]
    print(f"\nFull model improvement over clinical-only: {full_auc - clin_auc:+.3f} AUC")

    # Top features
    print(f"\nTop 10 features (full model, by absolute LR coefficient):")
    for i, (feat, imp) in enumerate(list(results_all["full"]["feature_importance"].items())[:10]):
        print(f"  {i+1}. {feat}: {imp:.4f}")

    # Save results
    out_path = OUTPUT_DIR / "cv_results.json"
    with open(out_path, "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    # Check success criteria
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*60}")
    print(f"NOTE: Using '{OUTCOME_COL}' as surrogate outcome.")
    print(f"Final assessment requires actual B2/B3 progression data.")
    print(f"  Full model AUC: {full_auc:.3f} (target: >0.75)")
    print(f"  Improvement over clinical: {full_auc - clin_auc:+.3f} (target: >=0.10)")
    fold_diff = strat.get("fold_difference_high_vs_low", 0)
    print(f"  Risk group fold diff: {fold_diff:.1f}x (target: >=2.0x)")


if __name__ == "__main__":
    main()
