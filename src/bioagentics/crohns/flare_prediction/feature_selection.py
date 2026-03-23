"""Within-patient paired analysis and stability selection for feature filtering.

Statistical feature filtering before multi-omic integration:
1. Paired Wilcoxon signed-rank tests (pre-flare vs stable per patient)
2. Stability selection via randomized lasso
3. Ranked feature reports with effect sizes and stability scores

Usage::

    from bioagentics.crohns.flare_prediction.feature_selection import (
        paired_feature_analysis,
        stability_selection,
        select_features,
    )

    report = paired_feature_analysis(features, windows)
    stable_feats = stability_selection(features, labels)
    reduced = select_features(features, report, stable_feats)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)


def paired_feature_analysis(
    features: pd.DataFrame,
    windows: list[Window],
    fdr_threshold: float = 0.1,
) -> pd.DataFrame:
    """Within-patient paired analysis of features (pre-flare vs stable).

    For each feature, compute paired difference (pre-flare minus stable)
    per patient, test significance via paired Wilcoxon signed-rank test,
    and correct for multiple testing (Benjamini-Hochberg FDR).

    Parameters
    ----------
    features:
        Feature matrix (instances x features).
    windows:
        Classification windows aligned with feature rows.
    fdr_threshold:
        FDR threshold for significance.

    Returns
    -------
    DataFrame with columns: feature, mean_diff, median_diff, p_value,
    p_adjusted, significant, omic_layer, n_pairs.
    """
    if len(features) == 0 or len(windows) == 0:
        return pd.DataFrame(
            columns=[
                "feature", "mean_diff", "median_diff", "p_value",
                "p_adjusted", "significant", "omic_layer", "n_pairs",
            ]
        )

    # Build per-patient paired data
    patient_windows: dict[str, dict[str, list[int]]] = {}
    for idx, w in enumerate(windows):
        patient_windows.setdefault(w.subject_id, {}).setdefault(w.label, []).append(idx)

    results = []
    for col in features.columns:
        paired_diffs = []
        for patient_id, label_map in patient_windows.items():
            pre_flare_idxs = label_map.get("pre_flare", [])
            stable_idxs = label_map.get("stable", [])
            if not pre_flare_idxs or not stable_idxs:
                continue

            pre_vals = features.loc[pre_flare_idxs, col].dropna()
            stable_vals = features.loc[stable_idxs, col].dropna()
            if len(pre_vals) == 0 or len(stable_vals) == 0:
                continue

            # Mean within each class for this patient, then diff
            diff = float(pre_vals.mean()) - float(stable_vals.mean())
            paired_diffs.append(diff)

        if len(paired_diffs) < 3:
            # Not enough pairs for Wilcoxon test
            results.append({
                "feature": col,
                "mean_diff": np.nanmean(paired_diffs) if paired_diffs else np.nan,
                "median_diff": np.nanmedian(paired_diffs) if paired_diffs else np.nan,
                "p_value": np.nan,
                "n_pairs": len(paired_diffs),
            })
            continue

        diffs_arr = np.array(paired_diffs)
        # Remove zero diffs for Wilcoxon
        nonzero = diffs_arr[diffs_arr != 0]
        if len(nonzero) < 3:
            p_val = np.nan
        else:
            try:
                _, p_val = wilcoxon(nonzero)
            except ValueError:
                p_val = np.nan

        results.append({
            "feature": col,
            "mean_diff": float(np.mean(diffs_arr)),
            "median_diff": float(np.median(diffs_arr)),
            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
            "n_pairs": len(paired_diffs),
        })

    report = pd.DataFrame(results)
    if report.empty:
        report["p_adjusted"] = []
        report["significant"] = []
        report["omic_layer"] = []
        return report

    # BH FDR correction
    report["p_adjusted"] = _bh_fdr(report["p_value"].values)
    report["significant"] = report["p_adjusted"] < fdr_threshold

    # Infer omic layer from feature prefix
    report["omic_layer"] = report["feature"].apply(_infer_layer)

    report = report.sort_values("p_adjusted").reset_index(drop=True)
    n_sig = report["significant"].sum()
    logger.info(
        "Paired analysis: %d/%d features significant at FDR < %.2f",
        n_sig, len(report), fdr_threshold,
    )
    return report


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.array(p_values, dtype=float)
    n = len(p)
    adjusted = np.full(n, np.nan)

    valid = np.isfinite(p)
    if not valid.any():
        return adjusted

    valid_p = p[valid]
    order = np.argsort(valid_p)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(valid_p) + 1)

    adj = valid_p * len(valid_p) / ranks
    # Enforce monotonicity
    adj_sorted_idx = np.argsort(-ranks)
    adj_cummin = np.minimum.accumulate(adj[adj_sorted_idx])
    adj[adj_sorted_idx] = adj_cummin
    adj = np.clip(adj, 0, 1)

    adjusted[valid] = adj
    return adjusted


def _infer_layer(feature_name: str) -> str:
    """Infer omic layer from feature name prefix."""
    if feature_name.startswith("mb_"):
        return "microbiome"
    elif feature_name.startswith("met_"):
        return "metabolomics"
    elif feature_name.startswith("pw_"):
        return "pathways"
    elif feature_name.startswith("tx_"):
        return "transcriptomics"
    elif feature_name.startswith("sero_"):
        return "serology"
    return "unknown"


def stability_selection(
    features: pd.DataFrame,
    labels: pd.Series,
    n_bootstrap: int = 500,
    threshold: float = 0.6,
    alpha_range: tuple[float, float] = (0.01, 1.0),
    random_state: int = 42,
) -> pd.DataFrame:
    """Stability selection via randomized lasso.

    Run logistic regression with L1 penalty across bootstrap samples
    with random feature scaling. Features selected in >threshold fraction
    of bootstrap models are considered stable.

    Parameters
    ----------
    features:
        Feature matrix (instances x features).
    labels:
        Binary labels (pre_flare=1, stable=0).
    n_bootstrap:
        Number of bootstrap samples.
    threshold:
        Minimum selection frequency to be considered stable.
    alpha_range:
        Range for random regularization scaling.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    DataFrame with columns: feature, selection_freq, stable, omic_layer.
    """
    if features.empty or len(labels) == 0:
        return pd.DataFrame(columns=["feature", "selection_freq", "stable", "omic_layer"])

    rng = np.random.default_rng(random_state)
    y = (labels == "pre_flare").astype(int).values

    # Impute NaN with column medians for modeling
    X = features.values.copy()
    col_medians = np.nanmedian(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_medians[j] if np.isfinite(col_medians[j]) else 0.0

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selection_counts = np.zeros(X.shape[1])

    for b in range(n_bootstrap):
        # Bootstrap resample
        idx = resample(np.arange(len(y)), random_state=rng.integers(0, 2**31))
        X_b, y_b = X_scaled[idx], y[idx]

        # Skip if only one class in bootstrap
        if len(np.unique(y_b)) < 2:
            continue

        # Random feature scaling (randomized lasso)
        scale = rng.uniform(alpha_range[0], alpha_range[1], size=X_b.shape[1])
        X_b_scaled = X_b * scale

        try:
            model = LogisticRegression(
                solver="saga", penalty="l1", C=1.0,
                max_iter=1000, random_state=int(rng.integers(0, 2**31)),
            )
            model.fit(X_b_scaled, y_b)
            selected = np.abs(model.coef_[0]) > 1e-8
            selection_counts += selected
        except Exception:
            continue

    selection_freq = selection_counts / n_bootstrap

    result = pd.DataFrame({
        "feature": features.columns.tolist(),
        "selection_freq": selection_freq,
        "stable": selection_freq >= threshold,
        "omic_layer": [_infer_layer(c) for c in features.columns],
    })
    result = result.sort_values("selection_freq", ascending=False).reset_index(drop=True)

    n_stable = result["stable"].sum()
    logger.info(
        "Stability selection: %d/%d features stable (freq >= %.2f) over %d bootstraps",
        n_stable, len(result), threshold, n_bootstrap,
    )
    return result


def select_features(
    features: pd.DataFrame,
    paired_report: pd.DataFrame,
    stability_report: pd.DataFrame,
    require_significant: bool = True,
    require_stable: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select features that pass both paired analysis and stability selection.

    Parameters
    ----------
    features:
        Full feature matrix.
    paired_report:
        Output from ``paired_feature_analysis``.
    stability_report:
        Output from ``stability_selection``.
    require_significant:
        If True, require FDR-significant in paired analysis.
    require_stable:
        If True, require stable in stability selection.

    Returns
    -------
    Tuple of (reduced feature matrix, combined report).
    """
    # Merge reports
    combined = paired_report.merge(
        stability_report[["feature", "selection_freq", "stable"]],
        on="feature",
        how="outer",
        suffixes=("_paired", "_stability"),
    )

    # Determine which features to keep
    keep = pd.Series(True, index=combined.index)
    if require_significant and "significant" in combined.columns:
        keep = keep & combined["significant"].fillna(False)
    if require_stable and "stable" in combined.columns:
        keep = keep & combined["stable"].fillna(False)

    selected_features = combined.loc[keep, "feature"].tolist()

    # Filter to columns that exist in the feature matrix
    available = [f for f in selected_features if f in features.columns]

    if not available:
        logger.warning("No features passed both filters. Falling back to stability-only selection.")
        stable_only = stability_report.loc[stability_report["stable"], "feature"].tolist()
        available = [f for f in stable_only if f in features.columns]

    reduced = features[available] if available else features.iloc[:, :0]

    logger.info("Selected %d/%d features", len(available), features.shape[1])
    return reduced, combined
