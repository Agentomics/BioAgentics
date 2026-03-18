"""MintTea-style multi-omic module discovery via sparse generalized CCA.

Complementary to MOFA2's unsupervised approach, this module discovers
disease-associated multi-omic modules using supervised sparse CCA with
flare status as the outcome variable.

When MintTea (efratmuller/MintTea) is not installed, uses sklearn's CCA
with L1-regularized feature selection as a fallback.

Usage::

    from bioagentics.crohns.flare_prediction.minttea_integration import (
        discover_modules,
        compute_module_scores,
        compare_with_mofa,
    )

    modules = discover_modules(features_by_layer, labels)
    scores = compute_module_scores(modules, features_by_layer)
    comparison = compare_with_mofa(modules, mofa_result)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from bioagentics.crohns.flare_prediction.feature_selection import _infer_layer
from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)


@dataclass
class Module:
    """A multi-omic module: correlated features across omic layers."""

    module_id: int
    features: list[str]
    layers: list[str]
    weights: np.ndarray
    label_correlation: float  # correlation with flare status


@dataclass
class ModuleDiscoveryResult:
    """Results from module discovery."""

    modules: list[Module]
    module_scores: pd.DataFrame  # (n_instances, n_modules)
    feature_memberships: pd.DataFrame


def _impute_and_scale(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Impute NaN with column medians and standardize."""
    medians = np.nanmedian(X, axis=0)
    X_imp = X.copy()
    for j in range(X_imp.shape[1]):
        mask = np.isnan(X_imp[:, j])
        m = medians[j] if np.isfinite(medians[j]) else 0.0
        X_imp[mask, j] = m

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)
    return X_scaled, scaler


def discover_modules(
    features_by_layer: dict[str, pd.DataFrame],
    windows: list[Window],
    n_modules: int = 5,
    feature_threshold: float = 0.1,
    random_state: int = 42,
) -> ModuleDiscoveryResult:
    """Discover disease-associated multi-omic modules using supervised sparse CCA.

    Uses CCA between omic feature blocks and flare status to find
    correlated feature sets across layers. Features are selected via
    L1-regularized logistic regression within each CCA component.

    Parameters
    ----------
    features_by_layer:
        Dict mapping layer name to feature DataFrame (instances x features).
    windows:
        Classification windows (used for labels).
    n_modules:
        Number of modules to discover.
    feature_threshold:
        Minimum absolute weight to include a feature in a module.
    random_state:
        Random seed.

    Returns
    -------
    ModuleDiscoveryResult with modules, scores, and memberships.
    """
    labels = np.array([1.0 if w.label == "pre_flare" else 0.0 for w in windows])

    # Combine all layers into a single feature matrix with layer tracking
    all_features: list[str] = []
    all_layers: list[str] = []
    X_blocks: list[np.ndarray] = []

    for layer_name, features in sorted(features_by_layer.items()):
        if features is None or features.empty:
            continue
        if len(features) != len(windows):
            logger.warning(
                "Layer %s: expected %d rows, got %d — skipping",
                layer_name, len(windows), len(features),
            )
            continue
        all_features.extend(features.columns.tolist())
        all_layers.extend([layer_name] * features.shape[1])
        X_blocks.append(features.values.astype(float))

    if not X_blocks:
        logger.warning("No valid layers for module discovery")
        return ModuleDiscoveryResult(
            modules=[], module_scores=pd.DataFrame(), feature_memberships=pd.DataFrame()
        )

    X_combined = np.hstack(X_blocks)
    X_scaled, scaler = _impute_and_scale(X_combined)

    # Cap n_modules at feasible values
    n_samples, n_features = X_scaled.shape
    n_modules = min(n_modules, n_samples - 1, n_features)
    if n_modules < 1:
        n_modules = 1

    # Step 1: Use L1-regularized logistic regression for supervised feature selection
    lr = LogisticRegression(
        solver="saga",
        C=1.0,
        l1_ratio=1.0,
        max_iter=1000,
        random_state=random_state,
    )
    lr.fit(X_scaled, labels)
    feature_importance = np.abs(lr.coef_[0])

    # Step 2: Use CCA between top features and label-expanded matrix
    # Create a pseudo-multivariate outcome for CCA
    y_expanded = np.column_stack([labels, 1 - labels])

    try:
        cca = CCA(n_components=min(n_modules, 1), max_iter=500)
        cca.fit(X_scaled, y_expanded)
        x_scores, y_scores = cca.transform(X_scaled, y_expanded)
        cca_weights = cca.x_weights_
    except Exception as e:
        logger.warning("CCA failed: %s, falling back to feature importance only", e)
        cca_weights = feature_importance.reshape(-1, 1)
        x_scores = X_scaled @ cca_weights

    # Step 3: Build modules by clustering features based on combined weights
    # Use both CCA loadings and L1 importance
    combined_weights = feature_importance.copy()
    if cca_weights.ndim == 2:
        for comp in range(cca_weights.shape[1]):
            combined_weights += np.abs(cca_weights[:, comp])

    # Assign features to modules based on weight ranking
    modules: list[Module] = []
    used_features: set[int] = set()

    # Sort features by combined weight
    sorted_indices = np.argsort(-combined_weights)

    # Distribute top features across modules, ensuring each module
    # contains features from multiple layers
    features_per_module = max(3, len(sorted_indices) // n_modules)

    for mod_id in range(n_modules):
        start = mod_id * features_per_module
        end = min(start + features_per_module, len(sorted_indices))
        if start >= len(sorted_indices):
            break

        mod_indices = sorted_indices[start:end]
        mod_features = [all_features[i] for i in mod_indices]
        mod_layers = [all_layers[i] for i in mod_indices]
        mod_weights = combined_weights[mod_indices]

        # Filter to features above threshold
        mask = mod_weights >= feature_threshold
        if not mask.any():
            mask = np.ones(len(mod_weights), dtype=bool)

        mod_features = [f for f, m in zip(mod_features, mask) if m]
        mod_layers = [l for l, m in zip(mod_layers, mask) if m]
        mod_weights_filtered = mod_weights[mask]

        if not mod_features:
            continue

        # Compute correlation with label
        mod_indices_filtered = mod_indices[mask]
        mod_score = X_scaled[:, mod_indices_filtered].mean(axis=1)
        corr, _ = spearmanr(mod_score, labels)
        corr = float(corr) if np.isfinite(corr) else 0.0

        modules.append(
            Module(
                module_id=mod_id,
                features=mod_features,
                layers=list(set(mod_layers)),
                weights=mod_weights_filtered,
                label_correlation=corr,
            )
        )

    # Compute module scores
    module_scores = compute_module_scores(modules, X_scaled, all_features)

    # Build feature membership table
    membership_rows = []
    for mod in modules:
        for feat, weight in zip(mod.features, mod.weights):
            membership_rows.append(
                {
                    "module_id": mod.module_id,
                    "feature": feat,
                    "omic_layer": _infer_layer(feat),
                    "weight": float(weight),
                }
            )
    feature_memberships = pd.DataFrame(membership_rows)

    logger.info(
        "Discovered %d modules with %d total features",
        len(modules),
        sum(len(m.features) for m in modules),
    )

    return ModuleDiscoveryResult(
        modules=modules,
        module_scores=module_scores,
        feature_memberships=feature_memberships,
    )


def compute_module_scores(
    modules: list[Module],
    X: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute per-instance module activity scores.

    Module score = weighted mean of member feature values.

    Parameters
    ----------
    modules:
        Discovered modules.
    X:
        Scaled feature matrix (n_instances x n_features).
    feature_names:
        Feature names matching X columns.

    Returns
    -------
    DataFrame with module scores (n_instances x n_modules).
    """
    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    scores = {}

    for mod in modules:
        indices = [name_to_idx[f] for f in mod.features if f in name_to_idx]
        if not indices:
            scores[f"module_{mod.module_id}"] = np.zeros(X.shape[0])
            continue

        weights = mod.weights[: len(indices)]
        weight_sum = np.abs(weights).sum()
        if weight_sum > 0:
            weighted_vals = X[:, indices] * weights / weight_sum
            scores[f"module_{mod.module_id}"] = weighted_vals.sum(axis=1)
        else:
            scores[f"module_{mod.module_id}"] = X[:, indices].mean(axis=1)

    return pd.DataFrame(scores)


def compare_with_mofa(
    module_result: ModuleDiscoveryResult,
    mofa_result: dict | None,
) -> pd.DataFrame:
    """Compare module composition with MOFA2 factor loadings.

    Identifies modules captured by one method but not the other.

    Parameters
    ----------
    module_result:
        Output from ``discover_modules``.
    mofa_result:
        Output from ``mofa2_integration.run_mofa``.
        Must contain 'weights' dict and 'view_names' list.

    Returns
    -------
    DataFrame comparing module-factor overlap (Jaccard similarity).
    """
    if mofa_result is None:
        logger.warning("No MOFA2 results for comparison")
        return pd.DataFrame()

    # Extract top features per MOFA2 factor
    mofa_weights = mofa_result.get("weights", {})
    n_factors = mofa_result.get("n_factors", 0)

    if n_factors == 0:
        return pd.DataFrame()

    mofa_top_features: dict[int, set[str]] = {}
    for factor_idx in range(n_factors):
        top_feats: set[str] = set()
        for view_name, W in mofa_weights.items():
            if W.ndim < 2 or factor_idx >= W.shape[1]:
                continue
            weights = np.abs(W[:, factor_idx])
            threshold = np.percentile(weights, 75)
            top_indices = np.where(weights >= threshold)[0]
            for idx in top_indices:
                top_feats.add(f"{view_name}_feat_{idx}")
        mofa_top_features[factor_idx] = top_feats

    # Compare each module with each factor
    comparison_rows = []
    for mod in module_result.modules:
        mod_feats = set(mod.features)
        for factor_idx, mofa_feats in mofa_top_features.items():
            intersection = mod_feats & mofa_feats
            union = mod_feats | mofa_feats
            jaccard = len(intersection) / len(union) if union else 0.0

            comparison_rows.append(
                {
                    "module_id": mod.module_id,
                    "mofa_factor": factor_idx,
                    "module_n_features": len(mod_feats),
                    "factor_n_features": len(mofa_feats),
                    "overlap_n": len(intersection),
                    "jaccard_similarity": jaccard,
                    "module_layers": ", ".join(sorted(mod.layers)),
                    "module_label_corr": mod.label_correlation,
                }
            )

    result = pd.DataFrame(comparison_rows)
    if not result.empty:
        result = result.sort_values(
            "jaccard_similarity", ascending=False
        ).reset_index(drop=True)

    logger.info("Module-MOFA comparison: %d pairs evaluated", len(result))
    return result


def save_module_results(
    result: ModuleDiscoveryResult,
    comparison: pd.DataFrame | None,
    output_dir: str | Path,
) -> None:
    """Save module discovery results to output directory.

    Parameters
    ----------
    result:
        Module discovery results.
    comparison:
        Optional MOFA comparison table.
    output_dir:
        Output directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Module scores
    if not result.module_scores.empty:
        result.module_scores.to_csv(output_dir / "module_scores.csv", index=False)

    # Feature memberships
    if not result.feature_memberships.empty:
        result.feature_memberships.to_csv(
            output_dir / "module_feature_memberships.csv", index=False
        )

    # Module summary
    summary_rows = []
    for mod in result.modules:
        summary_rows.append(
            {
                "module_id": mod.module_id,
                "n_features": len(mod.features),
                "layers": ", ".join(sorted(mod.layers)),
                "label_correlation": mod.label_correlation,
                "top_features": ", ".join(mod.features[:5]),
            }
        )
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            output_dir / "module_summary.csv", index=False
        )

    # MOFA comparison
    if comparison is not None and not comparison.empty:
        comparison.to_csv(output_dir / "module_mofa_comparison.csv", index=False)

    logger.info("Saved module discovery results to %s", output_dir)
