"""MOFA2 multi-omic integration for latent factor discovery.

Stage 1 integration: discover latent factors across omic layers using
MOFA2, then compute per-patient factor trajectories for temporal
modeling in Phase 4.

Usage::

    from bioagentics.crohns.flare_prediction.mofa2_integration import (
        prepare_mofa_input,
        run_mofa,
        extract_factor_trajectories,
    )

    views = prepare_mofa_input(features_by_layer, windows)
    model = run_mofa(views, n_factors=10)
    trajectories = extract_factor_trajectories(model, windows)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

from bioagentics.crohns.flare_prediction.flare_events import Window

logger = logging.getLogger(__name__)

# Omic layer view names
VIEW_NAMES = ["microbiome", "metabolomics", "pathways", "transcriptomics", "serology"]


def prepare_mofa_input(
    features_by_layer: dict[str, pd.DataFrame],
    windows: list[Window],
) -> dict[str, np.ndarray]:
    """Prepare multi-omic data as MOFA2 input views.

    Parameters
    ----------
    features_by_layer:
        Dict mapping layer name to feature matrix (instances x features).
        All matrices must have the same number of rows (instances).
    windows:
        Classification windows (used for metadata only).

    Returns
    -------
    Dict mapping view names to numpy arrays (samples x features).
    NaN values are preserved for MOFA2's missing data handling.
    """
    views: dict[str, np.ndarray] = {}
    n_instances = len(windows)

    for layer_name, features in features_by_layer.items():
        if features is None or features.empty:
            continue
        if len(features) != n_instances:
            logger.warning(
                "Layer %s has %d rows but expected %d, skipping",
                layer_name, len(features), n_instances,
            )
            continue
        views[layer_name] = features.values.astype(float)
        logger.info("View '%s': %d samples x %d features", layer_name, *features.shape)

    return views


def run_mofa(
    views: dict[str, np.ndarray],
    n_factors: int = 10,
    seed: int = 42,
    convergence_mode: str = "fast",
    max_iter: int = 1000,
    output_path: str | Path | None = None,
) -> dict:
    """Run MOFA2 factor analysis on multi-view data.

    Parameters
    ----------
    views:
        Dict of view name -> (samples x features) arrays.
    n_factors:
        Number of latent factors to learn.
    seed:
        Random seed.
    convergence_mode:
        MOFA convergence mode ("fast" or "medium").
    max_iter:
        Maximum training iterations.
    output_path:
        Optional path to save the trained model HDF5 file.

    Returns
    -------
    Dict with keys: factors, weights, variance_explained, view_names.
    """
    from mofapy2.run.entry_point import entry_point

    n_samples = None
    for v in views.values():
        if n_samples is None:
            n_samples = v.shape[0]
        elif v.shape[0] != n_samples:
            raise ValueError("All views must have the same number of samples")

    if n_samples is None or n_samples == 0:
        raise ValueError("No valid views provided")

    # Cap factors at min(n_samples, min_features)
    min_features = min(v.shape[1] for v in views.values())
    n_factors = min(n_factors, n_samples - 1, min_features)
    if n_factors < 1:
        n_factors = 1

    ent = entry_point()

    # Set data: MOFA2 expects list of views, each view is list of groups
    # Single group: [[view1_group1], [view2_group1], ...]
    view_names = list(views.keys())
    data = [[views[v]] for v in view_names]

    ent.set_data_options(scale_groups=False, scale_views=True)
    ent.set_data_matrix(data, views_names=view_names)

    ent.set_model_options(factors=n_factors, spikeslab_weights=True, ard_weights=True)
    ent.set_train_options(
        iter=max_iter,
        convergence_mode=convergence_mode,
        seed=seed,
        verbose=False,
        dropR2=0.001,
    )

    if output_path is not None:
        ent.set_stochastic_options()
        ent.set_train_options(outfile=str(output_path), save_interrupted=True)

    ent.build()

    try:
        ent.run()
    except SystemExit:
        # mofapy2 calls exit() when all factors are dropped
        logger.warning("MOFA2 dropped all factors — no shared structure found")

    # Extract results — model may be uninitialised if run() aborted
    try:
        expectations = ent.model.getExpectations()
    except (AttributeError, RuntimeError):
        logger.warning("MOFA2 model has no expectations — returning zero factors")
        return {
            "factors": np.zeros((n_samples, 1)),
            "weights": {v: np.zeros((1, 1)) for v in view_names},
            "variance_explained": {v: [0.0] for v in view_names},
            "view_names": view_names,
            "n_factors": 1,
        }

    if expectations is None or "Z" not in expectations:
        logger.warning("MOFA2 expectations missing Z — returning zero factors")
        return {
            "factors": np.zeros((n_samples, 1)),
            "weights": {v: np.zeros((1, 1)) for v in view_names},
            "variance_explained": {v: [0.0] for v in view_names},
            "view_names": view_names,
            "n_factors": 1,
        }

    # Z: factors matrix — either ndarray directly or dict with 'E' key
    Z_raw = expectations["Z"]
    if isinstance(Z_raw, dict):
        factors = Z_raw["E"]
    elif isinstance(Z_raw, list):
        factors = Z_raw[0]
    else:
        factors = Z_raw

    # Handle case where all factors were dropped
    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)
    if factors.shape[1] == 0:
        factors = np.zeros((n_samples, 1))

    # W: weights — list of dicts (one per view), each with 'E' key
    W_raw = expectations["W"]
    weights = {}
    for i, vname in enumerate(view_names):
        w_item = W_raw[i]
        weights[vname] = w_item["E"] if isinstance(w_item, dict) else w_item

    # Variance explained per factor per view
    variance_explained: dict[str, list[float]] = {v: [] for v in view_names}
    try:
        r2 = ent.model.calculate_variance_explained()
        if r2 is not None and len(r2) > 0:
            r2_arr = r2[0]  # single group
            for i, vname in enumerate(view_names):
                if isinstance(r2_arr, np.ndarray) and r2_arr.ndim == 2:
                    variance_explained[vname] = r2_arr[:, i].tolist()
    except Exception:
        pass

    result = {
        "factors": factors,
        "weights": weights,
        "variance_explained": variance_explained,
        "view_names": view_names,
        "n_factors": factors.shape[1] if factors.ndim == 2 else 1,
    }

    logger.info(
        "MOFA2 completed: %d factors across %d views, %d samples",
        result["n_factors"], len(view_names), n_samples,
    )
    return result


def extract_factor_trajectories(
    mofa_result: dict,
    windows: list[Window],
) -> pd.DataFrame:
    """Compute per-patient latent factor trajectories.

    For each factor, compute trajectory features within each window:
    mean factor value and slope (if multiple timepoints map to the window).

    Parameters
    ----------
    mofa_result:
        Output from ``run_mofa``.
    windows:
        Classification windows.

    Returns
    -------
    DataFrame with factor trajectory features per instance.
    """
    factors = mofa_result["factors"]
    n_factors = mofa_result["n_factors"]

    feature_rows = []
    for idx, window in enumerate(windows):
        row: dict[str, float] = {}
        for f in range(n_factors):
            val = float(factors[idx, f]) if idx < len(factors) else np.nan
            row[f"mofa_factor_{f}_mean"] = val
        feature_rows.append(row)

    result = pd.DataFrame(feature_rows, index=range(len(windows)))
    result.index.name = "instance_id"

    logger.info(
        "Extracted %d factor trajectory features for %d instances",
        result.shape[1], len(windows),
    )
    return result


def factor_flare_association(
    mofa_result: dict,
    windows: list[Window],
) -> pd.DataFrame:
    """Assess which factors differ between pre-flare and stable windows.

    Parameters
    ----------
    mofa_result:
        Output from ``run_mofa``.
    windows:
        Classification windows.

    Returns
    -------
    DataFrame with factor-level statistics (mean_diff, p_value per factor).
    """
    from scipy.stats import mannwhitneyu

    factors = mofa_result["factors"]
    n_factors = mofa_result["n_factors"]
    labels = np.array([w.label for w in windows])

    results = []
    for f in range(n_factors):
        vals = factors[:, f]
        pre_flare = vals[labels == "pre_flare"]
        stable = vals[labels == "stable"]

        mean_diff = float(np.mean(pre_flare) - np.mean(stable))

        if len(pre_flare) >= 2 and len(stable) >= 2:
            try:
                _, p_val = mannwhitneyu(pre_flare, stable, alternative="two-sided")
            except ValueError:
                p_val = np.nan
        else:
            p_val = np.nan

        results.append({
            "factor": f,
            "mean_pre_flare": float(np.mean(pre_flare)) if len(pre_flare) > 0 else np.nan,
            "mean_stable": float(np.mean(stable)) if len(stable) > 0 else np.nan,
            "mean_diff": mean_diff,
            "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
        })

    report = pd.DataFrame(results)
    logger.info(
        "Factor-flare association: %d factors tested",
        len(report),
    )
    return report


def save_mofa_results(
    mofa_result: dict,
    association: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Save MOFA2 results to output directory.

    Parameters
    ----------
    mofa_result:
        Output from ``run_mofa``.
    association:
        Output from ``factor_flare_association``.
    output_dir:
        Directory to save results.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Factor values
    factors_df = pd.DataFrame(
        mofa_result["factors"],
        columns=[f"factor_{i}" for i in range(mofa_result["n_factors"])],
    )
    factors_df.to_csv(output_dir / "mofa_factors.csv", index=False)

    # Weights per view
    for vname, W in mofa_result["weights"].items():
        w_df = pd.DataFrame(
            W, columns=[f"factor_{i}" for i in range(mofa_result["n_factors"])],
        )
        w_df.to_csv(output_dir / f"mofa_weights_{vname}.csv", index=False)

    # Variance explained
    var_df = pd.DataFrame(mofa_result["variance_explained"])
    var_df.index.name = "factor"
    var_df.to_csv(output_dir / "mofa_variance_explained.csv")

    # Association results
    association.to_csv(output_dir / "mofa_factor_flare_association.csv", index=False)

    logger.info("Saved MOFA2 results to %s", output_dir)
