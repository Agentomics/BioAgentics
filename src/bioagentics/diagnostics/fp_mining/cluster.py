"""FP clustering pipeline to identify coherent subgroups.

Clusters false positive cases by feature similarity using K-Means and DBSCAN
to determine whether FPs form coherent subgroups or are random noise.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from bioagentics.diagnostics.fp_mining.extract import ExtractionResult, get_feature_columns

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/diagnostics/false-positive-biomarker-mining/clusters")


def prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, list[str], StandardScaler]:
    """Scale features for clustering.

    Args:
        df: DataFrame with feature columns.
        feature_cols: Columns to use. If None, auto-detected.

    Returns:
        Tuple of (scaled_array, feature_names, fitted_scaler).
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X = df[feature_cols].values.astype(float)

    # Impute NaNs with column medians
    col_medians = np.nanmedian(X, axis=0)
    nan_mask = np.isnan(X)
    for j in range(X.shape[1]):
        X[nan_mask[:, j], j] = col_medians[j]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, feature_cols, scaler


def find_optimal_k(
    X: np.ndarray,
    k_range: range | None = None,
) -> tuple[int, list[float]]:
    """Find optimal number of clusters via silhouette score.

    Args:
        X: Scaled feature matrix.
        k_range: Range of k values to try. Defaults to 2-min(10, n//5).

    Returns:
        Tuple of (best_k, silhouette_scores).
    """
    n = X.shape[0]
    if k_range is None:
        max_k = min(10, max(3, n // 5))
        k_range = range(2, max_k + 1)

    scores = []
    for k in k_range:
        if k >= n:
            break
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            scores.append(-1.0)
        else:
            scores.append(float(silhouette_score(X, labels)))

    if not scores:
        return 2, []

    best_idx = int(np.argmax(scores))
    best_k = list(k_range)[best_idx]
    return best_k, scores


def cluster_kmeans(
    X: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, KMeans]:
    """Run K-Means clustering.

    Returns:
        Tuple of (cluster_labels, fitted_model).
    """
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    return labels, km


def cluster_dbscan(
    X: np.ndarray,
    eps: float | None = None,
    min_samples: int = 5,
) -> np.ndarray:
    """Run DBSCAN clustering.

    Args:
        X: Scaled feature matrix.
        eps: Neighborhood radius. If None, estimated from data.
        min_samples: Minimum cluster size.

    Returns:
        Cluster labels (-1 = noise).
    """
    if eps is None:
        # Estimate eps from k-nearest neighbor distances
        from sklearn.neighbors import NearestNeighbors

        k = min(min_samples, X.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k, n_jobs=1)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        sorted_dists = np.sort(distances[:, -1])
        # Use the "knee" point — approximate as the distance at 90th percentile
        eps = float(sorted_dists[int(0.9 * len(sorted_dists))])

    db = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=1)
    return db.fit_predict(X)


def compute_cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute centroid feature profiles for each cluster.

    Returns:
        DataFrame with cluster_id, n_samples, and mean feature values.
    """
    df_with_labels = df.copy()
    df_with_labels["cluster"] = labels

    profiles = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue  # Skip noise points from DBSCAN
        mask = df_with_labels["cluster"] == cluster_id
        cluster_data = df_with_labels.loc[mask, feature_cols]

        profile = {"cluster_id": cluster_id, "n_samples": int(mask.sum())}
        for col in feature_cols:
            profile[f"{col}_mean"] = float(cluster_data[col].mean())
            profile[f"{col}_std"] = float(cluster_data[col].std())

        profiles.append(profile)

    return pd.DataFrame(profiles)


def reduce_for_visualization(
    X: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """Reduce to 2D using PCA for visualization.

    Args:
        X: Scaled feature matrix.
        n_components: Number of PCA components.

    Returns:
        Reduced array of shape (n_samples, n_components).
    """
    n_comp = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_comp, random_state=42)
    return pca.fit_transform(X)


def run_clustering(
    result: ExtractionResult,
    feature_cols: list[str] | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run full clustering pipeline on false positives.

    Steps:
    1. Prepare and scale features
    2. Find optimal K via silhouette analysis
    3. Run K-Means clustering
    4. Run DBSCAN for comparison
    5. Compute cluster profiles
    6. Generate 2D projection for visualization
    7. Save results

    Args:
        result: ExtractionResult containing false positives.
        feature_cols: Features to cluster on. If None, auto-detected.
        output_dir: Directory to save outputs.

    Returns:
        Dict with clustering results and metadata.
    """
    save_dir = output_dir or OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    fp = result.false_positives
    if len(fp) < 5:
        logger.warning("Too few false positives (%d) for clustering", len(fp))
        return {"error": "insufficient_samples", "n_fp": len(fp)}

    # Prepare features
    X, feat_names, scaler = prepare_features(fp, feature_cols)

    # Find optimal K
    best_k, sil_scores = find_optimal_k(X)
    logger.info("Optimal K=%d (silhouette scores: %s)", best_k, sil_scores)

    # K-Means
    km_labels, km_model = cluster_kmeans(X, best_k)
    km_sil = float(silhouette_score(X, km_labels)) if len(set(km_labels)) > 1 else -1.0

    # DBSCAN
    db_labels = cluster_dbscan(X)
    n_dbscan_clusters = len(set(db_labels) - {-1})
    n_noise = int((db_labels == -1).sum())
    db_sil = float(silhouette_score(X, db_labels)) if n_dbscan_clusters > 1 else -1.0

    # Cluster profiles (using K-Means labels)
    profiles = compute_cluster_profiles(fp, km_labels, feat_names)

    # 2D projection
    X_2d = reduce_for_visualization(X)

    # Save results
    domain = result.domain
    op_name = result.operating_point.name

    # Cluster assignments
    if "sample_id" in fp.columns:
        assignments = fp[["sample_id"]].copy()
    else:
        assignments = pd.DataFrame(index=range(len(km_labels)))
    assignments["kmeans_cluster"] = km_labels
    assignments["dbscan_cluster"] = db_labels
    assignments["pca_x"] = X_2d[:, 0] if X_2d.ndim > 1 else X_2d
    assignments["pca_y"] = X_2d[:, 1] if X_2d.ndim > 1 and X_2d.shape[1] > 1 else 0.0
    assignments.to_parquet(
        save_dir / f"{domain}_{op_name}_assignments.parquet",
        index=False,
    )

    # Cluster profiles
    profiles.to_csv(
        save_dir / f"{domain}_{op_name}_profiles.csv",
        index=False,
    )

    summary = {
        "domain": domain,
        "operating_point": op_name,
        "n_fp": len(fp),
        "n_features": len(feat_names),
        "kmeans_k": best_k,
        "kmeans_silhouette": km_sil,
        "dbscan_n_clusters": n_dbscan_clusters,
        "dbscan_n_noise": n_noise,
        "dbscan_silhouette": db_sil,
        "silhouette_scores": sil_scores,
    }

    logger.info(
        "%s @ %s: K-Means k=%d (sil=%.3f), DBSCAN clusters=%d noise=%d",
        domain,
        op_name,
        best_k,
        km_sil,
        n_dbscan_clusters,
        n_noise,
    )

    return {
        "assignments": assignments,
        "profiles": profiles,
        "summary": summary,
    }
