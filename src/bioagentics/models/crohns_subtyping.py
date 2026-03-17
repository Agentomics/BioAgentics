"""Consensus clustering and subtype discovery on integrated latent factors.

Takes factor matrices from CMTF and/or MOFA2 and discovers robust
CD subtypes using consensus clustering with multiple algorithms.

Methods:
- Consensus clustering (k=2 to 6) with k-means, hierarchical, spectral
- Optimal k via silhouette analysis and gap statistic
- Adjusted Rand Index for comparing CMTF vs MOFA2 cluster assignments
- PERMANOVA for verifying clusters aren't driven by technical confounders

Success criterion: silhouette score > 0.3 for at least 2 subtypes.

Usage::

    from bioagentics.models.crohns_subtyping import ConsensusSubtyping

    subtyping = ConsensusSubtyping()
    results = subtyping.fit(cmtf_factors)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "microbiome-metabolome-subtyping"


# ── Consensus Clustering ──


def consensus_matrix(
    X: np.ndarray,
    k: int,
    n_iterations: int = 100,
    subsample_fraction: float = 0.8,
    random_state: int = 42,
) -> np.ndarray:
    """Compute consensus matrix via repeated subsampled k-means.

    Parameters
    ----------
    X : ndarray
        Feature matrix (samples × features).
    k : int
        Number of clusters.
    n_iterations : int
        Number of subsampling iterations.
    subsample_fraction : float
        Fraction of samples to use per iteration.

    Returns
    -------
    Consensus matrix (n_samples × n_samples): fraction of times pairs
    co-clustered.
    """
    n = X.shape[0]
    rng = np.random.default_rng(random_state)
    cooccurrence = np.zeros((n, n))
    cosampled = np.zeros((n, n))

    subsample_size = max(int(n * subsample_fraction), k + 1)

    for it in range(n_iterations):
        idx = rng.choice(n, size=subsample_size, replace=False)
        km = KMeans(n_clusters=k, n_init=5, random_state=random_state + it)
        labels = km.fit_predict(X[idx])

        for i in range(len(idx)):
            for j in range(i + 1, len(idx)):
                cosampled[idx[i], idx[j]] += 1
                cosampled[idx[j], idx[i]] += 1
                if labels[i] == labels[j]:
                    cooccurrence[idx[i], idx[j]] += 1
                    cooccurrence[idx[j], idx[i]] += 1

    # Normalize
    mask = cosampled > 0
    result = np.zeros_like(cooccurrence)
    result[mask] = cooccurrence[mask] / cosampled[mask]
    np.fill_diagonal(result, 1.0)

    return result


def multi_algorithm_clustering(
    X: np.ndarray,
    k: int,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Cluster with multiple algorithms for consensus.

    Returns dict mapping algorithm name to cluster labels.
    """
    results: dict[str, np.ndarray] = {}

    # K-means
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    results["kmeans"] = km.fit_predict(X)

    # Hierarchical (Ward)
    hc = AgglomerativeClustering(n_clusters=k, linkage="ward")
    results["hierarchical"] = hc.fit_predict(X)

    # Spectral
    if X.shape[0] > k + 1:
        try:
            sc = SpectralClustering(
                n_clusters=k, random_state=random_state, n_init=5
            )
            results["spectral"] = sc.fit_predict(X)
        except Exception:
            logger.warning("Spectral clustering failed for k=%d", k)

    return results


def gap_statistic(
    X: np.ndarray,
    k_range: range,
    n_references: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute gap statistic for optimal k selection.

    Gap(k) = E*[log(W_k)] - log(W_k) where E* is over reference
    distributions.

    Returns DataFrame with k, gap, gap_se.
    """
    rng = np.random.default_rng(random_state)

    results = []
    for k in k_range:
        # Observed within-cluster dispersion
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X)
        wk = _within_cluster_dispersion(X, km.labels_)
        log_wk = np.log(max(wk, 1e-10))

        # Reference distributions (uniform in bounding box)
        ref_log_wks = []
        for b in range(n_references):
            X_ref = rng.uniform(X.min(axis=0), X.max(axis=0), size=X.shape)
            km_ref = KMeans(n_clusters=k, n_init=5, random_state=random_state + b)
            km_ref.fit(X_ref)
            wk_ref = _within_cluster_dispersion(X_ref, km_ref.labels_)
            ref_log_wks.append(np.log(max(wk_ref, 1e-10)))

        gap = np.mean(ref_log_wks) - log_wk
        gap_se = np.std(ref_log_wks) * np.sqrt(1 + 1 / n_references)

        results.append({"k": k, "gap": float(gap), "gap_se": float(gap_se)})

    return pd.DataFrame(results)


def _within_cluster_dispersion(X: np.ndarray, labels: np.ndarray) -> float:
    """Compute total within-cluster sum of squared distances."""
    wk = 0.0
    for c in np.unique(labels):
        members = X[labels == c]
        center = members.mean(axis=0)
        wk += np.sum((members - center) ** 2)
    return wk


# ── PERMANOVA ──


def permanova_test(
    X: np.ndarray,
    labels: np.ndarray,
    covariate: np.ndarray | pd.Series,
    n_permutations: int = 999,
    random_state: int = 42,
) -> dict[str, float]:
    """PERMANOVA to check if clusters are driven by a covariate.

    Tests whether a technical covariate (e.g., site, batch) explains
    the cluster structure. Low p-value = clusters may be confounded.

    Returns dict with F_statistic and p_value.
    """
    rng = np.random.default_rng(random_state)

    if isinstance(covariate, pd.Series):
        covariate = covariate.values

    dist = squareform(pdist(X, metric="euclidean"))

    # Observed F-statistic
    f_obs = _pseudo_f(dist, covariate)

    # Permutation test
    n_greater = 0
    for _ in range(n_permutations):
        perm_cov = rng.permutation(covariate)
        f_perm = _pseudo_f(dist, perm_cov)
        if f_perm >= f_obs:
            n_greater += 1

    p_value = (n_greater + 1) / (n_permutations + 1)

    return {"F_statistic": float(f_obs), "p_value": float(p_value)}


def _pseudo_f(dist_matrix: np.ndarray, groups: np.ndarray) -> float:
    """Compute pseudo-F statistic for PERMANOVA."""
    n = len(groups)
    unique_groups = np.unique(groups)
    k = len(unique_groups)

    if k < 2 or k >= n:
        return 0.0

    # Total sum of squared distances
    ss_total = np.sum(dist_matrix**2) / (2 * n)

    # Within-group sum of squared distances
    ss_within = 0.0
    for g in unique_groups:
        mask = groups == g
        n_g = np.sum(mask)
        if n_g > 1:
            sub_dist = dist_matrix[np.ix_(mask, mask)]
            ss_within += np.sum(sub_dist**2) / (2 * n_g)

    ss_between = ss_total - ss_within

    # Pseudo-F
    f_stat = (ss_between / (k - 1)) / (ss_within / (n - k)) if ss_within > 0 else 0.0
    return f_stat


# ── Subtyping Pipeline ──


class ConsensusSubtyping:
    """Consensus clustering pipeline for CD subtype discovery."""

    def __init__(
        self,
        k_range: range = range(2, 7),
        n_consensus_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        self.k_range = k_range
        self.n_consensus_iter = n_consensus_iter
        self.random_state = random_state

    def fit(
        self,
        factors: pd.DataFrame,
        output_dir: Path | None = None,
    ) -> dict:
        """Discover subtypes from integrated latent factors.

        Parameters
        ----------
        factors : DataFrame
            Factor matrix (participants × components) from CMTF or MOFA2.

        Returns
        -------
        dict with:
            - "optimal_k": int
            - "labels": Series (participant → subtype)
            - "silhouette_scores": dict[int, float]
            - "consensus_matrices": dict[int, ndarray]
            - "gap_statistic": DataFrame
            - "multi_algorithm_labels": dict[int, dict[str, ndarray]]
        """
        output_dir = output_dir or OUTPUT_DIR
        X = factors.values.astype(float)
        n_samples = X.shape[0]

        logger.info(
            "Consensus subtyping: %d participants, %d factors, k=%d-%d",
            n_samples, X.shape[1], self.k_range.start, self.k_range.stop - 1,
        )

        silhouette_scores: dict[int, float] = {}
        consensus_matrices: dict[int, np.ndarray] = {}
        all_labels: dict[int, dict[str, np.ndarray]] = {}
        best_labels: dict[int, np.ndarray] = {}

        for k in self.k_range:
            if k >= n_samples:
                continue

            # Consensus matrix
            cm = consensus_matrix(
                X, k,
                n_iterations=self.n_consensus_iter,
                random_state=self.random_state,
            )
            consensus_matrices[k] = cm

            # Multi-algorithm clustering
            algo_labels = multi_algorithm_clustering(
                X, k, random_state=self.random_state
            )
            all_labels[k] = algo_labels

            # Use k-means labels as primary
            labels = algo_labels["kmeans"]
            best_labels[k] = labels

            # Silhouette score
            if len(np.unique(labels)) > 1:
                sil = silhouette_score(X, labels)
                silhouette_scores[k] = float(sil)
                logger.info("k=%d: silhouette=%.3f", k, sil)
            else:
                silhouette_scores[k] = -1.0

        # Gap statistic
        gap_df = gap_statistic(
            X, self.k_range,
            n_references=20,
            random_state=self.random_state,
        )

        # Select optimal k
        optimal_k = self._select_optimal_k(silhouette_scores, gap_df)

        # Final labels
        final_labels = pd.Series(
            best_labels[optimal_k],
            index=factors.index,
            name="subtype",
        )

        results = {
            "optimal_k": optimal_k,
            "labels": final_labels,
            "silhouette_scores": silhouette_scores,
            "consensus_matrices": consensus_matrices,
            "gap_statistic": gap_df,
            "multi_algorithm_labels": all_labels,
        }

        # Check success criterion
        best_sil = silhouette_scores.get(optimal_k, -1)
        if best_sil > 0.3:
            logger.info(
                "SUCCESS: %d subtypes with silhouette=%.3f (>0.3)",
                optimal_k, best_sil,
            )
        else:
            logger.warning(
                "Below threshold: %d subtypes with silhouette=%.3f (<0.3)",
                optimal_k, best_sil,
            )

        # Save outputs
        self._save_results(results, output_dir)

        return results

    def _select_optimal_k(
        self,
        silhouette_scores: dict[int, float],
        gap_df: pd.DataFrame,
    ) -> int:
        """Select optimal k using silhouette + gap statistic.

        Primary: highest silhouette. Tie-break: gap statistic.
        """
        if not silhouette_scores:
            return 2

        # Best silhouette
        best_k = max(silhouette_scores, key=lambda k: silhouette_scores[k])

        # Check gap statistic 1-SE rule
        if len(gap_df) > 0:
            for _, row in gap_df.iterrows():
                k = int(row["k"])
                if k >= gap_df["k"].max():
                    break
                next_row = gap_df[gap_df["k"] == k + 1]
                if len(next_row) > 0:
                    gap_k = row["gap"]
                    gap_k1 = next_row.iloc[0]["gap"]
                    se_k1 = next_row.iloc[0]["gap_se"]
                    if gap_k >= gap_k1 - se_k1:
                        # k satisfies 1-SE rule
                        if silhouette_scores.get(k, -1) > 0.3:
                            best_k = k
                            break

        return best_k

    def compare_methods(
        self,
        cmtf_factors: pd.DataFrame,
        mofa2_factors: pd.DataFrame,
    ) -> dict[str, float | pd.DataFrame]:
        """Compare CMTF vs MOFA2 cluster assignments.

        Runs subtyping on both factor matrices and computes adjusted
        Rand index for each k.

        Returns dict with ARI scores and per-method labels.
        """
        shared = cmtf_factors.index.intersection(mofa2_factors.index)
        cmtf = cmtf_factors.loc[shared]
        mofa = mofa2_factors.loc[shared]

        X_cmtf = cmtf.values.astype(float)
        X_mofa = mofa.values.astype(float)

        ari_scores: dict[int, float] = {}
        for k in self.k_range:
            if k >= len(shared):
                continue
            km_cmtf = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)
            km_mofa = KMeans(n_clusters=k, n_init=10, random_state=self.random_state)

            labels_cmtf = km_cmtf.fit_predict(X_cmtf)
            labels_mofa = km_mofa.fit_predict(X_mofa)

            ari = adjusted_rand_score(labels_cmtf, labels_mofa)
            ari_scores[k] = float(ari)
            logger.info("k=%d: CMTF vs MOFA2 ARI=%.3f", k, ari)

        return {
            "ari_scores": ari_scores,
            "mean_ari": float(np.mean(list(ari_scores.values()))) if ari_scores else 0.0,
        }

    def _save_results(self, results: dict, output_dir: Path) -> None:
        """Save subtyping results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Labels
        results["labels"].to_csv(output_dir / "subtype_labels.csv")

        # Silhouette scores
        sil_df = pd.DataFrame(
            list(results["silhouette_scores"].items()),
            columns=["k", "silhouette"],
        )
        sil_df.to_csv(output_dir / "silhouette_scores.csv", index=False)

        # Gap statistic
        results["gap_statistic"].to_csv(
            output_dir / "gap_statistic.csv", index=False
        )

        logger.info(
            "Saved subtyping results (optimal_k=%d) to %s",
            results["optimal_k"], output_dir,
        )

    def plot_consensus_heatmap(
        self,
        consensus_mat: np.ndarray,
        labels: np.ndarray,
        k: int,
        output_path: Path | None = None,
    ) -> None:
        """Plot consensus matrix heatmap ordered by cluster."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Reorder by cluster
        order = np.argsort(labels)
        cm_ordered = consensus_mat[np.ix_(order, order)]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(cm_ordered, cmap="RdBu_r", vmin=0, vmax=1)
        ax.set_title(f"Consensus Matrix (k={k})")
        fig.colorbar(im, label="Co-clustering frequency")

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
