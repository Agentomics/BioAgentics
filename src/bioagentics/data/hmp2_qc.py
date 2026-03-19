"""QC and normalization pipeline for HMP2 multi-omics data.

Applies quality control filters and normalization to metagenomic and
metabolomic data for downstream integration in the CD subtyping pipeline.

Steps:
1. Filter low-prevalence species (<10% of samples)
2. Filter low-prevalence metabolites (<20% of samples)
3. CLR (centered log-ratio) normalization for metagenomics (compositional data)
4. Log-transform + median normalization for metabolomics
5. Missing value handling (KNN imputation for metabolomics, zero-handling for metagenomics)
6. Batch effect assessment (PCoA visualization)
7. Metabolite feature selection (variance-based or sPLS-guided)
8. Site batch correction (ComBat)

Usage::

    from bioagentics.data.hmp2_download import HMP2Loader
    from bioagentics.data.hmp2_qc import HMP2QCPipeline

    loader = HMP2Loader()
    species, pathways, metabolomics, metadata = loader.get_aligned_matrices()

    qc = HMP2QCPipeline()
    species_qc = qc.process_metagenomics(species)
    metab_qc = qc.process_metabolomics(metabolomics)

    # With feature selection and batch correction:
    metab_qc = qc.select_metabolite_features(metab_qc, max_features=1000)
    species_qc = qc.batch_correct(species_qc, metadata["site_name"])
    metab_qc = qc.batch_correct(metab_qc, metadata["site_name"])
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.impute import KNNImputer

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "data" / "hmp2" / "qc"


# ── Prevalence Filtering ──


def filter_low_prevalence(
    df: pd.DataFrame,
    min_prevalence: float = 0.1,
    zero_threshold: float = 0.0,
) -> pd.DataFrame:
    """Remove features present in fewer than min_prevalence fraction of samples.

    Parameters
    ----------
    df : DataFrame
        Samples as rows, features as columns.
    min_prevalence : float
        Minimum fraction of samples where feature must be detected (default: 0.1).
    zero_threshold : float
        Values <= this threshold are considered absent (default: 0.0).

    Returns
    -------
    DataFrame with low-prevalence features removed.
    """
    n_samples = len(df)
    prevalence = (df > zero_threshold).sum() / n_samples
    keep = prevalence[prevalence >= min_prevalence].index
    n_removed = len(df.columns) - len(keep)
    logger.info(
        "Prevalence filter (>%.0f%%): %d → %d features (%d removed)",
        min_prevalence * 100,
        len(df.columns),
        len(keep),
        n_removed,
    )
    return df[keep]


# ── CLR Normalization (for compositional metagenomics data) ──


def clr_transform(df: pd.DataFrame, pseudocount: float = 1e-6) -> pd.DataFrame:
    """Apply centered log-ratio (CLR) transformation.

    CLR is the standard transformation for compositional data (metagenomics).
    For each sample: CLR(x_i) = log(x_i / geometric_mean(x))

    Parameters
    ----------
    df : DataFrame
        Non-negative abundance matrix (samples × features).
    pseudocount : float
        Added to zeros before log transform (default: 1e-6).

    Returns
    -------
    CLR-transformed DataFrame. Values sum to approximately 0 per sample.
    """
    mat = df.values.copy().astype(float)

    # Replace zeros with pseudocount
    mat[mat <= 0] = pseudocount

    # Log transform
    log_mat = np.log(mat)

    # Subtract geometric mean (mean of log values) per sample
    geo_mean = log_mat.mean(axis=1, keepdims=True)
    clr_mat = log_mat - geo_mean

    result = pd.DataFrame(clr_mat, index=df.index, columns=df.columns)

    # Verify: CLR values should sum to ~0 per sample
    row_sums = result.sum(axis=1)
    max_deviation = row_sums.abs().max()
    if max_deviation > 1e-6:
        logger.warning("CLR row sums deviate from 0: max=%.6f", max_deviation)
    else:
        logger.info("CLR transform applied: row sums ~0 (max deviation: %.2e)", max_deviation)

    return result


# ── Metabolomics Normalization ──


def log_median_normalize(
    df: pd.DataFrame,
    pseudocount: float = 1.0,
) -> pd.DataFrame:
    """Log-transform and median-normalize metabolomics data.

    Steps:
    1. Add pseudocount to handle zeros
    2. Log2 transform
    3. Median-center each sample (subtract sample median)

    Parameters
    ----------
    df : DataFrame
        Non-negative metabolomics matrix (samples × metabolites).
    pseudocount : float
        Added before log transform (default: 1.0 for metabolomics).

    Returns
    -------
    Log2, median-centered DataFrame.
    """
    mat = df.values.copy().astype(float)

    # Replace NaN with 0 before adding pseudocount
    mat = np.nan_to_num(mat, nan=0.0)

    # Add pseudocount and log2 transform
    log_mat = np.log2(mat + pseudocount)

    # Median-center each sample
    sample_medians = np.nanmedian(log_mat, axis=1, keepdims=True)
    normalized = log_mat - sample_medians

    result = pd.DataFrame(normalized, index=df.index, columns=df.columns)
    logger.info(
        "Log2-median normalization applied: shape %s, value range [%.2f, %.2f]",
        result.shape,
        result.min().min(),
        result.max().max(),
    )
    return result


# ── Missing Value Imputation ──


def impute_knn(
    df: pd.DataFrame,
    n_neighbors: int = 5,
) -> pd.DataFrame:
    """Impute missing values using KNN imputation.

    Appropriate for metabolomics where missing values may be below
    detection limit or randomly missing.

    Parameters
    ----------
    df : DataFrame
        Matrix with potential NaN values.
    n_neighbors : int
        Number of neighbors for KNN imputation (default: 5).

    Returns
    -------
    DataFrame with no NaN values.
    """
    missing_frac = df.isna().mean().mean()
    if missing_frac == 0:
        logger.info("No missing values — skipping KNN imputation")
        return df

    logger.info("KNN imputation (k=%d): %.1f%% missing values", n_neighbors, missing_frac * 100)

    # Drop columns that are entirely NaN
    all_nan_cols = df.columns[df.isna().all()]
    if len(all_nan_cols) > 0:
        logger.warning("Dropping %d all-NaN columns", len(all_nan_cols))
        df = df.drop(columns=all_nan_cols)

    # Drop rows that are entirely NaN
    all_nan_rows = df.index[df.isna().all(axis=1)]
    if len(all_nan_rows) > 0:
        logger.warning("Dropping %d all-NaN rows", len(all_nan_rows))
        df = df.drop(index=all_nan_rows)

    imputer = KNNImputer(n_neighbors=min(n_neighbors, len(df) - 1))
    imputed = imputer.fit_transform(df.values)

    result = pd.DataFrame(imputed, index=df.index, columns=df.columns)
    assert result.isna().sum().sum() == 0, "KNN imputation failed: NaN values remain"
    logger.info("Imputation complete: no NaN values remain")
    return result


def handle_metagenomics_zeros(
    df: pd.DataFrame,
    strategy: str = "pseudocount",
    pseudocount: float = 1e-6,
) -> pd.DataFrame:
    """Handle zeros in metagenomics data.

    Parameters
    ----------
    df : DataFrame
        Metagenomic abundance matrix.
    strategy : str
        "pseudocount" (default) — replace zeros with pseudocount.
        "keep" — leave zeros as-is (handled in CLR transform).
    pseudocount : float
        Value to replace zeros with (default: 1e-6).

    Returns
    -------
    DataFrame with zeros handled.
    """
    n_zeros = (df == 0).sum().sum()
    total = df.size
    logger.info("Metagenomics zeros: %d / %d (%.1f%%)", n_zeros, total, n_zeros / total * 100)

    if strategy == "pseudocount":
        result = df.copy()
        result[result == 0] = pseudocount
        return result
    return df


# ── Metabolite Feature Selection ──


def select_features_by_variance(
    df: pd.DataFrame,
    max_features: int = 1000,
) -> pd.DataFrame:
    """Select top metabolite features by variance.

    Addresses the overcomplete problem (80K metabolites vs 76 samples)
    by keeping only the most variable features.

    Parameters
    ----------
    df : DataFrame
        Normalized metabolomics matrix (samples x metabolites).
    max_features : int
        Maximum number of features to retain (default: 1000).

    Returns
    -------
    DataFrame with top-variance features only.
    """
    if df.shape[1] <= max_features:
        logger.info(
            "Feature selection: %d features <= max %d, keeping all",
            df.shape[1], max_features,
        )
        return df

    variances = df.var(axis=0)
    top_features = variances.nlargest(max_features).index
    logger.info(
        "Variance-based feature selection: %d → %d metabolites (min var=%.4f)",
        df.shape[1], max_features, variances[top_features[-1]],
    )
    return df[top_features]


def select_features_by_spls(
    df: pd.DataFrame,
    spls_pairs_path: Path | str,
    max_features: int = 1000,
) -> pd.DataFrame:
    """Select metabolite features guided by sPLS species-metabolite correlations.

    Uses the top species-metabolite pairs from sPLS-Regression to retain
    metabolites with known cross-omic signal.

    Parameters
    ----------
    df : DataFrame
        Normalized metabolomics matrix (samples x metabolites).
    spls_pairs_path : Path
        Path to spls_top_pairs.csv with columns: species, metabolite, component, score.
    max_features : int
        Maximum number of features to retain.

    Returns
    -------
    DataFrame with sPLS-selected features (falls back to variance if insufficient).
    """
    spls_path = Path(spls_pairs_path)
    if not spls_path.exists():
        logger.warning("sPLS pairs file not found: %s — falling back to variance selection", spls_path)
        return select_features_by_variance(df, max_features)

    pairs = pd.read_csv(spls_path)
    if "metabolite" not in pairs.columns:
        logger.warning("sPLS pairs file missing 'metabolite' column — falling back to variance")
        return select_features_by_variance(df, max_features)

    # Get unique metabolites from sPLS, keeping order by score
    spls_metabolites = pairs.sort_values("score", ascending=False)["metabolite"].unique()
    # Keep only those present in our data
    spls_in_data = [m for m in spls_metabolites if m in df.columns]

    if len(spls_in_data) < 10:
        logger.warning(
            "Only %d sPLS metabolites found in data — falling back to variance",
            len(spls_in_data),
        )
        return select_features_by_variance(df, max_features)

    # If sPLS gives fewer than max_features, supplement with high-variance features
    if len(spls_in_data) < max_features:
        remaining = df.drop(columns=spls_in_data, errors="ignore")
        variances = remaining.var(axis=0)
        n_extra = max_features - len(spls_in_data)
        extra = variances.nlargest(n_extra).index.tolist()
        selected = spls_in_data + extra
    else:
        selected = spls_in_data[:max_features]

    logger.info(
        "sPLS-guided feature selection: %d → %d metabolites (%d from sPLS, %d from variance)",
        df.shape[1], len(selected), len(spls_in_data), len(selected) - len(spls_in_data),
    )
    return df[selected]


# ── Site Batch Correction ──


def combat_correct(
    df: pd.DataFrame,
    batch: pd.Series,
) -> pd.DataFrame:
    """Apply ComBat batch correction.

    Removes batch effects (e.g., collection site) from omics data while
    preserving biological variation.

    Parameters
    ----------
    df : DataFrame
        Normalized omics matrix (samples x features). Must not contain NaN.
    batch : Series
        Batch labels indexed by sample ID (e.g., site_name).
        Must cover all samples in df.

    Returns
    -------
    Batch-corrected DataFrame with same shape as input.
    """
    from combat.pycombat import pycombat

    # Align batch to df index
    common = df.index.intersection(batch.index)
    if len(common) < len(df):
        logger.warning(
            "Batch labels missing for %d / %d samples — using available only",
            len(df) - len(common), len(df),
        )
    if len(common) < 3:
        logger.warning("Too few samples with batch labels (%d) — skipping ComBat", len(common))
        return df

    df_aligned = df.loc[common]
    batch_aligned = batch.loc[common]

    # ComBat needs at least 2 batches
    n_batches = batch_aligned.nunique()
    if n_batches < 2:
        logger.info("Only 1 batch found — skipping ComBat correction")
        return df

    # pycombat expects features x samples (transposed) and integer batch labels
    batch_codes = batch_aligned.astype("category").cat.codes.values

    logger.info(
        "ComBat correction: %d samples, %d features, %d batches",
        len(common), df_aligned.shape[1], n_batches,
    )

    # pycombat(data, batch): data is features x samples
    corrected = pycombat(df_aligned.T, batch_codes)

    result = pd.DataFrame(
        corrected.T,
        index=df_aligned.index,
        columns=df_aligned.columns,
    )

    # If some samples were dropped, add them back uncorrected
    if len(common) < len(df):
        missing = df.index.difference(common)
        result = pd.concat([result, df.loc[missing]])
        result = result.loc[df.index]

    logger.info("ComBat correction complete: shape %s", result.shape)
    return result


# ── Batch Effect Assessment ──


def compute_pcoa(
    df: pd.DataFrame,
    metric: str = "braycurtis",
    n_components: int = 2,
) -> pd.DataFrame:
    """Compute PCoA (principal coordinates analysis) for batch effect assessment.

    Uses PCA on distance matrix as a simple PCoA approximation.

    Parameters
    ----------
    df : DataFrame
        Normalized abundance matrix (samples × features).
    metric : str
        Distance metric for pdist (default: "braycurtis").
    n_components : int
        Number of coordinates to return (default: 2).

    Returns
    -------
    DataFrame with PCoA coordinates (samples × n_components).
    """
    # For CLR-transformed data, use Euclidean (Aitchison distance)
    # For raw compositional data, use Bray-Curtis
    mat = df.values.copy()
    mat = np.nan_to_num(mat, nan=0.0)

    if metric == "braycurtis":
        # Ensure non-negative for Bray-Curtis
        mat = mat - mat.min(axis=1, keepdims=True) if (mat < 0).any() else mat

    dist_matrix = squareform(pdist(mat, metric=metric))

    # Classical MDS via PCA on double-centered distance matrix
    n = len(dist_matrix)
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ (dist_matrix**2) @ H

    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Take top components
    coords = eigvecs[:, :n_components] * np.sqrt(np.maximum(eigvals[:n_components], 0))

    # Variance explained
    total_var = np.sum(np.maximum(eigvals, 0))
    var_explained = eigvals[:n_components] / total_var * 100 if total_var > 0 else np.zeros(n_components)

    result = pd.DataFrame(
        coords,
        index=df.index,
        columns=[f"PCoA{i + 1} ({var_explained[i]:.1f}%)" for i in range(n_components)],
    )
    logger.info(
        "PCoA: %.1f%% variance explained by top %d components",
        sum(var_explained),
        n_components,
    )
    return result


def plot_pcoa(
    pcoa_df: pd.DataFrame,
    groups: pd.Series | None = None,
    title: str = "PCoA — Batch Effect Assessment",
    output_path: Path | None = None,
) -> None:
    """Plot PCoA coordinates colored by group for batch effect visualization.

    Parameters
    ----------
    pcoa_df : DataFrame
        PCoA coordinates (samples × 2).
    groups : Series, optional
        Group labels for coloring (e.g., diagnosis, site).
    title : str
        Plot title.
    output_path : Path, optional
        Save plot to this path if provided.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    if groups is not None:
        # Align groups to pcoa_df index
        common = pcoa_df.index.intersection(groups.index)
        pcoa_plot = pcoa_df.loc[common]
        groups_plot = groups.loc[common]

        for label in groups_plot.unique():
            mask = groups_plot == label
            ax.scatter(
                pcoa_plot.iloc[:, 0][mask],
                pcoa_plot.iloc[:, 1][mask],
                label=str(label),
                alpha=0.6,
                s=40,
            )
        ax.legend(title=groups.name or "Group")
    else:
        ax.scatter(pcoa_df.iloc[:, 0], pcoa_df.iloc[:, 1], alpha=0.6, s=40)

    ax.set_xlabel(pcoa_df.columns[0])
    ax.set_ylabel(pcoa_df.columns[1])
    ax.set_title(title)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info("Saved PCoA plot: %s", output_path)
    plt.close(fig)


# ── QC Pipeline ──


class HMP2QCPipeline:
    """Combined QC and normalization pipeline for HMP2 multi-omics data."""

    def __init__(
        self,
        species_min_prevalence: float = 0.10,
        metabolite_min_prevalence: float = 0.20,
        knn_neighbors: int = 5,
        clr_pseudocount: float = 1e-6,
        log_pseudocount: float = 1.0,
    ) -> None:
        self.species_min_prevalence = species_min_prevalence
        self.metabolite_min_prevalence = metabolite_min_prevalence
        self.knn_neighbors = knn_neighbors
        self.clr_pseudocount = clr_pseudocount
        self.log_pseudocount = log_pseudocount

    def process_metagenomics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full QC pipeline for metagenomic species abundances.

        Steps:
        1. Filter low-prevalence species (<10% of samples)
        2. Handle zeros (pseudocount)
        3. CLR normalization

        Returns CLR-transformed DataFrame.
        """
        logger.info("=== Metagenomics QC Pipeline ===")
        logger.info("Input: %d samples × %d species", *df.shape)

        # 1. Prevalence filter
        df = filter_low_prevalence(df, min_prevalence=self.species_min_prevalence)

        # 2. Handle zeros (done inside CLR, but also replace explicit NaN)
        df = df.fillna(0)

        # 3. CLR transform
        df = clr_transform(df, pseudocount=self.clr_pseudocount)

        logger.info("Output: %d samples × %d species", *df.shape)
        return df

    def process_pathways(self, df: pd.DataFrame) -> pd.DataFrame:
        """QC pipeline for pathway abundances.

        Same as metagenomics: prevalence filter + CLR.
        """
        logger.info("=== Pathway QC Pipeline ===")
        logger.info("Input: %d samples × %d pathways", *df.shape)

        df = filter_low_prevalence(df, min_prevalence=self.species_min_prevalence)
        df = df.fillna(0)
        df = clr_transform(df, pseudocount=self.clr_pseudocount)

        logger.info("Output: %d samples × %d pathways", *df.shape)
        return df

    def process_metabolomics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full QC pipeline for metabolomics data.

        Steps:
        1. Filter low-prevalence metabolites (<20% of samples)
        2. KNN imputation for missing values
        3. Log2 + median normalization

        Returns normalized DataFrame.
        """
        logger.info("=== Metabolomics QC Pipeline ===")
        logger.info("Input: %d samples × %d metabolites", *df.shape)

        # 1. Prevalence filter
        df = filter_low_prevalence(df, min_prevalence=self.metabolite_min_prevalence)

        # 2. KNN imputation
        df = impute_knn(df, n_neighbors=self.knn_neighbors)

        # 3. Log2 + median normalization
        df = log_median_normalize(df, pseudocount=self.log_pseudocount)

        logger.info("Output: %d samples × %d metabolites", *df.shape)
        return df

    def select_metabolite_features(
        self,
        df: pd.DataFrame,
        max_features: int = 1000,
        method: str = "variance",
        spls_pairs_path: Path | str | None = None,
    ) -> pd.DataFrame:
        """Select top metabolite features for integration.

        Parameters
        ----------
        df : DataFrame
            Normalized metabolomics matrix.
        max_features : int
            Maximum features to retain.
        method : str
            "variance" for variance-based, "spls" for sPLS-guided selection.
        spls_pairs_path : Path, optional
            Path to spls_top_pairs.csv (required for method="spls").
        """
        if method == "spls" and spls_pairs_path is not None:
            return select_features_by_spls(df, spls_pairs_path, max_features)
        return select_features_by_variance(df, max_features)

    def batch_correct(
        self,
        df: pd.DataFrame,
        batch: pd.Series,
    ) -> pd.DataFrame:
        """Apply ComBat batch correction to an omic layer.

        Parameters
        ----------
        df : DataFrame
            Normalized omics matrix (samples x features).
        batch : Series
            Batch labels (e.g., site_name) indexed by sample ID.
        """
        return combat_correct(df, batch)

    def process_all(
        self,
        species: pd.DataFrame,
        pathways: pd.DataFrame,
        metabolomics: pd.DataFrame,
        metadata: pd.DataFrame | None = None,
        max_metabolite_features: int = 1000,
        feature_selection_method: str = "variance",
        spls_pairs_path: Path | str | None = None,
        batch_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run full QC pipeline on all omic layers.

        Parameters
        ----------
        metadata : DataFrame, optional
            Clinical metadata with batch info. Required if batch_column set.
        max_metabolite_features : int
            Max metabolite features (default: 1000). Set 0 to disable.
        feature_selection_method : str
            "variance" or "spls" for metabolite selection.
        spls_pairs_path : Path, optional
            Path to spls_top_pairs.csv for sPLS-guided selection.
        batch_column : str, optional
            Column in metadata for batch correction (e.g., "site_name").

        Returns (species_qc, pathways_qc, metabolomics_qc).
        """
        species_qc = self.process_metagenomics(species)
        pathways_qc = self.process_pathways(pathways)
        metabolomics_qc = self.process_metabolomics(metabolomics)

        # Feature selection on metabolomics
        if max_metabolite_features > 0:
            metabolomics_qc = self.select_metabolite_features(
                metabolomics_qc,
                max_features=max_metabolite_features,
                method=feature_selection_method,
                spls_pairs_path=spls_pairs_path,
            )

        # Batch correction
        if batch_column and metadata is not None and batch_column in metadata.columns:
            # Build per-participant batch series
            if "Participant ID" in metadata.columns:
                batch_series = metadata.drop_duplicates(
                    subset=["Participant ID"]
                ).set_index("Participant ID")[batch_column]
            else:
                batch_series = metadata[batch_column]

            species_qc = self.batch_correct(species_qc, batch_series)
            metabolomics_qc = self.batch_correct(metabolomics_qc, batch_series)
            logger.info("Batch correction applied using '%s'", batch_column)

        return species_qc, pathways_qc, metabolomics_qc

    def assess_batch_effects(
        self,
        species_qc: pd.DataFrame,
        metabolomics_qc: pd.DataFrame,
        metadata: pd.DataFrame,
        output_dir: Path | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Compute PCoA for batch effect assessment.

        Returns dict of PCoA coordinate DataFrames.
        """
        output_dir = output_dir or OUTPUT_DIR

        results: dict[str, pd.DataFrame] = {}

        # PCoA on CLR-transformed species (Euclidean = Aitchison distance)
        pcoa_species = compute_pcoa(species_qc, metric="euclidean")
        results["species_pcoa"] = pcoa_species

        # PCoA on normalized metabolomics
        pcoa_metab = compute_pcoa(metabolomics_qc, metric="euclidean")
        results["metabolomics_pcoa"] = pcoa_metab

        # Plot if metadata has diagnosis info
        diag_col = None
        for col in metadata.columns:
            if col.lower() in ("diagnosis", "dx"):
                diag_col = col
                break

        if diag_col:
            # Build participant → diagnosis mapping
            if "Participant ID" in metadata.columns:
                diag_series = metadata.drop_duplicates(subset=["Participant ID"]).set_index(
                    "Participant ID"
                )[diag_col]
            else:
                diag_series = metadata[diag_col]
            diag_series.name = "Diagnosis"

            plot_pcoa(
                pcoa_species,
                groups=diag_series,
                title="Species PCoA (CLR, Aitchison) — by Diagnosis",
                output_path=output_dir / "pcoa_species_diagnosis.png",
            )
            plot_pcoa(
                pcoa_metab,
                groups=diag_series,
                title="Metabolomics PCoA — by Diagnosis",
                output_path=output_dir / "pcoa_metabolomics_diagnosis.png",
            )

        return results


def summarize_qc(
    raw: pd.DataFrame,
    processed: pd.DataFrame,
    name: str = "data",
) -> dict[str, object]:
    """Return QC summary statistics for a single omic layer."""
    stats: dict[str, object] = {
        "name": name,
        "raw_shape": raw.shape,
        "processed_shape": processed.shape,
        "features_removed": raw.shape[1] - processed.shape[1],
        "raw_nan_frac": float(raw.isna().mean().mean()),
        "processed_nan_frac": float(processed.isna().mean().mean()),
        "processed_value_range": (
            float(processed.min().min()),
            float(processed.max().max()),
        ),
    }
    if name == "species" or name == "pathways":
        # CLR: verify row sums ~ 0
        row_sums = processed.sum(axis=1)
        stats["clr_max_row_deviation"] = float(row_sums.abs().max())
    return stats
