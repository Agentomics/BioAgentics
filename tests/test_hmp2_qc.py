"""Tests for HMP2 QC and normalization pipeline."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.data.hmp2_qc import (
    HMP2QCPipeline,
    clr_transform,
    combat_correct,
    compute_pcoa,
    filter_low_prevalence,
    handle_metagenomics_zeros,
    impute_knn,
    log_median_normalize,
    select_features_by_spls,
    select_features_by_variance,
    summarize_qc,
)


@pytest.fixture
def species_df():
    """Synthetic species abundance matrix (10 samples × 8 species)."""
    rng = np.random.default_rng(42)
    n_samples, n_species = 10, 8
    data = rng.random((n_samples, n_species)) * 100
    # Make 2 species low-prevalence (present in <10% of samples = 1 sample)
    data[:, -2] = 0  # species_7: absent in all
    data[:, -1] = 0
    data[0, -1] = 5.0  # species_8: present in 1/10 = 10%
    cols = [f"species_{i}" for i in range(n_species)]
    return pd.DataFrame(data, index=[f"S{i}" for i in range(n_samples)], columns=cols)


@pytest.fixture
def metabolomics_df():
    """Synthetic metabolomics matrix with missing values."""
    rng = np.random.default_rng(42)
    n_samples, n_metab = 10, 12
    data = rng.random((n_samples, n_metab)) * 1000
    # Add some missing values
    data[1, 3] = np.nan
    data[4, 7] = np.nan
    data[6, 2] = np.nan
    # Make 2 metabolites low-prevalence (<20% = 2 samples)
    data[:, -2] = np.nan  # all missing
    data[:, -1] = np.nan
    data[0, -1] = 50.0  # present in 1/10 = 10%
    cols = [f"metab_{i}" for i in range(n_metab)]
    return pd.DataFrame(data, index=[f"S{i}" for i in range(n_samples)], columns=cols)


# ── Prevalence Filter ──


def test_filter_low_prevalence(species_df):
    filtered = filter_low_prevalence(species_df, min_prevalence=0.1)
    # species_6 (all zeros) should be removed
    # species_7 (1/10 = 10%) should be kept at boundary
    assert "species_6" not in filtered.columns
    assert "species_7" in filtered.columns
    assert filtered.shape[0] == species_df.shape[0]  # samples unchanged


def test_filter_low_prevalence_strict(species_df):
    filtered = filter_low_prevalence(species_df, min_prevalence=0.5)
    # Only species present in >=50% of samples should remain
    assert len(filtered.columns) <= len(species_df.columns)
    for col in filtered.columns:
        assert (species_df[col] > 0).mean() >= 0.5


# ── CLR Transform ──


def test_clr_row_sums_near_zero(species_df):
    # Remove zero-only columns first
    df = species_df.iloc[:, :6]  # Use only non-zero species
    clr = clr_transform(df)
    row_sums = clr.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0, atol=1e-10)


def test_clr_shape_preserved(species_df):
    clr = clr_transform(species_df)
    assert clr.shape == species_df.shape


def test_clr_no_nan(species_df):
    clr = clr_transform(species_df)
    assert clr.isna().sum().sum() == 0


# ── Log-Median Normalization ──


def test_log_median_normalize_shape(metabolomics_df):
    # Remove columns with all NaN for this test
    df = metabolomics_df.iloc[:, :10]
    normalized = log_median_normalize(df)
    assert normalized.shape == df.shape


def test_log_median_normalize_no_nan(metabolomics_df):
    df = metabolomics_df.iloc[:, :10].fillna(0)
    normalized = log_median_normalize(df)
    assert normalized.isna().sum().sum() == 0


# ── KNN Imputation ──


def test_impute_knn_removes_nan(metabolomics_df):
    # Use subset with some NaN but not all-NaN columns
    df = metabolomics_df.iloc[:, :10]
    imputed = impute_knn(df, n_neighbors=3)
    assert imputed.isna().sum().sum() == 0


def test_impute_knn_no_missing():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
    imputed = impute_knn(df)
    pd.testing.assert_frame_equal(imputed, df)


# ── Handle Metagenomics Zeros ──


def test_handle_zeros_pseudocount():
    df = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [3.0, 0.0, 4.0]})
    result = handle_metagenomics_zeros(df, strategy="pseudocount", pseudocount=1e-6)
    assert (result > 0).all().all()


def test_handle_zeros_keep():
    df = pd.DataFrame({"a": [0.0, 1.0], "b": [3.0, 0.0]})
    result = handle_metagenomics_zeros(df, strategy="keep")
    pd.testing.assert_frame_equal(result, df)


# ── PCoA ──


def test_compute_pcoa_shape(species_df):
    df = species_df.iloc[:, :6]
    pcoa = compute_pcoa(df, metric="euclidean", n_components=2)
    assert pcoa.shape == (len(df), 2)
    assert all("PCoA" in col for col in pcoa.columns)


# ── Full Pipeline ──


def test_pipeline_metagenomics(species_df):
    qc = HMP2QCPipeline()
    result = qc.process_metagenomics(species_df)
    assert result.isna().sum().sum() == 0
    # CLR: row sums ~ 0
    row_sums = result.sum(axis=1)
    np.testing.assert_allclose(row_sums, 0, atol=1e-10)


def test_pipeline_metabolomics(metabolomics_df):
    qc = HMP2QCPipeline()
    result = qc.process_metabolomics(metabolomics_df)
    assert result.isna().sum().sum() == 0
    # Low-prevalence metabolites should be removed
    assert result.shape[1] <= metabolomics_df.shape[1]


def test_pipeline_process_all(species_df, metabolomics_df):
    qc = HMP2QCPipeline()
    pathways = species_df.copy()  # Use species as stand-in for pathways
    s_qc, p_qc, m_qc = qc.process_all(species_df, pathways, metabolomics_df)
    assert s_qc.isna().sum().sum() == 0
    assert p_qc.isna().sum().sum() == 0
    assert m_qc.isna().sum().sum() == 0


# ── Summarize QC ──


def test_summarize_qc(species_df):
    qc = HMP2QCPipeline()
    processed = qc.process_metagenomics(species_df)
    stats = summarize_qc(species_df, processed, name="species")
    assert stats["name"] == "species"
    assert stats["processed_nan_frac"] == 0.0
    assert "clr_max_row_deviation" in stats


# ── Metabolite Feature Selection ──


@pytest.fixture
def large_metab_df():
    """Metabolomics matrix with many features to test selection."""
    rng = np.random.default_rng(42)
    n_samples, n_metab = 20, 200
    data = rng.random((n_samples, n_metab)) * 1000
    # Make some features high-variance
    data[:, :10] *= 100
    cols = [f"mb_{i}" for i in range(n_metab)]
    return pd.DataFrame(data, index=[f"S{i}" for i in range(n_samples)], columns=cols)


def test_variance_selection_reduces_features(large_metab_df):
    result = select_features_by_variance(large_metab_df, max_features=50)
    assert result.shape[1] == 50
    assert result.shape[0] == large_metab_df.shape[0]


def test_variance_selection_noop_when_small():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    result = select_features_by_variance(df, max_features=10)
    assert result.shape[1] == 2


def test_variance_selection_keeps_high_variance(large_metab_df):
    result = select_features_by_variance(large_metab_df, max_features=15)
    # High-variance features (mb_0 to mb_9) should be in the result
    for i in range(10):
        assert f"mb_{i}" in result.columns


def test_spls_selection_with_valid_file(large_metab_df, tmp_path):
    # Create a mock sPLS pairs file with enough metabolites (>10 threshold)
    metab_names = [f"mb_{i}" for i in range(50, 65)]
    pairs = pd.DataFrame({
        "species": [f"sp_{i}" for i in range(15)],
        "metabolite": metab_names,
        "component": ["PLS_1"] * 8 + ["PLS_2"] * 7,
        "score": [0.9 - i * 0.05 for i in range(15)],
    })
    pairs_path = tmp_path / "spls_top_pairs.csv"
    pairs.to_csv(pairs_path, index=False)

    result = select_features_by_spls(large_metab_df, pairs_path, max_features=20)
    assert result.shape[1] == 20
    # sPLS metabolites should be present
    assert "mb_50" in result.columns
    assert "mb_55" in result.columns
    assert "mb_60" in result.columns


def test_spls_selection_falls_back_on_missing_file(large_metab_df):
    result = select_features_by_spls(large_metab_df, "/nonexistent/path.csv", max_features=50)
    assert result.shape[1] == 50


# ── ComBat Batch Correction ──


def test_combat_correct_basic(species_df):
    batch = pd.Series(
        ["site_A"] * 5 + ["site_B"] * 5,
        index=species_df.index,
        name="site",
    )
    result = combat_correct(species_df.iloc[:, :6], batch)
    assert result.shape == species_df.iloc[:, :6].shape
    assert result.isna().sum().sum() == 0


def test_combat_correct_single_batch(species_df):
    batch = pd.Series(
        ["site_A"] * 10,
        index=species_df.index,
        name="site",
    )
    result = combat_correct(species_df.iloc[:, :6], batch)
    # Should skip correction and return original
    pd.testing.assert_frame_equal(result, species_df.iloc[:, :6])


def test_combat_correct_partial_labels(species_df):
    # Only provide batch for half the samples
    batch = pd.Series(
        ["site_A"] * 3 + ["site_B"] * 3,
        index=species_df.index[:6],
        name="site",
    )
    result = combat_correct(species_df.iloc[:, :6], batch)
    assert result.shape == species_df.iloc[:, :6].shape


# ── Pipeline with Feature Selection and Batch Correction ──


def test_pipeline_process_all_with_feature_selection(species_df, metabolomics_df):
    qc = HMP2QCPipeline()
    pathways = species_df.copy()
    s_qc, p_qc, m_qc = qc.process_all(
        species_df, pathways, metabolomics_df,
        max_metabolite_features=5,
    )
    assert m_qc.shape[1] <= 5
    assert s_qc.isna().sum().sum() == 0


def test_pipeline_process_all_with_batch_correction(species_df, metabolomics_df):
    qc = HMP2QCPipeline()
    pathways = species_df.copy()
    metadata = pd.DataFrame({
        "Participant ID": species_df.index,
        "site_name": ["site_A"] * 5 + ["site_B"] * 5,
    })
    s_qc, p_qc, m_qc = qc.process_all(
        species_df, pathways, metabolomics_df,
        metadata=metadata,
        batch_column="site_name",
    )
    assert s_qc.isna().sum().sum() == 0
    assert m_qc.isna().sum().sum() == 0
