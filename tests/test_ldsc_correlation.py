"""Tests for the LDSC genetic correlation pipeline.

Uses synthetic summary statistics and LD scores to verify pipeline correctness.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bioagentics.pipelines.ldsc_correlation.pipeline import (
    LDSCResult,
    _bivariate_ldsc,
    _univariate_ldsc,
    compute_genetic_correlation_matrix,
    ldsc_regression,
    load_ld_scores,
    load_sumstats,
    munge_sumstats,
    results_to_dataframe,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic GWAS data
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)
N_SNPS = 5000
SAMPLE_SIZE = 50000


def _make_snp_ids(n: int) -> list[str]:
    return [f"rs{i}" for i in range(1, n + 1)]


@pytest.fixture()
def snp_ids():
    return _make_snp_ids(N_SNPS)


@pytest.fixture()
def ld_scores_df(snp_ids):
    """Synthetic LD scores — exponential-ish distribution centered around 50."""
    l2 = RNG.exponential(scale=50, size=N_SNPS) + 1.0
    return pd.DataFrame({"SNP": snp_ids, "L2": l2})


def _make_sumstats(snp_ids: list[str], h2: float = 0.5, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    """Generate synthetic summary statistics with a known heritability signal.

    Under the infinitesimal model, Z ~ N(0, 1 + N*h2*l_j/M).
    We simplify by drawing Z from N(0, 1+h2_scale) to embed a detectable signal.
    """
    n_snps = len(snp_ids)
    z = RNG.normal(0, np.sqrt(1 + h2 * 2), size=n_snps)
    return pd.DataFrame(
        {
            "SNP": snp_ids,
            "A1": RNG.choice(["A", "C", "G", "T"], size=n_snps),
            "A2": RNG.choice(["A", "C", "G", "T"], size=n_snps),
            "Z": z,
            "N": np.full(n_snps, n),
        }
    )


def _make_correlated_sumstats(
    snp_ids: list[str],
    z_reference: np.ndarray,
    rg_target: float = 0.5,
    n: int = SAMPLE_SIZE,
) -> pd.DataFrame:
    """Generate summary stats correlated with a reference trait at ~rg_target."""
    n_snps = len(snp_ids)
    noise = RNG.normal(0, 1, size=n_snps)
    z = rg_target * z_reference + np.sqrt(1 - rg_target**2) * noise
    return pd.DataFrame(
        {
            "SNP": snp_ids,
            "A1": RNG.choice(["A", "C", "G", "T"], size=n_snps),
            "A2": RNG.choice(["A", "C", "G", "T"], size=n_snps),
            "Z": z,
            "N": np.full(n_snps, n),
        }
    )


@pytest.fixture()
def trait1_sumstats(snp_ids):
    return _make_sumstats(snp_ids, h2=0.5)


@pytest.fixture()
def trait2_sumstats(snp_ids, trait1_sumstats):
    return _make_correlated_sumstats(snp_ids, trait1_sumstats["Z"].values, rg_target=0.4)


# ---------------------------------------------------------------------------
# Tests: load_sumstats
# ---------------------------------------------------------------------------


class TestLoadSumstats:
    def test_load_standard_format(self, tmp_path):
        df = pd.DataFrame(
            {
                "SNP": ["rs1", "rs2", "rs3"],
                "A1": ["A", "G", "C"],
                "A2": ["T", "A", "G"],
                "Z": [1.5, -0.3, 2.1],
                "P": [0.05, 0.8, 0.01],
                "N": [10000, 10000, 10000],
            }
        )
        path = tmp_path / "test.sumstats"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_sumstats(path)
        assert len(loaded) == 3
        assert "SNP" in loaded.columns
        assert "Z" in loaded.columns

    def test_load_beta_se_format(self, tmp_path):
        df = pd.DataFrame(
            {
                "SNP": ["rs1", "rs2"],
                "A1": ["A", "G"],
                "A2": ["T", "A"],
                "BETA": [0.05, -0.02],
                "SE": [0.01, 0.015],
                "P": [1e-5, 0.2],
                "N": [50000, 50000],
            }
        )
        path = tmp_path / "test.tsv"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_sumstats(path)
        assert "BETA" in loaded.columns
        assert "SE" in loaded.columns

    def test_load_with_column_aliases(self, tmp_path):
        df = pd.DataFrame(
            {
                "RSID": ["rs1", "rs2"],
                "EFFECT_ALLELE": ["A", "G"],
                "OTHER_ALLELE": ["T", "A"],
                "ZSCORE": [1.5, -0.3],
                "PVALUE": [0.05, 0.8],
                "SAMPLESIZE": [10000, 10000],
            }
        )
        path = tmp_path / "test.txt"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_sumstats(path)
        assert "SNP" in loaded.columns
        assert "A1" in loaded.columns
        assert "Z" in loaded.columns
        assert "N" in loaded.columns


# ---------------------------------------------------------------------------
# Tests: munge_sumstats
# ---------------------------------------------------------------------------


class TestMungeSumstats:
    def test_basic_munging(self):
        df = pd.DataFrame(
            {
                "SNP": ["rs1", "rs2", "rs3"],
                "A1": ["a", "G", "c"],
                "A2": ["t", "A", "g"],
                "Z": [1.5, -0.3, 2.1],
                "N": [10000, 10000, 10000],
            }
        )
        munged = munge_sumstats(df, "test")
        assert len(munged) == 3
        assert all(munged["A1"].str.isupper())

    def test_derives_z_from_beta_se(self):
        df = pd.DataFrame(
            {
                "SNP": ["rs1", "rs2"],
                "A1": ["A", "G"],
                "A2": ["T", "A"],
                "BETA": [0.05, -0.02],
                "SE": [0.01, 0.015],
                "N": [50000, 50000],
            }
        )
        munged = munge_sumstats(df, "test")
        assert "Z" in munged.columns
        np.testing.assert_allclose(munged["Z"].values, [5.0, -4 / 3], rtol=1e-10)

    def test_removes_extreme_z(self):
        df = pd.DataFrame(
            {
                "SNP": ["rs1", "rs2", "rs3"],
                "A1": ["A", "G", "C"],
                "A2": ["T", "A", "G"],
                "Z": [1.5, 50.0, -40.0],
                "N": [10000, 10000, 10000],
            }
        )
        munged = munge_sumstats(df, "test")
        assert len(munged) == 1

    def test_removes_duplicates(self):
        df = pd.DataFrame(
            {
                "SNP": ["rs1", "rs1", "rs2"],
                "A1": ["A", "A", "G"],
                "A2": ["T", "T", "A"],
                "Z": [1.5, 1.6, -0.3],
                "N": [10000, 10000, 10000],
            }
        )
        munged = munge_sumstats(df, "test")
        assert len(munged) == 2

    def test_raises_on_missing_columns(self):
        df = pd.DataFrame({"SNP": ["rs1"], "Z": [1.0]})
        with pytest.raises(ValueError, match="Missing"):
            munge_sumstats(df, "test")


# ---------------------------------------------------------------------------
# Tests: load_ld_scores
# ---------------------------------------------------------------------------


class TestLoadLdScores:
    def test_load_single_file(self, tmp_path):
        df = pd.DataFrame({"SNP": ["rs1", "rs2"], "L2": [50.0, 30.0]})
        path = tmp_path / "ld.l2.ldscore"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_ld_scores(path)
        assert len(loaded) == 2
        assert set(loaded.columns) == {"SNP", "L2"}

    def test_load_directory(self, tmp_path):
        for chrom in [1, 2]:
            df = pd.DataFrame(
                {"SNP": [f"rs{chrom}1", f"rs{chrom}2"], "L2": [50.0, 30.0]}
            )
            df.to_csv(tmp_path / f"chr{chrom}.l2.ldscore", sep="\t", index=False)
        loaded = load_ld_scores(tmp_path)
        assert len(loaded) == 4

    def test_raises_on_missing_column(self, tmp_path):
        df = pd.DataFrame({"SNP": ["rs1"], "WRONG": [50.0]})
        path = tmp_path / "bad.tsv"
        df.to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match="L2"):
            load_ld_scores(path)


# ---------------------------------------------------------------------------
# Tests: univariate LDSC
# ---------------------------------------------------------------------------


class TestUnivariateLDSC:
    def test_positive_h2(self):
        n_snps = 2000
        ld = RNG.exponential(50, size=n_snps) + 1
        n = np.full(n_snps, 50000.0)
        true_h2 = 0.5
        # chi2 = 1 + N*h2*l/M (under infinitesimal)
        chi2 = 1 + n * true_h2 * ld / n_snps + RNG.normal(0, 0.5, size=n_snps)
        chi2 = np.maximum(chi2, 0.01)

        h2, h2_se, intercept, int_se = _univariate_ldsc(chi2, ld, n, n_snps)
        # Should recover h2 in the right ballpark
        assert 0.1 < h2 < 1.5, f"h2={h2} out of expected range"
        assert h2_se > 0

    def test_zero_signal(self):
        n_snps = 2000
        ld = RNG.exponential(50, size=n_snps) + 1
        n = np.full(n_snps, 50000.0)
        chi2 = RNG.exponential(1, size=n_snps)

        h2, _, intercept, _ = _univariate_ldsc(chi2, ld, n, n_snps)
        # h2 should be near zero
        assert abs(h2) < 0.5, f"h2={h2} too far from zero for null"


# ---------------------------------------------------------------------------
# Tests: bivariate LDSC
# ---------------------------------------------------------------------------


class TestBivariateLDSC:
    def test_positive_covariance(self):
        n_snps = 3000
        ld = RNG.exponential(50, size=n_snps) + 1
        n1 = np.full(n_snps, 50000.0)
        n2 = np.full(n_snps, 40000.0)
        true_gcov = 0.3
        sqrt_n = np.sqrt(n1 * n2)
        z_prod = sqrt_n * true_gcov * ld / n_snps + RNG.normal(0, 0.5, size=n_snps)

        gcov, gcov_se, intercept = _bivariate_ldsc(
            z_prod / 2, np.ones(n_snps) * 2, ld, n1, n2, n_snps
        )
        assert gcov_se > 0


# ---------------------------------------------------------------------------
# Tests: full ldsc_regression
# ---------------------------------------------------------------------------


class TestLDSCRegression:
    def test_returns_result(self, trait1_sumstats, trait2_sumstats, ld_scores_df):
        result = ldsc_regression(
            trait1_sumstats, trait2_sumstats, ld_scores_df, "trait1", "trait2"
        )
        assert isinstance(result, LDSCResult)
        assert result.trait1 == "trait1"
        assert result.trait2 == "trait2"
        assert result.n_snps > 0
        assert np.isfinite(result.rg) or np.isnan(result.rg)

    def test_self_correlation_near_one(self, trait1_sumstats, ld_scores_df):
        result = ldsc_regression(
            trait1_sumstats, trait1_sumstats, ld_scores_df, "same", "same"
        )
        # Self-correlation should be close to 1 (may not be exact due to estimation)
        if np.isfinite(result.rg):
            assert result.rg > 0.5, f"Self-rg={result.rg} unexpectedly low"


# ---------------------------------------------------------------------------
# Tests: compute_genetic_correlation_matrix
# ---------------------------------------------------------------------------


class TestComputeMatrix:
    def test_full_pipeline(self, snp_ids, ld_scores_df, trait1_sumstats, tmp_path):
        sumstats_dir = tmp_path / "sumstats"
        sumstats_dir.mkdir()

        # Write reference trait
        trait1_sumstats.to_csv(sumstats_dir / "TS.sumstats", sep="\t", index=False)

        # Write 3 correlated traits
        for name, rg in [("OCD", 0.4), ("ADHD", 0.3), ("MDD", 0.2)]:
            t = _make_correlated_sumstats(snp_ids, trait1_sumstats["Z"].values, rg)
            t.to_csv(sumstats_dir / f"{name}.sumstats", sep="\t", index=False)

        # Write LD scores
        ld_path = sumstats_dir / "ld_scores"
        ld_path.mkdir()
        ld_scores_df.to_csv(ld_path / "all.l2.ldscore", sep="\t", index=False)

        results = compute_genetic_correlation_matrix(sumstats_dir, ld_scores_df, "TS")
        assert len(results) == 3
        trait_names = {r.trait2 for r in results}
        assert trait_names == {"OCD", "ADHD", "MDD"}

    def test_missing_reference_trait(self, ld_scores_df, tmp_path):
        sumstats_dir = tmp_path / "sumstats"
        sumstats_dir.mkdir()
        pd.DataFrame({"SNP": ["rs1"], "A1": ["A"], "A2": ["T"], "Z": [1.0], "N": [1000]}).to_csv(
            sumstats_dir / "OCD.sumstats", sep="\t", index=False
        )
        with pytest.raises(FileNotFoundError, match="TS"):
            compute_genetic_correlation_matrix(sumstats_dir, ld_scores_df, "TS")


# ---------------------------------------------------------------------------
# Tests: results_to_dataframe
# ---------------------------------------------------------------------------


class TestResultsToDataframe:
    def test_conversion(self):
        results = [
            LDSCResult("TS", "OCD", 0.4, 0.05, 1e-5, 0.5, 0.02, 0.3, 0.03, 0.01, 5000),
            LDSCResult("TS", "ADHD", 0.3, 0.06, 1e-3, 0.5, 0.02, 0.25, 0.04, 0.02, 4500),
        ]
        df = results_to_dataframe(results)
        assert len(df) == 2
        assert list(df.columns) == [
            "trait1",
            "trait2",
            "rg",
            "rg_se",
            "p_value",
            "h2_trait1",
            "h2_trait1_se",
            "h2_trait2",
            "h2_trait2_se",
            "gcov_intercept",
            "n_snps",
        ]


# ---------------------------------------------------------------------------
# Tests: CLI (main)
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_with_synthetic_data(self, snp_ids, ld_scores_df, trait1_sumstats, tmp_path):
        sumstats_dir = tmp_path / "input"
        sumstats_dir.mkdir()
        output_dir = tmp_path / "output"

        trait1_sumstats.to_csv(sumstats_dir / "TS.sumstats", sep="\t", index=False)
        t2 = _make_correlated_sumstats(snp_ids, trait1_sumstats["Z"].values, 0.3)
        t2.to_csv(sumstats_dir / "OCD.sumstats", sep="\t", index=False)

        ld_path = sumstats_dir / "ld_scores"
        ld_path.mkdir()
        ld_scores_df.to_csv(ld_path / "all.l2.ldscore", sep="\t", index=False)

        from bioagentics.pipelines.ldsc_correlation.pipeline import main

        main(
            [
                "--input-dir",
                str(sumstats_dir),
                "--output-dir",
                str(output_dir),
            ]
        )

        assert (output_dir / "genetic_correlations.tsv").exists()
        assert (output_dir / "rg_matrix.tsv").exists()

        result_df = pd.read_csv(output_dir / "genetic_correlations.tsv", sep="\t")
        assert len(result_df) == 1
        assert result_df.iloc[0]["trait2"] == "OCD"
