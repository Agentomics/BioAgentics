"""Tests for the stratified PRS construction pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.pipelines.stratified_prs.pipeline import (
    PRSResult,
    PRSWeights,
    StratifiedPRSComparison,
    compare_stratified_vs_aggregate,
    compute_prs_weights,
    evaluate_prs,
    ld_clump,
    load_factor_gwas,
    run_stratified_prs,
    score_individuals,
    threshold_snps,
)

RNG = np.random.default_rng(42)

N_SNPS = 500
N_INDIVIDUALS = 200
SAMPLE_SIZE = 50000


def _make_snp_ids(n: int) -> list[str]:
    return [f"rs{i}" for i in range(1, n + 1)]


def _make_gwas_df(
    snp_ids: list[str] | None = None,
    n_snps: int = N_SNPS,
    h2: float = 0.3,
    with_position: bool = False,
) -> pd.DataFrame:
    """Generate synthetic GWAS summary statistics with embedded signal."""
    if snp_ids is None:
        snp_ids = _make_snp_ids(n_snps)
    n = len(snp_ids)

    # Generate Z-scores with some signal
    z = RNG.normal(0, np.sqrt(1 + h2 * 2), size=n)
    se = 1.0 / np.sqrt(SAMPLE_SIZE)
    beta = z * se
    p = 2 * (1 - np.abs(np.vectorize(lambda x: float(
        __import__("scipy").stats.norm.cdf(x)
    ))(z)))
    # Simpler P-value computation
    from scipy import stats as sp_stats
    p = 2 * sp_stats.norm.sf(np.abs(z))

    df = pd.DataFrame({
        "SNP": snp_ids,
        "A1": RNG.choice(["A", "C", "G", "T"], size=n),
        "A2": RNG.choice(["A", "C", "G", "T"], size=n),
        "BETA": beta,
        "SE": np.full(n, se),
        "Z": z,
        "P": p,
        "N": np.full(n, SAMPLE_SIZE, dtype=int),
    })

    if with_position:
        df["CHR"] = RNG.integers(1, 23, size=n)
        df["BP"] = RNG.integers(1, 250_000_000, size=n)

    return df


def _make_genotypes(
    snp_ids: list[str],
    n_individuals: int = N_INDIVIDUALS,
) -> pd.DataFrame:
    """Generate synthetic genotype dosage matrix."""
    # Simulate allele frequencies and genotype dosages
    afs = RNG.uniform(0.05, 0.95, size=len(snp_ids))
    dosages = np.zeros((n_individuals, len(snp_ids)))
    for j, af in enumerate(afs):
        dosages[:, j] = RNG.binomial(2, af, size=n_individuals).astype(float)

    iids = [f"IND{i:04d}" for i in range(n_individuals)]
    return pd.DataFrame(dosages, index=iids, columns=snp_ids)


def _make_phenotype_from_prs(
    genotypes: pd.DataFrame,
    gwas_df: pd.DataFrame,
    h2: float = 0.3,
) -> np.ndarray:
    """Generate phenotype correlated with PRS (genetic signal + noise)."""
    shared = [s for s in gwas_df["SNP"].values if s in genotypes.columns]
    if not shared:
        return RNG.normal(0, 1, size=len(genotypes))

    betas = gwas_df.set_index("SNP").loc[shared, "BETA"].values
    dosage = genotypes[shared].values
    genetic = dosage @ betas
    # Standardize genetic component
    if np.std(genetic) > 0:
        genetic = (genetic - np.mean(genetic)) / np.std(genetic)
    noise = RNG.normal(0, np.sqrt(1 - h2), size=len(genotypes))
    pheno = np.sqrt(h2) * genetic + noise
    return pheno


@pytest.fixture()
def snp_ids():
    return _make_snp_ids(N_SNPS)


@pytest.fixture()
def gwas_df(snp_ids):
    return _make_gwas_df(snp_ids)


@pytest.fixture()
def gwas_with_pos(snp_ids):
    return _make_gwas_df(snp_ids, with_position=True)


@pytest.fixture()
def genotypes(snp_ids):
    return _make_genotypes(snp_ids)


# ---------------------------------------------------------------------------
# Tests: load_factor_gwas
# ---------------------------------------------------------------------------


class TestLoadFactorGWAS:
    def test_load_standard_format(self, tmp_path):
        df = _make_gwas_df()
        path = tmp_path / "gwas.tsv"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_factor_gwas(path)
        assert "SNP" in loaded.columns
        assert "BETA" in loaded.columns
        assert "P" in loaded.columns
        assert len(loaded) == N_SNPS

    def test_load_with_column_aliases(self, tmp_path):
        df = _make_gwas_df()
        df = df.rename(columns={"SNP": "RSID", "BETA": "EFFECT", "P": "PVAL"})
        path = tmp_path / "gwas.tsv"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_factor_gwas(path)
        assert "SNP" in loaded.columns
        assert "BETA" in loaded.columns

    def test_derives_beta_from_z(self, tmp_path):
        df = _make_gwas_df()
        df = df.drop(columns=["BETA"])
        path = tmp_path / "gwas.tsv"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_factor_gwas(path)
        assert "BETA" in loaded.columns
        assert not loaded["BETA"].isna().all()

    def test_removes_duplicates(self, tmp_path):
        df = _make_gwas_df()
        # Add duplicate SNP
        df = pd.concat([df, df.iloc[:1]], ignore_index=True)
        path = tmp_path / "gwas.tsv"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_factor_gwas(path)
        assert loaded["SNP"].nunique() == len(loaded)

    def test_csv_format(self, tmp_path):
        df = _make_gwas_df()
        path = tmp_path / "gwas.csv"
        df.to_csv(path, index=False)
        loaded = load_factor_gwas(path)
        assert len(loaded) == N_SNPS


# ---------------------------------------------------------------------------
# Tests: ld_clump
# ---------------------------------------------------------------------------


class TestLDClump:
    def test_no_ld_info_keeps_all(self, gwas_df):
        result = ld_clump(gwas_df)
        assert len(result) == len(gwas_df)

    def test_position_based_clumping(self, gwas_with_pos):
        result = ld_clump(gwas_with_pos, kb_window=50000)
        # Should remove some SNPs that are close on the same chromosome
        assert len(result) <= len(gwas_with_pos)

    def test_ld_matrix_clumping(self, snp_ids, gwas_df):
        # Create a simple LD matrix with some correlated pairs
        n = min(50, len(snp_ids))
        small_snps = snp_ids[:n]
        small_gwas = gwas_df[gwas_df["SNP"].isin(small_snps)].reset_index(drop=True)

        ld = pd.DataFrame(
            np.eye(n),
            index=small_snps,
            columns=small_snps,
        )
        # Make some SNPs correlated
        for i in range(0, n - 1, 2):
            ld.iloc[i, i + 1] = 0.5
            ld.iloc[i + 1, i] = 0.5

        result = ld_clump(small_gwas, ld_matrix=ld, r2_threshold=0.1)
        assert len(result) < len(small_gwas)

    def test_sorted_by_p_value(self, gwas_with_pos):
        result = ld_clump(gwas_with_pos)
        # Clumping sorts by P
        assert result["P"].iloc[0] <= result["P"].iloc[-1]

    def test_clumping_respects_r2_threshold(self, snp_ids):
        n = 10
        small_snps = snp_ids[:n]
        gwas = _make_gwas_df(small_snps)

        # LD matrix where all SNPs are weakly correlated (r=0.2, r²=0.04)
        ld_arr = np.full((n, n), 0.2)
        np.fill_diagonal(ld_arr, 1.0)
        ld = pd.DataFrame(ld_arr, index=small_snps, columns=small_snps)

        # With r²=0.01 threshold, r²=0.04 > threshold → should clump
        strict = ld_clump(gwas, ld_matrix=ld, r2_threshold=0.01)
        # With r²=0.1 threshold, r²=0.04 < threshold → should keep all
        lenient = ld_clump(gwas, ld_matrix=ld, r2_threshold=0.1)
        assert len(strict) <= len(lenient)


# ---------------------------------------------------------------------------
# Tests: threshold_snps
# ---------------------------------------------------------------------------


class TestThresholdSNPs:
    def test_genome_wide_significant(self, gwas_df):
        result = threshold_snps(gwas_df, 5e-8)
        assert all(result["P"] <= 5e-8)

    def test_liberal_threshold(self, gwas_df):
        result = threshold_snps(gwas_df, 1.0)
        assert len(result) == len(gwas_df)

    def test_zero_threshold(self, gwas_df):
        result = threshold_snps(gwas_df, 0.0)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# Tests: compute_prs_weights
# ---------------------------------------------------------------------------


class TestComputePRSWeights:
    def test_basic_weights(self, gwas_df):
        weights = compute_prs_weights(gwas_df, "compulsive", p_thresholds=[0.05, 1.0])
        assert len(weights) >= 1
        # At P<1.0, should get all SNPs
        last = weights[-1]
        assert last.stratum == "compulsive"
        assert last.n_snps > 0

    def test_weights_at_strict_threshold(self, gwas_df):
        weights = compute_prs_weights(gwas_df, "test", p_thresholds=[5e-8])
        # May get 0 weights at strict threshold with small synthetic data
        if len(weights) > 0:
            assert weights[0].p_threshold == 5e-8

    def test_increasing_snps_with_threshold(self, gwas_df):
        weights = compute_prs_weights(gwas_df, "test", p_thresholds=[0.01, 0.1, 1.0])
        # Number of SNPs should be non-decreasing as threshold increases
        n_snps = [w.n_snps for w in weights]
        for i in range(len(n_snps) - 1):
            assert n_snps[i] <= n_snps[i + 1]

    def test_weight_values_match_betas(self, gwas_df):
        weights = compute_prs_weights(gwas_df, "test", p_thresholds=[1.0])
        assert len(weights) == 1
        w = weights[0]
        assert len(w.weights) == w.n_snps
        assert len(w.snp_ids) == w.n_snps


# ---------------------------------------------------------------------------
# Tests: score_individuals
# ---------------------------------------------------------------------------


class TestScoreIndividuals:
    def test_basic_scoring(self, snp_ids, genotypes):
        weights = PRSWeights(
            stratum="test",
            p_threshold=1.0,
            snp_ids=snp_ids[:10],
            weights=np.ones(10),
            effect_alleles=["A"] * 10,
        )
        scores = score_individuals(genotypes, weights)
        assert len(scores) == N_INDIVIDUALS
        assert np.all(np.isfinite(scores))

    def test_no_overlap_returns_zeros(self, genotypes):
        weights = PRSWeights(
            stratum="test",
            p_threshold=1.0,
            snp_ids=["fake_rs1", "fake_rs2"],
            weights=np.array([1.0, 1.0]),
            effect_alleles=["A", "A"],
        )
        scores = score_individuals(genotypes, weights)
        assert np.all(scores == 0)

    def test_score_magnitude(self, snp_ids, genotypes):
        # Positive weights should produce positive mean score
        weights = PRSWeights(
            stratum="test",
            p_threshold=1.0,
            snp_ids=snp_ids[:50],
            weights=np.ones(50),
            effect_alleles=["A"] * 50,
        )
        scores = score_individuals(genotypes, weights)
        assert np.mean(scores) > 0  # positive weights → positive mean

    def test_partial_overlap(self, snp_ids, genotypes):
        # Mix of real and fake SNPs
        mixed_snps = snp_ids[:5] + ["fake_rs1", "fake_rs2"]
        weights = PRSWeights(
            stratum="test",
            p_threshold=1.0,
            snp_ids=mixed_snps,
            weights=np.ones(7),
            effect_alleles=["A"] * 7,
        )
        scores = score_individuals(genotypes, weights)
        # Should still produce valid scores using the 5 real SNPs
        assert np.all(np.isfinite(scores))
        assert np.any(scores != 0)


# ---------------------------------------------------------------------------
# Tests: evaluate_prs
# ---------------------------------------------------------------------------


class TestEvaluatePRS:
    def test_perfect_correlation(self):
        scores = np.arange(100, dtype=float)
        pheno = scores * 2 + 1  # perfect linear
        result = evaluate_prs(scores, pheno, stratum="test", p_threshold=1.0)
        assert result.r_squared > 0.99
        assert result.p_value < 0.001

    def test_no_correlation(self):
        rng = np.random.default_rng(123)
        scores = rng.normal(0, 1, size=200)
        pheno = rng.normal(0, 1, size=200)
        result = evaluate_prs(scores, pheno, stratum="test", p_threshold=1.0)
        assert result.r_squared < 0.1

    def test_with_covariates(self):
        rng = np.random.default_rng(456)
        n = 300
        cov = rng.normal(0, 1, size=(n, 2))
        scores = rng.normal(0, 1, size=n)
        # Phenotype depends on covariates + PRS
        pheno = 0.5 * cov[:, 0] + 0.3 * scores + rng.normal(0, 0.5, size=n)
        result = evaluate_prs(scores, pheno, cov, stratum="test", p_threshold=1.0)
        assert result.r_squared > 0
        assert result.p_value < 0.05

    def test_too_few_samples(self):
        scores = np.array([1.0, 2.0, 3.0])
        pheno = np.array([1.0, 2.0, 3.0])
        result = evaluate_prs(scores, pheno, stratum="test", p_threshold=1.0)
        assert result.r_squared == 0.0  # too few samples

    def test_nan_handling(self):
        rng = np.random.default_rng(789)
        scores = rng.normal(0, 1, size=100)
        pheno = rng.normal(0, 1, size=100)
        scores[0] = np.nan
        pheno[1] = np.nan
        result = evaluate_prs(scores, pheno, stratum="test", p_threshold=1.0)
        assert result.n_snps == 98  # two removed


# ---------------------------------------------------------------------------
# Tests: compare_stratified_vs_aggregate
# ---------------------------------------------------------------------------


class TestCompareStratifiedVsAggregate:
    def test_basic_comparison(self):
        rng = np.random.default_rng(111)
        n = 200
        pheno = rng.normal(0, 1, size=n)
        strata = {
            "compulsive": 0.3 * pheno + rng.normal(0, 0.7, size=n),
            "neurodevelopmental": 0.2 * pheno + rng.normal(0, 0.8, size=n),
        }
        aggregate = 0.25 * pheno + rng.normal(0, 0.75, size=n)

        result = compare_stratified_vs_aggregate(strata, aggregate, pheno)
        assert isinstance(result, StratifiedPRSComparison)
        assert len(result.strata_results) == 2
        assert result.aggregate_result is not None
        assert result.combined_r_squared >= 0

    def test_no_aggregate(self):
        rng = np.random.default_rng(222)
        n = 200
        pheno = rng.normal(0, 1, size=n)
        strata = {
            "compulsive": rng.normal(0, 1, size=n),
        }
        result = compare_stratified_vs_aggregate(strata, None, pheno)
        assert result.aggregate_result is None
        assert result.combined_r_squared >= 0

    def test_improvement_positive_when_better(self):
        rng = np.random.default_rng(333)
        n = 500
        # Create phenotype with two distinct genetic components
        comp1 = rng.normal(0, 1, size=n)
        comp2 = rng.normal(0, 1, size=n)
        pheno = 0.4 * comp1 + 0.4 * comp2 + rng.normal(0, 0.5, size=n)

        strata = {
            "compulsive": comp1 + rng.normal(0, 0.3, size=n),
            "neurodevelopmental": comp2 + rng.normal(0, 0.3, size=n),
        }
        # Aggregate is a weaker single signal
        aggregate = 0.3 * (comp1 + comp2) + rng.normal(0, 0.8, size=n)

        result = compare_stratified_vs_aggregate(strata, aggregate, pheno)
        # Stratified should explain more variance than aggregate
        assert result.combined_r_squared > 0
        assert result.r_squared_improvement > 0


# ---------------------------------------------------------------------------
# Tests: run_stratified_prs (integration)
# ---------------------------------------------------------------------------


class TestRunStratifiedPRS:
    def test_weights_only(self, tmp_path):
        """Test pipeline with GWAS files but no genotypes."""
        gwas_dir = tmp_path / "gwas"
        gwas_dir.mkdir()
        output_dir = tmp_path / "output"

        # Write factor GWAS files
        for name in ["compulsive", "neurodevelopmental", "ts_specific"]:
            df = _make_gwas_df(h2=0.3)
            df.to_csv(gwas_dir / f"{name}.tsv", sep="\t", index=False)

        result = run_stratified_prs(
            factor_gwas_dir=gwas_dir,
            output_dir=output_dir,
            p_thresholds=[0.05, 1.0],
        )

        # No genotypes → no comparison
        assert result is None

        # But weights should be written
        assert (output_dir / "prs_weights_summary.tsv").exists()
        summary = pd.read_csv(output_dir / "prs_weights_summary.tsv", sep="\t")
        assert len(summary) > 0
        assert "compulsive" in summary["stratum"].values

    def test_full_pipeline(self, tmp_path, snp_ids):
        """Test full pipeline with genotypes and phenotype."""
        gwas_dir = tmp_path / "gwas"
        gwas_dir.mkdir()
        geno_dir = tmp_path / "geno"
        geno_dir.mkdir()
        output_dir = tmp_path / "output"

        # Write factor GWAS files
        gwas_dfs = {}
        for name in ["compulsive", "neurodevelopmental", "ts_specific"]:
            df = _make_gwas_df(snp_ids, h2=0.3)
            df.to_csv(gwas_dir / f"{name}.tsv", sep="\t", index=False)
            gwas_dfs[name] = df

        # Write aggregate GWAS
        agg_df = _make_gwas_df(snp_ids, h2=0.3)
        agg_df.to_csv(gwas_dir / "aggregate.tsv", sep="\t", index=False)

        # Write genotypes
        geno_df = _make_genotypes(snp_ids, N_INDIVIDUALS)
        geno_path = geno_dir / "genotypes.tsv"
        geno_df.to_csv(geno_path, sep="\t")

        # Write phenotype
        pheno = _make_phenotype_from_prs(geno_df, gwas_dfs["compulsive"], h2=0.3)
        pheno_df = pd.DataFrame({"PHENO": pheno}, index=geno_df.index)
        pheno_df.to_csv(geno_dir / "phenotype.tsv", sep="\t")

        result = run_stratified_prs(
            factor_gwas_dir=gwas_dir,
            target_genotypes=geno_path,
            output_dir=output_dir,
            p_thresholds=[0.05, 1.0],
        )

        assert result is not None
        assert isinstance(result, StratifiedPRSComparison)
        assert len(result.strata_results) > 0
        assert result.aggregate_result is not None

        # Check output files
        assert (output_dir / "prs_r_squared.tsv").exists()
        assert (output_dir / "prs_comparison_summary.tsv").exists()
        assert (output_dir / "prs_weights_summary.tsv").exists()

    def test_missing_stratum(self, tmp_path):
        """Pipeline should handle missing strata gracefully."""
        gwas_dir = tmp_path / "gwas"
        gwas_dir.mkdir()
        output_dir = tmp_path / "output"

        # Only write one stratum
        df = _make_gwas_df(h2=0.3)
        df.to_csv(gwas_dir / "compulsive.tsv", sep="\t", index=False)

        result = run_stratified_prs(
            factor_gwas_dir=gwas_dir,
            output_dir=output_dir,
            p_thresholds=[1.0],
        )

        assert result is None
        assert (output_dir / "prs_weights_summary.tsv").exists()


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_with_synthetic_data(self, tmp_path):
        from bioagentics.pipelines.stratified_prs.pipeline import main

        gwas_dir = tmp_path / "gwas"
        gwas_dir.mkdir()
        output_dir = tmp_path / "output"

        for name in ["compulsive", "neurodevelopmental", "ts_specific"]:
            df = _make_gwas_df(h2=0.3)
            df.to_csv(gwas_dir / f"{name}.tsv", sep="\t", index=False)

        main([
            "--factor-gwas-dir", str(gwas_dir),
            "--output-dir", str(output_dir),
            "--p-thresholds", "0.05", "1.0",
            "--log-level", "WARNING",
        ])

        assert (output_dir / "prs_weights_summary.tsv").exists()
