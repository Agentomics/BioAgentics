"""Tests for Grotzinger factor integration into stratified PRS."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from tourettes.ts_comorbidity_genetic_architecture.grotzinger_factors import (
    GROTZINGER_FACTORS,
    PRS_STRATA,
    TS_PRIMARY_FACTORS,
    GrotzingerDataPackage,
    build_factor_prs_configs,
    create_synthetic_data_package,
    filter_snps_by_pleiotropic_loci,
    generate_reference_weights,
    generate_synthetic_factor_gwas,
    generate_synthetic_loadings,
    generate_synthetic_pleiotropic_loci,
    get_ts_factor_loading,
    load_factor_loadings,
    load_grotzinger_package,
    load_pleiotropic_loci,
    run_grotzinger_prs,
    write_config_templates,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_loadings():
    return generate_synthetic_loadings()


@pytest.fixture()
def synthetic_loci():
    return generate_synthetic_pleiotropic_loci(n_loci=50)


@pytest.fixture()
def synthetic_gwas():
    return generate_synthetic_factor_gwas(n_snps=200, seed=42, with_position=True)


@pytest.fixture()
def synthetic_package(tmp_path):
    """Create a full synthetic data package on disk."""
    return create_synthetic_data_package(tmp_path)


# ---------------------------------------------------------------------------
# Tests: synthetic data generators
# ---------------------------------------------------------------------------


class TestSyntheticData:
    def test_generate_factor_gwas(self):
        df = generate_synthetic_factor_gwas(n_snps=100, seed=1)
        assert len(df) == 100
        assert "SNP" in df.columns
        assert "BETA" in df.columns
        assert "P" in df.columns
        assert all(df["P"] > 0)

    def test_gwas_with_position(self):
        df = generate_synthetic_factor_gwas(n_snps=50, with_position=True)
        assert "CHR" in df.columns
        assert "BP" in df.columns

    def test_generate_loadings(self, synthetic_loadings):
        assert len(synthetic_loadings) > 0
        assert "trait" in synthetic_loadings.columns
        assert "factor" in synthetic_loadings.columns
        assert "loading" in synthetic_loadings.columns
        # TS should have loadings on multiple factors
        ts_rows = synthetic_loadings[synthetic_loadings["trait"] == "TS"]
        assert len(ts_rows) >= 2

    def test_generate_pleiotropic_loci(self, synthetic_loci):
        assert len(synthetic_loci) == 50
        assert "locus_id" in synthetic_loci.columns
        assert "chr" in synthetic_loci.columns
        assert "factor" in synthetic_loci.columns
        assert synthetic_loci["locus_id"].nunique() == 50

    def test_create_synthetic_package(self, synthetic_package):
        assert (synthetic_package / "factor_loadings.tsv").exists()
        assert (synthetic_package / "pleiotropic_loci.tsv").exists()
        assert (synthetic_package / "factor_gwas").is_dir()
        for factor in GROTZINGER_FACTORS:
            assert (synthetic_package / "factor_gwas" / f"{factor}.tsv").exists()
        assert (synthetic_package / "factor_gwas" / "ts_residual.tsv").exists()


# ---------------------------------------------------------------------------
# Tests: data ingestion
# ---------------------------------------------------------------------------


class TestLoadFactorLoadings:
    def test_load_from_file(self, tmp_path, synthetic_loadings):
        path = tmp_path / "loadings.tsv"
        synthetic_loadings.to_csv(path, sep="\t", index=False)
        loaded = load_factor_loadings(path)
        assert len(loaded) == len(synthetic_loadings)
        assert "loading" in loaded.columns

    def test_missing_columns_raises(self, tmp_path):
        df = pd.DataFrame({"foo": [1], "bar": [2]})
        path = tmp_path / "bad.tsv"
        df.to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_factor_loadings(path)

    def test_csv_format(self, tmp_path, synthetic_loadings):
        path = tmp_path / "loadings.csv"
        synthetic_loadings.to_csv(path, index=False)
        loaded = load_factor_loadings(path)
        assert len(loaded) == len(synthetic_loadings)


class TestLoadPleiotropicLoci:
    def test_load_from_file(self, tmp_path, synthetic_loci):
        path = tmp_path / "loci.tsv"
        synthetic_loci.to_csv(path, sep="\t", index=False)
        loaded = load_pleiotropic_loci(path)
        assert len(loaded) == 50
        assert "locus_id" in loaded.columns

    def test_missing_columns_raises(self, tmp_path):
        df = pd.DataFrame({"x": [1]})
        path = tmp_path / "bad.tsv"
        df.to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_pleiotropic_loci(path)


class TestLoadGrotzingerPackage:
    def test_full_package(self, synthetic_package):
        pkg = load_grotzinger_package(synthetic_package)
        assert isinstance(pkg, GrotzingerDataPackage)
        assert len(pkg.loadings) > 0
        assert len(pkg.pleiotropic_loci) > 0
        assert len(pkg.factor_gwas) == len(GROTZINGER_FACTORS)
        assert pkg.ts_residual_gwas is not None

    def test_missing_loadings_raises(self, tmp_path):
        d = tmp_path / "processed"
        d.mkdir()
        (d / "pleiotropic_loci.tsv").write_text("locus_id\tchr\tfactor\nL1\t1\tcompulsive\n")
        (d / "factor_gwas").mkdir()
        with pytest.raises(FileNotFoundError, match="Factor loadings"):
            load_grotzinger_package(d)

    def test_partial_factor_gwas(self, synthetic_package):
        # Remove one GWAS file
        (synthetic_package / "factor_gwas" / "sb.tsv").unlink()
        pkg = load_grotzinger_package(synthetic_package)
        assert "sb" not in pkg.factor_gwas
        assert len(pkg.factor_gwas) == len(GROTZINGER_FACTORS) - 1


# ---------------------------------------------------------------------------
# Tests: SNP filtering
# ---------------------------------------------------------------------------


class TestFilterSNPsByPleiotropicLoci:
    def test_annotates_pleiotropic_column(self, synthetic_gwas, synthetic_loci):
        result = filter_snps_by_pleiotropic_loci(synthetic_gwas, synthetic_loci, "compulsive")
        assert "pleiotropic" in result.columns
        assert result["pleiotropic"].dtype == bool

    def test_no_chr_bp_sets_false(self):
        gwas = pd.DataFrame({"SNP": ["rs1"], "BETA": [0.1], "P": [0.01]})
        loci = generate_synthetic_pleiotropic_loci(n_loci=5)
        result = filter_snps_by_pleiotropic_loci(gwas, loci, "compulsive")
        assert result["pleiotropic"].sum() == 0

    def test_empty_loci_for_factor(self, synthetic_gwas, synthetic_loci):
        result = filter_snps_by_pleiotropic_loci(synthetic_gwas, synthetic_loci, "nonexistent")
        assert result["pleiotropic"].sum() == 0


# ---------------------------------------------------------------------------
# Tests: factor loading helpers
# ---------------------------------------------------------------------------


class TestGetTSFactorLoading:
    def test_ts_compulsive_loading(self, synthetic_loadings):
        val = get_ts_factor_loading(synthetic_loadings, "compulsive")
        assert val > 0

    def test_ts_neurodevelopmental_loading(self, synthetic_loadings):
        val = get_ts_factor_loading(synthetic_loadings, "neurodevelopmental")
        assert val > 0

    def test_missing_factor_returns_zero(self, synthetic_loadings):
        val = get_ts_factor_loading(synthetic_loadings, "nonexistent_factor")
        assert val == 0.0


# ---------------------------------------------------------------------------
# Tests: config templates
# ---------------------------------------------------------------------------


class TestBuildFactorPRSConfigs:
    def test_returns_all_factors_plus_residual(self):
        configs = build_factor_prs_configs()
        names = [c.factor_name for c in configs]
        for factor in GROTZINGER_FACTORS:
            assert factor in names
        assert "ts_residual" in names
        assert len(configs) == len(GROTZINGER_FACTORS) + 1

    def test_with_loadings(self, synthetic_loadings):
        configs = build_factor_prs_configs(loadings_df=synthetic_loadings)
        comp_cfg = next(c for c in configs if c.factor_name == "compulsive")
        assert "ts_loading" in comp_cfg.metadata
        assert float(comp_cfg.metadata["ts_loading"]) > 0

    def test_ts_primary_factors_flagged(self):
        configs = build_factor_prs_configs()
        for cfg in configs:
            if cfg.factor_name in TS_PRIMARY_FACTORS:
                assert cfg.metadata.get("is_ts_primary") == "True"
            elif cfg.factor_name in GROTZINGER_FACTORS:
                assert cfg.metadata.get("is_ts_primary") == "False"

    def test_custom_thresholds(self):
        configs = build_factor_prs_configs(p_thresholds=[0.01, 0.05])
        for cfg in configs:
            assert cfg.p_thresholds == [0.01, 0.05]

    def test_residual_metadata(self):
        configs = build_factor_prs_configs()
        res_cfg = next(c for c in configs if c.factor_name == "ts_residual")
        assert "residual_variance_fraction" in res_cfg.metadata
        assert float(res_cfg.metadata["residual_variance_fraction"]) == 0.87


class TestWriteConfigTemplates:
    def test_writes_json(self, tmp_path):
        configs = build_factor_prs_configs()
        path = write_config_templates(configs, tmp_path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data) == len(configs)
        assert data[0]["factor_name"] in GROTZINGER_FACTORS

    def test_config_roundtrip(self, tmp_path):
        configs = build_factor_prs_configs(p_thresholds=[0.05, 1.0])
        path = write_config_templates(configs, tmp_path)
        with open(path) as f:
            data = json.load(f)
        for item in data:
            assert "factor_name" in item
            assert "gwas_filename" in item
            assert "metadata" in item


# ---------------------------------------------------------------------------
# Tests: reference weight generation
# ---------------------------------------------------------------------------


class TestGenerateReferenceWeights:
    def test_generates_weight_files(self, synthetic_package, tmp_path):
        gwas_dir = synthetic_package / "factor_gwas"
        configs = [
            c for c in build_factor_prs_configs(p_thresholds=[0.05, 1.0])
            if c.factor_name in PRS_STRATA
        ]
        output = tmp_path / "weights_out"
        result = generate_reference_weights(gwas_dir, configs, output_dir=output)
        assert len(result) > 0
        for factor, factor_dir in result.items():
            assert factor_dir.exists()
            assert (factor_dir / "metadata.json").exists()
            # At least one weight file
            weight_files = list(factor_dir.glob("weights_p*.tsv"))
            assert len(weight_files) > 0

    def test_weight_file_format(self, synthetic_package, tmp_path):
        gwas_dir = synthetic_package / "factor_gwas"
        configs = [
            c for c in build_factor_prs_configs(p_thresholds=[1.0])
            if c.factor_name == "compulsive"
        ]
        output = tmp_path / "weights_out"
        result = generate_reference_weights(gwas_dir, configs, output_dir=output)
        wf = list(result["compulsive"].glob("weights_p*.tsv"))[0]
        df = pd.read_csv(wf, sep="\t")
        assert "SNP" in df.columns
        assert "A1" in df.columns
        assert "WEIGHT" in df.columns
        assert len(df) > 0

    def test_missing_gwas_skipped(self, tmp_path):
        gwas_dir = tmp_path / "empty_gwas"
        gwas_dir.mkdir()
        configs = build_factor_prs_configs(p_thresholds=[1.0])
        output = tmp_path / "weights_out"
        result = generate_reference_weights(gwas_dir, configs, output_dir=output)
        assert len(result) == 0

    def test_pleiotropic_annotation(self, synthetic_package, tmp_path):
        gwas_dir = synthetic_package / "factor_gwas"
        loci = load_pleiotropic_loci(synthetic_package / "pleiotropic_loci.tsv")
        configs = [
            c for c in build_factor_prs_configs(p_thresholds=[1.0])
            if c.factor_name == "compulsive"
        ]
        output = tmp_path / "weights_out"
        generate_reference_weights(gwas_dir, configs, loci_df=loci, output_dir=output)
        wf = list((output / "compulsive").glob("weights_p*.tsv"))[0]
        df = pd.read_csv(wf, sep="\t")
        assert "PLEIOTROPIC" in df.columns


# ---------------------------------------------------------------------------
# Tests: end-to-end runner
# ---------------------------------------------------------------------------


class TestRunGrotzingerPRS:
    def test_gwas_only_mode(self, synthetic_package, tmp_path):
        gwas_dir = synthetic_package / "factor_gwas"
        output = tmp_path / "output"
        result = run_grotzinger_prs(
            gwas_dir=gwas_dir,
            output_dir=output,
            p_thresholds=[0.05, 1.0],
        )
        assert len(result) > 0
        assert (output / "prs_factor_configs.json").exists()
        assert (output / "README.md").exists()

    def test_full_package_mode(self, synthetic_package, tmp_path):
        output = tmp_path / "output"
        result = run_grotzinger_prs(
            processed_dir=synthetic_package,
            output_dir=output,
            p_thresholds=[0.05, 1.0],
        )
        assert len(result) > 0
        assert (output / "prs_factor_configs.json").exists()

    def test_subset_of_factors(self, synthetic_package, tmp_path):
        output = tmp_path / "output"
        result = run_grotzinger_prs(
            processed_dir=synthetic_package,
            output_dir=output,
            factors=["compulsive"],
            p_thresholds=[1.0],
        )
        assert "compulsive" in result
        # Config file should still include all factors (full template)
        with open(output / "prs_factor_configs.json") as f:
            configs = json.load(f)
        assert len(configs) == len(GROTZINGER_FACTORS) + 1

    def test_no_input_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Must provide"):
            run_grotzinger_prs(output_dir=tmp_path / "out", p_thresholds=[1.0])

    def test_readme_content(self, synthetic_package, tmp_path):
        output = tmp_path / "output"
        run_grotzinger_prs(
            gwas_dir=synthetic_package / "factor_gwas",
            output_dir=output,
            p_thresholds=[1.0],
        )
        readme = (output / "README.md").read_text()
        assert "Grotzinger" in readme
        assert "87%" in readme
        assert "UK Biobank" in readme


# ---------------------------------------------------------------------------
# Tests: constants and module-level
# ---------------------------------------------------------------------------


class TestConstants:
    def test_grotzinger_factors_count(self):
        assert len(GROTZINGER_FACTORS) == 5

    def test_prs_strata(self):
        assert "compulsive" in PRS_STRATA
        assert "neurodevelopmental" in PRS_STRATA
        assert "ts_residual" in PRS_STRATA

    def test_ts_primary_factors(self):
        assert "neurodevelopmental" in TS_PRIMARY_FACTORS
        assert "compulsive" in TS_PRIMARY_FACTORS
        assert len(TS_PRIMARY_FACTORS) == 2
