"""Tests for the S-LDSC functional annotation partitioning pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.pipelines.sldsc_partition.pipeline import (
    AnnotationEnrichment,
    PartitionedResult,
    compute_partitioned_correlations,
    load_annotations,
    load_stratified_ld_scores,
    results_to_dataframe,
    sldsc_regression,
)

RNG = np.random.default_rng(123)
N_SNPS = 3000
SAMPLE_SIZE = 50000

ANNOTATIONS = ["coding", "regulatory", "brain_expressed", "cstc_circuit"]


def _make_snp_ids(n: int) -> list[str]:
    return [f"rs{i}" for i in range(1, n + 1)]


@pytest.fixture()
def snp_ids():
    return _make_snp_ids(N_SNPS)


@pytest.fixture()
def stratified_ld(snp_ids):
    """Synthetic stratified LD scores with 4 annotation categories."""
    data = {"SNP": snp_ids}
    for annot in ANNOTATIONS:
        data[f"L2_{annot}"] = RNG.exponential(20, size=N_SNPS) + 0.5
    return pd.DataFrame(data)


@pytest.fixture()
def sumstats_df(snp_ids):
    """Synthetic summary statistics with a heritable signal."""
    z = RNG.normal(0, np.sqrt(1.5), size=N_SNPS)
    return pd.DataFrame(
        {
            "SNP": snp_ids,
            "A1": RNG.choice(["A", "C", "G", "T"], size=N_SNPS),
            "A2": RNG.choice(["A", "C", "G", "T"], size=N_SNPS),
            "Z": z,
            "N": np.full(N_SNPS, SAMPLE_SIZE),
        }
    )


# ---------------------------------------------------------------------------
# Tests: load_annotations
# ---------------------------------------------------------------------------


class TestLoadAnnotations:
    def test_load_single_file(self, tmp_path, snp_ids):
        df = pd.DataFrame(
            {
                "SNP": snp_ids[:5],
                "coding": [1, 0, 1, 0, 1],
                "regulatory": [0, 1, 0, 1, 0],
            }
        )
        path = tmp_path / "test.annot"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_annotations(path)
        assert len(loaded) == 5
        assert "coding" in loaded.columns

    def test_load_directory(self, tmp_path, snp_ids):
        for chrom in [1, 2]:
            df = pd.DataFrame(
                {
                    "SNP": [f"rs{chrom}1", f"rs{chrom}2"],
                    "coding": [1, 0],
                    "regulatory": [0, 1],
                }
            )
            df.to_csv(tmp_path / f"chr{chrom}.annot", sep="\t", index=False)
        loaded = load_annotations(tmp_path)
        assert len(loaded) == 4

    def test_raises_on_empty_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_annotations(tmp_path)


# ---------------------------------------------------------------------------
# Tests: load_stratified_ld_scores
# ---------------------------------------------------------------------------


class TestLoadStratifiedLD:
    def test_load_single_file(self, tmp_path, snp_ids):
        df = pd.DataFrame(
            {
                "SNP": snp_ids[:3],
                "L2_coding": [10.0, 20.0, 15.0],
                "L2_regulatory": [5.0, 8.0, 12.0],
            }
        )
        path = tmp_path / "ld.l2.ldscore"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_stratified_ld_scores(path)
        assert len(loaded) == 3
        assert "L2_coding" in loaded.columns

    def test_load_directory(self, tmp_path):
        for chrom in [1, 2]:
            df = pd.DataFrame(
                {
                    "SNP": [f"rs{chrom}1", f"rs{chrom}2"],
                    "L2_coding": [10.0, 20.0],
                }
            )
            df.to_csv(tmp_path / f"chr{chrom}.l2.ldscore", sep="\t", index=False)
        loaded = load_stratified_ld_scores(tmp_path)
        assert len(loaded) == 4

    def test_raises_missing_snp(self, tmp_path):
        df = pd.DataFrame({"L2_coding": [10.0]})
        path = tmp_path / "bad.l2.ldscore"
        df.to_csv(path, sep="\t", index=False)
        with pytest.raises(ValueError, match="SNP"):
            load_stratified_ld_scores(path)


# ---------------------------------------------------------------------------
# Tests: sldsc_regression
# ---------------------------------------------------------------------------


class TestSLDSCRegression:
    def test_returns_result(self, sumstats_df, stratified_ld):
        result = sldsc_regression(sumstats_df, stratified_ld, ANNOTATIONS, "TS")
        assert isinstance(result, PartitionedResult)
        assert result.trait1 == "TS"
        assert result.n_snps > 0
        assert len(result.annotations) == len(ANNOTATIONS)

    def test_annotation_names(self, sumstats_df, stratified_ld):
        result = sldsc_regression(sumstats_df, stratified_ld, ANNOTATIONS, "TS")
        names = {a.annotation for a in result.annotations}
        assert names == set(ANNOTATIONS)

    def test_enrichments_are_finite(self, sumstats_df, stratified_ld):
        result = sldsc_regression(sumstats_df, stratified_ld, ANNOTATIONS, "TS")
        for a in result.annotations:
            assert np.isfinite(a.coefficient)
            assert np.isfinite(a.coefficient_se)
            assert np.isfinite(a.enrichment)

    def test_raises_on_missing_annotations(self, sumstats_df, stratified_ld):
        with pytest.raises(ValueError, match="No matching"):
            sldsc_regression(sumstats_df, stratified_ld, ["nonexistent"], "TS")


# ---------------------------------------------------------------------------
# Tests: compute_partitioned_correlations
# ---------------------------------------------------------------------------


class TestComputePartitioned:
    def test_full_pipeline(self, snp_ids, sumstats_df, stratified_ld, tmp_path):
        sumstats_dir = tmp_path / "sumstats"
        sumstats_dir.mkdir()
        sumstats_df.to_csv(sumstats_dir / "TS.sumstats", sep="\t", index=False)
        sumstats_df.to_csv(sumstats_dir / "OCD.sumstats", sep="\t", index=False)

        results = compute_partitioned_correlations(
            sumstats_dir, stratified_ld, ANNOTATIONS
        )
        assert len(results) == 2
        assert {r.trait1 for r in results} == {"OCD", "TS"}

    def test_filter_traits(self, sumstats_df, stratified_ld, tmp_path):
        sumstats_dir = tmp_path / "sumstats"
        sumstats_dir.mkdir()
        sumstats_df.to_csv(sumstats_dir / "TS.sumstats", sep="\t", index=False)
        sumstats_df.to_csv(sumstats_dir / "OCD.sumstats", sep="\t", index=False)
        sumstats_df.to_csv(sumstats_dir / "ADHD.sumstats", sep="\t", index=False)

        results = compute_partitioned_correlations(
            sumstats_dir, stratified_ld, ANNOTATIONS, traits=["TS", "OCD"]
        )
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Tests: results_to_dataframe
# ---------------------------------------------------------------------------


class TestResultsToDataframe:
    def test_conversion(self):
        results = [
            PartitionedResult(
                trait1="TS",
                trait2="partitioned",
                annotations=[
                    AnnotationEnrichment("coding", 0.1, 0.02, 1.5, 0.01, 0.1, 0.15),
                    AnnotationEnrichment("regulatory", 0.05, 0.03, 0.8, 0.3, 0.2, 0.16),
                ],
                total_h2=0.5,
                n_snps=5000,
            )
        ]
        df = results_to_dataframe(results)
        assert len(df) == 2
        assert "annotation" in df.columns
        assert "enrichment" in df.columns
        assert "enrichment_p" in df.columns


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_with_synthetic_data(self, snp_ids, sumstats_df, stratified_ld, tmp_path):
        sumstats_dir = tmp_path / "input"
        sumstats_dir.mkdir()
        annotations_dir = tmp_path / "annotations"
        annotations_dir.mkdir()
        output_dir = tmp_path / "output"

        sumstats_df.to_csv(sumstats_dir / "TS.sumstats", sep="\t", index=False)

        # Write stratified LD scores
        ld_dir = annotations_dir / "ld_scores"
        ld_dir.mkdir()
        stratified_ld.to_csv(ld_dir / "all.l2.ldscore", sep="\t", index=False)

        from bioagentics.pipelines.sldsc_partition.pipeline import main

        main(
            [
                "--input-dir", str(sumstats_dir),
                "--annotations-dir", str(annotations_dir),
                "--output-dir", str(output_dir),
            ]
        )

        assert (output_dir / "sldsc_enrichment.tsv").exists()
        result_df = pd.read_csv(output_dir / "sldsc_enrichment.tsv", sep="\t")
        assert len(result_df) == len(ANNOTATIONS)
        assert result_df.iloc[0]["trait"] == "TS"
