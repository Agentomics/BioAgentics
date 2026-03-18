"""Tests for bulk tissue fibrosis signature derivation pipeline."""

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bioagentics.data.cd_fibrosis.bulk_signature import (
    KNOWN_FIBROSIS_GENES,
    differential_expression,
    fdr_correction,
    meta_combine_de,
    parse_expression_matrix,
    parse_gpl_annotation,
    parse_platform_id,
    probes_to_genes,
)


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


def _make_series_matrix(path: Path, probes: list[str], samples: list[str],
                        data: np.ndarray, platform: str = "GPL570",
                        sample_titles: list[str] | None = None,
                        sample_chars: list[list[str]] | None = None) -> Path:
    """Create a minimal GEO series matrix file for testing."""
    lines = []
    lines.append(f'!Series_platform_id\t"{platform}"')
    lines.append(f'!Sample_geo_accession\t' + '\t'.join(f'"{s}"' for s in samples))
    if sample_titles:
        lines.append(f'!Sample_title\t' + '\t'.join(f'"{t}"' for t in sample_titles))
    if sample_chars:
        for char_idx in range(len(sample_chars[0]) if sample_chars else 0):
            chars = [sc[char_idx] if char_idx < len(sc) else "" for sc in sample_chars]
            lines.append(f'!Sample_characteristics_ch1\t' + '\t'.join(f'"{c}"' for c in chars))
    lines.append("!series_matrix_table_begin")
    lines.append('"ID_REF"\t' + '\t'.join(f'"{s}"' for s in samples))
    for i, probe in enumerate(probes):
        vals = '\t'.join(str(data[i, j]) for j in range(data.shape[1]))
        lines.append(f'"{probe}"\t{vals}')
    lines.append("!series_matrix_table_end")

    content = '\n'.join(lines)
    gz_path = path / "test_series_matrix.txt.gz"
    with gzip.open(gz_path, "wt") as f:
        f.write(content)
    return gz_path


def _make_gpl_annotation(path: Path, mapping: dict[str, str]) -> Path:
    """Create a minimal GPL annotation file."""
    lines = ["# Comment line", "ID\tGene Symbol\tGene Title"]
    for probe_id, gene in mapping.items():
        lines.append(f"{probe_id}\t{gene}\tSome title")
    annot_path = path / "GPL570.annot.gz"
    with gzip.open(annot_path, "wt") as f:
        f.write('\n'.join(lines))
    return annot_path


class TestParseExpressionMatrix:
    def test_basic_parsing(self, tmp_dir):
        probes = ["1007_s_at", "1053_at", "117_at"]
        samples = ["GSM1", "GSM2", "GSM3"]
        data = np.array([[5.1, 6.2, 7.3],
                         [3.4, 4.5, 5.6],
                         [8.7, 9.8, 10.9]])
        path = _make_series_matrix(tmp_dir, probes, samples, data)

        result = parse_expression_matrix(path)
        assert result.shape == (3, 3)
        assert list(result.index) == probes
        assert list(result.columns) == samples
        np.testing.assert_allclose(result.values, data, atol=0.01)

    def test_nan_handling(self, tmp_dir):
        probes = ["p1", "p2"]
        samples = ["s1", "s2"]
        path = _make_series_matrix(tmp_dir, probes, samples,
                                   np.array([[1.0, 2.0], [3.0, 4.0]]))
        # Manually inject a null value
        with gzip.open(path, "rt") as f:
            content = f.read()
        content = content.replace("3.0", "null")
        with gzip.open(path, "wt") as f:
            f.write(content)

        result = parse_expression_matrix(path)
        assert np.isnan(result.loc["p2", "s1"])


class TestParsePlatformId:
    def test_extracts_platform(self, tmp_dir):
        probes = ["p1"]
        samples = ["s1"]
        data = np.array([[1.0]])
        path = _make_series_matrix(tmp_dir, probes, samples, data, platform="GPL6244")
        assert parse_platform_id(path) == "GPL6244"


class TestParseGplAnnotation:
    def test_basic(self, tmp_dir):
        mapping = {"1007_s_at": "DDR1", "1053_at": "RFC2", "117_at": "HSPA6"}
        annot_path = _make_gpl_annotation(tmp_dir, mapping)
        result = parse_gpl_annotation(annot_path)
        assert result["1007_s_at"] == "DDR1"
        assert result["117_at"] == "HSPA6"

    def test_multi_gene_takes_first(self, tmp_dir):
        annot_path = tmp_dir / "test.annot.gz"
        with gzip.open(annot_path, "wt") as f:
            f.write("# Comment\nID\tGene Symbol\n")
            f.write("probe1\tTP53 /// MDM2\n")
        result = parse_gpl_annotation(annot_path)
        assert result["probe1"] == "TP53"

    def test_skips_missing_symbols(self, tmp_dir):
        annot_path = tmp_dir / "test.annot.gz"
        with gzip.open(annot_path, "wt") as f:
            f.write("# Comment\nID\tGene Symbol\n")
            f.write("probe1\tTP53\n")
            f.write("probe2\t---\n")
            f.write("probe3\t\n")
        result = parse_gpl_annotation(annot_path)
        assert "probe1" in result
        assert "probe2" not in result
        assert "probe3" not in result


class TestProbesToGenes:
    def test_collapse_by_max_mean(self):
        # Two probes map to same gene; keep the one with higher mean
        expr = pd.DataFrame(
            {"s1": [1.0, 5.0, 3.0], "s2": [2.0, 6.0, 4.0]},
            index=["p_low", "p_high", "p_unique"]
        )
        mapping = {"p_low": "GENE_A", "p_high": "GENE_A", "p_unique": "GENE_B"}
        result = probes_to_genes(expr, mapping)
        assert set(result.index) == {"GENE_A", "GENE_B"}
        # GENE_A should use p_high (mean=5.5 > 1.5)
        assert result.loc["GENE_A", "s1"] == 5.0

    def test_unmapped_probes_dropped(self):
        expr = pd.DataFrame({"s1": [1.0, 2.0]}, index=["p1", "p2"])
        result = probes_to_genes(expr, {"p1": "GENE_A"})
        assert len(result) == 1
        assert "GENE_A" in result.index


class TestFdrCorrection:
    def test_basic(self):
        pvals = np.array([0.01, 0.04, 0.03, 0.20])
        fdr = fdr_correction(pvals)
        # After BH correction, all should be >= original
        for i in range(len(pvals)):
            assert fdr[i] >= pvals[i] or np.isnan(fdr[i])
        # FDR should be <= 1
        assert all(f <= 1.0 for f in fdr if not np.isnan(f))

    def test_monotonicity(self):
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
        fdr = fdr_correction(pvals)
        sorted_fdr = fdr[np.argsort(pvals)]
        # Sorted by p-value, FDR should be non-decreasing
        for i in range(1, len(sorted_fdr)):
            assert sorted_fdr[i] >= sorted_fdr[i - 1] - 1e-10

    def test_nan_handling(self):
        pvals = np.array([0.01, np.nan, 0.05])
        fdr = fdr_correction(pvals)
        assert not np.isnan(fdr[0])
        assert np.isnan(fdr[1])
        assert not np.isnan(fdr[2])

    def test_empty(self):
        assert len(fdr_correction(np.array([]))) == 0


class TestDifferentialExpression:
    def test_detects_difference(self):
        np.random.seed(42)
        n_genes = 50
        n_a, n_b = 10, 10
        # First gene has clear difference, rest are noise
        data = np.random.normal(5, 1, (n_genes, n_a + n_b))
        data[0, :n_a] += 3  # gene 0 upregulated in group A

        genes = [f"gene_{i}" for i in range(n_genes)]
        samples_a = [f"a_{i}" for i in range(n_a)]
        samples_b = [f"b_{i}" for i in range(n_b)]

        expr = pd.DataFrame(data, index=genes, columns=samples_a + samples_b)
        de = differential_expression(expr, samples_a, samples_b)

        # gene_0 should be the most significant
        assert de.iloc[0]["gene"] == "gene_0"
        assert de.iloc[0]["log2fc"] > 0
        assert de.iloc[0]["pvalue"] < 0.01

    def test_requires_min_samples(self):
        expr = pd.DataFrame({"s1": [1.0], "s2": [2.0], "s3": [3.0]},
                            index=["g1"])
        with pytest.raises(ValueError, match="Too few"):
            differential_expression(expr, ["s1"], ["s2", "s3"])


class TestMetaCombine:
    def test_single_dataset(self):
        de = pd.DataFrame({"gene": ["A", "B"], "log2fc": [1.0, -0.5],
                           "pvalue": [0.01, 0.1], "fdr": [0.02, 0.15],
                           "mean_a": [5.0, 4.0], "mean_b": [4.0, 4.5],
                           "n_a": [5, 5], "n_b": [5, 5]})
        result = meta_combine_de([(de, "DS1", "test")])
        assert "meta_pvalue" in result.columns
        assert len(result) == 2

    def test_two_datasets(self):
        de1 = pd.DataFrame({"gene": ["A", "B", "C"], "log2fc": [1.0, -0.5, 0.3],
                            "pvalue": [0.001, 0.1, 0.05], "fdr": [0.01, 0.2, 0.1],
                            "mean_a": [5, 4, 5], "mean_b": [4, 4.5, 4.7],
                            "n_a": [5, 5, 5], "n_b": [5, 5, 5]})
        de2 = pd.DataFrame({"gene": ["A", "B", "D"], "log2fc": [0.8, -0.3, 0.7],
                            "pvalue": [0.01, 0.2, 0.02], "fdr": [0.05, 0.3, 0.1],
                            "mean_a": [5, 4, 5], "mean_b": [4.2, 4.3, 4.3],
                            "n_a": [5, 5, 5], "n_b": [5, 5, 5]})
        result = meta_combine_de([(de1, "DS1", "c1"), (de2, "DS2", "c2")])
        assert "meta_pvalue" in result.columns
        # Gene A should have the strongest combined signal
        assert result.iloc[0]["gene"] == "A"


class TestKnownFibrosisGenes:
    def test_core_markers_present(self):
        expected = {"COL1A1", "ACTA2", "TGFB1", "SERPINE1", "POSTN", "CTHRC1"}
        assert expected.issubset(KNOWN_FIBROSIS_GENES)
