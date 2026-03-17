"""Tests for ts-neuroimmune-subtyping data acquisition scripts.

Tests the module structure, manifest definitions, and LDSC conversion logic
without requiring network access or large downloads.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.data.tourettes.neuroimmune.download_autoimmune_gwas import (
    AUTOIMMUNE_STUDIES,
    convert_to_ldsc_format,
    GWASStudy,
)
from bioagentics.data.tourettes.neuroimmune.download_immune_references import (
    CELLTYPE_GROUPS,
    compute_specificity_scores,
)
from bioagentics.data.tourettes.neuroimmune.download_mr_instruments import (
    MR_EXPOSURES,
)


class TestAutoimmunGWASManifest:
    """Test the curated autoimmune GWAS study definitions."""

    def test_minimum_diseases_covered(self):
        """Plan requires 15+ autoimmune diseases."""
        assert len(AUTOIMMUNE_STUDIES) >= 15

    def test_unique_abbreviations(self):
        abbrs = [s.abbreviation for s in AUTOIMMUNE_STUDIES]
        assert len(abbrs) == len(set(abbrs))

    def test_all_have_download_urls(self):
        for s in AUTOIMMUNE_STUDIES:
            assert s.download_url.startswith("https://"), f"{s.abbreviation}: bad URL"

    def test_required_diseases_present(self):
        """Diseases specified in the research plan must be included."""
        required = {"ra", "sle", "ms", "t1d", "ibd", "cel", "pso"}
        present = {s.abbreviation for s in AUTOIMMUNE_STUDIES}
        missing = required - present
        assert not missing, f"Missing required diseases: {missing}"

    def test_sample_sizes_positive(self):
        for s in AUTOIMMUNE_STUDIES:
            assert s.sample_size > 0, f"{s.abbreviation}: zero sample size"
            assert s.n_cases > 0
            assert s.n_controls > 0
            assert s.n_cases + s.n_controls <= s.sample_size * 1.1  # allow rounding


class TestLDSCConversion:
    """Test LDSC format conversion logic."""

    def test_convert_beta_se_format(self, tmp_path):
        """Test conversion from beta/SE format to Z scores."""
        # Create a mock harmonized GWAS file
        data = {
            "hm_rsid": ["rs1", "rs2", "rs3", "rs4"],
            "hm_effect_allele": ["A", "C", "g", "T"],
            "hm_other_allele": ["G", "T", "a", "C"],
            "hm_beta": ["0.05", "-0.10", "0.02", "NA"],
            "standard_error": ["0.01", "0.02", "0.005", "0.01"],
            "p_value": ["1e-6", "1e-6", "1e-4", "0.5"],
        }
        df = pd.DataFrame(data)
        raw_path = tmp_path / "test.tsv.gz"
        df.to_csv(raw_path, sep="\t", index=False, compression="gzip")

        study = GWASStudy(
            disease="Test", abbreviation="test", gwas_catalog_id="GCST000001",
            pmid="12345", first_author="Test", year=2020,
            sample_size=10000, n_cases=5000, n_controls=5000,
            ancestry="EUR", download_url="", filename="test.tsv.gz",
            source="test",
        )

        ldsc_path = convert_to_ldsc_format(raw_path, study, tmp_path)
        assert ldsc_path is not None

        result = pd.read_csv(ldsc_path, sep="\t")
        assert list(result.columns) == ["SNP", "A1", "A2", "Z", "N"]
        assert len(result) == 3  # rs4 dropped (NA beta)
        assert result["N"].iloc[0] == 10000
        assert result["A1"].iloc[0] == "A"  # uppercased

        # Check Z = beta / SE
        z1 = result[result["SNP"] == "rs1"]["Z"].iloc[0]
        assert abs(z1 - 5.0) < 0.01  # 0.05 / 0.01 = 5.0


class TestImmuneReferences:
    """Test immune cell-type reference processing."""

    def test_celltype_groups_cover_priority(self):
        """Th17 and NK must be in the cell type groups."""
        assert "Th17" in CELLTYPE_GROUPS
        assert "NK" in CELLTYPE_GROUPS

    def test_specificity_scores_sum_to_one(self):
        """Specificity scores should approximately sum to 1 per gene."""
        expr = pd.DataFrame(
            {"Th17": [10, 0, 5], "NK": [5, 10, 5], "B": [5, 10, 0]},
            index=["GENE1", "GENE2", "GENE3"],
        )
        spec = compute_specificity_scores(expr, Path("/tmp"))
        row_sums = spec.sum(axis=1)
        np.testing.assert_allclose(row_sums[row_sums > 0], 1.0, atol=1e-10)


class TestMRInstruments:
    """Test MR instrument definitions and extraction."""

    def test_minimum_exposures(self):
        """Plan requires 11 immune biomarkers."""
        assert len(MR_EXPOSURES) >= 11

    def test_priority_biomarkers_present(self):
        """Key biomarkers from the plan must be included."""
        required = {"crp", "il6", "tnfa", "lymph", "nk", "cd4", "il17", "ccl2"}
        present = {e.abbreviation for e in MR_EXPOSURES}
        missing = required - present
        assert not missing, f"Missing biomarkers: {missing}"

    def test_all_have_gwas_ids(self):
        for e in MR_EXPOSURES:
            assert e.gwas_id.startswith("GCST"), f"{e.abbreviation}: bad GWAS ID"
