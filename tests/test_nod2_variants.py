"""Tests for NOD2 variant data collection pipeline."""

from __future__ import annotations

import pandas as pd

from bioagentics.data.nod2.variants import (
    _parse_clinvar_xml,
    _review_status_to_stars,
    merge_variants,
    validate_known_variants,
)


class TestReviewStatusToStars:
    def test_practice_guideline(self):
        assert _review_status_to_stars("practice guideline") == 4

    def test_expert_panel(self):
        assert _review_status_to_stars("reviewed by expert panel") == 3

    def test_multiple_submitters(self):
        assert _review_status_to_stars("criteria provided, multiple submitters, no conflicts") == 2

    def test_single_submitter(self):
        assert _review_status_to_stars("criteria provided, single submitter") == 1

    def test_conflicting(self):
        assert _review_status_to_stars("criteria provided, conflicting interpretations") == 1

    def test_no_criteria(self):
        assert _review_status_to_stars("no assertion criteria provided") == 0

    def test_empty(self):
        assert _review_status_to_stars("") == 0


class TestMergeVariants:
    def _make_clinvar(self) -> pd.DataFrame:
        return pd.DataFrame({
            "variant_id": ["ClinVar:VCV000005321", "ClinVar:VCV000005322"],
            "chrom": ["16", "16"],
            "pos": [50745926, 50756540],
            "ref": ["C", "G"],
            "alt": ["T", "A"],
            "hgvs_p": ["p.Arg702Trp", "p.Gly908Arg"],
            "clinvar_significance": ["Pathogenic", "Pathogenic"],
            "review_stars": [3, 2],
        })

    def _make_gnomad(self) -> pd.DataFrame:
        return pd.DataFrame({
            "chrom": ["16", "16", "16"],
            "pos": [50745926, 50756540, 50750000],
            "ref": ["C", "G", "A"],
            "alt": ["T", "A", "G"],
            "gnomad_af": [0.02, 0.005, 0.0001],
            "gnomad_af_popmax": [0.04, 0.01, 0.0003],
            "gnomad_hom_count": [100, 10, 0],
        })

    def test_merge_both_sources(self):
        merged = merge_variants(self._make_clinvar(), self._make_gnomad())
        assert len(merged) == 3  # 2 overlapping + 1 gnomAD-only
        # Check R702W has both ClinVar and gnomAD data
        r702w = merged[merged["hgvs_p"].str.contains("Arg702Trp", na=False)]
        assert len(r702w) == 1
        assert r702w.iloc[0]["clinvar_significance"] == "Pathogenic"
        assert r702w.iloc[0]["gnomad_af"] == 0.02

    def test_merge_clinvar_only(self):
        merged = merge_variants(self._make_clinvar(), pd.DataFrame())
        assert len(merged) == 2
        assert pd.isna(merged.iloc[0]["gnomad_af"])

    def test_merge_gnomad_only(self):
        merged = merge_variants(pd.DataFrame(), self._make_gnomad())
        assert len(merged) == 3
        assert merged.iloc[0]["clinvar_significance"] == ""

    def test_merge_both_empty(self):
        merged = merge_variants(pd.DataFrame(), pd.DataFrame())
        assert merged.empty

    def test_sorted_by_position(self):
        merged = merge_variants(self._make_clinvar(), self._make_gnomad())
        positions = merged["pos"].tolist()
        assert positions == sorted(positions)


class TestValidateKnownVariants:
    def test_all_found(self):
        df = pd.DataFrame({
            "hgvs_p": ["p.Arg702Trp", "p.Gly908Arg", "p.Leu1007fs"],
        })
        results = validate_known_variants(df)
        assert results["R702W"] is True
        assert results["G908R"] is True
        assert results["L1007fs"] is True

    def test_partial_found(self):
        df = pd.DataFrame({
            "hgvs_p": ["p.Arg702Trp", "p.Ala100Val"],
        })
        results = validate_known_variants(df)
        assert results["R702W"] is True
        assert results["G908R"] is False

    def test_single_letter_notation(self):
        df = pd.DataFrame({
            "hgvs_p": ["R702W", "G908R", "L1007fsinsC"],
        })
        results = validate_known_variants(df)
        assert all(results.values())


class TestParseClinvarXml:
    def test_empty_xml(self):
        assert _parse_clinvar_xml("<ClinVarResult-Set/>") == []

    def test_malformed_xml(self):
        assert _parse_clinvar_xml("not xml at all") == []
