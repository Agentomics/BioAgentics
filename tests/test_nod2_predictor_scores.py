"""Tests for NOD2 dbNSFP predictor score fetching."""

from __future__ import annotations

from bioagentics.data.nod2.predictor_scores import (
    _build_variant_id,
    _extract_score,
)


class TestBuildVariantId:
    def test_snv(self):
        assert _build_variant_id("16", 50712015, "C", "T") == "chr16:g.50712015C>T"

    def test_str_chrom(self):
        assert _build_variant_id("X", 12345, "A", "G") == "chrX:g.12345A>G"


class TestExtractScore:
    def test_simple_nested(self):
        data = {"dbnsfp": {"cadd": {"phred": 25.3}}}
        assert _extract_score(data, "dbnsfp.cadd.phred") == 25.3

    def test_missing_field(self):
        data = {"dbnsfp": {"cadd": {}}}
        assert _extract_score(data, "dbnsfp.cadd.phred") is None

    def test_empty_data(self):
        assert _extract_score({}, "dbnsfp.cadd.phred") is None

    def test_list_value_takes_max(self):
        data = {"dbnsfp": {"revel": {"score": [0.3, 0.8, 0.5]}}}
        assert _extract_score(data, "dbnsfp.revel.score") == 0.8

    def test_single_value_in_list(self):
        data = {"dbnsfp": {"revel": {"score": [0.5]}}}
        assert _extract_score(data, "dbnsfp.revel.score") == 0.5

    def test_deeply_nested(self):
        data = {"dbnsfp": {"phylo": {"p100way": {"vertebrate": 7.5}}}}
        assert _extract_score(data, "dbnsfp.phylo.p100way.vertebrate") == 7.5

    def test_none_in_path(self):
        data = {"dbnsfp": None}
        assert _extract_score(data, "dbnsfp.cadd.phred") is None
