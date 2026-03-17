"""Tests for VarMeter2 structure-based features."""

from __future__ import annotations

import math

from bioagentics.data.nod2.varmeter2 import (
    _parse_protein_change,
    estimate_ddg,
    grantham_distance,
)


class TestGranthamDistance:
    def test_identical_aa(self):
        assert grantham_distance("A", "A") == 0.0

    def test_known_pair(self):
        assert grantham_distance("R", "W") == 101

    def test_symmetric(self):
        assert grantham_distance("A", "R") == grantham_distance("R", "A")

    def test_conservative_small(self):
        # I -> L is very conservative (Grantham = 5)
        assert grantham_distance("I", "L") == 5

    def test_radical_large(self):
        # C -> W is radical (Grantham = 215)
        assert grantham_distance("C", "W") == 215

    def test_unknown_pair(self):
        assert math.isnan(grantham_distance("X", "A"))


class TestEstimateDdg:
    def test_buried_radical(self):
        # Buried (nSASA=0), radical (grantham=200), high confidence
        ddg = estimate_ddg(0.0, 200.0, 90.0)
        assert ddg > 0.5  # should be highly destabilizing

    def test_exposed_conservative(self):
        # Exposed (nSASA=1), conservative (grantham=5), high confidence
        ddg = estimate_ddg(1.0, 5.0, 90.0)
        assert ddg < 0.05  # should be tolerated

    def test_nan_input(self):
        assert math.isnan(estimate_ddg(float("nan"), 100.0, 90.0))

    def test_buried_more_destabilizing_than_exposed(self):
        ddg_buried = estimate_ddg(0.1, 100.0, 90.0)
        ddg_exposed = estimate_ddg(0.9, 100.0, 90.0)
        assert ddg_buried > ddg_exposed


class TestParseProteinChange:
    def test_three_letter(self):
        assert _parse_protein_change("p.Arg702Trp") == ("R", 702, "W")

    def test_single_letter(self):
        assert _parse_protein_change("R702W") == ("R", 702, "W")

    def test_p_prefix_single_letter(self):
        assert _parse_protein_change("p.R702W") == ("R", 702, "W")

    def test_frameshift_returns_none(self):
        assert _parse_protein_change("p.Leu1007fs") is None

    def test_stop_gain_returns_none(self):
        assert _parse_protein_change("p.Arg702*") is None

    def test_synonymous_returns_none(self):
        assert _parse_protein_change("p.Arg702Arg") is None

    def test_empty_returns_none(self):
        assert _parse_protein_change("") is None
