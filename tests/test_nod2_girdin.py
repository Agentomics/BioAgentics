"""Tests for NOD2-girdin binding domain features."""

from __future__ import annotations

import math

import numpy as np

from bioagentics.data.nod2.girdin import (
    GIRDIN_INTERFACE_END,
    GIRDIN_INTERFACE_START,
    _compute_min_distance,
    _parse_frameshift,
    _predict_binding_effect,
)


class TestParseFrameshift:
    def test_three_letter_fs(self):
        is_fs, pos = _parse_frameshift("p.Leu1007fs")
        assert is_fs is True
        assert pos == 1007

    def test_single_letter_fs(self):
        is_fs, pos = _parse_frameshift("L1007fs")
        assert is_fs is True
        assert pos == 1007

    def test_not_frameshift(self):
        is_fs, pos = _parse_frameshift("p.Arg702Trp")
        assert is_fs is False
        assert pos is None

    def test_l1007fsinsc(self):
        is_fs, pos = _parse_frameshift("p.Leu1007fsinsC")
        assert is_fs is True
        assert pos == 1007


class TestComputeMinDistance:
    def test_at_interface(self):
        coords = {1007: np.array([0.0, 0.0, 0.0])}
        dist = _compute_min_distance(coords, 1007, [1007])
        assert dist == 0.0

    def test_known_distance(self):
        coords = {
            100: np.array([0.0, 0.0, 0.0]),
            1007: np.array([3.0, 4.0, 0.0]),
        }
        dist = _compute_min_distance(coords, 100, [1007])
        assert abs(dist - 5.0) < 0.001

    def test_missing_target(self):
        coords = {1007: np.array([0.0, 0.0, 0.0])}
        assert math.isnan(_compute_min_distance(coords, 999, [1007]))


class TestPredictBindingEffect:
    def test_in_domain_abolishes(self):
        assert _predict_binding_effect(1007, 0.0, True) == "abolishes"

    def test_close_reduces(self):
        assert _predict_binding_effect(985, 8.0, False) == "reduces"

    def test_moderate_minimal(self):
        assert _predict_binding_effect(950, 15.0, False) == "minimal"

    def test_far_none(self):
        assert _predict_binding_effect(100, 50.0, False) == "none"


class TestGirdinDomainBoundaries:
    def test_l1007_in_girdin_domain(self):
        """L1007fs should be within the girdin binding domain."""
        assert GIRDIN_INTERFACE_START <= 1007 <= GIRDIN_INTERFACE_END

    def test_domain_in_lrr(self):
        """Girdin binding domain should be in the LRR region."""
        from bioagentics.data.nod2.structure import NOD2_DOMAINS
        lrr_start, lrr_end = NOD2_DOMAINS["LRR"]
        assert GIRDIN_INTERFACE_START >= lrr_start
        assert GIRDIN_INTERFACE_END <= lrr_end
