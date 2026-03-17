"""Tests for NOD2 structural features extraction."""

from __future__ import annotations

import numpy as np

from bioagentics.data.nod2.structure import (
    ACTIVE_SITE_RESIDUES,
    NOD2_DOMAINS,
    compute_active_site_distance,
    get_domain,
)


class TestGetDomain:
    def test_card1(self):
        assert get_domain(1) == "CARD1"
        assert get_domain(50) == "CARD1"
        assert get_domain(110) == "CARD1"

    def test_card2(self):
        assert get_domain(124) == "CARD2"
        assert get_domain(170) == "CARD2"
        assert get_domain(220) == "CARD2"

    def test_nacht(self):
        assert get_domain(273) == "NACHT"
        assert get_domain(400) == "NACHT"
        assert get_domain(576) == "NACHT"

    def test_wh(self):
        assert get_domain(577) == "WH"
        assert get_domain(700) == "WH"

    def test_lrr(self):
        assert get_domain(744) == "LRR"
        assert get_domain(900) == "LRR"
        assert get_domain(1040) == "LRR"

    def test_linker(self):
        assert get_domain(111) == "linker"
        assert get_domain(123) == "linker"
        assert get_domain(221) == "linker"

    def test_known_pathogenic_variants(self):
        # R702W is in LRR domain (pos 702 is actually in WH domain)
        assert get_domain(702) == "WH"
        # G908R is in LRR domain
        assert get_domain(908) == "LRR"
        # L1007fs is in LRR domain
        assert get_domain(1007) == "LRR"

    def test_blau_variants_in_nacht(self):
        # R334Q/W and L469F are Blau syndrome GOF variants in NACHT
        assert get_domain(334) == "NACHT"
        assert get_domain(469) == "NACHT"


class TestActiveSiteDistance:
    def test_active_site_residue_returns_zero(self):
        ca_coords = {305: np.array([0.0, 0.0, 0.0])}
        assert compute_active_site_distance(ca_coords, 305) == 0.0

    def test_nearby_residue(self):
        ca_coords = {
            305: np.array([0.0, 0.0, 0.0]),
            100: np.array([3.0, 4.0, 0.0]),  # distance = 5.0
        }
        dist = compute_active_site_distance(ca_coords, 100)
        assert abs(dist - 5.0) < 0.001

    def test_missing_residue_returns_nan(self):
        ca_coords = {305: np.array([0.0, 0.0, 0.0])}
        assert np.isnan(compute_active_site_distance(ca_coords, 999))

    def test_no_active_site_coords_returns_nan(self):
        ca_coords = {100: np.array([0.0, 0.0, 0.0])}
        dist = compute_active_site_distance(ca_coords, 100)
        assert np.isnan(dist)


class TestDomainBoundaries:
    def test_no_overlapping_domains(self):
        """Ensure domain boundaries don't overlap."""
        ranges = list(NOD2_DOMAINS.values())
        for i, (s1, e1) in enumerate(ranges):
            for j, (s2, e2) in enumerate(ranges):
                if i >= j:
                    continue
                assert e1 < s2 or e2 < s1, (
                    f"Domains {list(NOD2_DOMAINS.keys())[i]} and "
                    f"{list(NOD2_DOMAINS.keys())[j]} overlap"
                )

    def test_active_site_residues_in_nacht(self):
        """Active site residues should be in NACHT domain."""
        for pos in ACTIVE_SITE_RESIDUES:
            assert get_domain(pos) == "NACHT", (
                f"Active site residue {pos} not in NACHT domain"
            )
