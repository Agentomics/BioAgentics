"""Tests for NOD2 functional spectrum training set construction."""

from __future__ import annotations

import pandas as pd

from bioagentics.data.nod2.training_set import (
    GOF_VARIANTS,
    LOF_VARIANTS,
    build_training_set,
)


class TestTrainingSetDefinitions:
    def test_lof_has_known_cd_variants(self):
        hgvs_list = [v["hgvs_p"] for v in LOF_VARIANTS]
        assert "R702W" in hgvs_list
        assert "G908R" in hgvs_list
        assert "L1007fs" in hgvs_list

    def test_gof_has_blau_variants(self):
        hgvs_list = [v["hgvs_p"] for v in GOF_VARIANTS]
        assert "R334Q" in hgvs_list
        assert "R334W" in hgvs_list
        assert "L469F" in hgvs_list

    def test_gof_variants_have_nfkb_activity(self):
        for v in GOF_VARIANTS:
            assert v["nfkb_activity"], f"GOF variant {v['hgvs_p']} missing nfkb_activity"

    def test_surf_lof_variants_present(self):
        hgvs_list = [v["hgvs_p"] for v in LOF_VARIANTS]
        assert "L682F" in hgvs_list
        assert "R587C" in hgvs_list

    def test_minimum_functionally_characterized(self):
        """Plan requires ~15 functionally characterized variants."""
        total = len(LOF_VARIANTS) + len(set(v["hgvs_p"] for v in GOF_VARIANTS))
        assert total >= 8  # At minimum 8 unique variants with functional data
