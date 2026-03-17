"""Tests for NOD2 variant functional impact classifier."""

from __future__ import annotations

from bioagentics.data.nod2.classifier import (
    ALL_FEATURES,
    _extract_fs_pos,
    _make_json_serializable,
)
import numpy as np


class TestExtractFsPos:
    def test_standard(self):
        assert _extract_fs_pos("p.Leu1007fs") == 1007

    def test_with_suffix(self):
        assert _extract_fs_pos("L1007fsinsC") == 1007

    def test_no_fs(self):
        assert _extract_fs_pos("p.Arg702Trp") is None


class TestFeatureList:
    def test_all_features_nonempty(self):
        assert len(ALL_FEATURES) > 20

    def test_includes_structure(self):
        assert "plddt" in ALL_FEATURES
        assert "rsasa" in ALL_FEATURES

    def test_includes_predictors(self):
        assert "cadd_phred" in ALL_FEATURES
        assert "revel" in ALL_FEATURES

    def test_includes_varmeter2(self):
        assert "mutation_energy" in ALL_FEATURES
        assert "nsasa" in ALL_FEATURES

    def test_includes_girdin(self):
        assert "girdin_interface_distance" in ALL_FEATURES
        assert "disrupts_girdin_domain" in ALL_FEATURES

    def test_includes_domains(self):
        assert "domain_NACHT" in ALL_FEATURES
        assert "domain_LRR" in ALL_FEATURES


class TestJsonSerializable:
    def test_numpy_int(self):
        assert _make_json_serializable(np.int64(42)) == 42

    def test_numpy_float(self):
        result = _make_json_serializable(np.float64(3.14))
        assert abs(result - 3.14) < 0.001

    def test_nested_dict(self):
        data = {"a": np.int64(1), "b": {"c": np.float64(2.0)}}
        result = _make_json_serializable(data)
        assert result == {"a": 1, "b": {"c": 2.0}}

    def test_list(self):
        data = [np.int64(1), np.float64(2.0)]
        result = _make_json_serializable(data)
        assert result == [1, 2.0]
