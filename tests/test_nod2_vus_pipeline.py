"""Tests for NOD2 VUS classification pipeline."""

from __future__ import annotations

from bioagentics.data.nod2.vus_pipeline import _extract_fs_pos


class TestExtractFsPos:
    def test_standard(self):
        assert _extract_fs_pos("p.Leu1007fs") == 1007

    def test_no_fs(self):
        assert _extract_fs_pos("p.Arg702Trp") is None

    def test_with_suffix(self):
        assert _extract_fs_pos("L1007fsinsC") == 1007
