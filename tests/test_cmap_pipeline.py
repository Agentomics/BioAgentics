"""Tests for CMAP/iLINCS connectivity scoring pipeline."""

import pandas as pd
import pytest

from bioagentics.data.cd_fibrosis.cmap_pipeline import (
    SIGNATURE_FILES,
    is_fibroblast_cell_line,
    load_all_signatures,
    parse_ilincs_results,
    rank_compounds_across_signatures,
)


class TestSignatureLoading:
    def test_signature_files_defined(self):
        assert len(SIGNATURE_FILES) == 6
        assert "bulk" in SIGNATURE_FILES
        assert "celltype" in SIGNATURE_FILES
        assert "transition" in SIGNATURE_FILES
        assert "glis3_il11" in SIGNATURE_FILES
        assert "cthrc1_yaptaz" in SIGNATURE_FILES
        assert "tl1a_dr3_rho" in SIGNATURE_FILES

    def test_load_all_signatures(self):
        """Load all real signatures from the output directory."""
        sigs = load_all_signatures()
        assert len(sigs) >= 3  # At least the original 3 should exist
        for name, (up, down) in sigs.items():
            assert len(up) + len(down) > 0, f"Signature {name} is empty"

    def test_load_subset(self):
        sigs = load_all_signatures(which=["transition"])
        assert len(sigs) == 1
        assert "transition" in sigs

    def test_unknown_signature_skipped(self):
        sigs = load_all_signatures(which=["nonexistent"])
        assert len(sigs) == 0


class TestFibroblastCellLine:
    def test_imr90_detected(self):
        assert is_fibroblast_cell_line("IMR90")
        assert is_fibroblast_cell_line("IMR-90")
        assert is_fibroblast_cell_line("imr90")

    def test_wi38_detected(self):
        assert is_fibroblast_cell_line("WI38")
        assert is_fibroblast_cell_line("WI-38")

    def test_non_fibroblast_rejected(self):
        assert not is_fibroblast_cell_line("MCF7")
        assert not is_fibroblast_cell_line("A549")
        assert not is_fibroblast_cell_line("HeLa")


class TestParseIlincsResults:
    def test_parse_empty(self):
        df = parse_ilincs_results([], "test")
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_parse_results(self):
        mock_results = [
            {
                "signatureid": "LINCSCP_001",
                "compound": "vorinostat",
                "similarity": -0.85,
                "cellline": "IMR90",
                "concentration": "10uM",
                "time": "24h",
            },
            {
                "signatureid": "LINCSCP_002",
                "compound": "trichostatin-a",
                "similarity": -0.72,
                "cellline": "MCF7",
                "concentration": "1uM",
                "time": "6h",
            },
        ]
        df = parse_ilincs_results(mock_results, "transition")
        assert len(df) == 2
        assert "compound" in df.columns
        assert "concordance" in df.columns
        assert "query_signature" in df.columns
        assert df.iloc[0]["concordance"] <= df.iloc[1]["concordance"]  # sorted ascending

    def test_results_sorted_ascending(self):
        """Most negative concordance (best reversal) should be first."""
        mock_results = [
            {"signatureid": "1", "compound": "A", "similarity": 0.5, "cellline": "X"},
            {"signatureid": "2", "compound": "B", "similarity": -0.9, "cellline": "X"},
            {"signatureid": "3", "compound": "C", "similarity": -0.3, "cellline": "X"},
        ]
        df = parse_ilincs_results(mock_results, "test")
        assert df.iloc[0]["compound"] == "B"  # Most negative first
        assert df.iloc[0]["concordance"] == pytest.approx(-0.9)


class TestRankCompounds:
    def _make_results(self) -> dict[str, pd.DataFrame]:
        """Create mock results from multiple signature queries."""
        return {
            "ilincs_transition": pd.DataFrame([
                {"compound": "vorinostat", "concordance": -0.85, "cell_line": "IMR90",
                 "query_signature": "transition", "signature_id": "1", "concentration": "", "time": ""},
                {"compound": "trichostatin-a", "concordance": -0.72, "cell_line": "MCF7",
                 "query_signature": "transition", "signature_id": "2", "concentration": "", "time": ""},
                {"compound": "pirfenidone", "concordance": -0.60, "cell_line": "A549",
                 "query_signature": "transition", "signature_id": "3", "concentration": "", "time": ""},
            ]),
            "ilincs_glis3_il11": pd.DataFrame([
                {"compound": "vorinostat", "concordance": -0.78, "cell_line": "WI38",
                 "query_signature": "glis3_il11", "signature_id": "4", "concentration": "", "time": ""},
                {"compound": "decitabine", "concordance": -0.65, "cell_line": "MCF7",
                 "query_signature": "glis3_il11", "signature_id": "5", "concentration": "", "time": ""},
                {"compound": "trichostatin-a", "concordance": 0.10, "cell_line": "MCF7",
                 "query_signature": "glis3_il11", "signature_id": "6", "concentration": "", "time": ""},
            ]),
        }

    def test_ranking_returns_dataframe(self):
        results = self._make_results()
        ranked = rank_compounds_across_signatures(results)
        assert isinstance(ranked, pd.DataFrame)
        assert len(ranked) > 0

    def test_convergent_hits_ranked_first(self):
        """Compounds with negative scores in multiple signatures should rank higher."""
        results = self._make_results()
        ranked = rank_compounds_across_signatures(results)
        # vorinostat is negative in both signatures
        assert ranked.iloc[0]["compound"] == "vorinostat"
        assert ranked.iloc[0]["convergent_anti_fibrotic"] == True  # noqa: E712

    def test_fibroblast_hits_detected(self):
        results = self._make_results()
        ranked = rank_compounds_across_signatures(results)
        vor = ranked[ranked["compound"] == "vorinostat"].iloc[0]
        assert vor["has_fibroblast_hit"] == True  # noqa: E712 — IMR90 and WI38

    def test_non_convergent_compound(self):
        """trichostatin-a is negative in transition but positive in glis3_il11."""
        results = self._make_results()
        ranked = rank_compounds_across_signatures(results)
        tsa = ranked[ranked["compound"] == "trichostatin-a"].iloc[0]
        assert tsa["n_negative_hits"] == 1  # Only negative in transition

    def test_empty_results(self):
        ranked = rank_compounds_across_signatures({})
        assert isinstance(ranked, pd.DataFrame)
        assert len(ranked) == 0

    def test_top_n_limit(self):
        results = self._make_results()
        ranked = rank_compounds_across_signatures(results, top_n=2)
        assert len(ranked) <= 2
