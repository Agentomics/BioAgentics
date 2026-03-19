"""Tests for Acharjee 8-gene stricture signature module."""

import pandas as pd

from bioagentics.data.cd_fibrosis.acharjee_stricture_signature import (
    ACHARJEE_STRICTURE_GENES,
    build_acharjee_stricture_signature,
)


class TestGeneSetDefinitions:
    def test_exactly_8_genes(self):
        assert len(ACHARJEE_STRICTURE_GENES) == 8

    def test_all_required_genes_present(self):
        expected = {"LY96", "AKAP11", "SRM", "GREM1", "EHD2", "SERPINE1", "HDAC1", "FGF2"}
        assert set(ACHARJEE_STRICTURE_GENES.keys()) == expected

    def test_all_directions_valid(self):
        for gene, info in ACHARJEE_STRICTURE_GENES.items():
            assert info["direction"] in ("up", "down"), f"{gene} has invalid direction"

    def test_all_upregulated_in_stricture(self):
        """All 8 genes are upregulated in stricturing CD."""
        for gene, info in ACHARJEE_STRICTURE_GENES.items():
            assert info["direction"] == "up", f"{gene} should be up in stricture"

    def test_all_have_descriptions(self):
        for gene, info in ACHARJEE_STRICTURE_GENES.items():
            assert "description" in info
            assert len(info["description"]) > 10, f"{gene} needs a meaningful description"

    def test_overlap_genes_present(self):
        """GREM1 and SERPINE1 overlap with cell-type-resolved signature."""
        assert "GREM1" in ACHARJEE_STRICTURE_GENES
        assert "SERPINE1" in ACHARJEE_STRICTURE_GENES

    def test_hdac1_present(self):
        """HDAC1 is key for L1000 querying — well-characterized drug class."""
        assert "HDAC1" in ACHARJEE_STRICTURE_GENES


class TestBuildAcharjeeStrictureSignature:
    def test_returns_dataframe(self, tmp_path):
        df = build_acharjee_stricture_signature(msigdb_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 8

    def test_has_required_columns(self, tmp_path):
        df = build_acharjee_stricture_signature(msigdb_dir=tmp_path)
        required = {"gene", "direction", "description", "msigdb_pathways",
                     "n_msigdb_pathways", "overlaps_celltype_signature",
                     "l1000_well_characterized"}
        assert required.issubset(set(df.columns))

    def test_no_duplicate_genes(self, tmp_path):
        df = build_acharjee_stricture_signature(msigdb_dir=tmp_path)
        assert df["gene"].is_unique

    def test_overlap_flags_correct(self, tmp_path):
        df = build_acharjee_stricture_signature(msigdb_dir=tmp_path)
        grem1 = df[df["gene"] == "GREM1"].iloc[0]
        serpine1 = df[df["gene"] == "SERPINE1"].iloc[0]
        ly96 = df[df["gene"] == "LY96"].iloc[0]
        assert grem1["overlaps_celltype_signature"] == True  # noqa: E712
        assert serpine1["overlaps_celltype_signature"] == True  # noqa: E712
        assert ly96["overlaps_celltype_signature"] == False  # noqa: E712

    def test_l1000_flags_correct(self, tmp_path):
        df = build_acharjee_stricture_signature(msigdb_dir=tmp_path)
        hdac1 = df[df["gene"] == "HDAC1"].iloc[0]
        srm = df[df["gene"] == "SRM"].iloc[0]
        assert hdac1["l1000_well_characterized"] == True  # noqa: E712
        assert srm["l1000_well_characterized"] == False  # noqa: E712
