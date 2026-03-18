"""Tests for CTHRC1+/YAP-TAZ mechanosensitive fibroblast signature module."""

import pandas as pd

from bioagentics.data.cd_fibrosis.cthrc1_yaptaz_signature import (
    CTHRC1_FIBROBLAST_GENES,
    YAPTAZ_TARGET_GENES,
    MECHANOSENSING_GENES,
    CREEPING_FAT_SPATIAL,
    build_cthrc1_yaptaz_signature,
)


class TestGeneSetDefinitions:
    def test_cthrc1_fibroblast_has_core_genes(self):
        assert "CTHRC1" in CTHRC1_FIBROBLAST_GENES
        assert "POSTN" in CTHRC1_FIBROBLAST_GENES
        assert "TNC" in CTHRC1_FIBROBLAST_GENES
        assert CTHRC1_FIBROBLAST_GENES["CTHRC1"] == "up"

    def test_yaptaz_targets_has_core_genes(self):
        assert "YAP1" in YAPTAZ_TARGET_GENES
        assert "WWTR1" in YAPTAZ_TARGET_GENES
        assert "CTGF" in YAPTAZ_TARGET_GENES
        assert YAPTAZ_TARGET_GENES["YAP1"] == "up"

    def test_mechanosensing_has_core_genes(self):
        assert "ITGB1" in MECHANOSENSING_GENES
        assert "RHOA" in MECHANOSENSING_GENES
        assert "ROCK1" in MECHANOSENSING_GENES

    def test_creeping_fat_has_spatial_markers(self):
        assert "PPARG" in CREEPING_FAT_SPATIAL
        assert "LEP" in CREEPING_FAT_SPATIAL
        assert CREEPING_FAT_SPATIAL["PPARG"] == "down"  # lost in fibrotic fat
        assert CREEPING_FAT_SPATIAL["ADIPOQ"] == "down"

    def test_all_directions_valid(self):
        for name, genes in [("CTHRC1", CTHRC1_FIBROBLAST_GENES),
                            ("YAPTAZ", YAPTAZ_TARGET_GENES),
                            ("MECH", MECHANOSENSING_GENES),
                            ("CFAT", CREEPING_FAT_SPATIAL)]:
            for gene, direction in genes.items():
                assert direction in ("up", "down"), f"{name}:{gene} has invalid direction"

    def test_hippo_suppressors_are_down(self):
        """Hippo kinases should be down (lost in fibrosis)."""
        assert YAPTAZ_TARGET_GENES["LATS1"] == "down"
        assert YAPTAZ_TARGET_GENES["LATS2"] == "down"
        assert YAPTAZ_TARGET_GENES["MST1"] == "down"

    def test_verteporfin_target_present(self):
        """TEAD family (verteporfin target) should be present."""
        all_genes = {**YAPTAZ_TARGET_GENES, **MECHANOSENSING_GENES}
        tead_genes = [g for g in all_genes if g.startswith("TEAD")]
        assert len(tead_genes) >= 3


class TestBuildCthrc1YaptazSignature:
    def test_returns_dataframe(self, tmp_path):
        df = build_cthrc1_yaptaz_signature(msigdb_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self, tmp_path):
        df = build_cthrc1_yaptaz_signature(msigdb_dir=tmp_path)
        required = {"gene", "direction", "cthrc1_yaptaz_components", "n_components",
                     "msigdb_pathways", "n_msigdb_pathways"}
        assert required.issubset(set(df.columns))

    def test_no_duplicate_genes(self, tmp_path):
        df = build_cthrc1_yaptaz_signature(msigdb_dir=tmp_path)
        assert df["gene"].is_unique

    def test_multi_component_genes_detected(self, tmp_path):
        df = build_cthrc1_yaptaz_signature(msigdb_dir=tmp_path)
        multi = df[df["n_components"] > 1]
        # Many genes should be shared between fibroblast markers and YAP targets
        assert len(multi) >= 5

    def test_serpine1_overlap_validated(self, tmp_path):
        """SERPINE1 should appear in signature (validates Acharjee stricture overlap)."""
        df = build_cthrc1_yaptaz_signature(msigdb_dir=tmp_path)
        assert "SERPINE1" in df["gene"].values

    def test_gene_count_reasonable(self, tmp_path):
        """Should have 50-100 unique genes."""
        df = build_cthrc1_yaptaz_signature(msigdb_dir=tmp_path)
        assert 40 <= len(df) <= 120
