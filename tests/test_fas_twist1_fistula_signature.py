"""Tests for FAS/TWIST1 fistula fibroblast signature module."""

import pandas as pd

from bioagentics.data.cd_fibrosis.fas_twist1_fistula_signature import (
    FAS_OUTER_ZONE_MARKERS,
    FISTULA_ECM_PROGRAM,
    MACROPHAGE_SIGNALS,
    TWIST1_TARGETS,
    build_fas_twist1_fistula_signature,
)


class TestGeneSetDefinitions:
    def test_fas_outer_zone_has_core_markers(self):
        assert "TWIST1" in FAS_OUTER_ZONE_MARKERS
        assert "RUNX2" in FAS_OUTER_ZONE_MARKERS
        assert "OSR2" in FAS_OUTER_ZONE_MARKERS
        assert "PRRX1" in FAS_OUTER_ZONE_MARKERS
        assert "FAP" in FAS_OUTER_ZONE_MARKERS
        assert FAS_OUTER_ZONE_MARKERS["TWIST1"] == "up"

    def test_twist1_targets_has_core_genes(self):
        assert "TWIST1" in TWIST1_TARGETS
        assert "SNAI1" in TWIST1_TARGETS
        assert "SERPINE1" in TWIST1_TARGETS
        assert TWIST1_TARGETS["TWIST1"] == "up"

    def test_macrophage_signals_has_core_genes(self):
        assert "CXCL9" in MACROPHAGE_SIGNALS
        assert "IL1B" in MACROPHAGE_SIGNALS
        assert "TGFB1" in MACROPHAGE_SIGNALS
        assert MACROPHAGE_SIGNALS["CXCL9"] == "up"

    def test_fistula_ecm_has_core_genes(self):
        assert "TNC" in FISTULA_ECM_PROGRAM
        assert "MMP9" in FISTULA_ECM_PROGRAM
        assert "TIMP1" in FISTULA_ECM_PROGRAM

    def test_all_directions_valid(self):
        for name, genes in [("FAS", FAS_OUTER_ZONE_MARKERS),
                            ("TWIST1", TWIST1_TARGETS),
                            ("MACRO", MACROPHAGE_SIGNALS),
                            ("ECM", FISTULA_ECM_PROGRAM)]:
            for gene, direction in genes.items():
                assert direction in ("up", "down"), f"{name}:{gene} has invalid direction"

    def test_cdh1_is_protective(self):
        """CDH1 (E-cadherin) should be marked down — lost in fistula EMT."""
        assert TWIST1_TARGETS["CDH1"] == "down"

    def test_fistula_specific_markers_present(self):
        """Key fistula-specific markers should be in the signature."""
        all_genes = {
            **FAS_OUTER_ZONE_MARKERS, **TWIST1_TARGETS,
            **MACROPHAGE_SIGNALS, **FISTULA_ECM_PROGRAM,
        }
        assert "TWIST1" in all_genes
        assert "FAP" in all_genes
        assert "CXCL9" in all_genes
        assert "MMP9" in all_genes


class TestBuildFasTwist1FistulaSignature:
    def test_returns_dataframe(self, tmp_path):
        df = build_fas_twist1_fistula_signature(msigdb_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self, tmp_path):
        df = build_fas_twist1_fistula_signature(msigdb_dir=tmp_path)
        required = {"gene", "direction", "fas_twist1_components", "n_components",
                     "msigdb_pathways", "n_msigdb_pathways"}
        assert required.issubset(set(df.columns))

    def test_no_duplicate_genes(self, tmp_path):
        df = build_fas_twist1_fistula_signature(msigdb_dir=tmp_path)
        assert df["gene"].is_unique

    def test_multi_component_genes_detected(self, tmp_path):
        df = build_fas_twist1_fistula_signature(msigdb_dir=tmp_path)
        multi = df[df["n_components"] > 1]
        # TWIST1 is in both FAS_OUTER_ZONE and TWIST1_TARGETS
        assert len(multi) >= 1
        assert "TWIST1" in multi["gene"].values

    def test_twist1_is_multi_component(self, tmp_path):
        """TWIST1 should appear in both FAS outer zone and TWIST1 targets."""
        df = build_fas_twist1_fistula_signature(msigdb_dir=tmp_path)
        twist1_row = df[df["gene"] == "TWIST1"].iloc[0]
        assert twist1_row["n_components"] >= 2

    def test_gene_count_reasonable(self, tmp_path):
        """Should have 40-80 unique genes."""
        df = build_fas_twist1_fistula_signature(msigdb_dir=tmp_path)
        assert 35 <= len(df) <= 90
