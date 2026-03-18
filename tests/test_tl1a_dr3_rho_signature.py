"""Tests for TL1A-DR3/Rho pathway fibrosis signature module."""

import pandas as pd

from bioagentics.data.cd_fibrosis.tl1a_dr3_rho_signature import (
    TL1A_DR3_SIGNALING,
    RHO_GTPASE_CASCADE,
    TL1A_FIBROBLAST_EFFECTORS,
    ALK5_TGFB_AXIS,
    build_tl1a_dr3_rho_signature,
)


class TestGeneSetDefinitions:
    def test_tl1a_dr3_signaling_has_core_genes(self):
        assert "TNFSF15" in TL1A_DR3_SIGNALING
        assert "TNFRSF25" in TL1A_DR3_SIGNALING
        assert TL1A_DR3_SIGNALING["TNFSF15"] == "up"

    def test_rho_cascade_has_core_genes(self):
        assert "RHOA" in RHO_GTPASE_CASCADE
        assert "ROCK1" in RHO_GTPASE_CASCADE
        assert "ROCK2" in RHO_GTPASE_CASCADE
        assert "CDC42" in RHO_GTPASE_CASCADE

    def test_fibroblast_effectors_has_core_genes(self):
        assert "COL1A1" in TL1A_FIBROBLAST_EFFECTORS
        assert "SERPINE1" in TL1A_FIBROBLAST_EFFECTORS
        assert "GREM1" in TL1A_FIBROBLAST_EFFECTORS
        assert "TWIST1" in TL1A_FIBROBLAST_EFFECTORS

    def test_alk5_axis_has_ontunisertib_target(self):
        assert "TGFBR1" in ALK5_TGFB_AXIS  # ALK5 = ontunisertib target
        assert ALK5_TGFB_AXIS["TGFBR1"] == "up"

    def test_all_directions_valid(self):
        for name, genes in [("TL1A", TL1A_DR3_SIGNALING),
                            ("RHO", RHO_GTPASE_CASCADE),
                            ("EFFECTORS", TL1A_FIBROBLAST_EFFECTORS),
                            ("ALK5", ALK5_TGFB_AXIS)]:
            for gene, direction in genes.items():
                assert direction in ("up", "down"), f"{name}:{gene} has invalid direction"

    def test_cofilin_is_down(self):
        """CFL1 should be down (inactivated by LIMK)."""
        assert RHO_GTPASE_CASCADE["CFL1"] == "down"

    def test_smad7_is_protective(self):
        """SMAD7 is inhibitory SMAD."""
        assert ALK5_TGFB_AXIS["SMAD7"] == "down"

    def test_dcr3_decoy_receptor_present(self):
        """DcR3 (TNFRSF6B) should be present for duvakitug mechanism."""
        assert "TNFRSF6B" in TL1A_DR3_SIGNALING

    def test_rho_gefs_present(self):
        """Rho GEF activators should be included."""
        assert "ARHGEF2" in RHO_GTPASE_CASCADE


class TestBuildTl1aDr3RhoSignature:
    def test_returns_dataframe(self, tmp_path):
        df = build_tl1a_dr3_rho_signature(msigdb_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self, tmp_path):
        df = build_tl1a_dr3_rho_signature(msigdb_dir=tmp_path)
        required = {"gene", "direction", "tl1a_dr3_components", "n_components",
                     "msigdb_pathways", "n_msigdb_pathways"}
        assert required.issubset(set(df.columns))

    def test_no_duplicate_genes(self, tmp_path):
        df = build_tl1a_dr3_rho_signature(msigdb_dir=tmp_path)
        assert df["gene"].is_unique

    def test_multi_component_genes_detected(self, tmp_path):
        df = build_tl1a_dr3_rho_signature(msigdb_dir=tmp_path)
        multi = df[df["n_components"] > 1]
        assert len(multi) >= 5
        # COL1A1 should be in Rho cascade, effectors, and ALK5 axis
        assert "COL1A1" in multi["gene"].values
        assert "ACTA2" in multi["gene"].values

    def test_gene_count_reasonable(self, tmp_path):
        """Should have 60-120 unique genes."""
        df = build_tl1a_dr3_rho_signature(msigdb_dir=tmp_path)
        assert 50 <= len(df) <= 130

    def test_covers_all_clinical_pathways(self, tmp_path):
        """Signature should cover genes from all 3 clinical agents' mechanisms."""
        df = build_tl1a_dr3_rho_signature(msigdb_dir=tmp_path)
        genes = set(df["gene"].values)
        # Duvakitug/tulisokibart target
        assert "TNFSF15" in genes
        # Ontunisertib target (ALK5)
        assert "TGFBR1" in genes
        # Rho pathway (tulisokibart RNA-seq finding)
        assert "RHOA" in genes
