"""Tests for GLIS3/IL-11 axis fibrosis signature module."""

import pandas as pd

from bioagentics.data.cd_fibrosis.glis3_il11_signature import (
    GLIS3_TF_TARGETS,
    IL11_SIGNALING,
    UPSTREAM_ACTIVATORS,
    build_glis3_il11_signature,
)


class TestGeneSetDefinitions:
    def test_glis3_tf_targets_has_core_genes(self):
        assert "GLIS3" in GLIS3_TF_TARGETS
        assert "IL11" in GLIS3_TF_TARGETS
        assert GLIS3_TF_TARGETS["GLIS3"] == "up"

    def test_il11_signaling_has_core_genes(self):
        assert "IL11" in IL11_SIGNALING
        assert "IL11RA" in IL11_SIGNALING
        assert "STAT3" in IL11_SIGNALING
        assert IL11_SIGNALING["IL11"] == "up"

    def test_upstream_activators_has_core_genes(self):
        assert "TGFB1" in UPSTREAM_ACTIVATORS
        assert "IL1B" in UPSTREAM_ACTIVATORS
        assert "TGFBR1" in UPSTREAM_ACTIVATORS

    def test_all_directions_valid(self):
        for name, genes in [("GLIS3_TF", GLIS3_TF_TARGETS),
                            ("IL11", IL11_SIGNALING),
                            ("UPSTREAM", UPSTREAM_ACTIVATORS)]:
            for gene, direction in genes.items():
                assert direction in ("up", "down"), f"{name}:{gene} has invalid direction"

    def test_smad7_is_protective(self):
        """SMAD7 is an inhibitory SMAD, should be marked down."""
        assert UPSTREAM_ACTIVATORS["SMAD7"] == "down"

    def test_druggable_targets_present(self):
        """IL-11 and SERPINE1 are key druggable targets."""
        all_genes = {**GLIS3_TF_TARGETS, **IL11_SIGNALING, **UPSTREAM_ACTIVATORS}
        assert "IL11" in all_genes
        assert "SERPINE1" in all_genes
        assert "JAK1" in all_genes
        assert "JAK2" in all_genes


class TestBuildGlis3Il11Signature:
    def test_returns_dataframe(self, tmp_path):
        df = build_glis3_il11_signature(msigdb_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self, tmp_path):
        df = build_glis3_il11_signature(msigdb_dir=tmp_path)
        required = {"gene", "direction", "glis3_il11_components", "n_components",
                     "msigdb_pathways", "n_msigdb_pathways"}
        assert required.issubset(set(df.columns))

    def test_no_duplicate_genes(self, tmp_path):
        df = build_glis3_il11_signature(msigdb_dir=tmp_path)
        assert df["gene"].is_unique

    def test_multi_component_genes_detected(self, tmp_path):
        df = build_glis3_il11_signature(msigdb_dir=tmp_path)
        multi = df[df["n_components"] > 1]
        # IL11 is in both GLIS3_TF_TARGETS and IL11_SIGNALING
        assert len(multi) >= 1
        assert "IL11" in multi["gene"].values

    def test_il11_is_multi_component(self, tmp_path):
        """IL11 should appear in both GLIS3 targets and IL11 signaling."""
        df = build_glis3_il11_signature(msigdb_dir=tmp_path)
        il11_row = df[df["gene"] == "IL11"].iloc[0]
        assert il11_row["n_components"] >= 2

    def test_gene_count_reasonable(self, tmp_path):
        """Should have 30-60 unique genes."""
        df = build_glis3_il11_signature(msigdb_dir=tmp_path)
        assert 25 <= len(df) <= 70
