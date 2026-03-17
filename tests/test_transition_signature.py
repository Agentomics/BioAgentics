"""Tests for inflammation-to-fibrosis transition signature module."""

import pandas as pd
import pytest

from bioagentics.data.cd_fibrosis.transition_signature import (
    CD38_PECAM1_GENES,
    TL1A_DR3_RHO_GENES,
    YAP_TAZ_GENES,
    build_transition_signature,
)


class TestGeneSetDefinitions:
    def test_cd38_pecam1_has_core_genes(self):
        assert "CD38" in CD38_PECAM1_GENES
        assert "PECAM1" in CD38_PECAM1_GENES
        assert CD38_PECAM1_GENES["CD38"] == "up"

    def test_yap_taz_has_core_genes(self):
        assert "YAP1" in YAP_TAZ_GENES
        assert "WWTR1" in YAP_TAZ_GENES
        assert "CTGF" in YAP_TAZ_GENES
        assert YAP_TAZ_GENES["YAP1"] == "up"

    def test_tl1a_dr3_has_core_genes(self):
        assert "TNFSF15" in TL1A_DR3_RHO_GENES
        assert "TNFRSF25" in TL1A_DR3_RHO_GENES
        assert "RHOA" in TL1A_DR3_RHO_GENES

    def test_all_directions_valid(self):
        for name, genes in [("CD38", CD38_PECAM1_GENES),
                            ("YAP", YAP_TAZ_GENES),
                            ("TL1A", TL1A_DR3_RHO_GENES)]:
            for gene, direction in genes.items():
                assert direction in ("up", "down"), f"{name}:{gene} has invalid direction"


class TestBuildTransitionSignature:
    def test_returns_dataframe(self, tmp_path):
        # Use empty msigdb dir (no GMT files)
        df = build_transition_signature(msigdb_dir=tmp_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self, tmp_path):
        df = build_transition_signature(msigdb_dir=tmp_path)
        required = {"gene", "direction", "transition_pathways", "n_transition_pathways",
                     "msigdb_pathways", "n_msigdb_pathways"}
        assert required.issubset(set(df.columns))

    def test_no_duplicate_genes(self, tmp_path):
        df = build_transition_signature(msigdb_dir=tmp_path)
        assert df["gene"].is_unique

    def test_multi_pathway_genes_detected(self, tmp_path):
        df = build_transition_signature(msigdb_dir=tmp_path)
        multi = df[df["n_transition_pathways"] > 1]
        # COL1A1, ACTA2, RHOA, ROCK1, ROCK2, COL3A1 should be multi-pathway
        assert len(multi) >= 5
        assert "COL1A1" in multi["gene"].values

    def test_convergence_genes_are_key_fibrosis_markers(self, tmp_path):
        df = build_transition_signature(msigdb_dir=tmp_path)
        multi = set(df[df["n_transition_pathways"] > 1]["gene"])
        # These should be in multiple pathways
        assert "ACTA2" in multi  # alpha-SMA
        assert "RHOA" in multi   # Rho GTPase
