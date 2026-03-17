"""Tests for bioagentics.data.gene_sets."""

from __future__ import annotations

from bioagentics.data.gene_sets import (
    get_curated_gene_sets,
    MONOCYTE_M1_M2_MARKERS,
    KIR_CD8_T_CELL_GENES,
    AE_CLASSIFIER_FEATURES,
    MHC_HLA_PATHWAY_GENES,
    CUNNINGHAM_PANEL_GENES,
)


class TestCuratedGeneSets:
    def test_get_curated_returns_all(self):
        gs = get_curated_gene_sets()
        assert "monocyte_m1_markers" in gs
        assert "monocyte_m2_markers" in gs
        assert "monocyte_surface_panel" in gs
        assert "kir_cd8_t_cell" in gs
        assert "ae_classifier_features" in gs
        assert "mhc_hla_pathway" in gs
        assert "cunningham_panel" in gs

    def test_no_empty_sets(self):
        gs = get_curated_gene_sets()
        for name, genes in gs.items():
            assert len(genes) > 0, f"Gene set '{name}' is empty"

    def test_no_duplicate_genes_within_sets(self):
        for name, genes in get_curated_gene_sets().items():
            assert len(genes) == len(set(genes)), f"Duplicates in '{name}'"

    def test_monocyte_markers_include_key_genes(self):
        m1 = MONOCYTE_M1_M2_MARKERS["monocyte_m1_markers"]
        assert "CD14" in m1
        assert "HLA-DRA" in m1
        assert "CCR2" in m1

        m2 = MONOCYTE_M1_M2_MARKERS["monocyte_m2_markers"]
        assert "CD163" in m2
        assert "MRC1" in m2

    def test_cunningham_panel_genes(self):
        assert "DRD1" in CUNNINGHAM_PANEL_GENES
        assert "DRD2" in CUNNINGHAM_PANEL_GENES
        assert "CAMK2A" in CUNNINGHAM_PANEL_GENES

    def test_hla_pathway_genes(self):
        assert "HLA-A" in MHC_HLA_PATHWAY_GENES
        assert "CIITA" in MHC_HLA_PATHWAY_GENES

    def test_kir_cd8_genes(self):
        assert "IKZF2" in KIR_CD8_T_CELL_GENES
        assert "TOX" in KIR_CD8_T_CELL_GENES

    def test_ae_features(self):
        assert "CD3D" in AE_CLASSIFIER_FEATURES
        assert "NCAM1" in AE_CLASSIFIER_FEATURES
