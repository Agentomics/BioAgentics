"""Tests for cell-type-resolved fibroblast fibrosis signature module."""

import numpy as np
import pandas as pd

from bioagentics.data.cd_fibrosis.celltype_signature import (
    CTHRC1_FIBROBLAST_GENES,
    IBD_DRUGGABLE_TARGETS,
    KONG_SPATIAL_GENES,
    TWIST1_FAP_GENES,
    build_celltype_signature,
)


class TestGeneSetDefinitions:
    def test_cthrc1_has_protein_validated_markers(self):
        for gene in ("CTHRC1", "POSTN", "TNC", "CPA3"):
            assert gene in CTHRC1_FIBROBLAST_GENES
            assert CTHRC1_FIBROBLAST_GENES[gene][1] == "protein_validated"

    def test_twist1_fap_has_master_regulators(self):
        assert "TWIST1" in TWIST1_FAP_GENES
        assert "FAP" in TWIST1_FAP_GENES
        assert TWIST1_FAP_GENES["TWIST1"][1] == "druggable_target"

    def test_kong_spatial_has_inflammatory_genes(self):
        assert "IL11" in KONG_SPATIAL_GENES
        assert "CXCL9" in KONG_SPATIAL_GENES

    def test_all_required_druggable_targets_present(self):
        required = {"HDAC1", "GREM1", "SERPINE1", "TWIST1", "FAP", "FGF2",
                     "LY96", "AKAP11", "SRM", "EHD2"}
        all_genes = set()
        for gene_dict in [CTHRC1_FIBROBLAST_GENES, TWIST1_FAP_GENES,
                          KONG_SPATIAL_GENES, IBD_DRUGGABLE_TARGETS]:
            all_genes.update(gene_dict.keys())
        assert required.issubset(all_genes)

    def test_all_directions_valid(self):
        for name, genes in [("CTHRC1", CTHRC1_FIBROBLAST_GENES),
                            ("TWIST1", TWIST1_FAP_GENES),
                            ("Kong", KONG_SPATIAL_GENES),
                            ("IBD", IBD_DRUGGABLE_TARGETS)]:
            for gene, (direction, evidence) in genes.items():
                assert direction in ("up", "down"), f"{name}:{gene}"

    def test_all_evidence_types_valid(self):
        valid = {"protein_validated", "scrna_de", "spatial_module",
                 "druggable_target", "pathway_gene"}
        for name, genes in [("CTHRC1", CTHRC1_FIBROBLAST_GENES),
                            ("TWIST1", TWIST1_FAP_GENES),
                            ("Kong", KONG_SPATIAL_GENES),
                            ("IBD", IBD_DRUGGABLE_TARGETS)]:
            for gene, (_, evidence) in genes.items():
                assert evidence in valid, f"{name}:{gene} has {evidence}"


class TestBuildCelltypeSignature:
    def test_returns_dataframe(self, tmp_path):
        df = build_celltype_signature(msigdb_dir=tmp_path, bulk_sig_path=tmp_path / "none.tsv")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_has_required_columns(self, tmp_path):
        df = build_celltype_signature(msigdb_dir=tmp_path, bulk_sig_path=tmp_path / "none.tsv")
        required = {"gene", "direction", "sources", "n_sources",
                     "evidence_level", "evidence_types"}
        assert required.issubset(set(df.columns))

    def test_no_duplicate_genes(self, tmp_path):
        df = build_celltype_signature(msigdb_dir=tmp_path, bulk_sig_path=tmp_path / "none.tsv")
        assert df["gene"].is_unique

    def test_bulk_validation_with_mock_data(self, tmp_path):
        # Create mock bulk signature
        mock_bulk = pd.DataFrame({
            "gene": ["CTHRC1", "POSTN", "FAKE_GENE"],
            "mean_log2fc": [2.5, 1.8, 0.1],
            "meta_pvalue": [1e-10, 1e-5, 0.5],
            "meta_fdr": [1e-8, 1e-3, 0.8],
        })
        bulk_path = tmp_path / "bulk.tsv"
        mock_bulk.to_csv(bulk_path, sep="\t", index=False)

        df = build_celltype_signature(msigdb_dir=tmp_path, bulk_sig_path=bulk_path)
        cthrc1 = df[df["gene"] == "CTHRC1"].iloc[0]
        assert cthrc1["bulk_validated"] == True
        assert abs(cthrc1["bulk_log2fc"] - 2.5) < 0.01
