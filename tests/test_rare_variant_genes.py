"""Tests for TS rare variant gene curation (Phase 1)."""

from __future__ import annotations

from bioagentics.tourettes.rare_variant_convergence.rare_variant_genes import (
    CLINICAL_EXOME_GENES,
    CNV_REGION_GENES,
    DE_NOVO_GENES,
    ESTABLISHED_GENES,
    get_all_rare_variant_genes,
    genes_to_json,
    genes_to_records,
    summary_stats,
)


class TestEstablishedGenes:
    def test_contains_core_four(self):
        symbols = [g.gene_symbol for g in ESTABLISHED_GENES]
        assert "SLITRK1" in symbols
        assert "HDC" in symbols
        assert "NRXN1" in symbols
        assert "CNTN6" in symbols

    def test_wwc1_is_strong(self):
        wwc1 = next(g for g in ESTABLISHED_GENES if g.gene_symbol == "WWC1")
        assert wwc1.evidence_strength == "strong"
        assert "functional" in wwc1.evidence_types
        assert "hippo_signaling" in wwc1.pathways

    def test_all_established_are_strong(self):
        for g in ESTABLISHED_GENES:
            assert g.evidence_strength == "strong", (
                f"{g.gene_symbol} should have strong evidence"
            )


class TestDeNovoGenes:
    def test_trio_study_genes_present(self):
        symbols = [g.gene_symbol for g in DE_NOVO_GENES]
        assert "PPP5C" in symbols
        assert "EXOC1" in symbols
        assert "GXYLT1" in symbols

    def test_celsr3_has_dual_support(self):
        celsr3 = next(g for g in DE_NOVO_GENES if g.gene_symbol == "CELSR3")
        assert "de_novo" in celsr3.evidence_types
        assert celsr3.evidence_strength == "moderate"


class TestClinicalExomeGenes:
    def test_functional_categories_present(self):
        symbols = [g.gene_symbol for g in CLINICAL_EXOME_GENES]
        assert "SLC6A1" in symbols  # synaptic
        assert "KMT2C" in symbols  # chromatin remodeling
        assert "SMARCA2" in symbols  # chromatin remodeling


class TestCNVRegionGenes:
    def test_16p13_region(self):
        symbols = [g.gene_symbol for g in CNV_REGION_GENES]
        assert "NDE1" in symbols

    def test_22q11_region(self):
        symbols = [g.gene_symbol for g in CNV_REGION_GENES]
        assert "COMT" in symbols


class TestFullGeneSet:
    def test_no_duplicate_symbols(self):
        genes = get_all_rare_variant_genes()
        symbols = [g.gene_symbol for g in genes]
        assert len(symbols) == len(set(symbols)), (
            f"Duplicate gene symbols: {[s for s in symbols if symbols.count(s) > 1]}"
        )

    def test_minimum_gene_count(self):
        genes = get_all_rare_variant_genes()
        assert len(genes) >= 15

    def test_strong_moderate_target(self):
        """Plan target: >=15 genes with strong/moderate evidence."""
        genes = get_all_rare_variant_genes()
        strong_mod = [g for g in genes if g.evidence_strength in ("strong", "moderate")]
        assert len(strong_mod) >= 15

    def test_valid_evidence_strengths(self):
        for g in get_all_rare_variant_genes():
            assert g.evidence_strength in ("strong", "moderate", "suggestive")

    def test_valid_evidence_types(self):
        valid = {"de_novo", "segregation", "functional", "cnv", "case_report"}
        for g in get_all_rare_variant_genes():
            for et in g.evidence_types:
                assert et in valid, f"{g.gene_symbol} has invalid evidence type: {et}"

    def test_valid_variant_types(self):
        valid = {"lof", "missense", "structural", "regulatory"}
        for g in get_all_rare_variant_genes():
            for vt in g.variant_types:
                assert vt in valid, f"{g.gene_symbol} has invalid variant type: {vt}"

    def test_all_have_pathways(self):
        for g in get_all_rare_variant_genes():
            assert len(g.pathways) > 0, f"{g.gene_symbol} has no pathways"

    def test_all_have_references(self):
        for g in get_all_rare_variant_genes():
            assert len(g.references) > 0, f"{g.gene_symbol} has no references"


class TestSerialization:
    def test_genes_to_records_flat(self):
        genes = get_all_rare_variant_genes()
        records = genes_to_records(genes)
        assert len(records) == len(genes)
        for rec in records:
            assert isinstance(rec["evidence_types"], str)
            assert ";" in rec["evidence_types"] or rec["evidence_types"]

    def test_genes_to_json_preserves_lists(self):
        genes = get_all_rare_variant_genes()
        json_data = genes_to_json(genes)
        assert len(json_data) == len(genes)
        for entry in json_data:
            assert isinstance(entry["evidence_types"], list)
            assert isinstance(entry["pathways"], list)

    def test_summary_stats(self):
        genes = get_all_rare_variant_genes()
        stats = summary_stats(genes)
        assert stats["total_genes"] == len(genes)
        assert stats["strong_moderate_count"] >= 15
        assert "strong" in stats["by_evidence_strength"]
        assert len(stats["gene_symbols"]) == len(genes)
