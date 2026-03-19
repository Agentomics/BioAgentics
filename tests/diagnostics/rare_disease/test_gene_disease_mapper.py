"""Tests for OMIM genemap2 gene-disease mapper."""

import json
import textwrap
from pathlib import Path

import pytest

from bioagentics.diagnostics.rare_disease.gene_disease_mapper import (
    GeneDiseaseAssociation,
    parse_genemap2,
    parse_phenotype_entry,
    build_gene_disease_map,
    build_disease_gene_map,
    parse_and_build,
)


# --- Fixtures: minimal genemap2 data ---

# genemap2 columns (tab-separated):
# 0: Chromosome  1: Genomic Position Start  2: Genomic Position End
# 3: Cyto Location  4: Computed Cyto Location  5: MIM Number (gene)
# 6: Gene Symbols  7: Gene Name  8: Approved Gene Symbol
# 9: Entrez Gene ID  10: Ensembl Gene ID  11: Comments
# 12: Phenotypes  13: Mouse Gene Symbol/ID

GENEMAP2_HEADER = "# Generated from OMIM genemap2\n# Copyright OMIM\n"

GENEMAP2_ROWS = textwrap.dedent("""\
chr7\t117559590\t117706482\t7q31.2\t7q31.2\t602421\tCFTR, ABCC7\tCF transmembrane conductance regulator\tCFTR\t1080\tENSG00000001626\t\tCystic fibrosis, 219700 (3), Autosomal recessive; Congenital bilateral absence of vas deferens, 277180 (3), Autosomal recessive\tCftr (MGI:88388)
chr4\t1801973\t1862999\t4p16.3\t4p16.3\t613004\tHTT, HD, IT15\tHuntingtin\tHTT\t3064\tENSG00000197386\t\tHuntington disease, 143100 (3), Autosomal dominant\tHtt (MGI:96067)
chr11\t5225464\t5229395\t11p15.4\t11p15.4\t141900\tHBB\tHemoglobin subunit beta\tHBB\t3043\tENSG00000244734\t\t{Sickle cell anemia}, 603903 (3), Autosomal recessive; Thalassemia-beta, 613985 (3), Autosomal recessive\tHbb-bs (MGI:96021)
chrX\t31097677\t31428307\tXp21.2-p21.1\tXp21.2-p21.1\t300377\tDMD\tDystrophin\tDMD\t1756\tENSG00000198947\t\tDuchenne muscular dystrophy, 310200 (3), X-linked recessive; Becker muscular dystrophy, 300376 (3), X-linked recessive\tDmd (MGI:94909)
chr1\t10000\t20000\t1p36\t1p36\t100100\tTEST1\tTest gene\tTEST1\t9999\tENSG00000000001\t\t\tTest1 (MGI:0)
""")


@pytest.fixture
def genemap2_file(tmp_path: Path) -> Path:
    """Write minimal genemap2 to a temp file."""
    p = tmp_path / "genemap2.txt"
    p.write_text(GENEMAP2_HEADER + GENEMAP2_ROWS, encoding="utf-8")
    return p


@pytest.fixture
def parsed_associations(genemap2_file: Path) -> list[GeneDiseaseAssociation]:
    return parse_genemap2(genemap2_file)


# --- Tests: parse_phenotype_entry ---


class TestParsePhenotypeEntry:
    def test_standard_entry(self):
        result = parse_phenotype_entry("Cystic fibrosis, 219700 (3), Autosomal recessive")
        assert result["disease_name"] == "Cystic fibrosis"
        assert result["disease_mim"] == "219700"
        assert result["mapping_key"] == 3
        assert "Autosomal recessive" in result["inheritance_patterns"]
        assert result["is_provisional"] is False

    def test_provisional_entry(self):
        result = parse_phenotype_entry("{Sickle cell anemia}, 603903 (3), Autosomal recessive")
        assert result["disease_name"] == "Sickle cell anemia"
        assert result["disease_mim"] == "603903"
        assert result["is_provisional"] is True

    def test_no_inheritance(self):
        result = parse_phenotype_entry("Some disease, 123456 (3)")
        assert result["disease_name"] == "Some disease"
        assert result["disease_mim"] == "123456"
        assert result["mapping_key"] == 3
        assert result["inheritance_patterns"] == []

    def test_multiple_inheritance(self):
        result = parse_phenotype_entry(
            "Test disease, 111111 (3), Autosomal dominant, Autosomal recessive"
        )
        assert result["disease_mim"] == "111111"
        assert len(result["inheritance_patterns"]) == 2

    def test_empty_string(self):
        result = parse_phenotype_entry("")
        assert result == {}

    def test_whitespace_only(self):
        result = parse_phenotype_entry("   ")
        assert result == {}

    def test_no_mim_number(self):
        result = parse_phenotype_entry("Unknown condition (1)")
        assert result["disease_name"] == "Unknown condition"
        assert result["disease_mim"] == ""
        assert result["mapping_key"] == 1

    def test_bracket_prefix(self):
        result = parse_phenotype_entry("[Blood group A], 110300 (3)")
        assert result["is_provisional"] is True
        assert result["disease_mim"] == "110300"


# --- Tests: parse_genemap2 ---


class TestParseGenemap2:
    def test_correct_count(self, parsed_associations):
        # CFTR: 2, HTT: 1, HBB: 2, DMD: 2, TEST1: 0 (empty phenotype) = 7
        assert len(parsed_associations) == 7

    def test_cftr_cystic_fibrosis(self, parsed_associations):
        cftr_cf = [
            a for a in parsed_associations
            if a.gene_symbol == "CFTR" and a.disease_mim == "219700"
        ]
        assert len(cftr_cf) == 1
        assert cftr_cf[0].disease_name == "Cystic fibrosis"
        assert cftr_cf[0].mapping_key == 3
        assert "Autosomal recessive" in cftr_cf[0].inheritance_patterns

    def test_cftr_cbavd(self, parsed_associations):
        cftr_cbavd = [
            a for a in parsed_associations
            if a.gene_symbol == "CFTR" and a.disease_mim == "277180"
        ]
        assert len(cftr_cbavd) == 1
        assert "Congenital bilateral absence of vas deferens" in cftr_cbavd[0].disease_name

    def test_htt_huntington(self, parsed_associations):
        htt = [a for a in parsed_associations if a.gene_symbol == "HTT"]
        assert len(htt) == 1
        assert htt[0].disease_mim == "143100"
        assert "Autosomal dominant" in htt[0].inheritance_patterns

    def test_hbb_sickle_cell_provisional(self, parsed_associations):
        sickle = [
            a for a in parsed_associations
            if a.gene_symbol == "HBB" and a.disease_mim == "603903"
        ]
        assert len(sickle) == 1
        assert sickle[0].is_provisional is True

    def test_dmd_xlinked(self, parsed_associations):
        dmd = [a for a in parsed_associations if a.gene_symbol == "DMD"]
        assert len(dmd) == 2
        for assoc in dmd:
            assert "X-linked recessive" in assoc.inheritance_patterns

    def test_skips_comments(self, genemap2_file):
        # The header has comment lines starting with #
        assocs = parse_genemap2(genemap2_file)
        genes = {a.gene_symbol for a in assocs}
        assert "#" not in str(genes)

    def test_skips_empty_phenotype(self, parsed_associations):
        # TEST1 row has empty phenotype column
        test1 = [a for a in parsed_associations if a.gene_symbol == "TEST1"]
        assert len(test1) == 0

    def test_gene_mim_numbers(self, parsed_associations):
        cftr = [a for a in parsed_associations if a.gene_symbol == "CFTR"]
        assert all(a.gene_mim == "602421" for a in cftr)


# --- Tests: build_gene_disease_map ---


class TestBuildGeneDiseaseMap:
    def test_all_genes_present(self, parsed_associations):
        gene_map = build_gene_disease_map(parsed_associations)
        assert "CFTR" in gene_map
        assert "HTT" in gene_map
        assert "HBB" in gene_map
        assert "DMD" in gene_map

    def test_cftr_has_two_diseases(self, parsed_associations):
        gene_map = build_gene_disease_map(parsed_associations)
        assert len(gene_map["CFTR"]) == 2

    def test_min_mapping_key_filter(self, parsed_associations):
        gene_map_all = build_gene_disease_map(parsed_associations, min_mapping_key=0)
        gene_map_3 = build_gene_disease_map(parsed_associations, min_mapping_key=3)
        # All our test data has key 3, so should be the same
        assert len(gene_map_all) == len(gene_map_3)

    def test_omim_id_format(self, parsed_associations):
        gene_map = build_gene_disease_map(parsed_associations)
        for diseases in gene_map.values():
            for d in diseases:
                if d["disease_mim"]:
                    assert d["omim_id"].startswith("OMIM:")

    def test_annotation_dict_keys(self, parsed_associations):
        gene_map = build_gene_disease_map(parsed_associations)
        entry = gene_map["CFTR"][0]
        assert "disease_name" in entry
        assert "disease_mim" in entry
        assert "omim_id" in entry
        assert "mapping_key" in entry
        assert "inheritance_patterns" in entry
        assert "is_provisional" in entry


# --- Tests: build_disease_gene_map ---


class TestBuildDiseaseGeneMap:
    def test_disease_to_gene(self, parsed_associations):
        disease_map = build_disease_gene_map(parsed_associations)
        # Cystic fibrosis -> CFTR
        assert "OMIM:219700" in disease_map
        genes = [g["gene_symbol"] for g in disease_map["OMIM:219700"]]
        assert "CFTR" in genes

    def test_huntington_to_htt(self, parsed_associations):
        disease_map = build_disease_gene_map(parsed_associations)
        assert "OMIM:143100" in disease_map
        genes = [g["gene_symbol"] for g in disease_map["OMIM:143100"]]
        assert "HTT" in genes

    def test_duchenne_to_dmd(self, parsed_associations):
        disease_map = build_disease_gene_map(parsed_associations)
        assert "OMIM:310200" in disease_map
        genes = [g["gene_symbol"] for g in disease_map["OMIM:310200"]]
        assert "DMD" in genes


# --- Tests: parse_and_build (end-to-end) ---


class TestParseAndBuild:
    def test_saves_files(self, genemap2_file, tmp_path):
        gene_map, disease_map = parse_and_build(genemap2_file, output_dir=tmp_path)

        gene_file = tmp_path / "gene_disease_map.json"
        disease_file = tmp_path / "disease_gene_map.json"
        assert gene_file.exists()
        assert disease_file.exists()

        with open(gene_file) as f:
            loaded = json.load(f)
        assert loaded == gene_map

    def test_returned_data(self, genemap2_file, tmp_path):
        gene_map, disease_map = parse_and_build(genemap2_file, output_dir=tmp_path)
        assert len(gene_map) == 4  # CFTR, HTT, HBB, DMD
        assert len(disease_map) == 7  # 7 unique OMIM disease IDs
