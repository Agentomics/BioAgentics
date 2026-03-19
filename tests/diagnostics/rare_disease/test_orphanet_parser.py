"""Tests for Orphanet disease-phenotype XML parser."""

import json
import textwrap
from pathlib import Path

import pytest

from bioagentics.diagnostics.rare_disease.orphanet_parser import (
    OrphanetAnnotation,
    OrphanetDisease,
    _normalize_frequency,
    build_omim_crossref,
    build_orphanet_hpo_map,
    get_frequency_weights,
    parse_orphanet_xml,
    parse_and_build,
)


# --- Fixtures: minimal Orphanet XML ---


MINIMAL_PRODUCT4_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="UTF-8"?>
    <JDBOR>
      <DisorderList>
        <Disorder>
          <OrphaCode>558</OrphaCode>
          <Name>Marfan syndrome</Name>
          <ExternalReferenceList>
            <ExternalReference>
              <Source>OMIM</Source>
              <Reference>154700</Reference>
            </ExternalReference>
          </ExternalReferenceList>
          <HPODisorderAssociationList>
            <HPODisorderAssociation>
              <HPO>
                <HPOId>HP:0001166</HPOId>
                <Name>Arachnodactyly</Name>
              </HPO>
              <HPOFrequency>
                <Name>Obligate (100%)</Name>
              </HPOFrequency>
            </HPODisorderAssociation>
            <HPODisorderAssociation>
              <HPO>
                <HPOId>HP:0000256</HPOId>
                <Name>Macrocephaly</Name>
              </HPO>
              <HPOFrequency>
                <Name>Occasional (29-5%)</Name>
              </HPOFrequency>
            </HPODisorderAssociation>
            <HPODisorderAssociation>
              <HPO>
                <HPOId>HP:0002999</HPOId>
                <Name>Patellar dislocation</Name>
              </HPO>
              <HPOFrequency>
                <Name>Excluded (0%)</Name>
              </HPOFrequency>
            </HPODisorderAssociation>
          </HPODisorderAssociationList>
        </Disorder>
        <Disorder>
          <OrphaCode>730</OrphaCode>
          <Name>Ehlers-Danlos syndrome</Name>
          <HPODisorderAssociationList>
            <HPODisorderAssociation>
              <HPO>
                <HPOId>HP:0001382</HPOId>
                <Name>Joint hypermobility</Name>
              </HPO>
              <HPOFrequency>
                <Name>Very frequent (99-80%)</Name>
              </HPOFrequency>
            </HPODisorderAssociation>
            <HPODisorderAssociation>
              <HPO>
                <HPOId>HP:0000974</HPOId>
                <Name>Hyperextensible skin</Name>
              </HPO>
              <HPOFrequency>
                <Name>Frequent (79-30%)</Name>
              </HPOFrequency>
            </HPODisorderAssociation>
          </HPODisorderAssociationList>
        </Disorder>
      </DisorderList>
    </JDBOR>
""")


@pytest.fixture
def product4_xml(tmp_path: Path) -> Path:
    """Write minimal product4 XML to a temp file."""
    p = tmp_path / "en_product4.xml"
    p.write_text(MINIMAL_PRODUCT4_XML, encoding="utf-8")
    return p


@pytest.fixture
def parsed_diseases(product4_xml: Path) -> list[OrphanetDisease]:
    return parse_orphanet_xml(product4_xml)


# --- Tests: frequency normalization ---


class TestNormalizeFrequency:
    def test_obligate(self):
        cat, val = _normalize_frequency("Obligate (100%)")
        assert cat == "obligate"
        assert val == 1.00

    def test_very_frequent(self):
        cat, val = _normalize_frequency("Very frequent (99-80%)")
        assert cat == "very_frequent"
        assert val == 0.90

    def test_frequent(self):
        cat, val = _normalize_frequency("Frequent (79-30%)")
        assert cat == "frequent"
        assert val == 0.50

    def test_occasional(self):
        cat, val = _normalize_frequency("Occasional (29-5%)")
        assert cat == "occasional"
        assert val == 0.12

    def test_very_rare(self):
        cat, val = _normalize_frequency("Very rare (<4-1%)")
        assert cat == "very_rare"
        assert val == 0.02

    def test_excluded(self):
        cat, val = _normalize_frequency("Excluded (0%)")
        assert cat == "excluded"
        assert val == 0.00

    def test_unknown_string(self):
        cat, val = _normalize_frequency("SomethingWeird")
        assert cat == "unknown"
        assert val == 0.50

    def test_empty_string(self):
        cat, val = _normalize_frequency("")
        assert cat == "unknown"
        assert val == 0.50

    def test_whitespace_handling(self):
        cat, val = _normalize_frequency("  Obligate (100%)  ")
        assert cat == "obligate"


# --- Tests: XML parsing ---


class TestParseOrphanetXml:
    def test_parses_two_diseases(self, parsed_diseases):
        assert len(parsed_diseases) == 2

    def test_disease_ids(self, parsed_diseases):
        ids = {d.orphanet_id for d in parsed_diseases}
        assert ids == {"ORPHA:558", "ORPHA:730"}

    def test_disease_names(self, parsed_diseases):
        names = {d.disease_name for d in parsed_diseases}
        assert "Marfan syndrome" in names
        assert "Ehlers-Danlos syndrome" in names

    def test_marfan_has_three_annotations(self, parsed_diseases):
        marfan = next(d for d in parsed_diseases if d.orphanet_id == "ORPHA:558")
        assert len(marfan.annotations) == 3

    def test_eds_has_two_annotations(self, parsed_diseases):
        eds = next(d for d in parsed_diseases if d.orphanet_id == "ORPHA:730")
        assert len(eds.annotations) == 2

    def test_annotation_hpo_id(self, parsed_diseases):
        marfan = next(d for d in parsed_diseases if d.orphanet_id == "ORPHA:558")
        hpo_ids = {a.hpo_id for a in marfan.annotations}
        assert "HP:0001166" in hpo_ids
        assert "HP:0000256" in hpo_ids

    def test_annotation_frequency_category(self, parsed_diseases):
        marfan = next(d for d in parsed_diseases if d.orphanet_id == "ORPHA:558")
        arachno = next(a for a in marfan.annotations if a.hpo_id == "HP:0001166")
        assert arachno.frequency_category == "obligate"
        assert arachno.frequency_value == 1.00

    def test_annotation_occasional_frequency(self, parsed_diseases):
        marfan = next(d for d in parsed_diseases if d.orphanet_id == "ORPHA:558")
        macro = next(a for a in marfan.annotations if a.hpo_id == "HP:0000256")
        assert macro.frequency_category == "occasional"
        assert macro.frequency_value == 0.12

    def test_annotation_excluded_frequency(self, parsed_diseases):
        marfan = next(d for d in parsed_diseases if d.orphanet_id == "ORPHA:558")
        patellar = next(a for a in marfan.annotations if a.hpo_id == "HP:0002999")
        assert patellar.frequency_category == "excluded"
        assert patellar.frequency_value == 0.00

    def test_omim_crossref(self, parsed_diseases):
        marfan = next(d for d in parsed_diseases if d.orphanet_id == "ORPHA:558")
        assert "OMIM:154700" in marfan.omim_ids

    def test_no_crossref_when_absent(self, parsed_diseases):
        eds = next(d for d in parsed_diseases if d.orphanet_id == "ORPHA:730")
        assert eds.omim_ids == []

    def test_annotation_hpo_term_name(self, parsed_diseases):
        marfan = next(d for d in parsed_diseases if d.orphanet_id == "ORPHA:558")
        arachno = next(a for a in marfan.annotations if a.hpo_id == "HP:0001166")
        assert arachno.hpo_term_name == "Arachnodactyly"


class TestParseEmptyXml:
    def test_empty_disorder_list(self, tmp_path):
        xml = '<?xml version="1.0"?><JDBOR><DisorderList></DisorderList></JDBOR>'
        p = tmp_path / "empty.xml"
        p.write_text(xml)
        diseases = parse_orphanet_xml(p)
        assert diseases == []


# --- Tests: build_orphanet_hpo_map ---


class TestBuildOrphanetHpoMap:
    def test_excludes_excluded_by_default(self, parsed_diseases):
        disease_map = build_orphanet_hpo_map(parsed_diseases, exclude_excluded=True)
        # Marfan had 3 annotations but "excluded" should be omitted
        assert len(disease_map["ORPHA:558"]) == 2

    def test_includes_excluded_when_asked(self, parsed_diseases):
        disease_map = build_orphanet_hpo_map(parsed_diseases, exclude_excluded=False)
        assert len(disease_map["ORPHA:558"]) == 3

    def test_all_diseases_present(self, parsed_diseases):
        disease_map = build_orphanet_hpo_map(parsed_diseases)
        assert "ORPHA:558" in disease_map
        assert "ORPHA:730" in disease_map

    def test_annotation_dict_keys(self, parsed_diseases):
        disease_map = build_orphanet_hpo_map(parsed_diseases)
        ann = disease_map["ORPHA:558"][0]
        assert "hpo_id" in ann
        assert "hpo_term_name" in ann
        assert "frequency_category" in ann
        assert "frequency_value" in ann


# --- Tests: build_omim_crossref ---


class TestBuildOmimCrossref:
    def test_crossref_present(self, parsed_diseases):
        crossref = build_omim_crossref(parsed_diseases)
        assert "ORPHA:558" in crossref
        assert "OMIM:154700" in crossref["ORPHA:558"]

    def test_no_crossref_for_eds(self, parsed_diseases):
        crossref = build_omim_crossref(parsed_diseases)
        assert "ORPHA:730" not in crossref


# --- Tests: get_frequency_weights ---


class TestGetFrequencyWeights:
    def test_returns_weights(self, parsed_diseases):
        disease_map = build_orphanet_hpo_map(parsed_diseases)
        weights = get_frequency_weights(disease_map, "ORPHA:558")
        assert weights["HP:0001166"] == 1.00
        assert weights["HP:0000256"] == 0.12

    def test_missing_disease(self, parsed_diseases):
        disease_map = build_orphanet_hpo_map(parsed_diseases)
        weights = get_frequency_weights(disease_map, "ORPHA:999999")
        assert weights == {}


# --- Tests: parse_and_build (end-to-end) ---


class TestParseAndBuild:
    def test_saves_files(self, product4_xml, tmp_path):
        disease_map, crossref = parse_and_build(product4_xml, output_dir=tmp_path)

        hpo_file = tmp_path / "orphanet_disease_hpo.json"
        xref_file = tmp_path / "orphanet_omim_crossref.json"
        assert hpo_file.exists()
        assert xref_file.exists()

        # Verify JSON is valid and matches returned data
        with open(hpo_file) as f:
            loaded_map = json.load(f)
        assert loaded_map == disease_map

        with open(xref_file) as f:
            loaded_xref = json.load(f)
        assert loaded_xref == crossref

    def test_returned_data(self, product4_xml, tmp_path):
        disease_map, crossref = parse_and_build(product4_xml, output_dir=tmp_path)
        assert len(disease_map) == 2
        assert "ORPHA:558" in crossref
