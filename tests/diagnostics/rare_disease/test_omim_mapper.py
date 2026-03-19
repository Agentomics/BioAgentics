"""Tests for the OMIM disease-to-HPO mapper (phenotype.hpoa parser)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from bioagentics.diagnostics.rare_disease.omim_mapper import (
    build_disease_hpo_map,
    get_disease_hpo_terms,
    parse_frequency,
    parse_hpoa,
)

# Synthetic phenotype.hpoa content matching the real file format
SAMPLE_HPOA = textwrap.dedent("""\
    #description: HPO annotations
    #date: 2026-03-01
    #tracker: https://github.com/obophenotype/human-phenotype-ontology
    database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\taspect
    OMIM:100300\tAdams-Oliver syndrome 1\t\tHP:0001156\tPMID:12345\tPCS\t\tHP:0040281\t\t\tP
    OMIM:100300\tAdams-Oliver syndrome 1\t\tHP:0001163\tPMID:12345\tPCS\t\tHP:0040282\t\t\tP
    OMIM:100300\tAdams-Oliver syndrome 1\t\tHP:0000006\tPMID:12345\tPCS\t\t\t\t\tI
    OMIM:100300\tAdams-Oliver syndrome 1\tNOT\tHP:0009999\tPMID:12345\tPCS\t\t\t\t\tP
    OMIM:200100\tMicrocephaly\t\tHP:0000252\tPMID:67890\tTAS\tHP:0003577\tHP:0040280\t\t\tP
    OMIM:200100\tMicrocephaly\t\tHP:0001249\tPMID:67890\tIEA\t\t3/7\t\t\tP
    OMIM:200100\tMicrocephaly\t\tHP:0001250\tPMID:67890\tPCS\t\t45%\t\t\tP
    ORPHA:123456\tOrphanet disease\t\tHP:0000118\tORPHA:123456\tTAS\t\tHP:0040283\t\t\tP
""")


@pytest.fixture
def hpoa_file(tmp_path: Path) -> Path:
    p = tmp_path / "phenotype.hpoa"
    p.write_text(SAMPLE_HPOA)
    return p


class TestParseFrequency:
    def test_hpo_obligate(self):
        val, label = parse_frequency("HP:0040280")
        assert val == 1.0
        assert label == "obligate"

    def test_hpo_very_frequent(self):
        val, label = parse_frequency("HP:0040281")
        assert val == 0.90
        assert label == "very_frequent"

    def test_hpo_occasional(self):
        val, label = parse_frequency("HP:0040283")
        assert val == 0.12
        assert label == "occasional"

    def test_hpo_excluded(self):
        val, label = parse_frequency("HP:0040285")
        assert val == 0.0
        assert label == "excluded"

    def test_fraction(self):
        val, label = parse_frequency("3/7")
        assert abs(val - 3 / 7) < 0.01
        assert label == "3/7"

    def test_percentage(self):
        val, label = parse_frequency("45%")
        assert abs(val - 0.45) < 0.01

    def test_empty_returns_unknown(self):
        val, label = parse_frequency("")
        assert val == 0.50
        assert label == "unknown"


class TestParseHpoa:
    def test_parses_all_annotations(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        assert len(annotations) == 8

    def test_filters_by_prefix(self, hpoa_file: Path):
        omim_only = parse_hpoa(hpoa_file, disease_prefix="OMIM")
        assert len(omim_only) == 7
        assert all(a.database_id.startswith("OMIM:") for a in omim_only)

    def test_parses_disease_id(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        ids = {a.database_id for a in annotations}
        assert "OMIM:100300" in ids
        assert "OMIM:200100" in ids
        assert "ORPHA:123456" in ids

    def test_parses_hpo_id(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        first = annotations[0]
        assert first.hpo_id == "HP:0001156"

    def test_parses_evidence_code(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        evidence_codes = {a.evidence_code for a in annotations}
        assert "PCS" in evidence_codes
        assert "TAS" in evidence_codes
        assert "IEA" in evidence_codes

    def test_parses_frequency(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        # First annotation has HP:0040281 (very_frequent)
        first = annotations[0]
        assert first.frequency_value == 0.90
        assert first.frequency_label == "very_frequent"

    def test_parses_fraction_frequency(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        frac_ann = next(a for a in annotations if a.hpo_id == "HP:0001249")
        assert abs(frac_ann.frequency_value - 3 / 7) < 0.01

    def test_parses_percentage_frequency(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        pct_ann = next(a for a in annotations if a.hpo_id == "HP:0001250")
        assert abs(pct_ann.frequency_value - 0.45) < 0.01

    def test_parses_qualifier(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        not_ann = next(a for a in annotations if a.hpo_id == "HP:0009999")
        assert not_ann.qualifier == "NOT"

    def test_parses_onset(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        onset_ann = next(a for a in annotations if a.hpo_id == "HP:0000252")
        assert onset_ann.onset == "HP:0003577"

    def test_parses_aspect(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        # HP:0000006 is inheritance (aspect I)
        inh = next(a for a in annotations if a.hpo_id == "HP:0000006")
        assert inh.aspect == "I"


class TestBuildDiseaseMap:
    def test_correct_disease_count(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        disease_map = build_disease_hpo_map(annotations)
        # 3 diseases: OMIM:100300, OMIM:200100, ORPHA:123456
        # But OMIM:100300 has 1 P-annotation + 1 NOT (excluded) + 1 I (excluded)
        assert "OMIM:100300" in disease_map
        assert "OMIM:200100" in disease_map
        assert "ORPHA:123456" in disease_map

    def test_excludes_inheritance_aspect(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        disease_map = build_disease_hpo_map(annotations, aspect_filter="P")
        # OMIM:100300 should have 2 P terms (1156, 1163), NOT 0000006 (I aspect)
        terms = get_disease_hpo_terms(disease_map, "OMIM:100300")
        assert "HP:0000006" not in terms

    def test_excludes_negated_by_default(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        disease_map = build_disease_hpo_map(annotations)
        terms = get_disease_hpo_terms(disease_map, "OMIM:100300")
        assert "HP:0009999" not in terms

    def test_includes_negated_when_requested(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        disease_map = build_disease_hpo_map(annotations, exclude_negated=False)
        terms = get_disease_hpo_terms(disease_map, "OMIM:100300")
        assert "HP:0009999" in terms

    def test_frequency_preserved(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        disease_map = build_disease_hpo_map(annotations)
        ann = next(
            a for a in disease_map["OMIM:100300"]
            if a["hpo_id"] == "HP:0001156"
        )
        assert ann["frequency"] == 0.90
        assert ann["frequency_label"] == "very_frequent"

    def test_evidence_preserved(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        disease_map = build_disease_hpo_map(annotations)
        ann = next(
            a for a in disease_map["OMIM:200100"]
            if a["hpo_id"] == "HP:0001249"
        )
        assert ann["evidence"] == "IEA"


class TestGetDiseaseHpoTerms:
    def test_returns_correct_terms(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        disease_map = build_disease_hpo_map(annotations)
        terms = get_disease_hpo_terms(disease_map, "OMIM:200100")
        assert "HP:0000252" in terms
        assert "HP:0001249" in terms
        assert "HP:0001250" in terms

    def test_nonexistent_disease_returns_empty(self, hpoa_file: Path):
        annotations = parse_hpoa(hpoa_file)
        disease_map = build_disease_hpo_map(annotations)
        terms = get_disease_hpo_terms(disease_map, "OMIM:FAKE")
        assert terms == []
