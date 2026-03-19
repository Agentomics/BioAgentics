"""Tests for the phenopacket store loader."""

from __future__ import annotations

import json
from pathlib import Path

from bioagentics.diagnostics.rare_disease.phenopacket_loader import (
    extract_disease_id,
    extract_hpo_terms,
    iter_phenopackets,
    load_all_phenopackets,
    load_phenopacket,
)


def _make_phenopacket(
    case_id: str = "test_case",
    hpo_terms: list[dict] | None = None,
    disease_id: str = "OMIM:100001",
    disease_label: str = "Test Disease",
    use_interpretation: bool = True,
    excluded_terms: list[str] | None = None,
) -> dict:
    """Build a minimal phenopacket dict for testing."""
    features = []
    if hpo_terms is None:
        hpo_terms = [
            {"id": "HP:0001250", "label": "Seizures"},
            {"id": "HP:0001249", "label": "Intellectual disability"},
            {"id": "HP:0000252", "label": "Microcephaly"},
        ]
    for term in hpo_terms:
        features.append({"type": term})

    if excluded_terms:
        for term_id in excluded_terms:
            features.append({
                "type": {"id": term_id, "label": "Excluded term"},
                "excluded": True,
            })

    pkt = {
        "id": case_id,
        "subject": {"id": "patient_1"},
        "phenotypicFeatures": features,
    }

    if use_interpretation:
        pkt["interpretations"] = [{
            "id": "interp_1",
            "progressStatus": "SOLVED",
            "diagnosis": {
                "disease": {"id": disease_id, "label": disease_label},
            },
        }]
    else:
        pkt["diseases"] = [{
            "term": {"id": disease_id, "label": disease_label},
        }]

    return pkt


def _write_phenopacket(tmp_path: Path, pkt: dict, subdir: str = "GENE1") -> Path:
    """Write a phenopacket dict to a JSON file in a gene subdirectory."""
    gene_dir = tmp_path / subdir
    gene_dir.mkdir(parents=True, exist_ok=True)
    path = gene_dir / f"{pkt['id']}.json"
    with open(path, "w") as f:
        json.dump(pkt, f)
    return path


class TestExtractHPOTerms:
    def test_extracts_observed_terms(self):
        pkt = _make_phenopacket(hpo_terms=[
            {"id": "HP:0001250", "label": "Seizures"},
            {"id": "HP:0000252", "label": "Microcephaly"},
        ])
        terms = extract_hpo_terms(pkt)
        assert terms == ["HP:0001250", "HP:0000252"]

    def test_skips_excluded_terms(self):
        pkt = _make_phenopacket(
            hpo_terms=[{"id": "HP:0001250", "label": "Seizures"}],
            excluded_terms=["HP:0000319"],
        )
        terms = extract_hpo_terms(pkt)
        assert "HP:0000319" not in terms
        assert terms == ["HP:0001250"]

    def test_empty_features(self):
        pkt = {"phenotypicFeatures": []}
        assert extract_hpo_terms(pkt) == []

    def test_no_features_key(self):
        pkt = {}
        assert extract_hpo_terms(pkt) == []

    def test_filters_non_hpo_terms(self):
        pkt = _make_phenopacket(hpo_terms=[
            {"id": "HP:0001250", "label": "Seizures"},
            {"id": "MONDO:0001", "label": "Not HPO"},
        ])
        terms = extract_hpo_terms(pkt)
        assert terms == ["HP:0001250"]


class TestExtractDiseaseId:
    def test_from_interpretation(self):
        pkt = _make_phenopacket(disease_id="OMIM:123456", use_interpretation=True)
        assert extract_disease_id(pkt) == "OMIM:123456"

    def test_from_diseases_fallback(self):
        pkt = _make_phenopacket(disease_id="OMIM:654321", use_interpretation=False)
        assert extract_disease_id(pkt) == "OMIM:654321"

    def test_interpretation_preferred_over_diseases(self):
        pkt = _make_phenopacket(disease_id="OMIM:111", use_interpretation=True)
        pkt["diseases"] = [{"term": {"id": "OMIM:222", "label": "Other"}}]
        assert extract_disease_id(pkt) == "OMIM:111"

    def test_no_diagnosis(self):
        pkt = {"id": "test", "phenotypicFeatures": []}
        assert extract_disease_id(pkt) is None

    def test_empty_interpretation(self):
        pkt = {"interpretations": [{"diagnosis": {"disease": {}}}]}
        assert extract_disease_id(pkt) is None


class TestLoadPhenopacket:
    def test_loads_valid_file(self, tmp_path: Path):
        pkt = _make_phenopacket()
        path = _write_phenopacket(tmp_path, pkt)
        case = load_phenopacket(path)
        assert case is not None
        assert case.case_id == "test_case"
        assert case.true_disease_id == "OMIM:100001"
        assert len(case.query_hpo_terms) == 3

    def test_returns_none_for_no_diagnosis(self, tmp_path: Path):
        pkt = {"id": "no_diag", "phenotypicFeatures": [
            {"type": {"id": "HP:0001250", "label": "Seizures"}}
        ]}
        path = _write_phenopacket(tmp_path, pkt)
        assert load_phenopacket(path) is None

    def test_returns_none_for_no_phenotypes(self, tmp_path: Path):
        pkt = _make_phenopacket(hpo_terms=[], excluded_terms=["HP:0000319"])
        path = _write_phenopacket(tmp_path, pkt)
        assert load_phenopacket(path) is None

    def test_returns_none_for_invalid_json(self, tmp_path: Path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json{{{")
        assert load_phenopacket(bad_file) is None

    def test_metadata_includes_source(self, tmp_path: Path):
        pkt = _make_phenopacket()
        path = _write_phenopacket(tmp_path, pkt, subdir="ESAM")
        case = load_phenopacket(path)
        assert case is not None
        assert case.metadata["source"] == "phenopacket_store"
        assert case.metadata["gene_dir"] == "ESAM"


class TestIterPhenopackets:
    def test_iterates_valid_files(self, tmp_path: Path):
        for i in range(3):
            pkt = _make_phenopacket(case_id=f"case_{i}", disease_id=f"OMIM:{i}")
            _write_phenopacket(tmp_path, pkt, subdir=f"GENE{i}")

        cases = list(iter_phenopackets(base_dir=tmp_path))
        assert len(cases) == 3

    def test_skips_invalid_files(self, tmp_path: Path):
        # One valid
        pkt = _make_phenopacket(case_id="valid")
        _write_phenopacket(tmp_path, pkt, subdir="G1")

        # One invalid (no diagnosis)
        bad = {"id": "bad", "phenotypicFeatures": [
            {"type": {"id": "HP:0001250", "label": "X"}}
        ]}
        _write_phenopacket(tmp_path, bad, subdir="G2")

        cases = list(iter_phenopackets(base_dir=tmp_path))
        assert len(cases) == 1

    def test_min_hpo_terms_filter(self, tmp_path: Path):
        pkt1 = _make_phenopacket(
            case_id="many", hpo_terms=[
                {"id": f"HP:000{i}", "label": f"T{i}"} for i in range(5)
            ]
        )
        pkt2 = _make_phenopacket(
            case_id="few", disease_id="OMIM:999",
            hpo_terms=[{"id": "HP:0001250", "label": "X"}],
        )
        _write_phenopacket(tmp_path, pkt1, subdir="G1")
        _write_phenopacket(tmp_path, pkt2, subdir="G2")

        cases = list(iter_phenopackets(base_dir=tmp_path, min_hpo_terms=3))
        assert len(cases) == 1
        assert cases[0].case_id == "many"

    def test_nonexistent_dir(self, tmp_path: Path):
        cases = list(iter_phenopackets(base_dir=tmp_path / "nonexistent"))
        assert len(cases) == 0


class TestLoadAllPhenopackets:
    def test_loads_all_and_summary(self, tmp_path: Path):
        for i in range(5):
            pkt = _make_phenopacket(case_id=f"case_{i}", disease_id=f"OMIM:{i}")
            _write_phenopacket(tmp_path, pkt, subdir=f"GENE{i}")

        cases, summary = load_all_phenopackets(base_dir=tmp_path)
        assert len(cases) == 5
        assert summary.total_files == 5
        assert summary.loaded == 5
        assert summary.unique_diseases == 5

    def test_counts_skips(self, tmp_path: Path):
        # Valid
        pkt = _make_phenopacket(case_id="ok")
        _write_phenopacket(tmp_path, pkt, subdir="G1")

        # No diagnosis
        bad1 = {"id": "no_diag", "phenotypicFeatures": [
            {"type": {"id": "HP:0001250", "label": "X"}}
        ]}
        _write_phenopacket(tmp_path, bad1, subdir="G2")

        # No phenotypes
        bad2 = _make_phenopacket(case_id="no_pheno", hpo_terms=[])
        _write_phenopacket(tmp_path, bad2, subdir="G3")

        # Parse error
        err_dir = tmp_path / "G4"
        err_dir.mkdir()
        (err_dir / "bad.json").write_text("{invalid")

        cases, summary = load_all_phenopackets(base_dir=tmp_path)
        assert summary.loaded == 1
        assert summary.skipped_no_diagnosis == 1
        assert summary.skipped_no_phenotypes == 1
        assert summary.skipped_parse_error == 1

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path):
        cases, summary = load_all_phenopackets(base_dir=tmp_path / "nope")
        assert len(cases) == 0
        assert summary.total_files == 0

    def test_shared_disease_counted_once(self, tmp_path: Path):
        for i in range(3):
            pkt = _make_phenopacket(
                case_id=f"case_{i}",
                disease_id="OMIM:100001",  # same disease
            )
            _write_phenopacket(tmp_path, pkt, subdir=f"G{i}")

        cases, summary = load_all_phenopackets(base_dir=tmp_path)
        assert len(cases) == 3
        assert summary.unique_diseases == 1
