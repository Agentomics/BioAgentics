"""Tests for the simulated patient profile generator."""

from __future__ import annotations

import networkx as nx
import pytest

from bioagentics.diagnostics.rare_disease.patient_simulator import (
    SimulationConfig,
    generate_benchmark_cases,
    simulate_patient,
)


def _build_test_dag() -> nx.DiGraph:
    """Build a minimal HPO DAG for testing."""
    g = nx.DiGraph()
    terms = {
        "HP:0000001": "All",
        "HP:0000118": "Phenotypic abnormality",
        "HP:0000707": "Neuro",
        "HP:0000152": "Head/neck",
        "HP:0000234": "Head",
        "HP:0002011": "CNS",
        "HP:0012443": "Brain",
        "HP:0001250": "Seizure",
        "HP:0001249": "Intellectual disability",
        "HP:0000252": "Microcephaly",
    }
    for tid, name in terms.items():
        g.add_node(tid, name=name, node_type="phenotype")

    edges = [
        ("HP:0000118", "HP:0000001"),
        ("HP:0000707", "HP:0000118"),
        ("HP:0000152", "HP:0000118"),
        ("HP:0000234", "HP:0000152"),
        ("HP:0002011", "HP:0000707"),
        ("HP:0012443", "HP:0002011"),
        ("HP:0001250", "HP:0000707"),
        ("HP:0001249", "HP:0000707"),
        ("HP:0000252", "HP:0000234"),
    ]
    for child, parent in edges:
        g.add_edge(child, parent, relation="is_a")
    return g


DISEASE_ANNOTATIONS = {
    "OMIM:100001": ["HP:0000707", "HP:0002011", "HP:0012443", "HP:0001250", "HP:0001249"],
    "OMIM:100002": ["HP:0000234", "HP:0000252", "HP:0001249", "HP:0001250"],
    "OMIM:100003": ["HP:0000152", "HP:0000234"],  # too few terms (< 3)
}


@pytest.fixture
def hpo_dag():
    return _build_test_dag()


class TestSimulatePatient:
    def test_completeness_controls_size(self, hpo_dag):
        import random
        rng = random.Random(42)
        terms = DISEASE_ANNOTATIONS["OMIM:100001"]
        all_terms = list(hpo_dag.nodes())

        profile_80 = simulate_patient(terms, all_terms, 0.8, 0.0, rng)
        rng = random.Random(42)
        profile_20 = simulate_patient(terms, all_terms, 0.2, 0.0, rng)

        assert len(profile_80) > len(profile_20)

    def test_no_noise_only_disease_terms(self, hpo_dag):
        import random
        rng = random.Random(42)
        terms = DISEASE_ANNOTATIONS["OMIM:100001"]
        all_terms = list(hpo_dag.nodes())

        profile = simulate_patient(terms, all_terms, 0.8, 0.0, rng)
        disease_set = set(terms)
        for t in profile:
            assert t in disease_set

    def test_noise_adds_non_disease_terms(self, hpo_dag):
        import random
        rng = random.Random(42)
        terms = DISEASE_ANNOTATIONS["OMIM:100001"]
        all_terms = list(hpo_dag.nodes())

        profile = simulate_patient(terms, all_terms, 0.8, 0.5, rng)
        disease_set = set(terms)
        noise_terms = [t for t in profile if t not in disease_set]
        assert len(noise_terms) > 0

    def test_noise_terms_are_valid_hpo(self, hpo_dag):
        import random
        rng = random.Random(42)
        terms = DISEASE_ANNOTATIONS["OMIM:100001"]
        all_terms = list(hpo_dag.nodes())

        profile = simulate_patient(terms, all_terms, 0.6, 0.2, rng)
        for t in profile:
            assert t in hpo_dag

    def test_minimum_one_term(self, hpo_dag):
        import random
        rng = random.Random(42)
        all_terms = list(hpo_dag.nodes())
        # Even at very low completeness, at least 1 term
        profile = simulate_patient(["HP:0000707"], all_terms, 0.1, 0.0, rng)
        assert len(profile) >= 1

    def test_deterministic_with_same_seed(self, hpo_dag):
        import random
        terms = DISEASE_ANNOTATIONS["OMIM:100001"]
        all_terms = list(hpo_dag.nodes())

        rng1 = random.Random(123)
        profile1 = simulate_patient(terms, all_terms, 0.6, 0.1, rng1)

        rng2 = random.Random(123)
        profile2 = simulate_patient(terms, all_terms, 0.6, 0.1, rng2)

        assert profile1 == profile2


class TestGenerateBenchmarkCases:
    def test_skips_diseases_with_few_terms(self, hpo_dag):
        config = SimulationConfig(
            completeness_levels=[0.5],
            noise_levels=[0.0],
            min_disease_terms=3,
        )
        cases = generate_benchmark_cases(DISEASE_ANNOTATIONS, hpo_dag, config)
        disease_ids = {c.true_disease_id for c in cases}
        assert "OMIM:100003" not in disease_ids  # only 2 terms

    def test_correct_case_count(self, hpo_dag):
        config = SimulationConfig(
            completeness_levels=[0.4, 0.8],
            noise_levels=[0.0, 0.1],
            cases_per_combination=2,
            min_disease_terms=3,
        )
        cases = generate_benchmark_cases(DISEASE_ANNOTATIONS, hpo_dag, config)
        # 2 eligible diseases x 2 completeness x 2 noise x 2 per = 16
        assert len(cases) == 16

    def test_case_has_correct_true_disease(self, hpo_dag):
        config = SimulationConfig(
            completeness_levels=[0.5],
            noise_levels=[0.0],
        )
        cases = generate_benchmark_cases(DISEASE_ANNOTATIONS, hpo_dag, config)
        for case in cases:
            assert case.true_disease_id in DISEASE_ANNOTATIONS

    def test_case_metadata(self, hpo_dag):
        config = SimulationConfig(
            completeness_levels=[0.4],
            noise_levels=[0.1],
        )
        cases = generate_benchmark_cases(DISEASE_ANNOTATIONS, hpo_dag, config)
        for case in cases:
            assert case.completeness == 0.4
            assert case.noise_level == 0.1

    def test_query_terms_are_valid(self, hpo_dag):
        config = SimulationConfig(
            completeness_levels=[0.6],
            noise_levels=[0.1],
        )
        cases = generate_benchmark_cases(DISEASE_ANNOTATIONS, hpo_dag, config)
        for case in cases:
            for t in case.query_hpo_terms:
                assert t in hpo_dag

    def test_reproducible(self, hpo_dag):
        config1 = SimulationConfig(completeness_levels=[0.5], noise_levels=[0.1], seed=99)
        config2 = SimulationConfig(completeness_levels=[0.5], noise_levels=[0.1], seed=99)
        cases1 = generate_benchmark_cases(DISEASE_ANNOTATIONS, hpo_dag, config1)
        cases2 = generate_benchmark_cases(DISEASE_ANNOTATIONS, hpo_dag, config2)
        for c1, c2 in zip(cases1, cases2):
            assert c1.query_hpo_terms == c2.query_hpo_terms
