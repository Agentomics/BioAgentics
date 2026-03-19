"""Tests for candidate prioritization and composite scoring module."""

import pandas as pd
import pytest

from bioagentics.data.cd_fibrosis.candidate_prioritization import (
    BENCHMARK_REFERENCE,
    CLINICAL_BENCHMARKS,
    COMPOUND_PK,
    CURATED_COMPOUNDS,
    PATHWAY_NOVELTY,
    SAFETY_SCORES,
    SCORE_WEIGHTS,
    SCORE_WEIGHTS_NO_CMAP,
    SIGNATURE_WEIGHTS,
    classify_pathway,
    compute_composite_score,
    generate_candidate_report,
    prioritize_candidates,
    score_clinical_benchmark,
    score_cmap_reversal,
    score_novelty,
    score_pharmacokinetics,
    score_safety,
)
from bioagentics.data.cd_fibrosis.positive_controls import POSITIVE_CONTROLS


# ── Data structure tests ──


class TestDataStructures:
    def test_score_weights_sum_to_one(self):
        assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-6

    def test_score_weights_no_cmap_sum_to_one(self):
        assert abs(sum(SCORE_WEIGHTS_NO_CMAP.values()) - 1.0) < 1e-6

    def test_score_weights_no_cmap_excludes_cmap(self):
        assert "cmap_reversal" not in SCORE_WEIGHTS_NO_CMAP

    def test_signature_weights_celltype_highest(self):
        assert SIGNATURE_WEIGHTS["celltype"] > SIGNATURE_WEIGHTS["bulk"]
        assert SIGNATURE_WEIGHTS["celltype"] >= max(SIGNATURE_WEIGHTS.values())

    def test_signature_weights_bulk_lowest(self):
        assert SIGNATURE_WEIGHTS["bulk"] <= min(SIGNATURE_WEIGHTS.values())

    def test_pathway_novelty_range(self):
        for pathway, score in PATHWAY_NOVELTY.items():
            assert 0 <= score <= 1, f"{pathway} novelty {score} out of range"

    def test_pathway_novelty_novel_is_highest(self):
        assert PATHWAY_NOVELTY["novel"] == 1.0

    def test_pathway_novelty_tgf_beta_is_lowest(self):
        assert PATHWAY_NOVELTY["TGF-beta"] == min(PATHWAY_NOVELTY.values())

    def test_safety_scores_approved_highest(self):
        assert SAFETY_SCORES["approved"] == 1.0

    def test_safety_scores_unknown_lowest(self):
        assert SAFETY_SCORES["unknown"] == min(SAFETY_SCORES.values())


class TestCuratedCompounds:
    def test_all_positive_controls_have_curated_data(self):
        for ctrl in POSITIVE_CONTROLS:
            assert ctrl.name in CURATED_COMPOUNDS, (
                f"Positive control {ctrl.name} missing from CURATED_COMPOUNDS"
            )

    def test_curated_compounds_have_targets(self):
        for name, data in CURATED_COMPOUNDS.items():
            assert "target_genes" in data, f"{name} missing target_genes"
            assert len(data["target_genes"]) > 0, f"{name} has empty target_genes"

    def test_curated_compounds_have_groups(self):
        for name, data in CURATED_COMPOUNDS.items():
            assert "groups" in data, f"{name} missing groups"
            assert data["groups"] in (
                "approved", "investigational", "experimental"
            ), f"{name} has unexpected group: {data['groups']}"

    def test_curated_compounds_have_indication(self):
        for name, data in CURATED_COMPOUNDS.items():
            assert "indication" in data, f"{name} missing indication"

    def test_compound_pk_coverage(self):
        for ctrl in POSITIVE_CONTROLS:
            assert ctrl.name in COMPOUND_PK, (
                f"Positive control {ctrl.name} missing from COMPOUND_PK"
            )


class TestClinicalBenchmarks:
    def test_benchmark_reference_exists(self):
        assert BENCHMARK_REFERENCE in CLINICAL_BENCHMARKS

    def test_duvakitug_55_percent(self):
        assert CLINICAL_BENCHMARKS["duvakitug"]["endoscopic_response"] == 0.55

    def test_benchmarks_have_required_fields(self):
        for name, data in CLINICAL_BENCHMARKS.items():
            assert "trial" in data, f"{name} missing trial"
            assert "endoscopic_response" in data, f"{name} missing response rate"
            assert 0 < data["endoscopic_response"] <= 1.0


# ── Scoring function tests ──


class TestScoreCmapReversal:
    def test_no_concordance_returns_zero(self):
        row = {"mean_concordance": None}
        assert score_cmap_reversal(row) == 0.0

    def test_strong_negative_concordance(self):
        row = {"mean_concordance": -0.9, "signatures_hit": "celltype;bulk"}
        score = score_cmap_reversal(row)
        assert score > 0.8

    def test_positive_concordance_low_score(self):
        row = {"mean_concordance": 0.8, "signatures_hit": "bulk"}
        score = score_cmap_reversal(row)
        assert score < 0.2

    def test_zero_concordance_midpoint(self):
        row = {"mean_concordance": 0.0, "signatures_hit": ""}
        score = score_cmap_reversal(row)
        assert 0.4 <= score <= 0.6

    def test_celltype_weighted_higher_than_bulk(self):
        row_celltype = {"mean_concordance": -0.5, "signatures_hit": "celltype"}
        row_bulk = {"mean_concordance": -0.5, "signatures_hit": "bulk"}
        assert score_cmap_reversal(row_celltype) > score_cmap_reversal(row_bulk)

    def test_convergent_bonus(self):
        row_no = {
            "mean_concordance": -0.5,
            "signatures_hit": "bulk;celltype",
            "convergent_anti_fibrotic": False,
        }
        row_yes = {
            "mean_concordance": -0.5,
            "signatures_hit": "bulk;celltype",
            "convergent_anti_fibrotic": True,
        }
        assert score_cmap_reversal(row_yes) >= score_cmap_reversal(row_no)

    def test_score_capped_at_one(self):
        row = {
            "mean_concordance": -1.0,
            "signatures_hit": "celltype;cthrc1_yaptaz;glis3_il11",
            "convergent_anti_fibrotic": True,
        }
        assert score_cmap_reversal(row) <= 1.0

    def test_score_never_negative(self):
        row = {"mean_concordance": 1.0, "signatures_hit": "bulk"}
        assert score_cmap_reversal(row) >= 0.0


class TestScoreSafety:
    def test_approved_drug(self):
        info = {"groups": "approved", "indication": "Approved for X"}
        assert score_safety(info) == 1.0

    def test_phase_3_drug(self):
        info = {"groups": "investigational", "indication": "Phase 3 for UC"}
        assert score_safety(info) == SAFETY_SCORES["phase_3"]

    def test_investigational_drug(self):
        info = {"groups": "investigational", "indication": "Phase 2 for CD"}
        assert score_safety(info) == SAFETY_SCORES["investigational"]

    def test_experimental_drug(self):
        info = {"groups": "experimental", "indication": "Research tool"}
        assert score_safety(info) == SAFETY_SCORES["experimental"]

    def test_unknown_drug(self):
        assert score_safety(None) == SAFETY_SCORES["unknown"]

    def test_curated_pirfenidone_approved(self):
        info = CURATED_COMPOUNDS["pirfenidone"]
        assert score_safety(info) == 1.0

    def test_curated_trichostatin_experimental(self):
        info = CURATED_COMPOUNDS["trichostatin-a"]
        assert score_safety(info) == SAFETY_SCORES["experimental"]


class TestScoreNovelty:
    def test_novel_pathway_highest(self):
        assert score_novelty("novel") == 1.0

    def test_tgf_beta_lowest(self):
        assert score_novelty("TGF-beta") == PATHWAY_NOVELTY["TGF-beta"]

    def test_unknown_pathway_defaults_novel(self):
        assert score_novelty("unknown_pathway_xyz") == 1.0

    def test_yaptaz_high_novelty(self):
        assert score_novelty("YAP/TAZ") >= 0.7


class TestScorePharmacokinetics:
    def test_oral_small_molecule_gut_relevant(self):
        # Pirfenidone: oral + small_molecule + gut_relevant
        score = score_pharmacokinetics("pirfenidone")
        assert score == 1.0

    def test_iv_biologic_not_gut(self):
        # daratumumab: iv + biologic + not gut_relevant
        score = score_pharmacokinetics("daratumumab")
        assert score < 0.5

    def test_unknown_compound(self):
        score = score_pharmacokinetics("totally_unknown_compound")
        assert score == 0.5

    def test_score_range(self):
        for compound in COMPOUND_PK:
            score = score_pharmacokinetics(compound)
            assert 0 <= score <= 1, f"{compound} PK score {score} out of range"


class TestScoreClinicalBenchmark:
    def test_duvakitug_meets_benchmark(self):
        score = score_clinical_benchmark("duvakitug")
        assert score == 1.0  # reference compound equals itself

    def test_unknown_compound_neutral(self):
        score = score_clinical_benchmark("some_unknown_drug")
        assert score == 0.5

    def test_upadacitinib_below_benchmark(self):
        score = score_clinical_benchmark("upadacitinib")
        assert score < 1.0
        assert score > 0

    def test_score_range(self):
        for compound in CLINICAL_BENCHMARKS:
            score = score_clinical_benchmark(compound)
            assert 0 <= score <= 1.0


class TestClassifyPathway:
    def test_known_positive_control_pirfenidone(self):
        assert classify_pathway("pirfenidone") == "TGF-beta"

    def test_known_positive_control_duvakitug(self):
        assert classify_pathway("duvakitug") == "TL1A-DR3"

    def test_known_positive_control_vorinostat(self):
        assert classify_pathway("vorinostat") == "epigenetic"

    def test_known_positive_control_upadacitinib(self):
        assert classify_pathway("upadacitinib") == "JAK-STAT"

    def test_known_positive_control_obefazimod(self):
        assert classify_pathway("obefazimod") == "miR-124"

    def test_known_positive_control_daratumumab(self):
        assert classify_pathway("daratumumab") == "CD38/NAD+"

    def test_unknown_compound_with_pathway_overlap(self):
        overlap = {
            "TGF-beta": {"n_targets_in_pathway": 3, "overlap_fraction": 0.2},
            "Wnt": {"n_targets_in_pathway": 0, "overlap_fraction": 0.0},
        }
        assert classify_pathway("unknown_drug", overlap) == "TGF-beta"

    def test_unknown_compound_no_overlap(self):
        overlap = {
            "TGF-beta": {"n_targets_in_pathway": 0, "overlap_fraction": 0.0},
        }
        assert classify_pathway("unknown_drug", overlap) == "novel"

    def test_unknown_compound_no_pathway_data(self):
        assert classify_pathway("unknown_drug") == "novel"


class TestComputeCompositeScore:
    def test_with_all_components(self):
        score = compute_composite_score(
            cmap_score=0.8,
            network_score=0.6,
            safety_score=1.0,
            novelty_score=0.7,
            pk_score=0.9,
        )
        assert 0 < score <= 1.0

    def test_without_cmap(self):
        score = compute_composite_score(
            cmap_score=None,
            network_score=0.6,
            safety_score=1.0,
            novelty_score=0.7,
            pk_score=0.9,
        )
        assert 0 < score <= 1.0

    def test_no_cmap_uses_different_weights(self):
        # Without CMAP, network weight is 0.35 (not 0.25)
        assert SCORE_WEIGHTS_NO_CMAP["network_pharmacology"] > SCORE_WEIGHTS["network_pharmacology"]
        # Verify: network-only scores differ between modes
        score_without = compute_composite_score(None, 1.0, 0.0, 0.0, 0.0)
        assert abs(score_without - SCORE_WEIGHTS_NO_CMAP["network_pharmacology"]) < 1e-6

    def test_perfect_scores(self):
        score = compute_composite_score(1.0, 1.0, 1.0, 1.0, 1.0)
        assert abs(score - 1.0) < 1e-6

    def test_zero_scores(self):
        score = compute_composite_score(None, 0.0, 0.0, 0.0, 0.0)
        assert score == 0.0

    def test_custom_weights(self):
        custom = {"network_pharmacology": 1.0}
        score = compute_composite_score(
            None, 0.5, 1.0, 1.0, 1.0, weights=custom
        )
        assert abs(score - 0.5) < 1e-6


# ── Integration tests ──


class TestPrioritizeCandidates:
    def test_without_cmap_uses_positive_controls(self):
        result = prioritize_candidates(ranked_hits=None, top_n=20)
        assert len(result) > 0
        assert len(result) <= 20

    def test_result_has_required_columns(self):
        result = prioritize_candidates(ranked_hits=None, top_n=5)
        required = [
            "compound", "composite_score", "pathway_class",
            "network_score", "safety_score", "novelty_score",
            "pk_score", "clinical_benchmark", "approval_status",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"

    def test_sorted_by_composite_score(self):
        result = prioritize_candidates(ranked_hits=None, top_n=20)
        scores = result["composite_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_all_scores_in_range(self):
        result = prioritize_candidates(ranked_hits=None, top_n=20)
        for col in ["composite_score", "network_score", "safety_score",
                     "novelty_score", "pk_score", "clinical_benchmark"]:
            for val in result[col]:
                assert 0 <= val <= 1.0, f"{col} value {val} out of range"

    def test_pathway_classes_assigned(self):
        result = prioritize_candidates(ranked_hits=None, top_n=20)
        assert all(result["pathway_class"].notna())
        assert all(len(str(p)) > 0 for p in result["pathway_class"])

    def test_approval_statuses_resolved(self):
        result = prioritize_candidates(ranked_hits=None, top_n=20)
        # With curated data, no compound should be "unknown"
        assert not any(result["approval_status"] == "unknown")

    def test_top_n_limits_output(self):
        result = prioritize_candidates(ranked_hits=None, top_n=3)
        assert len(result) <= 3

    def test_with_cmap_data(self):
        cmap_hits = pd.DataFrame([
            {
                "compound": "pirfenidone",
                "mean_concordance": -0.7,
                "n_signatures_queried": 3,
                "convergent_anti_fibrotic": True,
                "signatures_hit": "celltype;bulk;transition",
            },
            {
                "compound": "vorinostat",
                "mean_concordance": -0.5,
                "n_signatures_queried": 2,
                "convergent_anti_fibrotic": True,
                "signatures_hit": "celltype;bulk",
            },
            {
                "compound": "novel_compound_x",
                "mean_concordance": -0.9,
                "n_signatures_queried": 4,
                "convergent_anti_fibrotic": True,
                "signatures_hit": "celltype;bulk;transition;cthrc1_yaptaz",
            },
        ])
        result = prioritize_candidates(cmap_hits, top_n=10)
        assert len(result) == 3
        # All should have CMAP reversal scores
        known_compounds = result[result["compound"] != "novel_compound_x"]
        assert all(known_compounds["cmap_reversal_score"].notna())

    def test_with_drugbank_overrides_curated(self):
        drug_targets = {
            "pirfenidone": {
                "target_genes": "TGFB1;TGFB2;TGFBR1;TGFBR2;SMAD2;SMAD3",
                "groups": "approved",
                "indication": "Approved for IPF",
            }
        }
        result = prioritize_candidates(
            ranked_hits=None, drug_targets=drug_targets, top_n=20
        )
        pir = result[result["compound"] == "pirfenidone"]
        assert len(pir) == 1
        # DrugBank record has more TGF-beta targets → higher network score
        assert pir.iloc[0]["n_targets"] == 6


class TestGenerateCandidateReport:
    def test_without_cmap(self, capsys):
        result = generate_candidate_report(top_n=10)
        assert len(result) > 0
        captured = capsys.readouterr()
        assert "Candidate Prioritization Report" in captured.out
        assert "positive control seed" in captured.out

    def test_with_cmap(self, capsys):
        cmap_hits = pd.DataFrame([
            {
                "compound": "pirfenidone",
                "mean_concordance": -0.6,
                "n_signatures_queried": 2,
                "convergent_anti_fibrotic": True,
                "signatures_hit": "celltype;bulk",
            },
        ])
        result = generate_candidate_report(ranked_hits=cmap_hits, top_n=5)
        assert len(result) == 1
        captured = capsys.readouterr()
        assert "1 CMAP/iLINCS hits" in captured.out

    def test_saves_tsv(self, tmp_path):
        result = generate_candidate_report(
            output_dir=tmp_path, top_n=5
        )
        tsv_path = tmp_path / "final_candidates.tsv"
        assert tsv_path.exists()
        loaded = pd.read_csv(tsv_path, sep="\t")
        assert len(loaded) == len(result)

    def test_success_criteria_reported(self, capsys):
        generate_candidate_report(top_n=15)
        captured = capsys.readouterr()
        assert "Success Criteria" in captured.out
        assert "approved/Phase3 safety + novel mechanism" in captured.out


class TestSuccessCriteria:
    """Verify the plan's success criteria are achievable with curated data."""

    def test_at_least_five_approved_with_novel_mechanism(self):
        result = prioritize_candidates(ranked_hits=None, top_n=20)
        approved_novel = result[
            (result["approval_status"].isin(["approved", "phase_3"]))
            & (result["novelty_score"] >= 0.5)
        ]
        assert len(approved_novel) >= 5

    def test_multiple_pathway_classes(self):
        result = prioritize_candidates(ranked_hits=None, top_n=20)
        n_pathways = result["pathway_class"].nunique()
        assert n_pathways >= 4

    def test_clinical_benchmark_included(self):
        result = prioritize_candidates(ranked_hits=None, top_n=20)
        duva = result[result["compound"] == "duvakitug"]
        assert len(duva) == 1
        assert duva.iloc[0]["clinical_benchmark"] == 1.0
