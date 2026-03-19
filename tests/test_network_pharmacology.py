"""Tests for network pharmacology validation module."""

import pandas as pd
import pytest

from bioagentics.data.cd_fibrosis.network_pharmacology import (
    ALL_FIBROSIS_GENES,
    CROSS_ORGAN_FIBROSIS_GENES,
    FIBROSIS_PATHWAYS,
    compute_cross_organ_overlap,
    compute_network_score,
    compute_pathway_overlap,
    generate_network_report,
    score_druggability,
    validate_candidates,
    validate_compound,
)


class TestFibrosisPathways:
    def test_pathways_defined(self):
        assert len(FIBROSIS_PATHWAYS) >= 7

    def test_key_pathways_present(self):
        expected = ["TGF-beta", "Wnt", "YAP/TAZ", "TL1A-DR3", "JAK-STAT",
                     "ECM_remodeling", "epigenetic"]
        for p in expected:
            assert p in FIBROSIS_PATHWAYS, f"Missing pathway: {p}"

    def test_cross_organ_sets_defined(self):
        assert "IPF_lung" in CROSS_ORGAN_FIBROSIS_GENES
        assert "liver_fibrosis" in CROSS_ORGAN_FIBROSIS_GENES
        assert "kidney_fibrosis" in CROSS_ORGAN_FIBROSIS_GENES

    def test_all_fibrosis_genes_union(self):
        assert len(ALL_FIBROSIS_GENES) > 50
        # Key genes should be in the union
        for gene in ["TGFB1", "COL1A1", "ACTA2", "SERPINE1"]:
            assert gene in ALL_FIBROSIS_GENES


class TestPathwayOverlap:
    def test_full_overlap(self):
        targets = {"TGFB1", "TGFBR1", "SMAD2", "SMAD3"}
        result = compute_pathway_overlap(targets)
        assert result["TGF-beta"]["n_targets_in_pathway"] == 4
        assert result["TGF-beta"]["overlap_fraction"] > 0

    def test_no_overlap(self):
        targets = {"TP53", "BRCA1", "EGFR"}
        result = compute_pathway_overlap(targets)
        for pathway_result in result.values():
            assert pathway_result["n_targets_in_pathway"] == 0

    def test_partial_overlap(self):
        targets = {"TGFB1", "JAK1", "TP53"}  # 2 pathways
        result = compute_pathway_overlap(targets)
        assert result["TGF-beta"]["n_targets_in_pathway"] == 1
        assert result["JAK-STAT"]["n_targets_in_pathway"] == 1

    def test_empty_targets(self):
        result = compute_pathway_overlap(set())
        for pathway_result in result.values():
            assert pathway_result["n_targets_in_pathway"] == 0

    def test_overlapping_genes_returned(self):
        targets = {"TGFB1", "COL1A1"}
        result = compute_pathway_overlap(targets)
        assert "TGFB1" in result["TGF-beta"]["overlapping_genes"]


class TestCrossOrganOverlap:
    def test_shared_fibrosis_genes(self):
        # TGFB1 and COL1A1 are shared across all organs
        targets = {"TGFB1", "COL1A1"}
        result = compute_cross_organ_overlap(targets)
        for organ in ["IPF_lung", "liver_fibrosis", "kidney_fibrosis"]:
            assert result[organ]["n_targets_in_pathway"] >= 1

    def test_organ_specific_gene(self):
        targets = {"MUC5B"}  # IPF-specific
        result = compute_cross_organ_overlap(targets)
        assert result["IPF_lung"]["n_targets_in_pathway"] == 1
        assert result["liver_fibrosis"]["n_targets_in_pathway"] == 0


class TestDruggabilityScore:
    def test_approved_drug(self):
        drug = {"groups": "approved;investigational", "n_targets": 3}
        result = score_druggability(drug)
        assert result["druggability_score"] == 5
        assert result["approval_status"] == "approved"
        assert result["in_drugbank"] is True

    def test_investigational_drug(self):
        drug = {"groups": "investigational", "indication": "", "n_targets": 1}
        result = score_druggability(drug)
        assert result["druggability_score"] == 3
        assert result["approval_status"] == "investigational"

    def test_experimental_drug(self):
        drug = {"groups": "experimental", "n_targets": 0}
        result = score_druggability(drug)
        assert result["druggability_score"] == 2

    def test_unknown_drug(self):
        result = score_druggability(None)
        assert result["druggability_score"] == 0
        assert result["in_drugbank"] is False

    def test_n_targets_extracted(self):
        drug = {"groups": "approved", "n_targets": 5}
        result = score_druggability(drug)
        assert result["n_known_targets"] == 5


class TestNetworkScore:
    def test_perfect_score(self):
        pathway_overlap = {
            p: {"n_targets_in_pathway": 3, "overlap_fraction": 1.0}
            for p in FIBROSIS_PATHWAYS
        }
        cross_organ = {
            o: {"n_targets_in_pathway": 2}
            for o in CROSS_ORGAN_FIBROSIS_GENES
        }
        druggability = {"druggability_score": 5}

        score = compute_network_score(pathway_overlap, cross_organ, druggability)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_zero_score(self):
        pathway_overlap = {
            p: {"n_targets_in_pathway": 0, "overlap_fraction": 0.0}
            for p in FIBROSIS_PATHWAYS
        }
        cross_organ = {
            o: {"n_targets_in_pathway": 0}
            for o in CROSS_ORGAN_FIBROSIS_GENES
        }
        druggability = {"druggability_score": 0}

        score = compute_network_score(pathway_overlap, cross_organ, druggability)
        assert score == 0.0

    def test_partial_score(self):
        # Only approved drug, no pathway overlap
        pathway_overlap = {
            p: {"n_targets_in_pathway": 0, "overlap_fraction": 0.0}
            for p in FIBROSIS_PATHWAYS
        }
        cross_organ = {
            o: {"n_targets_in_pathway": 0}
            for o in CROSS_ORGAN_FIBROSIS_GENES
        }
        druggability = {"druggability_score": 5}

        score = compute_network_score(pathway_overlap, cross_organ, druggability)
        assert score == pytest.approx(0.20)  # only druggability component


class TestValidateCompound:
    def test_compound_with_fibrosis_targets(self):
        targets = {"TGFB1", "TGFBR1", "SMAD2", "SMAD3", "COL1A1"}
        drug_info = {"groups": "approved", "n_targets": 5}
        result = validate_compound("pirfenidone", targets, drug_info)

        assert result["compound"] == "pirfenidone"
        assert result["n_targets"] == 5
        assert result["n_fibrosis_targets"] >= 4
        assert result["n_pathways_hit"] >= 1
        assert result["network_score"] > 0

    def test_compound_without_targets(self):
        result = validate_compound("unknown-drug", set())
        assert result["n_targets"] == 0
        assert result["network_score"] == 0

    def test_non_fibrosis_targets(self):
        targets = {"TP53", "BRCA1", "EGFR", "ERBB2"}
        result = validate_compound("cancer-drug", targets)
        assert result["n_fibrosis_targets"] == 0
        assert result["n_pathways_hit"] == 0


class TestValidateCandidates:
    def _make_ranked_hits(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"compound": "pirfenidone", "mean_concordance": -0.85,
             "n_signatures_queried": 3, "convergent_anti_fibrotic": True},
            {"compound": "vorinostat", "mean_concordance": -0.75,
             "n_signatures_queried": 4, "convergent_anti_fibrotic": True},
            {"compound": "unknown-compound", "mean_concordance": -0.70,
             "n_signatures_queried": 2, "convergent_anti_fibrotic": False},
        ])

    def _make_drug_targets(self) -> dict[str, dict]:
        return {
            "pirfenidone": {
                "target_genes": "TGFB1;TGFBR1;SMAD2;SMAD3",
                "groups": "approved",
                "n_targets": 4,
            },
            "vorinostat": {
                "target_genes": "HDAC1;HDAC2;HDAC3;HDAC6",
                "groups": "approved",
                "n_targets": 4,
            },
        }

    def test_validates_all_candidates(self):
        result = validate_candidates(
            self._make_ranked_hits(),
            self._make_drug_targets(),
        )
        assert len(result) == 3

    def test_sorted_by_network_score(self):
        result = validate_candidates(
            self._make_ranked_hits(),
            self._make_drug_targets(),
        )
        scores = result["network_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_carries_cmap_scores(self):
        result = validate_candidates(
            self._make_ranked_hits(),
            self._make_drug_targets(),
        )
        assert "mean_concordance" in result.columns
        assert "convergent_anti_fibrotic" in result.columns

    def test_top_n_limit(self):
        result = validate_candidates(
            self._make_ranked_hits(),
            self._make_drug_targets(),
            top_n=2,
        )
        assert len(result) == 2

    def test_no_drugbank_data(self):
        result = validate_candidates(self._make_ranked_hits())
        assert len(result) == 3
        # All compounds should have 0 targets without DrugBank
        assert all(result["n_targets"] == 0)


class TestGenerateReport:
    def test_report_generation(self, tmp_path):
        ranked = pd.DataFrame([
            {"compound": "pirfenidone", "mean_concordance": -0.85,
             "n_signatures_queried": 3, "convergent_anti_fibrotic": True},
        ])
        drug_targets = {
            "pirfenidone": {
                "target_genes": "TGFB1;TGFBR1",
                "groups": "approved",
                "n_targets": 2,
            },
        }
        result = generate_network_report(
            ranked, drug_targets, output_dir=tmp_path,
        )
        assert len(result) == 1
        assert (tmp_path / "network_validation.tsv").exists()

    def test_empty_hits(self, tmp_path):
        ranked = pd.DataFrame(columns=[
            "compound", "mean_concordance",
        ])
        result = generate_network_report(ranked, output_dir=tmp_path)
        assert len(result) == 0
