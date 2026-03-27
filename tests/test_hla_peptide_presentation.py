"""Tests for HLA peptide presentation modeling pipeline."""

import csv
import json

import pytest


class TestHLAAllelePanel:
    """Tests for Phase 1: HLA allele panel."""

    def test_panel_has_expected_alleles(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.hla_allele_panel import HLA_PANEL

        assert len(HLA_PANEL) == 9
        names = {a.allele_name for a in HLA_PANEL}
        assert "HLA-DRB1*07:01" in names
        assert "HLA-A*02:01" in names

    def test_allele_categories(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.hla_allele_panel import (
            get_alleles_by_category,
        )

        assert len(get_alleles_by_category("susceptibility")) == 3
        assert len(get_alleles_by_category("protective")) == 2
        assert len(get_alleles_by_category("control")) == 2
        assert len(get_alleles_by_category("exploratory")) == 2

    def test_mhc_class_split(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.hla_allele_panel import (
            get_mhc_i_alleles,
            get_mhc_ii_alleles,
        )

        assert len(get_mhc_ii_alleles()) == 7
        assert len(get_mhc_i_alleles()) == 2

    def test_export_panel_json(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.hla_allele_panel import export_panel_json

        out = tmp_path / "panel.json"
        export_panel_json(out)

        data = json.loads(out.read_text())
        assert len(data["alleles"]) == 9
        assert "allele_groups" in data
        assert "susceptibility" in data["allele_groups"]

    def test_invalid_category_raises(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.hla_allele_panel import (
            get_alleles_by_category,
        )

        with pytest.raises(ValueError, match="Unknown category"):
            get_alleles_by_category("invalid")


class TestGASProteomeDigest:
    """Tests for Phase 2: Peptide digestion."""

    def test_parse_fasta(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.gas_proteome_digest import (
            parse_fasta_streaming,
        )

        fasta = tmp_path / "test.fasta"
        fasta.write_text(
            ">sp|P0001|TEST_STRPY Test protein OS=GAS\n"
            "MKTLLILAVLCLGFASSALA\n"
            ">sp|P0002|SLO_STRPY Streptolysin O OS=GAS\n"
            "ACDEFGHIKLMNPQRSTVWY\n"
        )

        proteins = list(parse_fasta_streaming(fasta))
        assert len(proteins) == 2
        assert proteins[0].accession == "P0001"
        assert proteins[0].sequence == "MKTLLILAVLCLGFASSALA"
        assert not proteins[0].is_virulence_factor
        assert proteins[1].is_virulence_factor  # "streptolysin" in description

    def test_generate_peptides(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.gas_proteome_digest import (
            generate_peptides,
        )

        seq = "ABCDEFGHIJ"  # 10 residues
        peps_9 = generate_peptides(seq, 9)
        assert len(peps_9) == 2  # positions 0-1
        assert peps_9[0] == "ABCDEFGHI"
        assert peps_9[1] == "BCDEFGHIJ"

    def test_digest_serotype(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.gas_proteome_digest import (
            digest_serotype,
        )

        fasta = tmp_path / "test.fasta"
        fasta.write_text(
            ">sp|P0001|TEST Test protein OS=GAS\n"
            "ACDEFGHIKLMNPQRSTVWY"  # 20 residues
        )

        out_dir = tmp_path / "peptides"
        stats = digest_serotype("TEST", fasta, out_dir)

        assert stats["protein_count"] == 1
        assert stats["mhc_i_peptides"] > 0
        assert stats["mhc_ii_peptides"] > 0

        # Check MHC-II file was created
        mhc_ii = out_dir / "test_mhc_ii_peptides.tsv"
        assert mhc_ii.exists()

        # Verify TSV structure
        with open(mhc_ii) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
            assert len(rows) == stats["mhc_ii_peptides"]
            assert "peptide" in rows[0]
            assert len(rows[0]["peptide"]) == 15


class TestIEDBBindingPrediction:
    """Tests for Phase 3: Binding prediction (non-API parts)."""

    def test_classify_binding(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.iedb_binding_prediction import (
            classify_binding,
        )

        assert classify_binding(0.1) == "strong_binder"
        assert classify_binding(0.5) == "weak_binder"
        assert classify_binding(1.5) == "weak_binder"
        assert classify_binding(2.0) == "non_binder"
        assert classify_binding(50.0) == "non_binder"

    def test_load_peptides_sampled(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.iedb_binding_prediction import (
            load_peptides_sampled,
        )

        tsv = tmp_path / "peptides.tsv"
        tsv.write_text(
            "peptide\tprotein_accession\tprotein_name\tis_virulence_factor\n"
            "AAAAAAAAAAAAAAAA\tP001\tM protein\tTrue\n"
            "BBBBBBBBBBBBBBBB\tP002\tC5a peptidase\tTrue\n"
            "AAAAAAAAAAAAAAAA\tP001\tM protein\tTrue\n"  # duplicate
            "CCCCCCCCCCCCCCCC\tP003\tHypothetical\tFalse\n"
        )

        peptides = load_peptides_sampled(tsv)
        assert len(peptides) == 3  # deduped
        assert peptides[0]["is_virulence_factor"] is True
        assert peptides[2]["is_virulence_factor"] is False

    def test_load_peptides_sampled_max(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.iedb_binding_prediction import (
            load_peptides_sampled,
        )

        tsv = tmp_path / "peptides.tsv"
        tsv.write_text(
            "peptide\tprotein_accession\tprotein_name\tis_virulence_factor\n"
            "AAAAAAAAAAAAAAAA\tP001\tM protein\tTrue\n"
            "BBBBBBBBBBBBBBBB\tP002\tC5a peptidase\tTrue\n"
            "CCCCCCCCCCCCCCCC\tP003\tHypothetical\tFalse\n"
        )

        peptides = load_peptides_sampled(tsv, max_peptides=2)
        assert len(peptides) == 2

    def test_load_peptides_virulence_only(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.iedb_binding_prediction import (
            load_peptides_sampled,
        )

        tsv = tmp_path / "peptides.tsv"
        tsv.write_text(
            "peptide\tprotein_accession\tprotein_name\tis_virulence_factor\n"
            "AAAAAAAAAAAAAAAA\tP001\tM protein\tTrue\n"
            "BBBBBBBBBBBBBBBB\tP002\tC5a peptidase\tTrue\n"
            "CCCCCCCCCCCCCCCC\tP003\tHypothetical\tFalse\n"
            "DDDDDDDDDDDDDDDD\tP004\tRibosomal\tFalse\n"
        )

        peptides = load_peptides_sampled(tsv, virulence_only=True)
        assert len(peptides) == 2
        assert all(p["is_virulence_factor"] for p in peptides)

        # virulence_only=False loads all
        all_peps = load_peptides_sampled(tsv, virulence_only=False)
        assert len(all_peps) == 4

    def test_load_peptides_old_schema(self, tmp_path):
        """Test load_peptides_sampled with Phase 2 old-format TSV (sequence/source_*)."""
        from src.pandas_pans.hla_peptide_presentation_modeling.iedb_binding_prediction import (
            load_peptides_sampled,
        )

        tsv = tmp_path / "peptides.tsv"
        tsv.write_text(
            "sequence\tlength\tposition_start\tposition_end\tsource_protein\tsource_accession\tserotype\tis_virulence_factor\tvirulence_flags\n"
            "AAAAAAAAAAAAAAAA\t15\t0\t15\tM protein\tP001\tM12\tTrue\tM protein\n"
            "BBBBBBBBBBBBBBBB\t15\t1\t16\tC5a peptidase\tP002\tM12\tTrue\tC5a peptidase\n"
            "CCCCCCCCCCCCCCCC\t15\t2\t17\tHypothetical\tP003\tM12\tFalse\t\n"
        )

        peptides = load_peptides_sampled(tsv)
        assert len(peptides) == 3
        assert peptides[0]["peptide"] == "AAAAAAAAAAAAAAAA"
        assert peptides[0]["protein_name"] == "M protein"
        assert peptides[0]["protein_accession"] == "P001"
        assert peptides[0]["is_virulence_factor"] is True
        assert peptides[2]["is_virulence_factor"] is False

        # virulence_only works with old schema too
        vf_only = load_peptides_sampled(tsv, virulence_only=True)
        assert len(vf_only) == 2

    def test_api_url_selection_by_mhc_class(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.iedb_binding_prediction import (
            IEDB_API_URLS,
        )

        assert "I" in IEDB_API_URLS
        assert "II" in IEDB_API_URLS
        assert "mhci" in IEDB_API_URLS["I"]
        assert "mhcii" in IEDB_API_URLS["II"]
        assert IEDB_API_URLS["I"] != IEDB_API_URLS["II"]

    def test_call_iedb_api_default_methods(self):
        """Verify that call_iedb_api selects the correct default method per MHC class."""
        from unittest.mock import patch, MagicMock

        from src.pandas_pans.hla_peptide_presentation_modeling.iedb_binding_prediction import (
            IEDB_API_URLS,
            call_iedb_api,
        )

        mock_resp = MagicMock()
        mock_resp.text = "allele\tpeptide\tic50\tpercentile_rank\nA*02:01\tACDEFGHIK\t500\t1.5"
        mock_resp.raise_for_status = MagicMock()

        with patch("src.pandas_pans.hla_peptide_presentation_modeling.iedb_binding_prediction.requests.post", return_value=mock_resp) as mock_post:
            # MHC-I should use netmhcpan and MHC-I URL
            call_iedb_api(["ACDEFGHIK"], "A*02:01", mhc_class="I")
            args, kwargs = mock_post.call_args
            assert args[0] == IEDB_API_URLS["I"]
            assert kwargs["data"]["method"] == "netmhcpan"

            # MHC-II should use netmhciipan and MHC-II URL
            call_iedb_api(["ACDEFGHIKLMNPQR"], "DRB1*07:01", mhc_class="II")
            args, kwargs = mock_post.call_args
            assert args[0] == IEDB_API_URLS["II"]
            assert kwargs["data"]["method"] == "netmhciipan"

    def test_supplemental_sets_constant(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.iedb_binding_prediction import (
            SUPPLEMENTAL_SETS,
        )

        assert "enn_mrp" in SUPPLEMENTAL_SETS

    def test_merge_serotype_predictions(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.iedb_binding_prediction import (
            PREDICTION_FIELDNAMES,
            merge_serotype_predictions,
        )

        pred_dir = tmp_path / "binding_predictions"
        pred_dir.mkdir(parents=True)

        # Create two fake per-serotype-allele files
        for serotype, pep in [("m1", "AAAAAAA"), ("m3", "BBBBBBB")]:
            path = pred_dir / f"{serotype}_DRB1_07_01_mhc_ii.tsv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDNAMES, delimiter="\t")
                writer.writeheader()
                writer.writerow({
                    "peptide": pep, "allele": "DRB1*07:01", "ic50": 100.0,
                    "percentile_rank": 0.3, "binding_class": "strong_binder",
                    "method": "netmhciipan", "source_protein": "M protein",
                    "source_accession": "P001", "serotype": serotype.upper(),
                    "is_virulence_factor": True,
                })

        merged = merge_serotype_predictions(
            serotypes=["M1", "M3"], mhc_class="II", output_base=tmp_path
        )

        assert merged.exists()
        with open(merged) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        assert len(rows) == 2
        assert {r["serotype"] for r in rows} == {"M1", "M3"}


    def test_merge_includes_supplemental_sets(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.iedb_binding_prediction import (
            PREDICTION_FIELDNAMES,
            merge_serotype_predictions,
        )

        pred_dir = tmp_path / "binding_predictions"
        pred_dir.mkdir(parents=True)

        # Create serotype + supplement prediction files
        for label, pep in [("m1", "AAAAAAA"), ("enn_mrp", "BBBBBBB")]:
            path = pred_dir / f"{label}_DRB1_07_01_mhc_ii.tsv"
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=PREDICTION_FIELDNAMES, delimiter="\t")
                writer.writeheader()
                writer.writerow({
                    "peptide": pep, "allele": "DRB1*07:01", "ic50": 100.0,
                    "percentile_rank": 0.3, "binding_class": "strong_binder",
                    "method": "netmhciipan", "source_protein": "test",
                    "source_accession": "P001", "serotype": label.upper(),
                    "is_virulence_factor": True,
                })

        merged = merge_serotype_predictions(
            serotypes=["M1", "enn_mrp"], mhc_class="II", output_base=tmp_path
        )

        assert merged.exists()
        with open(merged) as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
        assert len(rows) == 2
        serotypes = {r["serotype"] for r in rows}
        assert "ENN_MRP" in serotypes


class TestDifferentialPresentation:
    """Tests for Phase 4: Differential analysis."""

    def _make_predictions(self):
        """Build synthetic binding predictions across allele groups."""
        predictions = []
        # Peptide A: strong binder to susceptibility, weak to control -> high fold change
        for allele in ["DRB1*07:01", "DRB1*04:01", "DRB1*01:01"]:
            predictions.append({
                "peptide": "PEPTIDEA_15MER_X",
                "allele": allele,
                "percentile_rank": 0.3,
                "ic50": 50.0,
                "source_protein": "M protein",
                "source_accession": "P001",
                "serotype": "M1",
                "is_virulence_factor": True,
            })
        for allele in ["DRB1*03:01", "DRB1*13:01"]:
            predictions.append({
                "peptide": "PEPTIDEA_15MER_X",
                "allele": allele,
                "percentile_rank": 15.0,
                "ic50": 5000.0,
                "source_protein": "M protein",
                "source_accession": "P001",
                "serotype": "M1",
                "is_virulence_factor": True,
            })
        # Peptide B: similar binding across groups -> fold change ~1
        for allele in ["DRB1*07:01", "DRB1*04:01", "DRB1*01:01"]:
            predictions.append({
                "peptide": "PEPTIDEB_15MER_Y",
                "allele": allele,
                "percentile_rank": 5.0,
                "ic50": 500.0,
                "source_protein": "Hypothetical",
                "source_accession": "P002",
                "serotype": "M1",
                "is_virulence_factor": False,
            })
        for allele in ["DRB1*03:01", "DRB1*13:01"]:
            predictions.append({
                "peptide": "PEPTIDEB_15MER_Y",
                "allele": allele,
                "percentile_rank": 5.0,
                "ic50": 500.0,
                "source_protein": "Hypothetical",
                "source_accession": "P002",
                "serotype": "M1",
                "is_virulence_factor": False,
            })
        return predictions

    def test_compute_differential_peptides_fold_change(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.differential_presentation import (
            compute_differential_peptides,
        )

        predictions = self._make_predictions()
        results = compute_differential_peptides(predictions)

        assert len(results) == 2
        # Results sorted by fold change descending
        assert results[0]["peptide"] == "PEPTIDEA_15MER_X"
        assert results[0]["fold_change"] == 50.0  # 15.0 / 0.3
        assert results[1]["peptide"] == "PEPTIDEB_15MER_Y"
        assert results[1]["fold_change"] == 1.0  # 5.0 / 5.0

    def test_compute_differential_peptides_best_allele(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.differential_presentation import (
            compute_differential_peptides,
        )

        predictions = self._make_predictions()
        results = compute_differential_peptides(predictions)

        # All susceptibility alleles had rank 0.3, so best is whichever was seen first
        assert results[0]["best_susceptibility_allele"] in {
            "DRB1*07:01", "DRB1*04:01", "DRB1*01:01"
        }

    def test_compute_differential_peptides_metadata(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.differential_presentation import (
            compute_differential_peptides,
        )

        predictions = self._make_predictions()
        results = compute_differential_peptides(predictions)

        vf_result = next(r for r in results if r["peptide"] == "PEPTIDEA_15MER_X")
        assert vf_result["is_virulence_factor"] is True
        assert vf_result["source_protein"] == "M protein"
        assert vf_result["serotype"] == "M1"

    def test_virulence_enrichment(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.differential_presentation import (
            virulence_enrichment_test,
        )

        diff = [
            {"peptide": "AAA", "is_virulence_factor": True},
            {"peptide": "BBB", "is_virulence_factor": False},
        ]
        all_preds = diff + [
            {"peptide": "CCC", "is_virulence_factor": False},
            {"peptide": "DDD", "is_virulence_factor": False},
        ]

        result = virulence_enrichment_test(diff, all_preds)
        assert "p_value" in result
        assert "odds_ratio" in result
        assert result["differential_vf"] == 1
        assert result["differential_non_vf"] == 1


class TestMimicryHLAIntegration:
    """Tests for Phase 5: Mimicry-HLA integration."""

    def test_compute_overlap_with_matches(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.mimicry_hla_integration import (
            compute_overlap,
        )

        mimicry_hits = [
            {"gas_protein": "P001", "human_protein": "DRD1", "score": 0.85},
            {"gas_protein": "P003", "human_protein": "tubulin", "score": 0.72},
        ]
        differential_peptides = [
            {"peptide": "AAAAAA", "source_accession": "P001", "source_protein": "M protein",
             "best_susceptibility_allele": "DRB1*07:01", "fold_change": "50.0"},
            {"peptide": "BBBBBB", "source_accession": "P002", "source_protein": "C5a peptidase",
             "best_susceptibility_allele": "DRB1*04:01", "fold_change": "12.0"},
        ]
        binding_predictions = [
            {"peptide": "AAAAAA", "source_accession": "P001", "source_protein": "M protein"},
            {"peptide": "BBBBBB", "source_accession": "P002", "source_protein": "C5a peptidase"},
            {"peptide": "CCCCCC", "source_accession": "P003", "source_protein": "Hypothetical"},
        ]

        result = compute_overlap(mimicry_hits, differential_peptides, binding_predictions)

        assert result["unique_mimicry_proteins"] == 2
        assert result["overlap"]["differential_and_mimicry"] == 1  # P001
        assert result["overlap"]["proteins_in_both"] == 2  # P001, P003
        assert result["network_edges"] == 1  # P001 -> DRD1
        assert result["significance_test"]["p_value"] <= 1.0
        assert result["pandas_target_analysis"]["pandas_targets_in_mimicry"] == 2  # DRD1, tubulin

    def test_compute_overlap_no_mimicry(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.mimicry_hla_integration import (
            compute_overlap,
        )

        result = compute_overlap([], [], [])
        assert result["mimicry_hits_total"] == 0
        assert result["overlap"]["differential_and_mimicry"] == 0
        assert result["network_edges"] == 0

    def test_load_mimicry_hits_empty_dir(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.mimicry_hla_integration import (
            load_mimicry_hits,
        )

        hits = load_mimicry_hits(tmp_path)
        assert hits == []

    def test_load_mimicry_hits_from_tsv(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.mimicry_hla_integration import (
            load_mimicry_hits,
        )

        tsv = tmp_path / "mimicry_results.tsv"
        tsv.write_text(
            "gas_protein\thuman_protein\tsimilarity_score\n"
            "P001\tDRD1\t0.85\n"
            "P002\ttubulin\t0.72\n"
        )

        hits = load_mimicry_hits(tmp_path)
        assert len(hits) == 2
        assert hits[0]["gas_protein"] == "P001"
        assert hits[0]["human_protein"] == "DRD1"
        assert hits[0]["score"] == 0.85


class TestPopulationRisk:
    """Tests for Phase 6: Population risk modeling."""

    def test_carrier_rate(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.population_risk import carrier_rate

        assert carrier_rate(0.0) == 0.0
        assert abs(carrier_rate(0.5) - 0.75) < 1e-10
        assert carrier_rate(1.0) == 1.0

    def test_weighted_risk_score(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.population_risk import (
            POPULATIONS,
            weighted_risk_score,
        )

        # All populations should produce a numeric score
        for pop in POPULATIONS:
            score = weighted_risk_score(pop)
            assert isinstance(score, float)

    def test_haplotype_combinations(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.population_risk import (
            POPULATIONS,
            compute_haplotype_combinations,
        )

        combos = compute_haplotype_combinations(POPULATIONS[:2])
        # 2 populations x C(3,2) = 6 combinations
        assert len(combos) == 6
        assert "combined_or" in combos[0]
        assert combos[0]["combined_or"] > 0

    def test_run_population_risk(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.population_risk import run_population_risk

        result = run_population_risk(output_base=tmp_path)

        assert result["populations_analyzed"] == 10
        assert (tmp_path / "population_risk" / "population_risk_scores.tsv").exists()
        assert (tmp_path / "population_risk" / "haplotype_risk_table.tsv").exists()
        assert (tmp_path / "population_risk" / "risk_analysis_summary.json").exists()


class TestEnnMrpIngestion:
    """Tests for Enn/Mrp M-like protein ingestion module."""

    def test_entry_metadata(self):
        from bioagentics.data.pandas_pans.enn_mrp_ingestion import (
            ENN_MRP_ENTRIES,
            EXISTING_ENN_MRP,
        )

        assert len(ENN_MRP_ENTRIES) == 4
        classes = {e["protein_class"] for e in ENN_MRP_ENTRIES}
        assert classes == {"mrp", "enn"}

        assert len(EXISTING_ENN_MRP) == 1
        assert EXISTING_ENN_MRP[0]["gene"] == "ennX"

    def test_write_and_load_metadata(self, tmp_path):
        from bioagentics.data.pandas_pans.enn_mrp_ingestion import (
            _load_existing_metadata,
            _write_metadata,
        )

        metadata = [
            {
                "accession": "X001",
                "protein_class": "mrp",
                "gene": "mrp",
                "protein_name": "Test Mrp",
                "length": 100,
                "serotype_association": "M3",
                "source": "uniprot",
                "already_in_proteome": False,
                "notes": "test",
            },
        ]

        _write_metadata(metadata, tmp_path)
        loaded = _load_existing_metadata(tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["accession"] == "X001"
        assert loaded[0]["protein_class"] == "mrp"

    def test_load_existing_metadata_missing(self, tmp_path):
        from bioagentics.data.pandas_pans.enn_mrp_ingestion import _load_existing_metadata

        assert _load_existing_metadata(tmp_path) == []

    def test_rebuild_combined_fasta(self, tmp_path):
        from bioagentics.data.pandas_pans.enn_mrp_ingestion import rebuild_combined_fasta

        # Create mock per-serotype FASTA
        (tmp_path / "gas_m1.fasta").write_text(">sp|P001|M1\nACDEFG\n")
        (tmp_path / "gas_m3.fasta").write_text(">sp|P002|M3\nHIKLMN\n")

        # Create mock supplement
        (tmp_path / "enn_mrp_supplement.fasta").write_text(">tr|X001|MRP\nPQRSTV\n")

        combined = rebuild_combined_fasta(tmp_path)
        assert combined.exists()

        text = combined.read_text()
        assert ">sp|P001|M1" in text
        assert ">sp|P002|M3" in text
        assert ">tr|X001|MRP" in text

    def test_rebuild_combined_fasta_no_supplement(self, tmp_path):
        from bioagentics.data.pandas_pans.enn_mrp_ingestion import rebuild_combined_fasta

        (tmp_path / "gas_m1.fasta").write_text(">sp|P001|M1\nACDEFG\n")

        combined = rebuild_combined_fasta(tmp_path)
        text = combined.read_text()
        assert ">sp|P001|M1" in text

    def test_rebuild_combined_fasta_no_serotypes(self, tmp_path):
        from bioagentics.data.pandas_pans.enn_mrp_ingestion import rebuild_combined_fasta

        with pytest.raises(FileNotFoundError):
            rebuild_combined_fasta(tmp_path)


class TestDigestEnnMrpIntegration:
    """Test that Enn/Mrp virulence factor keywords are detected."""

    def test_enn_mrp_virulence_keywords(self):
        from src.pandas_pans.hla_peptide_presentation_modeling.gas_proteome_digest import (
            VIRULENCE_FACTOR_KEYWORDS,
        )

        keywords_lower = {kw.lower() for kw in VIRULENCE_FACTOR_KEYWORDS}
        assert "mrp" in keywords_lower
        assert "m-related protein" in keywords_lower
        assert "protein enn" in keywords_lower

    def test_enn_mrp_detected_as_virulence_factor(self, tmp_path):
        from src.pandas_pans.hla_peptide_presentation_modeling.gas_proteome_digest import (
            parse_fasta_streaming,
        )

        fasta = tmp_path / "enn_mrp.fasta"
        fasta.write_text(
            ">tr|X001|MRP_STRPY M-related protein Mrp OS=GAS\n"
            "ACDEFGHIKLMNPQRSTVWY\n"
            ">tr|X002|ENN_STRPY M-related protein Enn OS=GAS\n"
            "ACDEFGHIKLMNPQRSTVWY\n"
        )

        proteins = list(parse_fasta_streaming(fasta))
        assert len(proteins) == 2
        assert proteins[0].is_virulence_factor  # "M-related protein" matched
        assert proteins[1].is_virulence_factor  # "Enn" matched
