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


class TestDifferentialPresentation:
    """Tests for Phase 4: Differential analysis."""

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
