"""Tests for PANDAS autoantibody seed protein definitions."""

import pandas as pd
import pytest

from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    GM1_NOTE,
    MECHANISM_CATEGORIES,
    SEED_PROTEINS,
    get_by_category,
    get_gene_symbols,
    get_seed_dataframe,
    get_seed_dict,
    get_uniprot_ids,
)


class TestSeedProteinDefinitions:
    """Verify seed protein data integrity."""

    def test_seed_count(self):
        """All 9 protein targets are defined (GM1 is a glycolipid, excluded)."""
        assert len(SEED_PROTEINS) == 9

    def test_all_have_uniprot_ids(self):
        for p in SEED_PROTEINS:
            assert p.uniprot_id, f"{p.name} missing UniProt ID"
            assert p.uniprot_id[0] in "POQA", f"{p.name} has unusual UniProt ID format"

    def test_all_have_gene_symbols(self):
        for p in SEED_PROTEINS:
            assert p.gene_symbol, f"{p.name} missing gene symbol"

    def test_unique_uniprot_ids(self):
        ids = [p.uniprot_id for p in SEED_PROTEINS]
        assert len(ids) == len(set(ids)), "Duplicate UniProt IDs found"

    def test_unique_gene_symbols(self):
        symbols = [p.gene_symbol for p in SEED_PROTEINS]
        assert len(symbols) == len(set(symbols)), "Duplicate gene symbols found"

    def test_valid_mechanism_categories(self):
        for p in SEED_PROTEINS:
            assert p.mechanism_category in MECHANISM_CATEGORIES, (
                f"{p.name} has invalid category '{p.mechanism_category}'"
            )

    def test_expected_cunningham_panel_targets(self):
        cp_targets = {p.gene_symbol for p in SEED_PROTEINS if p.cunningham_panel}
        assert cp_targets == {"DRD1", "DRD2", "TUBB3", "CAMK2A"}

    def test_expected_proteins_present(self):
        symbols = {p.gene_symbol for p in SEED_PROTEINS}
        expected = {"DRD1", "DRD2", "TUBB3", "CAMK2A", "PKM", "ALDOC", "ENO1", "ENO2", "FOLR1"}
        assert symbols == expected

    def test_mechanism_category_coverage(self):
        categories = {p.mechanism_category for p in SEED_PROTEINS}
        assert "dopaminergic" in categories
        assert "calcium" in categories
        assert "metabolic" in categories
        assert "structural" in categories
        assert "folate" in categories

    def test_specific_uniprot_mappings(self):
        d = get_seed_dict()
        assert d["DRD1"]["uniprot_id"] == "P21728"
        assert d["DRD2"]["uniprot_id"] == "P14416"
        assert d["TUBB3"]["uniprot_id"] == "Q13509"
        assert d["CAMK2A"]["uniprot_id"] == "Q9UQM7"
        assert d["PKM"]["uniprot_id"] == "P14618"
        assert d["ALDOC"]["uniprot_id"] == "P09972"
        assert d["ENO1"]["uniprot_id"] == "P06733"
        assert d["ENO2"]["uniprot_id"] == "P09104"
        assert d["FOLR1"]["uniprot_id"] == "P15328"


class TestSeedProteinAccessors:
    """Test accessor functions."""

    def test_get_seed_dataframe(self):
        df = get_seed_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 9
        assert set(df.columns) == {
            "name", "uniprot_id", "gene_symbol",
            "mechanism_category", "cunningham_panel", "notes",
        }

    def test_get_seed_dict_keys(self):
        d = get_seed_dict()
        assert len(d) == 9
        assert all(isinstance(v, dict) for v in d.values())

    def test_get_gene_symbols(self):
        symbols = get_gene_symbols()
        assert len(symbols) == 9
        assert "DRD1" in symbols
        assert "FOLR1" in symbols

    def test_get_uniprot_ids(self):
        ids = get_uniprot_ids()
        assert len(ids) == 9
        assert "P21728" in ids

    def test_get_by_category_dopaminergic(self):
        dopa = get_by_category("dopaminergic")
        assert len(dopa) == 2
        assert {p.gene_symbol for p in dopa} == {"DRD1", "DRD2"}

    def test_get_by_category_metabolic(self):
        met = get_by_category("metabolic")
        assert len(met) == 4
        assert {p.gene_symbol for p in met} == {"PKM", "ALDOC", "ENO1", "ENO2"}

    def test_get_by_category_folate(self):
        fol = get_by_category("folate")
        assert len(fol) == 1
        assert fol[0].gene_symbol == "FOLR1"

    def test_get_by_category_invalid(self):
        with pytest.raises(ValueError, match="Unknown category"):
            get_by_category("invalid")


class TestGM1Note:
    """Verify GM1 glycolipid documentation."""

    def test_gm1_note_exists(self):
        assert GM1_NOTE
        assert "glycolipid" in GM1_NOTE
        assert "UniProt" in GM1_NOTE
