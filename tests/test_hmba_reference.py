"""Tests for HMBA Basal Ganglia cell-type taxonomy reference module."""

import pytest

from bioagentics.data.tourettes.hmba_reference import (
    CellType,
    get_cell_type,
    get_cstc_region_cell_types,
    get_marker_panel,
    get_striosome_matrix_types,
    get_taxonomy,
    list_cell_types,
    map_to_hmba,
    reset,
)


@pytest.fixture(autouse=True)
def _reset_index():
    """Reset taxonomy index before each test to ensure isolation."""
    reset()
    yield
    reset()


# ---------------------------------------------------------------------------
# get_taxonomy
# ---------------------------------------------------------------------------

class TestGetTaxonomy:
    def test_returns_dict(self):
        tax = get_taxonomy()
        assert isinstance(tax, dict)
        assert len(tax) > 0

    def test_all_values_are_cell_type(self):
        for label, ct in get_taxonomy().items():
            assert isinstance(ct, CellType)
            assert ct.label == label

    def test_builtin_has_expected_cell_types(self):
        tax = get_taxonomy()
        expected = [
            "D1_MSN_matrix", "D1_MSN_striosome",
            "D2_MSN_matrix", "D2_MSN_striosome",
            "cholinergic_interneuron", "pv_interneuron", "sst_interneuron",
            "GPe_prototypic", "GPe_arkypallidal", "GPi_projection",
            "STN_glutamatergic",
            "thalamic_relay", "thalamic_reticular",
            "astrocyte", "oligodendrocyte", "OPC", "microglia",
        ]
        for label in expected:
            assert label in tax, f"Missing expected cell type: {label}"

    def test_taxonomy_count(self):
        assert len(get_taxonomy()) == 17


# ---------------------------------------------------------------------------
# list_cell_types
# ---------------------------------------------------------------------------

class TestListCellTypes:
    def test_all(self):
        all_types = list_cell_types()
        assert len(all_types) == 17
        assert all_types == sorted(all_types)

    def test_filter_by_class(self):
        gaba = list_cell_types(cell_class="GABAergic")
        assert "D1_MSN_matrix" in gaba
        assert "astrocyte" not in gaba

    def test_filter_by_region(self):
        striatal = list_cell_types(region="striatum")
        assert "D1_MSN_matrix" in striatal
        assert "GPi_projection" not in striatal

    def test_filter_by_subclass(self):
        d1 = list_cell_types(subclass="D1 MSN")
        assert set(d1) == {"D1_MSN_matrix", "D1_MSN_striosome"}

    def test_combined_filters(self):
        gaba_striatum = list_cell_types(cell_class="GABAergic", region="striatum")
        assert "D1_MSN_matrix" in gaba_striatum
        assert "cholinergic_interneuron" not in gaba_striatum  # Cholinergic class

    def test_no_matches(self):
        assert list_cell_types(cell_class="nonexistent") == []


# ---------------------------------------------------------------------------
# get_cell_type
# ---------------------------------------------------------------------------

class TestGetCellType:
    def test_valid_label(self):
        ct = get_cell_type("D1_MSN_matrix")
        assert ct.hmba_id == "CS20260101_D1_MSN_matrix"
        assert ct.cell_class == "GABAergic"
        assert ct.region == "striatum"

    def test_invalid_label(self):
        with pytest.raises(KeyError, match="Unknown HMBA cell type"):
            get_cell_type("nonexistent_type")

    def test_markers_are_tuple(self):
        ct = get_cell_type("D1_MSN_matrix")
        assert isinstance(ct.markers, tuple)
        assert "DRD1" in ct.markers


# ---------------------------------------------------------------------------
# get_marker_panel
# ---------------------------------------------------------------------------

class TestGetMarkerPanel:
    def test_d1_msn_markers(self):
        markers = get_marker_panel("D1_MSN_matrix")
        assert "DRD1" in markers
        assert "TAC1" in markers

    def test_d2_msn_markers(self):
        markers = get_marker_panel("D2_MSN_matrix")
        assert "DRD2" in markers
        assert "PENK" in markers

    def test_cholinergic_markers(self):
        markers = get_marker_panel("cholinergic_interneuron")
        assert "CHAT" in markers

    def test_invalid_label(self):
        with pytest.raises(KeyError):
            get_marker_panel("nonexistent")


# ---------------------------------------------------------------------------
# map_to_hmba
# ---------------------------------------------------------------------------

class TestMapToHmba:
    def test_exact_match(self):
        result = map_to_hmba(["D1_MSN_matrix"])
        assert result["D1_MSN_matrix"] == "D1_MSN_matrix"

    def test_case_insensitive(self):
        result = map_to_hmba(["d1_msn_matrix"])
        assert result["d1_msn_matrix"] == "D1_MSN_matrix"

    def test_alias_mapping(self):
        result = map_to_hmba(["D1 MSN", "cholinergic interneuron", "PV+"])
        assert result["D1 MSN"] == "D1_MSN_matrix"
        assert result["cholinergic interneuron"] == "cholinergic_interneuron"
        assert result["PV+"] == "pv_interneuron"

    def test_unknown_returns_none(self):
        result = map_to_hmba(["unknown_cell_type"])
        assert result["unknown_cell_type"] is None

    def test_mixed_known_unknown(self):
        result = map_to_hmba(["D1 MSN", "alien_cell"])
        assert result["D1 MSN"] == "D1_MSN_matrix"
        assert result["alien_cell"] is None

    def test_striosome_aliases(self):
        result = map_to_hmba(["D1 striosome", "D2 patch", "D1 matrix"])
        assert result["D1 striosome"] == "D1_MSN_striosome"
        assert result["D2 patch"] == "D2_MSN_striosome"
        assert result["D1 matrix"] == "D1_MSN_matrix"

    def test_gp_aliases(self):
        result = map_to_hmba(["GPe", "GPi", "STN"])
        assert result["GPe"] == "GPe_prototypic"
        assert result["GPi"] == "GPi_projection"
        assert result["STN"] == "STN_glutamatergic"


# ---------------------------------------------------------------------------
# get_striosome_matrix_types
# ---------------------------------------------------------------------------

class TestStriosomeMatrixTypes:
    def test_compartments(self):
        compartments = get_striosome_matrix_types()
        assert "striosome" in compartments
        assert "matrix" in compartments

    def test_striosome_contains_d1_d2(self):
        compartments = get_striosome_matrix_types()
        labels = {ct.label for ct in compartments["striosome"]}
        assert "D1_MSN_striosome" in labels
        assert "D2_MSN_striosome" in labels

    def test_matrix_contains_d1_d2(self):
        compartments = get_striosome_matrix_types()
        labels = {ct.label for ct in compartments["matrix"]}
        assert "D1_MSN_matrix" in labels
        assert "D2_MSN_matrix" in labels


# ---------------------------------------------------------------------------
# get_cstc_region_cell_types
# ---------------------------------------------------------------------------

class TestCstcRegionCellTypes:
    def test_striatum(self):
        cts = get_cstc_region_cell_types("striatum")
        labels = {ct.label for ct in cts}
        assert "D1_MSN_matrix" in labels
        assert "D2_MSN_matrix" in labels
        assert "cholinergic_interneuron" in labels
        assert "pv_interneuron" in labels
        assert "sst_interneuron" in labels

    def test_gpe(self):
        cts = get_cstc_region_cell_types("GPe")
        labels = {ct.label for ct in cts}
        assert "GPe_prototypic" in labels
        assert "GPe_arkypallidal" in labels

    def test_gpi(self):
        cts = get_cstc_region_cell_types("GPi")
        labels = {ct.label for ct in cts}
        assert "GPi_projection" in labels

    def test_stn(self):
        cts = get_cstc_region_cell_types("STN")
        labels = {ct.label for ct in cts}
        assert "STN_glutamatergic" in labels

    def test_thalamus(self):
        cts = get_cstc_region_cell_types("thalamus")
        labels = {ct.label for ct in cts}
        assert "thalamic_relay" in labels
        assert "thalamic_reticular" in labels

    def test_unknown_region_empty(self):
        assert get_cstc_region_cell_types("nonexistent") == []

    def test_results_sorted(self):
        cts = get_cstc_region_cell_types("striatum")
        labels = [ct.label for ct in cts]
        assert labels == sorted(labels)


# ---------------------------------------------------------------------------
# Integration: circuit_vulnerability
# ---------------------------------------------------------------------------

class TestCircuitVulnerabilityIntegration:
    def test_get_node_cell_types(self):
        from bioagentics.analysis.tourettes.circuit_vulnerability import (
            get_node_cell_types,
        )
        # Basal ganglia nodes should have cell types
        caudate_cts = get_node_cell_types("caudate")
        assert len(caudate_cts) > 0
        labels = {ct.label for ct in caudate_cts}
        assert "D1_MSN_matrix" in labels

        # Cortical nodes have no HMBA BG cell types
        assert get_node_cell_types("prefrontal_cortex") == []

    def test_annotate_cell_type_composition(self):
        import pandas as pd
        from bioagentics.analysis.tourettes.circuit_vulnerability import (
            CSTC_NODES,
            annotate_cell_type_composition,
        )
        # Create a minimal vulnerability DataFrame
        df = pd.DataFrame(
            {"composite_score": [0.5] * len(CSTC_NODES)},
            index=CSTC_NODES,
        )
        result = annotate_cell_type_composition(df)
        assert "hmba_cell_types" in result.columns
        assert "n_cell_types" in result.columns
        assert "hmba_marker_genes" in result.columns

        # caudate should have striatal cell types
        caudate_row = result.loc["caudate"]
        assert caudate_row["n_cell_types"] > 0
        assert "D1_MSN_matrix" in caudate_row["hmba_cell_types"]

        # cortex should have no cell types
        pfc_row = result.loc["prefrontal_cortex"]
        assert pfc_row["n_cell_types"] == 0
        assert pfc_row["hmba_cell_types"] == ""


# ---------------------------------------------------------------------------
# Integration: dbs_tracts
# ---------------------------------------------------------------------------

class TestDbsTractsIntegration:
    def test_get_tract_cell_types(self):
        from bioagentics.analysis.tourettes.dbs_tracts import get_tract_cell_types
        cts = get_tract_cell_types("ansa_lenticularis")
        labels = {ct.label for ct in cts}
        # AL goes through GPi and thalamus
        assert "GPi_projection" in labels
        assert "thalamic_relay" in labels

    def test_get_all_tract_cell_type_map(self):
        from bioagentics.analysis.tourettes.dbs_tracts import get_all_tract_cell_type_map
        mapping = get_all_tract_cell_type_map()
        assert len(mapping) == 3
        assert "ansa_lenticularis" in mapping
        assert "fasciculus_lenticularis" in mapping
        assert "posterior_intralaminar_lentiform" in mapping
        # All tracts should have at least one cell type
        for tract, labels in mapping.items():
            assert len(labels) > 0, f"No cell types for tract {tract}"

    def test_fl_includes_stn(self):
        from bioagentics.analysis.tourettes.dbs_tracts import get_tract_cell_types
        cts = get_tract_cell_types("fasciculus_lenticularis")
        labels = {ct.label for ct in cts}
        assert "STN_glutamatergic" in labels

    def test_invalid_tract(self):
        from bioagentics.analysis.tourettes.dbs_tracts import get_tract_cell_types
        with pytest.raises(KeyError):
            get_tract_cell_types("nonexistent_tract")


# ---------------------------------------------------------------------------
# Integration: wgcna_cstc
# ---------------------------------------------------------------------------

class TestWgcnaIntegration:
    def test_annotate_modules_with_markers(self):
        import pandas as pd
        from bioagentics.analysis.tourettes.wgcna_cstc import (
            annotate_modules_cell_types,
        )
        # Module 1 contains DRD1 (D1 MSN marker), module 2 has no markers
        modules = pd.DataFrame({
            "gene_symbol": ["DRD1", "TAC1", "GENE_A", "GENE_B", "GENE_C"],
            "module": [1, 1, 1, 2, 2],
        })
        result = annotate_modules_cell_types(modules)
        assert len(result) == 2

        mod1 = next(r for r in result if r["module"] == 1)
        assert len(mod1["cell_type_annotations"]) > 0
        # DRD1 is a D1 MSN marker
        ct_labels = [a["cell_type"] for a in mod1["cell_type_annotations"]]
        assert "D1_MSN_matrix" in ct_labels or "D1_MSN_striosome" in ct_labels

    def test_annotate_modules_no_markers(self):
        import pandas as pd
        from bioagentics.analysis.tourettes.wgcna_cstc import (
            annotate_modules_cell_types,
        )
        modules = pd.DataFrame({
            "gene_symbol": ["GENE_X", "GENE_Y"],
            "module": [1, 1],
        })
        result = annotate_modules_cell_types(modules)
        assert len(result) == 1
        assert result[0]["cell_type_annotations"] == []
        assert result[0]["top_cell_type"] is None

    def test_annotate_skips_unassigned_module(self):
        import pandas as pd
        from bioagentics.analysis.tourettes.wgcna_cstc import (
            annotate_modules_cell_types,
        )
        modules = pd.DataFrame({
            "gene_symbol": ["DRD1", "GENE_X"],
            "module": [0, 0],
        })
        result = annotate_modules_cell_types(modules)
        assert len(result) == 0  # module 0 is unassigned
