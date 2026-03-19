"""Tests for compartment scoring module."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.tourettes.striosomal_matrix.compartment_scoring import (
    compute_compartment_scores,
    score_genes_by_compartment,
)


def _make_zone_expression():
    """Create synthetic zone expression with known compartment structure."""
    zones = [f"zone_{i}" for i in range(1, 7)]
    genes = ["OPRM1", "TAC1", "CALB1", "PENK", "SST", "GENE_A", "GENE_B"]

    # Zone 1-2: striosome-like (high OPRM1/TAC1, low CALB1/PENK)
    # Zone 3-4: neutral
    # Zone 5-6: matrix-like (low OPRM1/TAC1, high CALB1/PENK)
    data = np.array([
        [10, 8, 1, 2, 6, 5, 3],   # zone_1 striosome
        [9,  7, 2, 1, 5, 4, 4],   # zone_2 striosome
        [5,  5, 5, 5, 3, 6, 5],   # zone_3 neutral
        [4,  4, 6, 4, 4, 5, 6],   # zone_4 neutral
        [1,  2, 9, 8, 2, 3, 7],   # zone_5 matrix
        [2,  1, 10, 9, 1, 2, 8],  # zone_6 matrix
    ], dtype=float)

    return pd.DataFrame(data, index=zones, columns=genes)


class TestComputeCompartmentScores:
    def test_striosome_zones_positive(self):
        df = _make_zone_expression()
        scores = compute_compartment_scores(df)
        # Zone 1 should have positive compartment_bias (striosome)
        z1 = scores[scores["zone_id"] == "zone_1"].iloc[0]
        assert z1["compartment_bias"] > 0
        assert z1["striosome_score"] > 0

    def test_matrix_zones_negative(self):
        df = _make_zone_expression()
        scores = compute_compartment_scores(df)
        z6 = scores[scores["zone_id"] == "zone_6"].iloc[0]
        assert z6["compartment_bias"] < 0
        assert z6["matrix_score"] > 0

    def test_all_zones_scored(self):
        df = _make_zone_expression()
        scores = compute_compartment_scores(df)
        assert len(scores) == 6

    def test_markers_reported(self):
        df = _make_zone_expression()
        scores = compute_compartment_scores(df)
        assert "OPRM1" in scores.iloc[0]["striosome_markers_found"]
        assert "CALB1" in scores.iloc[0]["matrix_markers_found"]

    def test_custom_markers(self):
        df = _make_zone_expression()
        scores = compute_compartment_scores(
            df, striosome_markers=["OPRM1"], matrix_markers=["PENK"],
        )
        assert scores.iloc[0]["striosome_markers_found"] == "OPRM1"
        assert scores.iloc[0]["matrix_markers_found"] == "PENK"

    def test_no_markers_raises(self):
        df = _make_zone_expression()[["GENE_A", "GENE_B"]]
        with pytest.raises(ValueError, match="No compartment markers"):
            compute_compartment_scores(df)


class TestScoreGenesByCompartment:
    def test_striosome_gene_positive(self):
        df = _make_zone_expression()
        zone_scores = compute_compartment_scores(df)
        gene_scores = score_genes_by_compartment(df, zone_scores)
        # OPRM1 is expressed in striosome zones -> positive score
        assert gene_scores["OPRM1"] > 0

    def test_matrix_gene_negative(self):
        df = _make_zone_expression()
        zone_scores = compute_compartment_scores(df)
        gene_scores = score_genes_by_compartment(df, zone_scores)
        # CALB1 is expressed in matrix zones -> negative score
        assert gene_scores["CALB1"] < 0

    def test_all_genes_scored(self):
        df = _make_zone_expression()
        zone_scores = compute_compartment_scores(df)
        gene_scores = score_genes_by_compartment(df, zone_scores)
        assert len(gene_scores) == len(df.columns)
