"""Tests for STRING PPI query pipeline (offline — no network calls)."""

import pandas as pd
import pytest

from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    get_gene_symbols,
)
from pandas_pans.autoantibody_target_network_mapping import string_ppi_query as mod


# Synthetic interaction data mimicking STRING API output
MOCK_EDGES = [
    {"source": "DRD1", "target": "GNB1", "source_string_id": "9606.A",
     "target_string_id": "9606.B", "combined_score": 0.95,
     "nscore": 0, "fscore": 0, "pscore": 0, "ascore": 0,
     "escore": 0.8, "dscore": 0, "tscore": 0.9, "seed_protein": "DRD1"},
    {"source": "DRD1", "target": "GNAI2", "source_string_id": "9606.A",
     "target_string_id": "9606.C", "combined_score": 0.85,
     "nscore": 0, "fscore": 0, "pscore": 0, "ascore": 0,
     "escore": 0.7, "dscore": 0, "tscore": 0.8, "seed_protein": "DRD1"},
    {"source": "DRD2", "target": "GNB1", "source_string_id": "9606.D",
     "target_string_id": "9606.B", "combined_score": 0.90,
     "nscore": 0, "fscore": 0, "pscore": 0, "ascore": 0,
     "escore": 0.6, "dscore": 0, "tscore": 0.85, "seed_protein": "DRD2"},
    {"source": "CAMK2A", "target": "CALM1", "source_string_id": "9606.E",
     "target_string_id": "9606.F", "combined_score": 0.99,
     "nscore": 0, "fscore": 0, "pscore": 0, "ascore": 0,
     "escore": 0.9, "dscore": 0, "tscore": 0.95, "seed_protein": "CAMK2A"},
]


class TestComputeNetworkStats:
    def test_basic_stats(self):
        df = pd.DataFrame(MOCK_EDGES)
        seeds = get_gene_symbols()
        stats = mod.compute_network_stats(df, seeds)
        assert stats["total_edges"] == 4
        assert stats["total_nodes"] == 6  # DRD1, DRD2, GNB1, GNAI2, CAMK2A, CALM1
        assert stats["unique_interactors"] >= 2  # GNB1, GNAI2, CALM1 are non-seed

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        stats = mod.compute_network_stats(df, ["DRD1"])
        assert stats["total_edges"] == 0
        assert stats["total_nodes"] == 0

    def test_seed_degree(self):
        df = pd.DataFrame(MOCK_EDGES)
        seeds = get_gene_symbols()
        stats = mod.compute_network_stats(df, seeds)
        assert stats["seed_degree"].get("DRD1", 0) == 2
        assert stats["seed_degree"].get("DRD2", 0) == 1
        assert stats["seed_degree"].get("CAMK2A", 0) == 1

    def test_score_stats(self):
        df = pd.DataFrame(MOCK_EDGES)
        seeds = get_gene_symbols()
        stats = mod.compute_network_stats(df, seeds)
        assert 0 < stats["mean_combined_score"] <= 1.0
        assert stats["min_combined_score"] > 0
        assert stats["max_combined_score"] <= 1.0


class TestModuleImports:
    def test_constants(self):
        assert mod.SPECIES_HUMAN == 9606
        assert mod.MIN_SCORE == 700
        assert mod.RATE_LIMIT_DELAY > 0

    def test_api_base_url(self):
        assert "string-db.org" in mod.STRING_API_BASE
