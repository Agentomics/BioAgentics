"""Tests for SigCom LINCS API client."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from bioagentics.data.cd_fibrosis.sigcom_client import (
    DATA_API,
    DEFAULT_DATABASE,
    METADATA_API,
    SigcomClient,
)
from bioagentics.data.cd_fibrosis.cmap_pipeline import (
    parse_sigcom_results,
    query_sigcom_signature,
)


# --- SigcomClient unit tests ---


class TestSigcomClientInit:
    def test_creates_session(self):
        client = SigcomClient()
        assert client.session is not None
        assert client.timeout > 0

    def test_default_database(self):
        assert DEFAULT_DATABASE == "l1000_cp"

    def test_gene_cache_starts_empty(self):
        client = SigcomClient()
        assert len(client._gene_cache) == 0


class TestResolveGenes:
    def test_resolve_caches_results(self):
        client = SigcomClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": "uuid-1", "meta": {"symbol": "TP53"}},
            {"id": "uuid-2", "meta": {"symbol": "BRCA1"}},
        ]
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client.session, "post", return_value=mock_resp):
            result = client.resolve_genes(["TP53", "BRCA1"])

        assert result == {"TP53": "uuid-1", "BRCA1": "uuid-2"}
        assert client._gene_cache["TP53"] == "uuid-1"

        # Second call should use cache, not make API call
        with patch.object(client.session, "post") as mock_post:
            result2 = client.resolve_genes(["TP53"])
            mock_post.assert_not_called()
        assert result2 == {"TP53": "uuid-1"}

    def test_resolve_handles_missing_genes(self):
        client = SigcomClient()
        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": "uuid-1", "meta": {"symbol": "TP53"}},
        ]
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client.session, "post", return_value=mock_resp):
            result = client.resolve_genes(["TP53", "FAKEGENE"])

        assert "TP53" in result
        assert "FAKEGENE" not in result

    def test_resolve_batches_large_lists(self):
        client = SigcomClient()
        genes = [f"GENE{i}" for i in range(250)]

        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            client.resolve_genes(genes)
            # Should make 3 batches: 100, 100, 50
            assert mock_post.call_count == 3


class TestEnrichTwosided:
    def _mock_client(self):
        client = SigcomClient()
        # Pre-fill gene cache
        client._gene_cache = {
            "SERPINE1": "uuid-s1",
            "COL1A1": "uuid-c1",
            "MMP9": "uuid-m9",
        }
        return client

    def test_enrichment_calls_correct_endpoint(self):
        client = self._mock_client()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "results": [
                {
                    "uuid": "sig-1",
                    "rank": 0,
                    "type": "reversers",
                    "z-up": 5.0,
                    "z-down": 4.0,
                    "z-sum": 9.0,
                    "p-up": 0.001,
                    "p-down": 0.002,
                    "fdr-up": 0.01,
                    "fdr-down": 0.02,
                    "logp-fisher": 10.5,
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            result = client.enrich_twosided(
                ["SERPINE1", "COL1A1"], ["MMP9"], "l1000_cp", 10
            )

        # Check endpoint
        call_args = mock_post.call_args
        assert DATA_API in call_args.args[0]
        assert "ranktwosided" in call_args.args[0]

        # Check payload
        payload = call_args.kwargs["json"]
        assert payload["database"] == "l1000_cp"
        assert payload["limit"] == 10
        assert len(payload["up_entities"]) == 2
        assert len(payload["down_entities"]) == 1

        assert len(result["results"]) == 1

    def test_enrichment_handles_no_mapped_genes(self):
        client = SigcomClient()
        # Empty gene cache
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client.session, "post", return_value=mock_resp):
            result = client.enrich_twosided(["NONEXISTENT"], [], "l1000_cp", 10)

        assert result["results"] == []
        assert "error" in result


class TestResolveSignatures:
    def test_resolve_batches_uuids(self):
        client = SigcomClient()
        uuids = [f"sig-{i}" for i in range(150)]

        mock_resp = MagicMock()
        mock_resp.json.return_value = [
            {"id": f"sig-{i}", "meta": {"pert_name": f"drug-{i}"}}
            for i in range(100)
        ]
        mock_resp.raise_for_status = MagicMock()

        with patch.object(client.session, "post", return_value=mock_resp) as mock_post:
            results = client.resolve_signatures(uuids)
            assert mock_post.call_count == 2

    def test_resolve_empty_list(self):
        client = SigcomClient()
        assert client.resolve_signatures([]) == []


class TestQueryFibrosisSignature:
    def test_full_workflow(self):
        client = SigcomClient()
        client._gene_cache = {
            "SERPINE1": "uuid-s1",
            "COL1A1": "uuid-c1",
            "MMP9": "uuid-m9",
        }

        enrich_resp = MagicMock()
        enrich_resp.json.return_value = {
            "results": [
                {
                    "uuid": "sig-abc",
                    "rank": 0,
                    "type": "reversers",
                    "z-up": -5.0,
                    "z-down": -4.0,
                    "z-sum": -9.0,
                    "p-up": 0.001,
                    "p-down": 0.002,
                    "fdr-up": 0.01,
                    "fdr-down": 0.02,
                    "logp-fisher": 10.5,
                }
            ]
        }
        enrich_resp.raise_for_status = MagicMock()

        meta_resp = MagicMock()
        meta_resp.json.return_value = [
            {
                "id": "sig-abc",
                "meta": {
                    "pert_name": "vorinostat",
                    "cell_line": "MCF7",
                    "pert_type": "trt_cp",
                },
            }
        ]
        meta_resp.raise_for_status = MagicMock()

        with patch.object(
            client.session, "post", side_effect=[enrich_resp, meta_resp]
        ):
            results = client.query_fibrosis_signature(
                ["SERPINE1", "COL1A1"], ["MMP9"]
            )

        assert len(results) == 1
        assert results[0]["compound"] == "vorinostat"
        assert results[0]["cell_line"] == "MCF7"
        assert results[0]["z_sum"] == -9.0
        assert results[0]["logp_fisher"] == 10.5


# --- Pipeline integration tests ---


class TestParseSigcomResults:
    def test_parse_results(self):
        results = [
            {
                "signature_id": "sig-1",
                "compound": "vorinostat",
                "cell_line": "MCF7",
                "z_sum": -8.5,
                "logp_fisher": 10.0,
            },
            {
                "signature_id": "sig-2",
                "compound": "trichostatin-a",
                "cell_line": "A549",
                "z_sum": 5.0,
                "logp_fisher": 6.0,
            },
        ]
        df = parse_sigcom_results(results, "bulk")

        assert len(df) == 2
        assert "concordance" in df.columns
        assert "query_signature" in df.columns
        assert df.iloc[0]["query_signature"] == "bulk"

        # Negative z_sum -> positive concordance value (after negation)
        # We negate z_sum so negative concordance = reversal
        vorinostat_row = df[df["compound"] == "vorinostat"].iloc[0]
        assert vorinostat_row["concordance"] == 8.5  # -(-8.5)

    def test_parse_empty_results(self):
        df = parse_sigcom_results([], "test")
        assert len(df) == 0

    def test_sorted_by_concordance(self):
        results = [
            {"signature_id": "a", "compound": "drug_a", "z_sum": 5.0},
            {"signature_id": "b", "compound": "drug_b", "z_sum": -10.0},
        ]
        df = parse_sigcom_results(results, "test")
        # drug_b has z_sum=-10 -> concordance=10 (positive, mimicker)
        # drug_a has z_sum=5 -> concordance=-5 (negative, reverser)
        assert df.iloc[0]["compound"] == "drug_a"  # Most negative concordance first


class TestQuerySigcomSignature:
    def test_delegates_to_client(self):
        mock_client = MagicMock()
        mock_client.query_fibrosis_signature.return_value = [
            {"compound": "drug_a", "z_sum": -5.0}
        ]

        results = query_sigcom_signature(
            mock_client, ["GENE1"], ["GENE2"], "l1000_cp", 10
        )

        mock_client.query_fibrosis_signature.assert_called_once_with(
            ["GENE1"], ["GENE2"], "l1000_cp", 10
        )
        assert len(results) == 1
