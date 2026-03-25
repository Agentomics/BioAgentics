"""SigCom LINCS API client for L1000 connectivity scoring.

Queries fibrosis gene signatures against L1000 chemical perturbation
signatures via the SigCom LINCS REST API (Ma'ayan Lab).

No authentication or API key required. No bulk download needed.

Workflow:
1. Resolve gene symbols to SigCom entity UUIDs
2. Submit two-sided enrichment (up/down gene sets) against l1000_cp database
3. Retrieve top reversing/mimicking compounds
4. Resolve result signature UUIDs to compound metadata

API docs: https://maayanlab.cloud/sigcom-lincs/metadata-api/explorer

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.sigcom_client --test
    uv run python -m bioagentics.data.cd_fibrosis.sigcom_client --query path/to/sig.tsv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

from bioagentics.config import REPO_ROOT
from bioagentics.data.cd_fibrosis.cmap_client import load_signature_tsv

METADATA_API = "https://maayanlab.cloud/sigcom-lincs/metadata-api"
DATA_API = "https://maayanlab.cloud/sigcom-lincs/data-api/api/v1"
TIMEOUT = 120
OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing"

# Primary database for drug repurposing: L1000 chemical perturbations
DEFAULT_DATABASE = "l1000_cp"


class SigcomClient:
    """Client for the SigCom LINCS enrichment API."""

    def __init__(self, timeout: int = TIMEOUT):
        self.session = requests.Session()
        self.timeout = timeout
        self._gene_cache: dict[str, str] = {}

    def list_databases(self) -> list[dict]:
        """List available enrichment databases."""
        resp = self.session.post(
            f"{DATA_API}/listdata",
            json={},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()

    def resolve_genes(self, symbols: list[str]) -> dict[str, str]:
        """Resolve gene symbols to SigCom entity UUIDs.

        Args:
            symbols: Gene symbols (e.g. ["SERPINE1", "COL1A1"]).

        Returns:
            Dict mapping symbol -> UUID for genes found in SigCom.
        """
        # Check cache first
        uncached = [s for s in symbols if s not in self._gene_cache]
        if uncached:
            # Query in batches of 100 to avoid oversized requests
            for i in range(0, len(uncached), 100):
                batch = uncached[i : i + 100]
                resp = self.session.post(
                    f"{METADATA_API}/entities/find",
                    json={
                        "filter": {
                            "where": {"meta.symbol": {"inq": batch}},
                            "fields": ["id", "meta.symbol"],
                        }
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                for entity in resp.json():
                    symbol = entity.get("meta", {}).get("symbol", "")
                    uid = entity.get("id", "")
                    if symbol and uid:
                        self._gene_cache[symbol] = uid

        return {s: self._gene_cache[s] for s in symbols if s in self._gene_cache}

    def enrich_twosided(
        self,
        up_genes: list[str],
        down_genes: list[str],
        database: str = DEFAULT_DATABASE,
        limit: int = 50,
    ) -> dict:
        """Run two-sided enrichment: find signatures that reverse the query.

        Submits up-regulated and down-regulated gene sets. Returns signatures
        ranked by reversal strength (reversers have the query's up-genes
        down-regulated and vice versa).

        Args:
            up_genes: Gene symbols upregulated in disease.
            down_genes: Gene symbols downregulated in disease.
            database: SigCom database to query (default: l1000_cp).
            limit: Max results to return per direction.

        Returns:
            Raw API response with 'results' list.
        """
        # Resolve genes to UUIDs
        all_genes = list(set(up_genes + down_genes))
        gene_map = self.resolve_genes(all_genes)

        up_uuids = [gene_map[g] for g in up_genes if g in gene_map]
        down_uuids = [gene_map[g] for g in down_genes if g in gene_map]

        mapped_up = [g for g in up_genes if g in gene_map]
        mapped_down = [g for g in down_genes if g in gene_map]
        unmapped_up = [g for g in up_genes if g not in gene_map]
        unmapped_down = [g for g in down_genes if g not in gene_map]

        if unmapped_up or unmapped_down:
            print(f"    Unmapped genes: {len(unmapped_up)} up, {len(unmapped_down)} down")

        if not up_uuids and not down_uuids:
            return {"results": [], "error": "No genes could be resolved"}

        payload: dict = {
            "database": database,
            "limit": limit,
        }
        if up_uuids:
            payload["up_entities"] = up_uuids
        if down_uuids:
            payload["down_entities"] = down_uuids

        resp = self.session.post(
            f"{DATA_API}/enrich/ranktwosided",
            json=payload,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        result = resp.json()

        # Attach mapping info for downstream use
        result["_mapped_up"] = mapped_up
        result["_mapped_down"] = mapped_down

        return result

    def resolve_signatures(self, sig_uuids: list[str]) -> list[dict]:
        """Resolve signature UUIDs to metadata (compound, cell line, etc.).

        Args:
            sig_uuids: List of signature UUIDs from enrichment results.

        Returns:
            List of signature metadata dicts.
        """
        if not sig_uuids:
            return []

        # Batch to avoid oversized requests
        all_sigs: list[dict] = []
        for i in range(0, len(sig_uuids), 100):
            batch = sig_uuids[i : i + 100]
            resp = self.session.post(
                f"{METADATA_API}/signatures/find",
                json={
                    "filter": {
                        "where": {"id": {"inq": batch}},
                    }
                },
                timeout=self.timeout,
            )
            resp.raise_for_status()
            all_sigs.extend(resp.json())

        return all_sigs

    def query_fibrosis_signature(
        self,
        up_genes: list[str],
        down_genes: list[str],
        database: str = DEFAULT_DATABASE,
        limit: int = 50,
    ) -> list[dict]:
        """Full workflow: enrich + resolve metadata for top reversing compounds.

        Returns list of dicts with compound name, cell line, scores, etc.
        """
        # Run enrichment
        enrichment = self.enrich_twosided(up_genes, down_genes, database, limit)
        results = enrichment.get("results", [])

        if not results:
            return []

        # Separate reversers and mimickers
        reversers = [r for r in results if r.get("type") == "reversers"]
        # We want reversers for drug repurposing (compounds that reverse disease sig)

        if not reversers:
            # Fall back to all results, filter by direction
            reversers = results

        # Resolve signature metadata
        sig_uuids = [r["uuid"] for r in reversers if "uuid" in r]
        sig_meta = self.resolve_signatures(sig_uuids)
        meta_map = {s["id"]: s for s in sig_meta}

        # Build combined records
        records = []
        for hit in reversers:
            uuid = hit.get("uuid", "")
            meta = meta_map.get(uuid, {})
            meta_inner = meta.get("meta", {})

            records.append({
                "signature_id": uuid,
                "compound": meta_inner.get("pert_name", "unknown"),
                "cell_line": meta_inner.get("cell_line", ""),
                "pert_type": meta_inner.get("pert_type", ""),
                "z_up": hit.get("z-up", 0),
                "z_down": hit.get("z-down", 0),
                "z_sum": hit.get("z-sum", 0),
                "p_up": hit.get("p-up", 1),
                "p_down": hit.get("p-down", 1),
                "fdr_up": hit.get("fdr-up", 1),
                "fdr_down": hit.get("fdr-down", 1),
                "logp_fisher": hit.get("logp-fisher", 0),
                "rank": hit.get("rank", 0),
                "direction": hit.get("type", ""),
            })

        return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="SigCom LINCS API client for CD fibrosis drug repurposing"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test API connectivity and list databases",
    )
    parser.add_argument(
        "--query",
        type=Path,
        help="Query a signature TSV file (gene, direction columns)",
    )
    parser.add_argument(
        "--database",
        default=DEFAULT_DATABASE,
        help=f"SigCom database (default: {DEFAULT_DATABASE})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max results per direction (default: 50)",
    )
    args = parser.parse_args(argv)

    client = SigcomClient()

    if args.test:
        print("Testing SigCom LINCS API...")
        dbs = client.list_databases()
        print(f"  Available databases: {len(dbs)}")
        for db in dbs:
            if isinstance(db, dict):
                print(f"    - {db.get('uuid', db.get('name', db))}")
            else:
                print(f"    - {db}")

        # Test gene resolution
        test_genes = ["SERPINE1", "COL1A1", "TGFB1", "ACTA2", "FN1"]
        print(f"\n  Resolving test genes: {test_genes}")
        resolved = client.resolve_genes(test_genes)
        for gene, uid in resolved.items():
            print(f"    {gene}: {uid[:12]}...")
        print(f"  Resolved {len(resolved)}/{len(test_genes)} genes")
        return

    if args.query:
        if not args.query.exists():
            print(f"File not found: {args.query}", file=sys.stderr)
            sys.exit(1)

        up_genes, down_genes = load_signature_tsv(args.query)
        print(f"Loaded: {len(up_genes)} up, {len(down_genes)} down genes")
        print(f"Database: {args.database}")
        print(f"Limit: {args.limit}")

        results = client.query_fibrosis_signature(
            up_genes, down_genes, args.database, args.limit
        )

        print(f"\nTop reversing compounds ({len(results)} results):")
        print(f"{'Compound':35s} {'Cell':12s} {'Z-sum':>8s} {'logP':>8s}")
        print(f"{'-'*35} {'-'*12} {'-'*8} {'-'*8}")
        for r in results[:20]:
            print(
                f"{r['compound']:35s} {r['cell_line']:12s} "
                f"{r['z_sum']:>+8.2f} {r['logp_fisher']:>8.1f}"
            )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
