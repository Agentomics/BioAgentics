"""CMAP/L1000 Connectivity Map API client for drug repurposing queries.

Queries fibrosis gene signatures against the Connectivity Map (clue.io)
to identify compounds that reverse fibrotic expression patterns.

The clue.io API requires a user key (free for academic use):
  https://clue.io/connectopedia/api_documentation

Set the API key via environment variable:
  export CLUE_API_KEY=your_key_here

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.cmap_client --test-connection
    uv run python -m bioagentics.data.cd_fibrosis.cmap_client --query-signature path/to/sig.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests

from bioagentics.config import REPO_ROOT

CLUE_API_BASE = "https://api.clue.io"
DEFAULT_DEST = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing"
TIMEOUT = 120

# L1000 landmark genes (978 genes measured directly on L1000 platform)
# The full list is available from clue.io; these are the most fibrosis-relevant
FIBROSIS_LANDMARK_GENES = [
    "SERPINE1", "FN1", "TGFB1", "TGFBR1", "TGFBR2", "SMAD2", "SMAD3",
    "SMAD4", "SMAD7", "COL1A1", "COL1A2", "COL3A1", "ACTA2", "CTGF",
    "MMP2", "MMP9", "TIMP1", "VIM", "CDH2", "SNAI1", "SNAI2", "TWIST1",
    "ZEB1", "ZEB2", "FGF2", "PDGFRA", "PDGFRB", "WNT5A", "CTNNB1",
    "YAP1", "WWTR1", "LATS1", "LATS2", "HDAC1", "HDAC2",
]

# Fibroblast-relevant cell lines in L1000 (lung fibroblasts as surrogates
# since no intestinal fibroblast data exists in CMAP)
FIBROBLAST_CELL_LINES = [
    "IMR90",    # Fetal lung fibroblasts — primary surrogate
    "WI38",     # Embryonic lung fibroblasts
    "BJ",       # Foreskin fibroblasts
    "HFF",      # Human foreskin fibroblasts
    "CCD18CO",  # Colon fibroblasts (if available)
]


class CmapClient:
    """Client for the clue.io Connectivity Map API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("CLUE_API_KEY", "")
        self.session = requests.Session()
        if self.api_key:
            self.session.headers["user_key"] = self.api_key

    def test_connection(self) -> bool:
        """Test API connectivity and authentication."""
        try:
            resp = self.session.get(
                f"{CLUE_API_BASE}/api/cell_lines",
                params={"filter": json.dumps({"limit": 1})},
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            print(f"  Connection OK. Sample response: {len(data)} record(s)")
            return True
        except requests.RequestException as e:
            print(f"  Connection failed: {e}", file=sys.stderr)
            return False

    def get_cell_lines(self, names: list[str] | None = None) -> list[dict]:
        """Fetch cell line metadata. Optionally filter by name."""
        filt: dict = {}
        if names:
            filt["where"] = {"cell_iname": {"inq": [n.lower() for n in names]}}

        resp = self.session.get(
            f"{CLUE_API_BASE}/api/cell_lines",
            params={"filter": json.dumps(filt)} if filt else None,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def query_signatures(
        self,
        up_genes: list[str],
        down_genes: list[str],
        cell_lines: list[str] | None = None,
    ) -> dict:
        """Submit a gene signature query to CMAP.

        Args:
            up_genes: Genes upregulated in fibrosis (to find compounds that downregulate)
            down_genes: Genes downregulated in fibrosis (to find compounds that upregulate)
            cell_lines: Optional list of cell lines to restrict results to

        Returns:
            Query result with connectivity scores per compound.
            Negative scores = signature reversal = potential anti-fibrotic.
        """
        payload = {
            "up_genes": up_genes,
            "down_genes": down_genes,
        }
        if cell_lines:
            payload["cell_lines"] = cell_lines

        # Submit query
        resp = self.session.post(
            f"{CLUE_API_BASE}/api/queries",
            json=payload,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json()

        job_id = result.get("job_id") or result.get("id")
        if not job_id:
            return result

        # Poll for completion
        return self._poll_query(job_id)

    def _poll_query(self, job_id: str, max_wait: int = 600) -> dict:
        """Poll for query completion."""
        print(f"  Waiting for query {job_id}...")
        start = time.time()
        while time.time() - start < max_wait:
            resp = self.session.get(
                f"{CLUE_API_BASE}/api/queries/{job_id}",
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            result = resp.json()

            status = result.get("status", "")
            if status == "completed":
                print(f"  Query completed in {time.time() - start:.0f}s")
                return result
            if status == "failed":
                raise RuntimeError(f"CMAP query failed: {result}")

            time.sleep(10)

        raise TimeoutError(f"Query {job_id} did not complete within {max_wait}s")

    def get_perturbagen_info(self, pert_ids: list[str]) -> list[dict]:
        """Get compound metadata for perturbagen IDs from CMAP results."""
        filt = {"where": {"pert_id": {"inq": pert_ids}}}
        resp = self.session.get(
            f"{CLUE_API_BASE}/api/perts",
            params={"filter": json.dumps(filt)},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def get_compound_signatures(
        self,
        compound_name: str,
        cell_lines: list[str] | None = None,
    ) -> list[dict]:
        """Get L1000 expression signatures for a specific compound.

        Useful for checking dose-response effects on fibrosis markers.
        """
        filt: dict = {
            "where": {
                "pert_iname": compound_name.lower(),
                "pert_type": "trt_cp",
            }
        }
        if cell_lines:
            filt["where"]["cell_iname"] = {"inq": [c.lower() for c in cell_lines]}

        resp = self.session.get(
            f"{CLUE_API_BASE}/api/sigs",
            params={"filter": json.dumps(filt)},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()


def load_signature_tsv(path: Path) -> tuple[list[str], list[str]]:
    """Load a gene signature from TSV file.

    Expects columns: gene, direction (up/down), and optionally logFC, pvalue.
    Returns (up_genes, down_genes).
    """
    up_genes = []
    down_genes = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row["gene"]
            direction = row.get("direction", "").lower()
            if direction == "up":
                up_genes.append(gene)
            elif direction == "down":
                down_genes.append(gene)
    return up_genes, down_genes


def save_cmap_results(results: list[dict], dest: Path) -> None:
    """Save CMAP query results to TSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(dest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CMAP/L1000 connectivity scoring for CD fibrosis signatures"
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test API connectivity",
    )
    parser.add_argument(
        "--query-signature",
        type=Path,
        help="Path to signature TSV file (gene, direction columns)",
    )
    parser.add_argument(
        "--cell-lines",
        nargs="+",
        default=FIBROBLAST_CELL_LINES,
        help="Cell lines to query (default: fibroblast lines)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Output directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args(argv)

    client = CmapClient()

    if not client.api_key:
        print(
            "No CLUE API key found.\n\n"
            "To use the CMAP/L1000 API:\n"
            "  1. Register at https://clue.io/\n"
            "  2. Get your API key from your profile\n"
            "  3. Set: export CLUE_API_KEY=your_key\n",
            file=sys.stderr,
        )
        if not args.test_connection:
            sys.exit(1)

    if args.test_connection:
        print("Testing CMAP API connection...")
        ok = client.test_connection()

        # Also check for fibroblast cell lines
        if ok:
            print("\nChecking fibroblast cell lines in L1000...")
            lines = client.get_cell_lines(FIBROBLAST_CELL_LINES)
            found = [l.get("cell_iname", "") for l in lines]
            for name in FIBROBLAST_CELL_LINES:
                status = "available" if name.lower() in [f.lower() for f in found] else "NOT FOUND"
                print(f"  {name}: {status}")

        sys.exit(0 if ok else 1)

    if args.query_signature:
        if not args.query_signature.exists():
            print(f"Signature file not found: {args.query_signature}", file=sys.stderr)
            sys.exit(1)

        up_genes, down_genes = load_signature_tsv(args.query_signature)
        print(f"Loaded signature: {len(up_genes)} up, {len(down_genes)} down genes")
        print(f"Cell lines: {args.cell_lines}")

        results = client.query_signatures(up_genes, down_genes, args.cell_lines)
        args.dest.mkdir(parents=True, exist_ok=True)
        out_path = args.dest / f"cmap_results_{args.query_signature.stem}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
