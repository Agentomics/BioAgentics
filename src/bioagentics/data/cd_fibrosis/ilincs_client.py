"""iLINCS (integrative LINCS) API client for complementary drug repurposing.

Queries fibrosis signatures against iLINCS as a complement to CMAP/L1000,
following the methodology from JCC 2025 (jjaf137) which demonstrated
this approach for CD fibrosis drug repurposing.

iLINCS API documentation: http://ilincs.org/ilincs/APIDocumentation

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.ilincs_client --test-connection
    uv run python -m bioagentics.data.cd_fibrosis.ilincs_client --query-signature path/to/sig.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import requests

from bioagentics.config import REPO_ROOT

ILINCS_API_BASE = "http://www.ilincs.org/api"
DEFAULT_DEST = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing"
TIMEOUT = 120


class IlincsClient:
    """Client for the iLINCS REST API."""

    def __init__(self):
        self.session = requests.Session()

    def test_connection(self) -> bool:
        """Test API connectivity."""
        try:
            resp = self.session.get(
                f"{ILINCS_API_BASE}/SignatureLibraries",
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            libraries = resp.json()
            print(f"  Connection OK. Available signature libraries: {len(libraries)}")
            for lib in libraries[:5]:
                print(f"    - {lib.get('libraryName', 'unknown')}: "
                      f"{lib.get('numberOfSignatures', '?')} signatures")
            return True
        except requests.RequestException as e:
            print(f"  Connection failed: {e}", file=sys.stderr)
            return False

    def get_signature_libraries(self) -> list[dict]:
        """List available signature libraries."""
        resp = self.session.get(
            f"{ILINCS_API_BASE}/SignatureLibraries",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def query_concordant_signatures(
        self,
        signature: list[dict],
        library: str = "LIB_5",
    ) -> list[dict]:
        """Find concordant/discordant signatures in iLINCS.

        Args:
            signature: List of dicts with keys 'Name_GeneSymbol' and 'Value_LogDiffExp'
                       (gene symbol and log fold change).
            library: Signature library to query against.
                     LIB_5 = LINCS L1000 chemical perturbagens.

        Returns:
            List of signature matches with concordance scores.
            Discordant signatures (negative concordance) = potential anti-fibrotic.
        """
        payload = {
            "signatures": json.dumps(signature),
            "library": library,
        }

        resp = self.session.post(
            f"{ILINCS_API_BASE}/SignatureMeta/findConcordantSignatures",
            data=payload,
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def get_signature_data(self, signature_id: str) -> dict:
        """Get full signature data (gene-level z-scores) for a specific signature."""
        resp = self.session.get(
            f"{ILINCS_API_BASE}/Signatures/{signature_id}",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def get_connected_drugs(self, signature_id: str) -> list[dict]:
        """Get drugs connected to a specific signature via DGIdb integration."""
        resp = self.session.get(
            f"{ILINCS_API_BASE}/Signatures/{signature_id}/ConnectedDrugs",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def get_connected_tfs(self, signature_id: str) -> list[dict]:
        """Get transcription factors connected to a signature via TRRUST."""
        resp = self.session.get(
            f"{ILINCS_API_BASE}/Signatures/{signature_id}/ConnectedTFs",
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()

    def search_compounds(self, name: str) -> list[dict]:
        """Search for compound signatures by name."""
        resp = self.session.get(
            f"{ILINCS_API_BASE}/SignatureMeta",
            params={
                "filter": json.dumps({
                    "where": {"compound": {"like": f"%{name}%"}},
                    "limit": 50,
                }),
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()


def load_signature_for_ilincs(path: Path) -> list[dict]:
    """Load a gene signature TSV and format for iLINCS query.

    Expects columns: gene, logFC (or direction with logFC).
    Returns list of dicts with Name_GeneSymbol and Value_LogDiffExp.
    """
    signature = []
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row["gene"]
            # Use logFC if available, otherwise map direction to ±1
            if "logFC" in row:
                value = float(row["logFC"])
            elif "direction" in row:
                value = 1.0 if row["direction"].lower() == "up" else -1.0
            else:
                continue
            signature.append({
                "Name_GeneSymbol": gene,
                "Value_LogDiffExp": value,
            })
    return signature


def save_ilincs_results(results: list[dict], dest: Path) -> None:
    """Save iLINCS concordance results to TSV."""
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(dest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="iLINCS connectivity scoring for CD fibrosis signatures"
    )
    parser.add_argument(
        "--test-connection",
        action="store_true",
        help="Test API connectivity",
    )
    parser.add_argument(
        "--query-signature",
        type=Path,
        help="Path to signature TSV file (gene, logFC or direction columns)",
    )
    parser.add_argument(
        "--library",
        default="LIB_5",
        help="iLINCS library to query (default: LIB_5 = LINCS chemical perturbagens)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Output directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args(argv)

    client = IlincsClient()

    if args.test_connection:
        print("Testing iLINCS API connection...")
        ok = client.test_connection()
        sys.exit(0 if ok else 1)

    if args.query_signature:
        if not args.query_signature.exists():
            print(f"Signature file not found: {args.query_signature}", file=sys.stderr)
            sys.exit(1)

        signature = load_signature_for_ilincs(args.query_signature)
        print(f"Loaded signature: {len(signature)} genes")
        print(f"Library: {args.library}")

        results = client.query_concordant_signatures(signature, args.library)
        args.dest.mkdir(parents=True, exist_ok=True)

        # Save full results
        out_path = args.dest / f"ilincs_results_{args.query_signature.stem}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")

        # Save as TSV if results are a list of dicts
        if isinstance(results, list) and results:
            tsv_path = args.dest / f"ilincs_results_{args.query_signature.stem}.tsv"
            save_ilincs_results(results, tsv_path)
            print(f"TSV saved to {tsv_path}")

            # Show top discordant hits (potential anti-fibrotics)
            scored = [r for r in results if "concordance" in r]
            if scored:
                scored.sort(key=lambda x: float(x.get("concordance", 0)))
                print(f"\nTop 10 discordant signatures (anti-fibrotic candidates):")
                for hit in scored[:10]:
                    print(f"  {hit.get('compound', 'N/A'):30s} "
                          f"concordance={hit.get('concordance', 'N/A'):>8s} "
                          f"cell_line={hit.get('cellLine', 'N/A')}")


if __name__ == "__main__":
    main()
