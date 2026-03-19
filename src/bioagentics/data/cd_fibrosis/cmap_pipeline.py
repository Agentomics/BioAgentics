"""CMAP/L1000 + iLINCS connectivity scoring pipeline for CD fibrosis drug repurposing.

Queries all fibrosis signatures against LINCS compound perturbation libraries
to identify compounds that reverse fibrotic gene expression patterns.

Pipeline:
1. Load all fibrosis signatures (bulk, cell-type, transition, GLIS3/IL-11,
   CTHRC1/YAP-TAZ, TL1A-DR3/Rho)
2. Query each signature against iLINCS LIB_5 (143k chemical perturbagen signatures)
3. Optionally query against clue.io CMAP (requires CLUE_API_KEY)
4. Rank compounds by negative concordance/connectivity (signature reversal)
5. Prioritize hits from fibroblast-relevant cell lines
6. Identify convergent hits across multiple signatures
7. Output ranked compound list

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.cmap_pipeline
    uv run python -m bioagentics.data.cd_fibrosis.cmap_pipeline --signatures transition
    uv run python -m bioagentics.data.cd_fibrosis.cmap_pipeline --top-n 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.cd_fibrosis.cmap_client import (
    FIBROBLAST_CELL_LINES,
    CmapClient,
    load_signature_tsv,
)
from bioagentics.data.cd_fibrosis.ilincs_client import IlincsClient
from bioagentics.data.cd_fibrosis.l1000_local import (
    DATA_DIR as L1000_DATA_DIR,
    run_local_scoring,
)

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing"
SIGNATURES_DIR = OUTPUT_DIR / "signatures"

# All available fibrosis signatures
SIGNATURE_FILES = {
    "bulk": "bulk_tissue_signature.tsv",
    "celltype": "celltype_fibroblast_signature.tsv",
    "transition": "transition_signature.tsv",
    "glis3_il11": "glis3_il11_signature.tsv",
    "cthrc1_yaptaz": "cthrc1_yaptaz_signature.tsv",
    "tl1a_dr3_rho": "tl1a_dr3_rho_signature.tsv",
    "fas_twist1_fistula": "fas_twist1_fistula_signature.tsv",
    "acharjee_stricture": "acharjee_stricture_signature.tsv",
}

# Fibroblast-relevant cell line identifiers in iLINCS
FIBROBLAST_CELL_LINE_PATTERNS = [
    "imr90", "imr-90", "wi38", "wi-38", "bj", "hff",
    "ccd18co", "ccd-18co", "lung fibroblast", "fibroblast",
]


def load_all_signatures(
    signatures_dir: Path | None = None,
    which: list[str] | None = None,
) -> dict[str, tuple[list[str], list[str]]]:
    """Load fibrosis signatures as (up_genes, down_genes) tuples.

    Args:
        signatures_dir: Directory containing signature TSV files.
        which: Subset of signature names to load (default: all).

    Returns:
        Dict mapping signature name -> (up_genes, down_genes).
    """
    signatures_dir = signatures_dir or SIGNATURES_DIR
    selected = which or list(SIGNATURE_FILES.keys())

    loaded = {}
    for name in selected:
        if name not in SIGNATURE_FILES:
            print(f"  Warning: unknown signature '{name}', skipping")
            continue
        path = signatures_dir / SIGNATURE_FILES[name]
        if not path.exists():
            print(f"  Warning: signature file not found: {path}")
            continue
        up, down = load_signature_tsv(path)
        loaded[name] = (up, down)
        print(f"  {name}: {len(up)} up, {len(down)} down genes")

    return loaded


def query_ilincs_signature(
    client: IlincsClient,
    up_genes: list[str],
    down_genes: list[str],
    library: str = "LIB_5",
) -> list[dict]:
    """Query a signature against iLINCS and return concordance results.

    Converts up/down gene lists to iLINCS format (logFC = +1/-1)
    and queries for discordant signatures (potential anti-fibrotics).
    """
    signature = []
    for gene in up_genes:
        signature.append({"Name_GeneSymbol": gene, "Value_LogDiffExp": 1.0})
    for gene in down_genes:
        signature.append({"Name_GeneSymbol": gene, "Value_LogDiffExp": -1.0})

    return client.query_concordant_signatures(signature, library)


def is_fibroblast_cell_line(cell_line: str) -> bool:
    """Check if a cell line name matches known fibroblast lines."""
    cell_lower = cell_line.lower().strip()
    return any(pat in cell_lower for pat in FIBROBLAST_CELL_LINE_PATTERNS)


def parse_ilincs_results(
    results: list[dict],
    signature_name: str,
) -> pd.DataFrame:
    """Parse iLINCS concordance results into a DataFrame.

    Extracts compound name, concordance score, cell line, concentration,
    and time point from the iLINCS response format.
    """
    records = []
    for hit in results:
        records.append({
            "signature_id": hit.get("signatureid", ""),
            "compound": hit.get("compound", hit.get("perturbagenName", "")),
            "concordance": float(hit.get("similarity", hit.get("concordance", 0))),
            "cell_line": hit.get("cellline", hit.get("cellLine", "")),
            "concentration": hit.get("concentration", ""),
            "time": hit.get("time", ""),
            "query_signature": signature_name,
        })

    df = pd.DataFrame(records)
    if len(df) > 0:
        df = df.sort_values("concordance", ascending=True)
    return df


def rank_compounds_across_signatures(
    all_results: dict[str, pd.DataFrame],
    top_n: int = 200,
) -> pd.DataFrame:
    """Rank compounds by convergent negative concordance across signatures.

    For each compound, computes:
    - Mean concordance across all queried signatures
    - Number of signatures where compound shows negative concordance
    - Whether hits come from fibroblast cell lines
    """
    if not all_results:
        return pd.DataFrame()

    # Combine all results
    combined = []
    for sig_name, df in all_results.items():
        if len(df) == 0:
            continue
        df_copy = df.copy()
        df_copy["query_signature"] = sig_name
        combined.append(df_copy)

    if not combined:
        return pd.DataFrame()

    full_df = pd.concat(combined, ignore_index=True)

    # Aggregate by compound
    compound_stats = []
    for compound, group in full_df.groupby("compound"):
        if not compound:
            continue

        mean_concordance = group["concordance"].mean()
        min_concordance = group["concordance"].min()
        n_signatures = group["query_signature"].nunique()
        n_negative = (group["concordance"] < 0).sum()
        signatures_hit = ";".join(sorted(group["query_signature"].unique()))

        # Check for fibroblast cell line hits
        fibroblast_hits = group[
            group["cell_line"].apply(is_fibroblast_cell_line)
        ]
        has_fibroblast = len(fibroblast_hits) > 0
        fibroblast_concordance = (
            fibroblast_hits["concordance"].mean() if has_fibroblast else None
        )

        compound_stats.append({
            "compound": compound,
            "mean_concordance": round(mean_concordance, 4),
            "min_concordance": round(min_concordance, 4),
            "n_signatures_queried": n_signatures,
            "n_negative_hits": n_negative,
            "signatures_hit": signatures_hit,
            "has_fibroblast_hit": has_fibroblast,
            "fibroblast_concordance": (
                round(fibroblast_concordance, 4)
                if fibroblast_concordance is not None
                else None
            ),
            "convergent_anti_fibrotic": n_negative >= 2,
        })

    ranked = pd.DataFrame(compound_stats)
    if len(ranked) == 0:
        return ranked

    # Sort by convergent signal: more signatures negative > lower mean concordance
    ranked = ranked.sort_values(
        ["n_negative_hits", "mean_concordance"],
        ascending=[False, True],
    )

    return ranked.head(top_n)


def run_pipeline(
    signatures_dir: Path | None = None,
    output_dir: Path | None = None,
    which_signatures: list[str] | None = None,
    use_ilincs: bool = True,
    use_cmap: bool = False,
    use_local_l1000: bool = False,
    l1000_data_dir: Path | None = None,
    ilincs_library: str = "LIB_5",
    top_n: int = 200,
    delay_between_queries: float = 5.0,
) -> pd.DataFrame:
    """Run the full CMAP/iLINCS connectivity scoring pipeline.

    Args:
        signatures_dir: Directory with signature TSV files.
        output_dir: Output directory for results.
        which_signatures: Subset of signatures to query (default: all).
        use_ilincs: Query iLINCS (default: True, no API key needed).
        use_cmap: Query clue.io CMAP (requires CLUE_API_KEY).
        use_local_l1000: Use local L1000 GCTX data (requires download).
        l1000_data_dir: Directory containing L1000 GCTX + metadata.
        ilincs_library: iLINCS library to query.
        top_n: Number of top compounds to return.
        delay_between_queries: Seconds between API calls (rate limiting).

    Returns:
        Ranked DataFrame of compounds with convergent anti-fibrotic signal.
    """
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("CMAP/L1000 + iLINCS Connectivity Scoring Pipeline")
    print("=" * 60)

    # Step 1: Load signatures
    print("\n[1/5] Loading fibrosis signatures...")
    signatures = load_all_signatures(signatures_dir, which_signatures)
    if not signatures:
        print("  ERROR: No signatures loaded. Aborting.")
        return pd.DataFrame()

    all_results: dict[str, pd.DataFrame] = {}

    # Step 2: Query iLINCS
    if use_ilincs:
        print(f"\n[2/5] Querying iLINCS ({ilincs_library})...")
        ilincs = IlincsClient()

        for sig_name, (up_genes, down_genes) in signatures.items():
            print(f"\n  Querying: {sig_name} ({len(up_genes)} up, {len(down_genes)} down)...")
            try:
                results = query_ilincs_signature(
                    ilincs, up_genes, down_genes, ilincs_library
                )
                df = parse_ilincs_results(results, sig_name)
                all_results[f"ilincs_{sig_name}"] = df
                print(f"    Got {len(df)} results")

                # Save per-signature results
                per_sig_path = output_dir / f"ilincs_{sig_name}_hits.tsv"
                df.to_csv(per_sig_path, sep="\t", index=False)

                if len(df) > 0:
                    top_anti = df[df["concordance"] < 0].head(5)
                    if len(top_anti) > 0:
                        print(f"    Top anti-fibrotic candidates:")
                        for _, row in top_anti.iterrows():
                            print(f"      {row['compound']:30s} "
                                  f"concordance={row['concordance']:+.4f} "
                                  f"cell={row['cell_line']}")

                if delay_between_queries > 0:
                    time.sleep(delay_between_queries)

            except Exception as e:
                print(f"    ERROR: {e}")
                continue
    else:
        print("\n[2/5] iLINCS: skipped")

    # Step 3: Query CMAP (if API key available)
    if use_cmap:
        print(f"\n[3/5] Querying CMAP/clue.io...")
        cmap = CmapClient()
        if not cmap.api_key:
            print("  WARNING: No CLUE_API_KEY set. Skipping CMAP queries.")
            print("  Set CLUE_API_KEY to enable clue.io queries.")
        else:
            for sig_name, (up_genes, down_genes) in signatures.items():
                print(f"\n  Querying: {sig_name}...")
                try:
                    results = cmap.query_signatures(
                        up_genes, down_genes, FIBROBLAST_CELL_LINES
                    )
                    # Save raw results
                    raw_path = output_dir / f"cmap_{sig_name}_raw.json"
                    with open(raw_path, "w") as f:
                        json.dump(results, f, indent=2)
                    print(f"    Saved raw results: {raw_path}")
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue
    else:
        print("\n[3/5] CMAP: skipped")

    # Step 4: Local L1000 scoring (alternative to clue.io API)
    if use_local_l1000:
        data_dir = l1000_data_dir or L1000_DATA_DIR
        gctx_files = list(data_dir.glob("*.gctx"))
        siginfo_files = list(data_dir.glob("*sig_info*"))
        geneinfo_files = list(data_dir.glob("*gene_info*"))

        if not gctx_files or not siginfo_files or not geneinfo_files:
            print(f"\n[4/5] Local L1000: data files not found in {data_dir}")
            print("  Run: uv run python -m bioagentics.data.cd_fibrosis.l1000_local --download")
        else:
            print(f"\n[4/5] Local L1000 scoring ({gctx_files[0].name})...")
            for sig_name, (up_genes, down_genes) in signatures.items():
                print(f"\n  Scoring: {sig_name} ({len(up_genes)} up, {len(down_genes)} down)...")
                try:
                    local_df = run_local_scoring(
                        gctx_path=gctx_files[0],
                        siginfo_path=siginfo_files[0],
                        geneinfo_path=geneinfo_files[0],
                        up_genes=up_genes,
                        down_genes=down_genes,
                        top_n=top_n,
                    )
                    if len(local_df) > 0:
                        # Convert to standard format for ranking
                        local_results = pd.DataFrame({
                            "signature_id": "",
                            "compound": local_df["compound"],
                            "concordance": local_df["mean_score"],
                            "cell_line": local_df["cell_lines"],
                            "concentration": "",
                            "time": "",
                            "query_signature": sig_name,
                        })
                        all_results[f"l1000_{sig_name}"] = local_results
                        print(f"    Got {len(local_df)} compound scores")

                        per_sig_path = output_dir / f"l1000_{sig_name}_hits.tsv"
                        local_df.to_csv(per_sig_path, sep="\t", index=False)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue
    else:
        print("\n[4/5] Local L1000: skipped")

    # Step 5: Rank compounds
    print(f"\n[5/5] Ranking compounds (top {top_n})...")
    ranked = rank_compounds_across_signatures(all_results, top_n)

    if len(ranked) == 0:
        print("  No results to rank.")
        return ranked

    # Summary stats
    convergent = ranked[ranked["convergent_anti_fibrotic"]]
    fibroblast = ranked[ranked["has_fibroblast_hit"]]

    print(f"\n  Total compounds ranked: {len(ranked)}")
    print(f"  Convergent anti-fibrotic (negative in >=2 signatures): {len(convergent)}")
    print(f"  With fibroblast cell line hits: {len(fibroblast)}")

    # Save
    out_path = output_dir / "cmap_hits.tsv"
    ranked.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Saved: {out_path}")

    # Print top 20
    print(f"\n  Top 20 anti-fibrotic candidates:")
    print(f"  {'Compound':35s} {'MeanConc':>10s} {'NSigs':>6s} {'NNeg':>5s} {'Fibro':>6s}")
    print(f"  {'-'*35} {'-'*10} {'-'*6} {'-'*5} {'-'*6}")
    for _, row in ranked.head(20).iterrows():
        print(f"  {row['compound']:35s} "
              f"{row['mean_concordance']:>+10.4f} "
              f"{row['n_signatures_queried']:>6d} "
              f"{row['n_negative_hits']:>5d} "
              f"{'yes' if row['has_fibroblast_hit'] else 'no':>6s}")

    return ranked


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="CMAP/iLINCS connectivity scoring pipeline for CD fibrosis"
    )
    parser.add_argument(
        "--signatures",
        nargs="+",
        choices=list(SIGNATURE_FILES.keys()),
        help="Signatures to query (default: all)",
    )
    parser.add_argument(
        "--no-ilincs",
        action="store_true",
        help="Skip iLINCS queries",
    )
    parser.add_argument(
        "--use-cmap",
        action="store_true",
        help="Enable CMAP/clue.io queries (requires CLUE_API_KEY)",
    )
    parser.add_argument(
        "--use-local-l1000",
        action="store_true",
        help="Use local L1000 GCTX data (requires prior download via l1000_local.py)",
    )
    parser.add_argument(
        "--l1000-data-dir",
        type=Path,
        default=None,
        help="Directory containing L1000 GCTX + metadata files",
    )
    parser.add_argument(
        "--library",
        default="LIB_5",
        help="iLINCS library (default: LIB_5 = LINCS chemical perturbagens)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=200,
        help="Number of top compounds to output (default: 200)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Delay between API queries in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Output directory",
    )
    args = parser.parse_args(argv)

    run_pipeline(
        output_dir=args.output_dir,
        which_signatures=args.signatures,
        use_ilincs=not args.no_ilincs,
        use_cmap=args.use_cmap,
        use_local_l1000=args.use_local_l1000,
        l1000_data_dir=args.l1000_data_dir,
        ilincs_library=args.library,
        top_n=args.top_n,
        delay_between_queries=args.delay,
    )


if __name__ == "__main__":
    main()
