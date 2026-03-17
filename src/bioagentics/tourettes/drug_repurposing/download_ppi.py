"""Download and parse STRING/BioGRID PPI data for human interactome.

Downloads:
- STRING v12.0 human protein links (combined score >= 700)
- BioGRID human physical interactions

Parses into a unified edge list: [protein_a, protein_b, source, confidence_score]
with gene symbol IDs (mapped from STRING protein IDs).

Output: data/tourettes/ts-drug-repurposing-network/human_ppi_network.tsv

Usage:
    uv run python -m bioagentics.tourettes.drug_repurposing.download_ppi
"""

from __future__ import annotations

import argparse
import gzip
import io
from pathlib import Path

import pandas as pd
import requests

from bioagentics.config import REPO_ROOT

DATA_DIR = REPO_ROOT / "data" / "tourettes" / "ts-drug-repurposing-network"
OUTPUT_PATH = DATA_DIR / "human_ppi_network.tsv"

# STRING v12.0 — human (taxon 9606)
STRING_LINKS_URL = (
    "https://stringdb-downloads.org/download/"
    "protein.links.v12.0/9606.protein.links.v12.0.txt.gz"
)
STRING_INFO_URL = (
    "https://stringdb-downloads.org/download/"
    "protein.info.v12.0/9606.protein.info.v12.0.txt.gz"
)
STRING_MIN_SCORE = 700  # High confidence threshold

# BioGRID — latest release, human physical interactions
BIOGRID_TAB_URL = (
    "https://downloads.thebiogrid.org/Download/BioGRID/Latest-Release/"
    "BIOGRID-ORGANISM-LATEST.tab3.zip"
)
BIOGRID_HUMAN_FILE = "BIOGRID-ORGANISM-Homo_sapiens"


def download_string_id_mapping(data_dir: Path) -> dict[str, str]:
    """Download STRING protein info and return mapping: STRING_id -> gene_symbol."""
    cache = data_dir / "string_protein_info.tsv.gz"
    if cache.exists():
        print("  Using cached STRING protein info")
    else:
        print("  Downloading STRING protein info...")
        resp = requests.get(STRING_INFO_URL, timeout=120, stream=True)
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

    mapping: dict[str, str] = {}
    with gzip.open(cache, "rt") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                string_id = parts[0]  # 9606.ENSPxxx
                gene_name = parts[1]  # preferred gene name
                if gene_name and gene_name != "":
                    mapping[string_id] = gene_name
    print(f"  STRING ID mapping: {len(mapping)} proteins")
    return mapping


def download_string_links(data_dir: Path, id_map: dict[str, str]) -> pd.DataFrame:
    """Download STRING links and filter to high-confidence, mapped to gene symbols."""
    cache = data_dir / "string_protein_links.txt.gz"
    if cache.exists():
        print("  Using cached STRING protein links")
    else:
        print("  Downloading STRING protein links (~400 MB compressed)...")
        resp = requests.get(STRING_LINKS_URL, timeout=600, stream=True)
        resp.raise_for_status()
        with open(cache, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

    print("  Parsing STRING links (filtering score >= {})...".format(STRING_MIN_SCORE))
    rows = []
    with gzip.open(cache, "rt") as f:
        header = f.readline()  # "protein1 protein2 combined_score"
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            p1, p2, score = parts[0], parts[1], int(parts[2])
            if score < STRING_MIN_SCORE:
                continue
            g1 = id_map.get(p1)
            g2 = id_map.get(p2)
            if g1 and g2 and g1 != g2:
                # Canonical ordering to avoid duplicates
                if g1 > g2:
                    g1, g2 = g2, g1
                rows.append((g1, g2, "STRING", score / 1000.0))

    df = pd.DataFrame(rows, columns=["protein_a", "protein_b", "source", "confidence_score"])
    df = df.drop_duplicates(subset=["protein_a", "protein_b"])
    print(f"  STRING: {len(df)} high-confidence interactions")
    return df


def download_biogrid_interactions(data_dir: Path) -> pd.DataFrame:
    """Download BioGRID human physical interactions."""
    import zipfile

    cache = data_dir / "biogrid_organism.tab3.zip"
    if cache.exists():
        print("  Using cached BioGRID data")
    else:
        print("  Downloading BioGRID data...")
        resp = requests.get(BIOGRID_TAB_URL, timeout=300, stream=True)
        resp.raise_for_status()
        with open(cache, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)

    print("  Parsing BioGRID human physical interactions...")
    rows = []
    with zipfile.ZipFile(cache) as zf:
        # Find the human file
        human_files = [n for n in zf.namelist() if BIOGRID_HUMAN_FILE in n and n.endswith(".tab3.txt")]
        if not human_files:
            print("  WARNING: No BioGRID human file found in archive")
            return pd.DataFrame(columns=["protein_a", "protein_b", "source", "confidence_score"])

        with zf.open(human_files[0]) as f:
            reader = io.TextIOWrapper(f, encoding="utf-8")
            header = reader.readline().strip().split("\t")
            # Find column indices
            try:
                gene_a_idx = header.index("Official Symbol Interactor A")
                gene_b_idx = header.index("Official Symbol Interactor B")
                exp_idx = header.index("Experimental System Type")
                org_a_idx = header.index("Organism ID Interactor A")
                org_b_idx = header.index("Organism ID Interactor B")
            except ValueError:
                # Try alternate column names
                gene_a_idx = header.index("OFFICIAL_SYMBOL_A") if "OFFICIAL_SYMBOL_A" in header else 7
                gene_b_idx = header.index("OFFICIAL_SYMBOL_B") if "OFFICIAL_SYMBOL_B" in header else 8
                exp_idx = header.index("EXPERIMENTAL_SYSTEM_TYPE") if "EXPERIMENTAL_SYSTEM_TYPE" in header else 12
                org_a_idx = header.index("ORGANISM_A_ID") if "ORGANISM_A_ID" in header else 15
                org_b_idx = header.index("ORGANISM_B_ID") if "ORGANISM_B_ID" in header else 16

            for line in reader:
                parts = line.strip().split("\t")
                if len(parts) <= max(gene_a_idx, gene_b_idx, exp_idx, org_a_idx, org_b_idx):
                    continue
                # Filter: human-human, physical interactions only
                if parts[org_a_idx] != "9606" or parts[org_b_idx] != "9606":
                    continue
                if parts[exp_idx] != "physical":
                    continue
                g1 = parts[gene_a_idx]
                g2 = parts[gene_b_idx]
                if g1 and g2 and g1 != g2:
                    if g1 > g2:
                        g1, g2 = g2, g1
                    rows.append((g1, g2, "BioGRID", 0.9))

    df = pd.DataFrame(rows, columns=["protein_a", "protein_b", "source", "confidence_score"])
    df = df.drop_duplicates(subset=["protein_a", "protein_b"])
    print(f"  BioGRID: {len(df)} physical interactions")
    return df


def merge_networks(string_df: pd.DataFrame, biogrid_df: pd.DataFrame) -> pd.DataFrame:
    """Merge STRING and BioGRID into unified PPI network.

    For edges present in both sources, keep the higher confidence score
    and annotate source as "STRING+BioGRID".
    """
    # Create merge keys
    string_df = string_df.copy()
    biogrid_df = biogrid_df.copy()
    string_df["edge_key"] = string_df["protein_a"] + "|" + string_df["protein_b"]
    biogrid_df["edge_key"] = biogrid_df["protein_a"] + "|" + biogrid_df["protein_b"]

    # Find overlapping edges
    overlap = set(string_df["edge_key"]) & set(biogrid_df["edge_key"])
    print(f"  Overlap: {len(overlap)} edges in both databases")

    # STRING-only edges
    string_only = string_df[~string_df["edge_key"].isin(overlap)].copy()
    # BioGRID-only edges
    biogrid_only = biogrid_df[~biogrid_df["edge_key"].isin(overlap)].copy()
    # Overlapping: take max score, mark dual source
    overlap_df = string_df[string_df["edge_key"].isin(overlap)].copy()
    biogrid_overlap = biogrid_df[biogrid_df["edge_key"].isin(overlap)].set_index("edge_key")
    for idx in overlap_df.index:
        key = overlap_df.at[idx, "edge_key"]
        bg_score = biogrid_overlap.at[key, "confidence_score"] if key in biogrid_overlap.index else 0
        overlap_df.at[idx, "confidence_score"] = max(overlap_df.at[idx, "confidence_score"], bg_score)
        overlap_df.at[idx, "source"] = "STRING+BioGRID"

    merged = pd.concat([string_only, biogrid_only, overlap_df], ignore_index=True)
    merged = merged.drop(columns=["edge_key"])
    merged = merged.sort_values("confidence_score", ascending=False).reset_index(drop=True)
    print(f"  Merged network: {len(merged)} unique edges")
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and parse PPI data")
    parser.add_argument("--string-only", action="store_true", help="Skip BioGRID download")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # STRING
    id_map = download_string_id_mapping(DATA_DIR)
    string_df = download_string_links(DATA_DIR, id_map)

    if args.string_only:
        merged = string_df
    else:
        biogrid_df = download_biogrid_interactions(DATA_DIR)
        merged = merge_networks(string_df, biogrid_df)

    merged.to_csv(args.output, sep="\t", index=False)
    print(f"  Saved to {args.output}")

    # Summary stats
    unique_proteins = set(merged["protein_a"]) | set(merged["protein_b"])
    print(f"  Unique proteins: {len(unique_proteins)}")
    print(f"  Source breakdown:")
    for src, count in merged["source"].value_counts().items():
        print(f"    {src}: {count}")


if __name__ == "__main__":
    main()
