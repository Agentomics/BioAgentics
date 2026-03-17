"""Download and parse MSigDB gene sets for fibrosis-relevant pathways.

Downloads GMT (Gene Matrix Transposed) files from MSigDB for pathway-level
filtering of fibrosis signatures: TGFβ, ECM, EMT, Wnt, collagen.

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.msigdb
    uv run python -m bioagentics.data.cd_fibrosis.msigdb --collections H C2 C5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from bioagentics.config import REPO_ROOT

DEFAULT_DEST = REPO_ROOT / "data" / "crohns" / "cd-fibrosis-drug-repurposing" / "msigdb"
TIMEOUT = 60
CHUNK_SIZE = 1024 * 64

# MSigDB download base (v2024.1.Hs — latest human gene symbols)
MSIGDB_BASE = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2024.1.Hs"

# Collections relevant to fibrosis drug repurposing
COLLECTIONS: dict[str, dict] = {
    "H": {
        "filename": "h.all.v2024.1.Hs.symbols.gmt",
        "description": "Hallmark gene sets (EMT, TGF-beta, Wnt, inflammatory response)",
    },
    "C2_CP_REACTOME": {
        "filename": "c2.cp.reactome.v2024.1.Hs.symbols.gmt",
        "description": "Reactome pathways (ECM organization, collagen, integrin signaling)",
    },
    "C2_CP_KEGG": {
        "filename": "c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt",
        "description": "KEGG pathways (TGF-beta, Wnt, focal adhesion)",
    },
    "C5_GO_BP": {
        "filename": "c5.go.bp.v2024.1.Hs.symbols.gmt",
        "description": "GO Biological Process (ECM assembly, wound healing, fibroblast proliferation)",
    },
}

# Fibrosis-relevant keywords for filtering gene sets
FIBROSIS_KEYWORDS = [
    "fibros",
    "collagen",
    "extracellular_matrix",
    "ecm",
    "tgf",
    "transforming_growth_factor",
    "wnt",
    "epithelial_mesenchymal",
    "emt",
    "wound_heal",
    "myofibroblast",
    "integrin",
    "matrix_metalloproteinase",
    "mmp",
    "yap",
    "hippo",
    "mechanotransduction",
    "smad",
    "bmp",
    "bone_morphogenetic",
    "connective_tissue",
    "basement_membrane",
    "laminin",
    "fibronectin",
    "elastin",
    "proteoglycan",
]


def download_gmt(collection: str, dest_dir: Path) -> Path:
    """Download a GMT file from MSigDB."""
    info = COLLECTIONS[collection]
    filename = info["filename"]
    url = f"{MSIGDB_BASE}/{filename}"
    dest_file = dest_dir / filename

    if dest_file.exists():
        print(f"  {filename} already exists, skipping")
        return dest_file

    print(f"  Downloading {filename}...")
    with requests.get(url, stream=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(dest_file, "wb") as f,
            tqdm(total=total or None, unit="B", unit_scale=True, desc=filename) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))

    return dest_file


def parse_gmt(gmt_path: Path) -> dict[str, list[str]]:
    """Parse a GMT file into a dict of gene_set_name -> gene list."""
    gene_sets: dict[str, list[str]] = {}
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            name = parts[0]
            # parts[1] is the description/URL, parts[2:] are genes
            genes = [g for g in parts[2:] if g]
            gene_sets[name] = genes
    return gene_sets


def filter_fibrosis_sets(
    gene_sets: dict[str, list[str]],
    keywords: list[str] | None = None,
) -> dict[str, list[str]]:
    """Filter gene sets to those matching fibrosis-relevant keywords."""
    kw = keywords or FIBROSIS_KEYWORDS
    filtered = {}
    for name, genes in gene_sets.items():
        name_lower = name.lower()
        if any(kw_item in name_lower for kw_item in kw):
            filtered[name] = genes
    return filtered


def save_filtered_sets(gene_sets: dict[str, list[str]], dest: Path) -> None:
    """Save filtered gene sets as a GMT file."""
    with open(dest, "w") as f:
        for name, genes in sorted(gene_sets.items()):
            f.write(f"{name}\tfibrosis_relevant\t{chr(9).join(genes)}\n")


def get_combined_fibrosis_genes(dest_dir: Path) -> set[str]:
    """Load all downloaded GMTs and return the union of fibrosis-relevant genes.

    Useful for filtering DE results to fibrosis pathways.
    """
    all_genes: set[str] = set()
    for gmt_file in dest_dir.glob("*.gmt"):
        if gmt_file.name == "fibrosis_relevant_sets.gmt":
            continue
        gene_sets = parse_gmt(gmt_file)
        fibrosis_sets = filter_fibrosis_sets(gene_sets)
        for genes in fibrosis_sets.values():
            all_genes.update(genes)
    return all_genes


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download MSigDB gene sets for fibrosis pathways"
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        choices=[*COLLECTIONS, "all"],
        default=["all"],
        help="Collections to download (default: all)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args(argv)

    selected = list(COLLECTIONS) if "all" in args.collections else args.collections
    args.dest.mkdir(parents=True, exist_ok=True)

    print("CD Fibrosis Drug Repurposing — MSigDB Gene Set Download")
    print(f"Destination: {args.dest}\n")

    all_fibrosis_sets: dict[str, list[str]] = {}
    failed = []

    for collection in selected:
        info = COLLECTIONS[collection]
        print(f"--- {collection}: {info['description']} ---")

        try:
            gmt_path = download_gmt(collection, args.dest)
            gene_sets = parse_gmt(gmt_path)
            fibrosis_sets = filter_fibrosis_sets(gene_sets)

            print(f"  Total sets: {len(gene_sets)}")
            print(f"  Fibrosis-relevant sets: {len(fibrosis_sets)}")
            if fibrosis_sets:
                total_genes = len({g for genes in fibrosis_sets.values() for g in genes})
                print(f"  Unique fibrosis genes: {total_genes}")

            all_fibrosis_sets.update(fibrosis_sets)

        except (requests.RequestException, OSError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failed.append(collection)

        print()

    # Save combined fibrosis-relevant gene sets
    if all_fibrosis_sets:
        combined_path = args.dest / "fibrosis_relevant_sets.gmt"
        save_filtered_sets(all_fibrosis_sets, combined_path)
        total_genes = len({g for genes in all_fibrosis_sets.values() for g in genes})
        print(f"Saved {len(all_fibrosis_sets)} fibrosis-relevant sets "
              f"({total_genes} unique genes) to {combined_path}")

    if failed:
        print(f"\nFailed collections: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\nAll MSigDB downloads complete.")


if __name__ == "__main__":
    main()
