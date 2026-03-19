"""Structural mimicry analysis for GAS-human protein pairs.

Downloads AlphaFold predicted structures for top mimicry hits, extracts
aligned regions, computes structural alignment (RMSD + TM-score), and
identifies surface-exposed mimicry regions.

Uses Biopython for structure handling and implements TM-score calculation
directly (no external TM-align binary required).

Usage:
    uv run python -m bioagentics.data.pandas_pans.structural_mimicry [--dest DIR]
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBParser, PDBIO, Superimposer
from Bio.PDB.DSSP import dssp_dict_from_pdb_file
from Bio.PDB.ResidueDepth import get_surface

logger = logging.getLogger(__name__)

PROJECT_DIR = Path("data/pandas_pans/gas-molecular-mimicry-mapping")
SCREEN_DIR = PROJECT_DIR / "mimicry_screen"
STRUCT_DIR = PROJECT_DIR / "structures"
OUTPUT_DIR = Path("output/pandas_pans/gas-molecular-mimicry-mapping/structural_mimicry")

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"

# Only analyze top hits (by bitscore) to keep runtime manageable
MAX_PAIRS = 20


def download_structure(uniprot_id: str, dest_dir: Path) -> Path | None:
    """Download AlphaFold predicted structure for a UniProt ID via API lookup."""
    pdb_path = dest_dir / f"AF-{uniprot_id}.pdb"
    if pdb_path.exists():
        return pdb_path

    try:
        # Query API for the correct PDB URL (handles version changes)
        api_url = ALPHAFOLD_API.format(uniprot=uniprot_id)
        api_resp = requests.get(api_url, timeout=15)
        if api_resp.status_code != 200:
            logger.warning("  No AlphaFold entry for %s (HTTP %d)", uniprot_id, api_resp.status_code)
            return None

        entries = api_resp.json()
        if not entries:
            logger.warning("  No AlphaFold entry for %s", uniprot_id)
            return None

        pdb_url = entries[0].get("pdbUrl")
        if not pdb_url:
            logger.warning("  No PDB URL for %s", uniprot_id)
            return None

        resp = requests.get(pdb_url, timeout=30)
        if resp.status_code == 200:
            pdb_path.write_bytes(resp.content)
            logger.info("  Downloaded: AF-%s", uniprot_id)
            return pdb_path
        else:
            logger.warning("  PDB download failed for %s (HTTP %d)", uniprot_id, resp.status_code)
            return None
    except requests.RequestException as e:
        logger.warning("  Download failed for %s: %s", uniprot_id, e)
        return None


def extract_ca_coords(structure, chain_id: str = "A",
                      start: int = 1, end: int | None = None) -> tuple[list, np.ndarray]:
    """Extract CA atom coordinates from a structure for a residue range."""
    model = structure[0]
    chain = model[chain_id]

    residues = []
    coords = []
    for res in chain:
        if res.id[0] != " ":  # skip hetero/water
            continue
        resnum = res.id[1]
        if resnum < start:
            continue
        if end is not None and resnum > end:
            continue
        if "CA" in res:
            residues.append(res)
            coords.append(res["CA"].get_vector().get_array())

    return residues, np.array(coords) if coords else np.empty((0, 3))


def compute_tm_score(coords1: np.ndarray, coords2: np.ndarray,
                     l_target: int) -> tuple[float, float]:
    """Compute TM-score after optimal superposition.

    TM-score = (1/L_target) * sum_i(1 / (1 + (d_i/d0)^2))
    where d0 = 1.24 * (L_target - 15)^(1/3) - 1.8
    """
    n = min(len(coords1), len(coords2))
    if n < 3:
        return 0.0, float("inf")

    # Truncate to same length
    c1 = coords1[:n].copy()
    c2 = coords2[:n].copy()

    # Superimpose using Biopython (needs Atom-like objects)
    # Build dummy atom lists for set_atoms
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue

    def _make_atoms(coords_arr: np.ndarray) -> list:
        atoms = []
        for i, c in enumerate(coords_arr):
            r = Residue((" ", i, " "), "ALA", " ")
            a = Atom("CA", c.tolist(), 0.0, 1.0, " ", "CA", i, "C")
            r.add(a)
            atoms.append(a)
        return atoms

    fixed_atoms = _make_atoms(c1)
    moving_atoms = _make_atoms(c2)

    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    rmsd = sup.rms

    # Apply rotation to get aligned coordinates
    rot, tran = sup.rotran
    rotated = (rot @ c2.T).T + tran

    # Compute TM-score
    d0 = 1.24 * (max(l_target, 16) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)  # floor

    distances = np.sqrt(np.sum((c1 - rotated) ** 2, axis=1))
    tm_score = np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / l_target

    return float(tm_score), float(rmsd)


def estimate_surface_accessibility(coords: np.ndarray, residues: list) -> list[bool]:
    """Simple surface accessibility estimate based on neighbor count.

    Residues with fewer CA neighbors within 10A are more likely surface-exposed.
    """
    n = len(coords)
    if n == 0:
        return []

    surface = []
    threshold = 12  # fewer than this many neighbors → surface
    for i in range(n):
        dists = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
        neighbor_count = np.sum((dists > 0) & (dists < 10.0))
        surface.append(neighbor_count < threshold)

    return surface


def get_gas_uniprot_id(qseqid: str) -> str | None:
    """Extract UniProt accession from GAS protein ID.

    Handles formats like:
    - sp|P0C0G7|G3P_STRP1
    - tr|Q9A200|Q9A200_STRP1
    - UPI0000165A3D (UniParc — no AlphaFold structure)
    """
    if qseqid.startswith("UPI"):
        return None  # UniParc IDs don't have AlphaFold structures
    parts = qseqid.split("|")
    if len(parts) >= 2:
        return parts[1]
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Structural mimicry analysis for GAS-human protein pairs",
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS,
                        help="Max pairs to analyze (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.dest.mkdir(parents=True, exist_ok=True)
    STRUCT_DIR.mkdir(parents=True, exist_ok=True)

    # Load filtered hits
    hits_path = SCREEN_DIR / "hits_filtered.tsv"
    if not hits_path.exists():
        raise FileNotFoundError(f"Run mimicry_screen.py first: {hits_path}")

    hits_df = pd.read_csv(hits_path, sep="\t")

    # Deduplicate and take top pairs by bitscore
    pairs = (
        hits_df.sort_values("bitscore", ascending=False)
        .drop_duplicates(subset=["qseqid", "human_accession"], keep="first")
        .head(args.max_pairs)
        .reset_index(drop=True)
    )
    logger.info("Analyzing structural mimicry for %d top pairs", len(pairs))

    parser_pdb = PDBParser(QUIET=True)
    results: list[dict] = []

    for idx, row in pairs.iterrows():
        gas_id = row["qseqid"]
        human_acc = row["human_accession"]
        human_gene = str(row.get("human_gene", "")) if pd.notna(row.get("human_gene")) else ""

        gas_uniprot = get_gas_uniprot_id(gas_id)
        if gas_uniprot is None:
            logger.info("  [%d] Skipping %s (UniParc, no structure)", idx + 1, gas_id)
            results.append({
                "gas_protein": gas_id,
                "gas_uniprot": "",
                "human_accession": human_acc,
                "human_gene": human_gene,
                "status": "skipped_uniparc",
                "tm_score": None,
                "rmsd": None,
                "aligned_residues": 0,
                "surface_exposed_fraction": None,
            })
            continue

        logger.info("  [%d/%d] %s vs %s (%s)",
                    idx + 1, len(pairs), gas_uniprot, human_acc, human_gene)

        # Download structures
        gas_pdb = download_structure(gas_uniprot, STRUCT_DIR)
        human_pdb = download_structure(human_acc, STRUCT_DIR)
        time.sleep(0.3)  # rate limit

        if gas_pdb is None or human_pdb is None:
            status = "no_gas_structure" if gas_pdb is None else "no_human_structure"
            results.append({
                "gas_protein": gas_id,
                "gas_uniprot": gas_uniprot,
                "human_accession": human_acc,
                "human_gene": human_gene,
                "status": status,
                "tm_score": None,
                "rmsd": None,
                "aligned_residues": 0,
                "surface_exposed_fraction": None,
            })
            continue

        try:
            gas_struct = parser_pdb.get_structure("gas", str(gas_pdb))
            human_struct = parser_pdb.get_structure("human", str(human_pdb))

            # Extract aligned region CA coords
            gas_residues, gas_coords = extract_ca_coords(
                gas_struct, start=int(row["qstart"]), end=int(row["qend"]),
            )
            human_residues, human_coords = extract_ca_coords(
                human_struct, start=int(row["sstart"]), end=int(row["send"]),
            )

            n_aligned = min(len(gas_coords), len(human_coords))
            if n_aligned < 3:
                results.append({
                    "gas_protein": gas_id,
                    "gas_uniprot": gas_uniprot,
                    "human_accession": human_acc,
                    "human_gene": human_gene,
                    "status": "too_few_residues",
                    "tm_score": None,
                    "rmsd": None,
                    "aligned_residues": n_aligned,
                    "surface_exposed_fraction": None,
                })
                continue

            # Compute TM-score using human protein length as reference
            human_full_res, _ = extract_ca_coords(human_struct)
            l_target = len(human_full_res)

            tm_score, rmsd = compute_tm_score(gas_coords, human_coords, l_target)

            # Surface accessibility of aligned human region
            surface = estimate_surface_accessibility(human_coords, human_residues)
            surface_fraction = sum(surface) / len(surface) if surface else 0.0

            results.append({
                "gas_protein": gas_id,
                "gas_uniprot": gas_uniprot,
                "human_accession": human_acc,
                "human_gene": human_gene,
                "status": "success",
                "tm_score": round(tm_score, 4),
                "rmsd": round(rmsd, 2),
                "aligned_residues": n_aligned,
                "gas_aligned_range": f"{int(row['qstart'])}-{int(row['qend'])}",
                "human_aligned_range": f"{int(row['sstart'])}-{int(row['send'])}",
                "surface_exposed_fraction": round(surface_fraction, 3),
                "pident": row["pident"],
                "bitscore": row["bitscore"],
            })

            logger.info("    TM-score=%.4f, RMSD=%.2f, aligned=%d, surface=%.1f%%",
                        tm_score, rmsd, n_aligned, surface_fraction * 100)

        except Exception as e:
            logger.warning("    Error processing pair: %s", e)
            results.append({
                "gas_protein": gas_id,
                "gas_uniprot": gas_uniprot,
                "human_accession": human_acc,
                "human_gene": human_gene,
                "status": f"error: {e}",
                "tm_score": None,
                "rmsd": None,
                "aligned_residues": 0,
                "surface_exposed_fraction": None,
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_path = args.dest / "structural_mimicry.tsv"
    results_df.to_csv(results_path, sep="\t", index=False)
    logger.info("Results saved: %s (%d entries)", results_path.name, len(results_df))

    # Summary
    success = results_df[results_df["status"] == "success"]
    if not success.empty:
        logger.info("\nSuccessful structural analyses: %d / %d", len(success), len(results_df))
        high_tm = success[success["tm_score"] >= 0.5]
        if not high_tm.empty:
            logger.info("High TM-score pairs (>=0.5, significant structural similarity):")
            for _, r in high_tm.iterrows():
                logger.info("  %s vs %s: TM=%.4f, RMSD=%.2f",
                            r["gas_protein"][:30], r["human_gene"] or r["human_accession"],
                            r["tm_score"], r["rmsd"])

    logger.info("Done. Structural mimicry in %s", args.dest)


if __name__ == "__main__":
    main()
