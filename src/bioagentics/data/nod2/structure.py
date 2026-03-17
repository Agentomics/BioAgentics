"""NOD2 structural features from AlphaFold predicted structure.

Downloads AlphaFold structure for NOD2 (UniProt Q9HC29), extracts
per-residue features: pLDDT, relative solvent accessibility, domain
assignment, and distance to active site residues.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)

# AlphaFold DB URL for NOD2 (UniProt Q9HC29)
ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-Q9HC29-F1-model_v6.pdb"
UNIPROT_ID = "Q9HC29"

OUTPUT_DIR = Path("data/crohns/nod2-variant-functional-impact")

# NOD2 domain boundaries (from UniProt Q9HC29 / InterPro)
# Verified against UniProt feature annotations
NOD2_DOMAINS = {
    "CARD1": (1, 110),
    "CARD2": (124, 220),
    "NACHT": (273, 576),
    "WH": (577, 743),  # Winged helix
    "LRR": (744, 1040),
}

# Key functional residues for NOD2
# Active site / nucleotide binding residues in NACHT domain
ACTIVE_SITE_RESIDUES = [
    # Walker A motif (P-loop, nucleotide binding)
    305, 306, 307, 308, 309, 310,
    # Walker B motif (Mg2+ coordination)
    379, 380, 381,
    # Sensor 1
    431,
    # Key catalytic residues
    382, 383,
]

# RIPK2 interaction interface (CARD-CARD)
RIPK2_INTERFACE_RESIDUES = list(range(27, 110))  # CARD1 domain


def download_alphafold_structure(output_dir: Path | None = None) -> Path:
    """Download AlphaFold predicted structure for NOD2.

    Returns path to downloaded PDB file.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = output_dir / f"AF-{UNIPROT_ID}-F1-model_v6.pdb"

    if pdb_path.exists():
        logger.info("AlphaFold structure already downloaded: %s", pdb_path)
        return pdb_path

    logger.info("Downloading AlphaFold structure for NOD2 (%s)...", UNIPROT_ID)
    resp = requests.get(ALPHAFOLD_URL, timeout=60)
    resp.raise_for_status()

    pdb_path.write_text(resp.text)
    logger.info("Saved AlphaFold structure to %s", pdb_path)
    return pdb_path


def get_domain(residue_pos: int) -> str:
    """Return the NOD2 domain for a given residue position."""
    for domain_name, (start, end) in NOD2_DOMAINS.items():
        if start <= residue_pos <= end:
            return domain_name
    return "linker"


def compute_active_site_distance(
    ca_coords: dict[int, np.ndarray],
    residue_pos: int,
) -> float:
    """Compute minimum distance from residue CA to any active site residue CA."""
    if residue_pos in ACTIVE_SITE_RESIDUES:
        return 0.0

    target_coord = ca_coords.get(residue_pos)
    if target_coord is None:
        return float("nan")

    min_dist = float("inf")
    for as_pos in ACTIVE_SITE_RESIDUES:
        as_coord = ca_coords.get(as_pos)
        if as_coord is not None:
            dist = float(np.linalg.norm(target_coord - as_coord))
            min_dist = min(min_dist, dist)

    return min_dist if min_dist != float("inf") else float("nan")


def compute_sasa_from_structure(pdb_path: Path) -> dict[int, float]:
    """Compute relative solvent accessible surface area per residue.

    Uses Shrake-Rupley algorithm via BioPython. Returns dict of
    residue_pos -> rSASA.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("NOD2", str(pdb_path))
    model = structure[0]

    # Maximum SASA values for each amino acid (Tien et al. 2013, empirical)
    max_sasa = {
        "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0,
        "CYS": 167.0, "GLN": 225.0, "GLU": 223.0, "GLY": 104.0,
        "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0,
        "MET": 224.0, "PHE": 240.0, "PRO": 159.0, "SER": 155.0,
        "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
    }

    # Compute SASA using Shrake-Rupley
    from Bio.PDB.SASA import ShrakeRupley
    sr = ShrakeRupley()
    sr.compute(model, level="R")

    rsasa_dict: dict[int, float] = {}
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue  # skip hetero atoms
            resname = residue.get_resname()
            res_pos = residue.id[1]
            sasa_val = residue.sasa
            max_val = max_sasa.get(resname, 200.0)
            rsasa_dict[res_pos] = min(sasa_val / max_val, 1.0) if max_val > 0 else 0.0

    return rsasa_dict


def extract_structure_features(pdb_path: Path | None = None) -> pd.DataFrame:
    """Extract per-residue structural features from AlphaFold NOD2 structure.

    Returns DataFrame with columns: residue_pos, aa, plddt, rsasa,
    domain, active_site_distance.
    """
    if pdb_path is None:
        pdb_path = download_alphafold_structure()

    logger.info("Extracting structural features from %s", pdb_path)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("NOD2", str(pdb_path))
    model = structure[0]

    # Three-letter to one-letter amino acid mapping
    aa3to1 = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }

    # Extract CA coordinates and B-factors (pLDDT in AlphaFold)
    ca_coords: dict[int, np.ndarray] = {}
    residue_data: list[dict] = []

    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue  # skip hetero atoms
            res_pos = residue.id[1]
            resname = residue.get_resname()
            aa = aa3to1.get(resname, "X")

            # pLDDT is stored as B-factor in AlphaFold PDB files
            ca = residue["CA"] if "CA" in residue else None
            if ca is not None:
                plddt = ca.get_bfactor()
                ca_coords[res_pos] = ca.get_vector().get_array()
            else:
                plddt = float("nan")

            residue_data.append({
                "residue_pos": res_pos,
                "aa": aa,
                "plddt": plddt,
                "domain": get_domain(res_pos),
            })

    # Compute SASA
    logger.info("Computing solvent accessible surface area...")
    rsasa_dict = compute_sasa_from_structure(pdb_path)

    # Compute active site distances
    logger.info("Computing active site distances...")
    for record in residue_data:
        pos = record["residue_pos"]
        record["rsasa"] = rsasa_dict.get(pos, float("nan"))
        record["active_site_distance"] = compute_active_site_distance(ca_coords, pos)

    df = pd.DataFrame(residue_data)

    # Reorder columns
    col_order = ["residue_pos", "aa", "plddt", "rsasa", "domain", "active_site_distance"]
    df = df[col_order]

    logger.info(
        "Extracted features for %d residues across domains: %s",
        len(df),
        df["domain"].value_counts().to_dict(),
    )

    return df


def compute_and_save_structure_features(
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Main pipeline: download structure, compute features, save TSV.

    Args:
        output_path: Where to save. Defaults to
            data/crohns/nod2-variant-functional-impact/nod2_structure_features.tsv

    Returns:
        DataFrame of per-residue structural features.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "nod2_structure_features.tsv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Download and extract
    pdb_path = download_alphafold_structure()
    df = extract_structure_features(pdb_path)

    # Validate domain boundaries
    _validate_domains(df)

    # Save
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("Saved structural features to %s (%d residues)", output_path, len(df))

    return df


def _validate_domains(df: pd.DataFrame) -> None:
    """Validate domain assignments match expected boundaries."""
    domain_counts = df["domain"].value_counts()
    logger.info("Domain residue counts: %s", domain_counts.to_dict())

    # Check that LRR domain has substantial residues (expected ~296)
    lrr_count = domain_counts.get("LRR", 0)
    if lrr_count < 200:
        logger.warning("LRR domain has fewer residues than expected: %d (expected ~296)", lrr_count)

    # Check that NACHT domain has substantial residues (expected ~304)
    nacht_count = domain_counts.get("NACHT", 0)
    if nacht_count < 200:
        logger.warning("NACHT domain has fewer residues than expected: %d (expected ~304)", nacht_count)

    # Check CARD domains
    card1_count = domain_counts.get("CARD1", 0)
    card2_count = domain_counts.get("CARD2", 0)
    if card1_count < 50 or card2_count < 50:
        logger.warning(
            "CARD domains may be too small: CARD1=%d, CARD2=%d",
            card1_count, card2_count,
        )
