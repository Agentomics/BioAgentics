"""VarMeter2-style structure-based features for NOD2 variants.

Computes normalized solvent-accessible surface area (nSASA), simplified
mutation energy (ddG proxy), and pLDDT for each missense variant position.
Based on VarMeter2 (Comp Struct Biotech J 2025, PMC11952791).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/crohns/nod2-variant-functional-impact")

# Grantham distance matrix for amino acid substitutions
# Higher values = more radical substitution = larger expected ddG
# Source: Grantham R (1974) Science 185:862-864
_GRANTHAM = {
    ("A", "R"): 112, ("A", "N"): 111, ("A", "D"): 126, ("A", "C"): 195,
    ("A", "Q"): 91, ("A", "E"): 107, ("A", "G"): 60, ("A", "H"): 86,
    ("A", "I"): 94, ("A", "L"): 96, ("A", "K"): 106, ("A", "M"): 84,
    ("A", "F"): 113, ("A", "P"): 27, ("A", "S"): 99, ("A", "T"): 58,
    ("A", "W"): 148, ("A", "Y"): 112, ("A", "V"): 64,
    ("R", "N"): 86, ("R", "D"): 96, ("R", "C"): 180, ("R", "Q"): 43,
    ("R", "E"): 54, ("R", "G"): 125, ("R", "H"): 29, ("R", "I"): 97,
    ("R", "L"): 102, ("R", "K"): 26, ("R", "M"): 91, ("R", "F"): 97,
    ("R", "P"): 103, ("R", "S"): 110, ("R", "T"): 71, ("R", "W"): 101,
    ("R", "Y"): 77, ("R", "V"): 96,
    ("N", "D"): 23, ("N", "C"): 139, ("N", "Q"): 46, ("N", "E"): 42,
    ("N", "G"): 80, ("N", "H"): 68, ("N", "I"): 149, ("N", "L"): 153,
    ("N", "K"): 94, ("N", "M"): 142, ("N", "F"): 158, ("N", "P"): 91,
    ("N", "S"): 46, ("N", "T"): 65, ("N", "W"): 174, ("N", "Y"): 143,
    ("N", "V"): 133,
    ("D", "C"): 154, ("D", "Q"): 61, ("D", "E"): 45, ("D", "G"): 94,
    ("D", "H"): 81, ("D", "I"): 168, ("D", "L"): 172, ("D", "K"): 101,
    ("D", "M"): 160, ("D", "F"): 177, ("D", "P"): 108, ("D", "S"): 65,
    ("D", "T"): 85, ("D", "W"): 181, ("D", "Y"): 160, ("D", "V"): 152,
    ("C", "Q"): 154, ("C", "E"): 170, ("C", "G"): 159, ("C", "H"): 174,
    ("C", "I"): 198, ("C", "L"): 198, ("C", "K"): 202, ("C", "M"): 196,
    ("C", "F"): 205, ("C", "P"): 169, ("C", "S"): 112, ("C", "T"): 149,
    ("C", "W"): 215, ("C", "Y"): 194, ("C", "V"): 192,
    ("Q", "E"): 29, ("Q", "G"): 87, ("Q", "H"): 24, ("Q", "I"): 109,
    ("Q", "L"): 113, ("Q", "K"): 53, ("Q", "M"): 101, ("Q", "F"): 116,
    ("Q", "P"): 76, ("Q", "S"): 68, ("Q", "T"): 42, ("Q", "W"): 130,
    ("Q", "Y"): 99, ("Q", "V"): 96,
    ("E", "G"): 98, ("E", "H"): 40, ("E", "I"): 134, ("E", "L"): 138,
    ("E", "K"): 56, ("E", "M"): 126, ("E", "F"): 140, ("E", "P"): 93,
    ("E", "S"): 80, ("E", "T"): 65, ("E", "W"): 152, ("E", "Y"): 122,
    ("E", "V"): 121,
    ("G", "H"): 98, ("G", "I"): 135, ("G", "L"): 138, ("G", "K"): 127,
    ("G", "M"): 127, ("G", "F"): 153, ("G", "P"): 42, ("G", "S"): 56,
    ("G", "T"): 59, ("G", "W"): 184, ("G", "Y"): 147, ("G", "V"): 109,
    ("H", "I"): 94, ("H", "L"): 99, ("H", "K"): 32, ("H", "M"): 87,
    ("H", "F"): 100, ("H", "P"): 77, ("H", "S"): 89, ("H", "T"): 47,
    ("H", "W"): 115, ("H", "Y"): 83, ("H", "V"): 84,
    ("I", "L"): 5, ("I", "K"): 102, ("I", "M"): 10, ("I", "F"): 21,
    ("I", "P"): 95, ("I", "S"): 142, ("I", "T"): 89, ("I", "W"): 61,
    ("I", "Y"): 33, ("I", "V"): 29,
    ("L", "K"): 107, ("L", "M"): 15, ("L", "F"): 22, ("L", "P"): 98,
    ("L", "S"): 145, ("L", "T"): 92, ("L", "W"): 61, ("L", "Y"): 36,
    ("L", "V"): 32,
    ("K", "M"): 95, ("K", "F"): 102, ("K", "P"): 103, ("K", "S"): 121,
    ("K", "T"): 78, ("K", "W"): 110, ("K", "Y"): 85, ("K", "V"): 97,
    ("M", "F"): 28, ("M", "P"): 87, ("M", "S"): 135, ("M", "T"): 81,
    ("M", "W"): 67, ("M", "Y"): 36, ("M", "V"): 21,
    ("F", "P"): 114, ("F", "S"): 155, ("F", "T"): 103, ("F", "W"): 40,
    ("F", "Y"): 22, ("F", "V"): 50,
    ("P", "S"): 74, ("P", "T"): 38, ("P", "W"): 147, ("P", "Y"): 110,
    ("P", "V"): 68,
    ("S", "T"): 58, ("S", "W"): 177, ("S", "Y"): 144, ("S", "V"): 124,
    ("T", "W"): 128, ("T", "Y"): 92, ("T", "V"): 69,
    ("W", "Y"): 37, ("W", "V"): 88,
    ("Y", "V"): 55,
}


def grantham_distance(aa_ref: str, aa_alt: str) -> float:
    """Return Grantham distance between two amino acids.

    Symmetric: grantham_distance(A, B) == grantham_distance(B, A).
    Returns 0 for identical amino acids, NaN for unknown.
    """
    if aa_ref == aa_alt:
        return 0.0
    pair = (aa_ref.upper(), aa_alt.upper())
    dist = _GRANTHAM.get(pair) or _GRANTHAM.get((pair[1], pair[0]))
    return float(dist) if dist is not None else float("nan")


def estimate_ddg(nsasa: float, grantham: float, plddt: float) -> float:
    """Estimate mutation energy change (ddG proxy).

    Simplified energy model based on VarMeter2 principles:
    - Buried residues (low nSASA) with radical substitutions (high Grantham)
      are most destabilizing
    - pLDDT weights the confidence of structural prediction

    Returns a proxy score where higher = more destabilizing.
    """
    if np.isnan(nsasa) or np.isnan(grantham) or np.isnan(plddt):
        return float("nan")

    # Burial factor: buried residues are more sensitive to mutation
    burial = 1.0 - nsasa  # 1 = fully buried, 0 = fully exposed

    # Normalize Grantham to 0-1 range (max Grantham distance is ~215)
    grantham_norm = grantham / 215.0

    # Confidence weight from pLDDT (0-100 scale)
    confidence = plddt / 100.0

    # Combined ddG proxy: burial * radical_change * confidence
    return burial * grantham_norm * confidence


def compute_varmeter2_features(
    variants_df: pd.DataFrame,
    structure_features_path: Path | None = None,
) -> pd.DataFrame:
    """Compute VarMeter2-style features for NOD2 missense variants.

    Args:
        variants_df: Variant DataFrame with hgvs_p column.
        structure_features_path: Path to structure features TSV.

    Returns:
        DataFrame with columns: variant_id, chrom, pos, ref, alt,
        nsasa, mutation_energy, plddt.
    """
    if structure_features_path is None:
        structure_features_path = OUTPUT_DIR / "nod2_structure_features.tsv"

    # Load structure features
    struct_df = pd.read_csv(structure_features_path, sep="\t")
    logger.info("Loaded structure features for %d residues", len(struct_df))

    # Build residue lookup
    struct_lookup = struct_df.set_index("residue_pos").to_dict("index")

    # Extract missense variants with protein change info
    records: list[dict] = []
    for _, row in variants_df.iterrows():
        hgvs_p = str(row.get("hgvs_p", ""))
        if not hgvs_p or hgvs_p == "nan":
            continue

        # Parse protein change: p.Arg702Trp or R702W
        parsed = _parse_protein_change(hgvs_p)
        if parsed is None:
            continue

        ref_aa, pos, alt_aa = parsed

        # Get structure features for this position
        struct_data = struct_lookup.get(pos)
        if struct_data is None:
            continue

        nsasa = struct_data.get("rsasa", float("nan"))
        plddt = struct_data.get("plddt", float("nan"))

        # Compute Grantham distance
        grantham = grantham_distance(ref_aa, alt_aa)

        # Compute ddG proxy
        ddg = estimate_ddg(nsasa, grantham, plddt)

        records.append({
            "variant_id": row.get("variant_id", ""),
            "chrom": row.get("chrom", ""),
            "pos": row.get("pos", ""),
            "ref": row.get("ref", ""),
            "alt": row.get("alt", ""),
            "residue_pos": pos,
            "ref_aa": ref_aa,
            "alt_aa": alt_aa,
            "nsasa": nsasa,
            "mutation_energy": ddg,
            "plddt": plddt,
            "grantham_distance": grantham,
        })

    df = pd.DataFrame(records)
    logger.info("Computed VarMeter2 features for %d missense variants", len(df))
    return df


# Three-letter to one-letter amino acid mapping
_AA3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

# One-letter AA set
_VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _parse_protein_change(hgvs_p: str) -> tuple[str, int, str] | None:
    """Parse protein change notation into (ref_aa, position, alt_aa).

    Handles both single-letter (R702W) and three-letter (p.Arg702Trp) notation.
    Returns None for non-missense changes (frameshifts, stop gains, etc.).
    """
    import re

    # Try three-letter notation: p.Arg702Trp
    m = re.match(r"p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})$", hgvs_p)
    if m:
        ref = _AA3TO1.get(m.group(1))
        alt = _AA3TO1.get(m.group(3))
        if ref and alt and ref != alt:
            return ref, int(m.group(2)), alt
        return None

    # Try single-letter notation: R702W
    m = re.match(r"([A-Z])(\d+)([A-Z])$", hgvs_p)
    if m:
        ref, alt = m.group(1), m.group(3)
        if ref in _VALID_AA and alt in _VALID_AA and ref != alt:
            return ref, int(m.group(2)), alt
        return None

    # Try p. prefix with single-letter: p.R702W
    m = re.match(r"p\.([A-Z])(\d+)([A-Z])$", hgvs_p)
    if m:
        ref, alt = m.group(1), m.group(3)
        if ref in _VALID_AA and alt in _VALID_AA and ref != alt:
            return ref, int(m.group(2)), alt
        return None

    return None


def collect_varmeter2_features(
    variants_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Main pipeline: load variants, compute features, save TSV."""
    if variants_path is None:
        variants_path = OUTPUT_DIR / "nod2_variants.tsv"
    if output_path is None:
        output_path = OUTPUT_DIR / "nod2_varmeter2_features.tsv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    variants_df = pd.read_csv(variants_path, sep="\t")
    logger.info("Loaded %d variants", len(variants_df))

    features_df = compute_varmeter2_features(variants_df)

    features_df.to_csv(output_path, sep="\t", index=False)
    logger.info("Saved VarMeter2 features to %s (%d variants)", output_path, len(features_df))

    return features_df
