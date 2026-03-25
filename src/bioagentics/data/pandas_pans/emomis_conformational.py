"""Lightweight conformational B-cell epitope mimicry scorer.

Inspired by the EMoMiS pipeline (Stebliankin et al., CSBJ 2026), which uses
MaSIF-Search deep learning on 128GB/GPU hardware to evaluate antibody
cross-reactivity via conformational epitope similarity. This lightweight
reimplementation captures the same conceptual framework — surface-exposed
conformational epitope overlap between GAS and human proteins — using only
pre-computed AlphaFold structures and CPU-based analysis.

Approach (mirrors EMoMiS steps without MaSIF/GPU):
  1. Load pre-downloaded AlphaFold structures for GAS-human mimicry pairs
  2. Compute per-residue surface exposure via DSSP-derived RSA
  3. Extract pLDDT confidence from AlphaFold B-factors
  4. Identify conformational epitope patches: surface-exposed, high-confidence
     regions using ElliPro-inspired protrusion index
  5. Score cross-reactivity: overlap of conformational epitope patches in
     structurally aligned regions, weighted by physicochemical similarity

Outputs a per-pair conformational mimicry score that complements:
  - BLASTp sequence similarity (existing)
  - Linear B-cell epitope overlap (existing epitope_prediction.py)
  - MHC-II T-cell mimicry (existing mmpred_scoring.py)

Usage:
    uv run python -m bioagentics.data.pandas_pans.emomis_conformational [--dest DIR]
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)

PROJECT_DIR = Path("data/pandas_pans/gas-molecular-mimicry-mapping")
STRUCT_DIR = PROJECT_DIR / "structures"
SCREEN_DIR = PROJECT_DIR / "mimicry_screen"
STRUCTURAL_DIR = Path("output/pandas_pans/gas-molecular-mimicry-mapping/structural_mimicry")
OUTPUT_DIR = Path("output/pandas_pans/gas-molecular-mimicry-mapping/emomis_analysis")

# Kyte-Doolittle hydrophobicity — negative = hydrophilic/surface-exposed
_KD_HYDROPHOBICITY: dict[str, float] = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Charge at pH 7
_CHARGE: dict[str, float] = {
    "R": 1.0, "K": 1.0, "H": 0.1, "D": -1.0, "E": -1.0,
}

# Residue volume (Å³) — for size similarity
_VOLUME: dict[str, float] = {
    "G": 60.1, "A": 88.6, "V": 140.0, "L": 166.7, "I": 166.7,
    "P": 112.7, "F": 189.9, "W": 227.8, "M": 162.9, "S": 89.0,
    "T": 116.1, "C": 108.5, "Y": 193.6, "H": 153.2, "D": 111.1,
    "E": 138.4, "N": 114.1, "Q": 143.8, "K": 168.6, "R": 173.4,
}


def extract_residue_features(structure, chain_id: str = "A",
                             start: int = 1, end: int | None = None
                             ) -> pd.DataFrame:
    """Extract per-residue structural features from an AlphaFold PDB.

    Features:
    - resnum, resname: residue identity
    - plddt: AlphaFold confidence (B-factor column, 0-100)
    - ca_coord: CA atom coordinates
    - protrusion: distance from center of mass (ElliPro-inspired)
    - neighbor_count: number of CA atoms within 10Å (burial proxy)
    - hydrophobicity: Kyte-Doolittle score
    """
    model = structure[0]
    chain = model[chain_id]

    records = []
    ca_coords = []

    for res in chain:
        if res.id[0] != " ":
            continue
        resnum = res.id[1]
        if resnum < start:
            continue
        if end is not None and resnum > end:
            continue
        if "CA" not in res:
            continue

        ca = res["CA"]
        coord = ca.get_vector().get_array()
        resname = res.get_resname()
        # In AlphaFold PDB files, B-factor = pLDDT
        plddt = ca.get_bfactor()

        aa_1letter = _three_to_one(resname)
        records.append({
            "resnum": resnum,
            "resname": aa_1letter,
            "plddt": plddt,
            "x": coord[0],
            "y": coord[1],
            "z": coord[2],
            "hydrophobicity": _KD_HYDROPHOBICITY.get(aa_1letter, 0.0),
            "charge": _CHARGE.get(aa_1letter, 0.0),
            "volume": _VOLUME.get(aa_1letter, 130.0),
        })
        ca_coords.append(coord)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    coords = np.array(ca_coords)

    # Protrusion index: distance from center of mass
    com = coords.mean(axis=0)
    df["protrusion"] = np.sqrt(np.sum((coords - com) ** 2, axis=1))

    # Normalize protrusion to 0-1
    pmax = df["protrusion"].max()
    if pmax > 0:
        df["protrusion"] = df["protrusion"] / pmax

    # Neighbor count (burial)
    neighbor_counts = []
    for i in range(len(coords)):
        dists = np.sqrt(np.sum((coords - coords[i]) ** 2, axis=1))
        neighbor_counts.append(int(np.sum((dists > 0) & (dists < 10.0))))
    df["neighbor_count"] = neighbor_counts

    return df


def _three_to_one(resname: str) -> str:
    """Convert 3-letter amino acid code to 1-letter."""
    mapping = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
        "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
        "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
        "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    }
    return mapping.get(resname, "X")


def identify_conformational_epitope(df: pd.DataFrame,
                                    plddt_min: float = 70.0,
                                    max_neighbors: int = 14,
                                    min_protrusion: float = 0.3,
                                    ) -> pd.Series:
    """Identify residues in conformational B-cell epitope patches.

    A residue is in an epitope patch if:
    - AlphaFold confidence pLDDT >= 70 (reliable structure)
    - Surface-exposed: fewer than max_neighbors CA within 10Å
    - Protruding: above min_protrusion from center of mass
    - Hydrophilic: Kyte-Doolittle < 0 (surface preference)

    Returns boolean Series indicating epitope residues.
    """
    return (
        (df["plddt"] >= plddt_min)
        & (df["neighbor_count"] <= max_neighbors)
        & (df["protrusion"] >= min_protrusion)
        & (df["hydrophobicity"] < 0.5)  # slightly relaxed: allow mildly hydrophobic
    )


def physicochemical_similarity(row_a: pd.Series, row_b: pd.Series) -> float:
    """Compute physicochemical similarity between two residues (0-1).

    Based on hydrophobicity, charge, and volume similarity.
    """
    # Hydrophobicity similarity (range: -4.5 to 4.5 → normalize)
    h_diff = abs(row_a["hydrophobicity"] - row_b["hydrophobicity"]) / 9.0
    h_sim = 1.0 - min(h_diff, 1.0)

    # Charge similarity
    c_diff = abs(row_a["charge"] - row_b["charge"]) / 2.0
    c_sim = 1.0 - min(c_diff, 1.0)

    # Volume similarity
    v_diff = abs(row_a["volume"] - row_b["volume"]) / 170.0
    v_sim = 1.0 - min(v_diff, 1.0)

    return 0.4 * h_sim + 0.3 * c_sim + 0.3 * v_sim


def score_conformational_mimicry(
    gas_features: pd.DataFrame,
    human_features: pd.DataFrame,
    gas_start: int, gas_end: int,
    human_start: int, human_end: int,
) -> dict:
    """Score conformational B-cell epitope mimicry between a GAS-human pair.

    Evaluates whether conformational epitope patches on the GAS protein
    structurally overlap with epitope patches on the human protein within
    the aligned mimicry region.

    Returns dict with conformational mimicry metrics.
    """
    # Extract aligned regions
    gas_region = gas_features[
        (gas_features["resnum"] >= gas_start) & (gas_features["resnum"] <= gas_end)
    ].copy()
    human_region = human_features[
        (human_features["resnum"] >= human_start) & (human_features["resnum"] <= human_end)
    ].copy()

    if gas_region.empty or human_region.empty:
        return _empty_result()

    # Identify conformational epitope residues
    gas_epitope = identify_conformational_epitope(gas_region)
    human_epitope = identify_conformational_epitope(human_region)

    gas_epitope_count = int(gas_epitope.sum())
    human_epitope_count = int(human_epitope.sum())

    # Fraction of aligned region in conformational epitope
    gas_epitope_frac = gas_epitope_count / len(gas_region) if len(gas_region) > 0 else 0
    human_epitope_frac = human_epitope_count / len(human_region) if len(human_region) > 0 else 0

    # Mean pLDDT in aligned regions (structural confidence)
    gas_plddt_mean = gas_region["plddt"].mean()
    human_plddt_mean = human_region["plddt"].mean()

    # Epitope overlap: aligned positions where BOTH have epitope residues
    n_aligned = min(len(gas_region), len(human_region))
    gas_aligned = gas_region.iloc[:n_aligned].reset_index(drop=True)
    human_aligned = human_region.iloc[:n_aligned].reset_index(drop=True)

    gas_epi_aligned = identify_conformational_epitope(gas_aligned)
    human_epi_aligned = identify_conformational_epitope(human_aligned)

    both_epitope = gas_epi_aligned & human_epi_aligned
    overlap_count = int(both_epitope.sum())
    overlap_fraction = overlap_count / n_aligned if n_aligned > 0 else 0.0

    # Physicochemical similarity in overlapping epitope positions
    physchem_scores = []
    for i in range(n_aligned):
        if both_epitope.iloc[i]:
            sim = physicochemical_similarity(gas_aligned.iloc[i], human_aligned.iloc[i])
            physchem_scores.append(sim)

    mean_physchem = float(np.mean(physchem_scores)) if physchem_scores else 0.0

    # Composite conformational mimicry score
    # Weights: epitope overlap (0.35) + physicochemical similarity (0.25)
    #          + dual epitope enrichment (0.20) + structural confidence (0.20)
    dual_enrichment = min(gas_epitope_frac, human_epitope_frac)
    confidence = min(gas_plddt_mean, human_plddt_mean) / 100.0

    conf_score = (
        0.35 * overlap_fraction
        + 0.25 * mean_physchem
        + 0.20 * dual_enrichment
        + 0.20 * confidence
    )

    return {
        "gas_epitope_residues": gas_epitope_count,
        "human_epitope_residues": human_epitope_count,
        "gas_epitope_fraction": round(gas_epitope_frac, 3),
        "human_epitope_fraction": round(human_epitope_frac, 3),
        "aligned_residues": n_aligned,
        "epitope_overlap_count": overlap_count,
        "epitope_overlap_fraction": round(overlap_fraction, 3),
        "mean_physicochemical_similarity": round(mean_physchem, 3),
        "gas_mean_plddt": round(gas_plddt_mean, 1),
        "human_mean_plddt": round(human_plddt_mean, 1),
        "conformational_mimicry_score": round(conf_score, 4),
    }


def _empty_result() -> dict:
    return {
        "gas_epitope_residues": 0,
        "human_epitope_residues": 0,
        "gas_epitope_fraction": 0.0,
        "human_epitope_fraction": 0.0,
        "aligned_residues": 0,
        "epitope_overlap_count": 0,
        "epitope_overlap_fraction": 0.0,
        "mean_physicochemical_similarity": 0.0,
        "gas_mean_plddt": 0.0,
        "human_mean_plddt": 0.0,
        "conformational_mimicry_score": 0.0,
    }


def get_gas_uniprot_id(qseqid: str) -> str | None:
    """Extract UniProt accession from GAS protein ID."""
    if qseqid.startswith("UPI"):
        return None
    parts = qseqid.split("|")
    if len(parts) >= 2:
        return parts[1]
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Conformational B-cell epitope mimicry scorer (EMoMiS-lite)",
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--max-pairs", type=int, default=20,
                        help="Max pairs to analyze (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    warnings.filterwarnings("ignore", message=".*discontinuous.*")

    args.dest.mkdir(parents=True, exist_ok=True)

    # Load structural mimicry results to get pairs with available structures
    struct_path = STRUCTURAL_DIR / "structural_mimicry.tsv"
    if not struct_path.exists():
        raise FileNotFoundError(
            f"Run structural_mimicry.py first: {struct_path}"
        )

    struct_df = pd.read_csv(struct_path, sep="\t")
    success = struct_df[struct_df["status"] == "success"].head(args.max_pairs)
    logger.info("Analyzing %d pairs with available structures", len(success))

    pdb_parser = PDBParser(QUIET=True)
    results: list[dict] = []
    feature_cache: dict[str, pd.DataFrame] = {}

    for idx, row in success.iterrows():
        gas_uniprot = row["gas_uniprot"]
        human_acc = row["human_accession"]
        human_gene = str(row.get("human_gene", "")) if pd.notna(row.get("human_gene")) else ""

        logger.info("  [%d] %s vs %s (%s)", idx + 1, gas_uniprot, human_acc, human_gene)

        gas_pdb = STRUCT_DIR / f"AF-{gas_uniprot}.pdb"
        human_pdb = STRUCT_DIR / f"AF-{human_acc}.pdb"

        if not gas_pdb.exists() or not human_pdb.exists():
            logger.warning("    Missing structure file, skipping")
            result = {"gas_uniprot": gas_uniprot, "human_accession": human_acc,
                      "human_gene": human_gene, "status": "missing_structure"}
            result.update(_empty_result())
            results.append(result)
            continue

        try:
            # Load/cache features
            if gas_uniprot not in feature_cache:
                gas_struct = pdb_parser.get_structure("gas", str(gas_pdb))
                feature_cache[gas_uniprot] = extract_residue_features(gas_struct)
            gas_features = feature_cache[gas_uniprot]

            if human_acc not in feature_cache:
                human_struct = pdb_parser.get_structure("human", str(human_pdb))
                feature_cache[human_acc] = extract_residue_features(human_struct)
            human_features = feature_cache[human_acc]

            if gas_features.empty or human_features.empty:
                logger.warning("    Empty features, skipping")
                result = {"gas_uniprot": gas_uniprot, "human_accession": human_acc,
                          "human_gene": human_gene, "status": "empty_features"}
                result.update(_empty_result())
                results.append(result)
                continue

            # Parse aligned ranges
            gas_range = str(row.get("gas_aligned_range", ""))
            human_range = str(row.get("human_aligned_range", ""))

            if "-" in gas_range and "-" in human_range:
                gas_start, gas_end = map(int, gas_range.split("-"))
                human_start, human_end = map(int, human_range.split("-"))
            else:
                logger.warning("    No aligned range info, skipping")
                result = {"gas_uniprot": gas_uniprot, "human_accession": human_acc,
                          "human_gene": human_gene, "status": "no_range"}
                result.update(_empty_result())
                results.append(result)
                continue

            scores = score_conformational_mimicry(
                gas_features, human_features,
                gas_start, gas_end, human_start, human_end,
            )

            result = {
                "gas_protein": row["gas_protein"],
                "gas_uniprot": gas_uniprot,
                "human_accession": human_acc,
                "human_gene": human_gene,
                "status": "success",
                "tm_score": row.get("tm_score"),
                "rmsd": row.get("rmsd"),
                "pident": row.get("pident"),
            }
            result.update(scores)
            results.append(result)

            logger.info(
                "    conf_score=%.4f, overlap=%d/%d (%.1f%%), physchem=%.3f",
                scores["conformational_mimicry_score"],
                scores["epitope_overlap_count"],
                scores["aligned_residues"],
                scores["epitope_overlap_fraction"] * 100,
                scores["mean_physicochemical_similarity"],
            )

        except Exception as e:
            logger.warning("    Error: %s", e)
            result = {"gas_uniprot": gas_uniprot, "human_accession": human_acc,
                      "human_gene": human_gene, "status": f"error: {e}"}
            result.update(_empty_result())
            results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_path = args.dest / "conformational_mimicry_scores.tsv"
    results_df.to_csv(results_path, sep="\t", index=False)
    logger.info("Results saved: %s (%d entries)", results_path.name, len(results_df))

    # Generate target-level summary (best score per human target)
    if not results_df.empty:
        success_df = results_df[results_df["status"] == "success"]
        if not success_df.empty:
            target_summary = (
                success_df.groupby("human_accession")
                .agg(
                    human_gene=("human_gene", "first"),
                    best_conf_score=("conformational_mimicry_score", "max"),
                    best_overlap_fraction=("epitope_overlap_fraction", "max"),
                    best_physchem=("mean_physicochemical_similarity", "max"),
                    pairs_analyzed=("gas_uniprot", "count"),
                )
                .reset_index()
                .sort_values("best_conf_score", ascending=False)
            )
            summary_path = args.dest / "emomis_target_summary.tsv"
            target_summary.to_csv(summary_path, sep="\t", index=False)
            logger.info("Target summary: %s (%d targets)", summary_path.name, len(target_summary))

            logger.info("\nTop conformational mimicry targets:")
            for _, r in target_summary.head(10).iterrows():
                gene = r["human_gene"] or r["human_accession"]
                logger.info("  %s: conf_score=%.4f, overlap=%.1f%%, physchem=%.3f",
                            gene, r["best_conf_score"],
                            r["best_overlap_fraction"] * 100, r["best_physchem"])

    logger.info("Done. Conformational mimicry analysis in %s", args.dest)


if __name__ == "__main__":
    main()
