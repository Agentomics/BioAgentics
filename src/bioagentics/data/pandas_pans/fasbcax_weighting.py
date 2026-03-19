"""FasBCAX invasive-phase protein weighting for GAS mimicry ranking.

Classifies GAS proteins by infection phase based on the FasBCAX/CovR/S
regulatory system and applies weighting multipliers for mimicry scoring.

Invasive-phase virulence factors (streptokinase, SLO, SpeB, DNases, etc.)
receive a 2x priority multiplier. Colonization-phase adhesins (pili, PrtF,
SfbI) rank lower with a 0.5x multiplier. Housekeeping proteins keep 1x.

Usage:
    uv run python -m bioagentics.data.pandas_pans.fasbcax_weighting [--dest DIR]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path("data/pandas_pans/gas-molecular-mimicry-mapping")
PROTEOME_DIR = PROJECT_DIR / "proteomes"
SCREEN_DIR = PROJECT_DIR / "mimicry_screen"
OUTPUT_DIR = PROJECT_DIR / "phase_weighting"

# ── Phase annotations ──
# Gene names / protein description patterns mapped to infection phase.
# Based on FasBCAX/CovR/S/Mga regulatory cascades in GAS.

# Invasive-phase proteins: upregulated during tissue invasion, immune evasion.
# These get 2x weight in mimicry scoring.
INVASIVE_GENES: dict[str, str] = {
    "ska": "Streptokinase A — activates human plasminogen, primary invasive factor",
    "slo": "Streptolysin O — cholesterol-dependent cytolysin, tissue damage",
    "sagA": "Streptolysin S — cytolytic toxin, SagA precursor peptide",
    "speB": "Streptococcal pyrogenic exotoxin B / streptopain — cysteine protease",
    "speC": "Streptococcal pyrogenic exotoxin C — superantigen",
    "speA": "Streptococcal pyrogenic exotoxin A — superantigen",
    "smeZ": "Streptococcal mitogenic exotoxin Z — superantigen",
    "endoS": "Endo-beta-N-acetylglucosaminidase — cleaves IgG, immune evasion",
    "ideS": "IgG-degrading enzyme — Mac/IdeS, immune evasion",
    "spyCEP": "IL-8 protease — neutrophil recruitment inhibitor",
    "sda1": "Streptodornase / DNase Sda1 — degrades NETs",
    "spd": "Streptodornase — degrades DNA/NETs",
    "grab": "Protein GRAB — alpha-2-macroglobulin binding, immune evasion",
    "sof": "Serum opacity factor — disrupts HDL, immune evasion",
    "dnaK": "DnaK/Hsp70 chaperone — surface-exposed during infection stress",
    "groEL": "GroEL/Hsp60 chaperonin — surface-exposed immunogen",
}

# M-protein family: central virulence factors, active across phases but
# especially important in immune evasion. Weighted as invasive (2x).
M_PROTEIN_GENES: dict[str, str] = {
    "emm": "M protein — anti-phagocytic, binds complement regulators",
    "mrp": "M-related protein — IgA binding, Fc receptor",
    "enn": "Protein Enn — IgA binding, mga regulon",
    "fcrA": "IgG Fc binding protein — alias for mrp in some serotypes",
}

# Colonization-phase proteins: adhesins and surface structures for
# initial host attachment. Lower mimicry priority (0.5x weight).
COLONIZATION_GENES: dict[str, str] = {
    "prtF": "Fibronectin-binding protein F / SfbI — epithelial adhesion",
    "sfbI": "Streptococcal fibronectin-binding protein I — adhesin",
    "sfbX": "Fibronectin-binding protein X — adhesin",
    "cpa": "Collagen-binding pilus backbone — adhesion pilus",
    "sipA": "Signal peptidase-processed pilus — adhesion pilus",
    "fba": "Fibronectin-binding protein — adhesion (note: also fructose-bisphosphate aldolase)",
    "lbp": "Laminin-binding protein — ECM adhesion",
    "scl": "Streptococcal collagen-like protein — adhesion/collagen mimicry",
    "hasA": "Hyaluronate synthase — capsule biosynthesis, colonization",
    "hasB": "UDP-glucose 6-dehydrogenase — capsule biosynthesis",
    "hasC": "UTP-glucose-1-phosphate uridylyltransferase — capsule biosynthesis",
}

# Compile regex patterns for matching FASTA headers
_INVASIVE_PATTERN = re.compile(
    "|".join(
        [
            # Gene name patterns (GN=xxx)
            *[rf"\bGN={re.escape(g)}\b" for g in INVASIVE_GENES],
            *[rf"\bGN={re.escape(g)}" for g in M_PROTEIN_GENES],
            # Protein description patterns
            r"[Ss]treptokinase",
            r"[Ss]treptolysin",
            r"[Ss]treptopain",
            r"[Ee]xotoxin",
            r"IgG.degrading.enzyme",
            r"Protein GRAB",
            r"Serum opacity factor",
            r"M protein type",
            r"M-related protein",
        ]
    ),
    re.IGNORECASE,
)

_COLONIZATION_PATTERN = re.compile(
    "|".join(
        [
            *[rf"\bGN={re.escape(g)}" for g in COLONIZATION_GENES],
            r"[Ff]ibronectin.binding protein",
            r"[Cc]ollagen.like protein",
            r"pilus",
            r"[Hh]yaluronate synthase",
        ]
    ),
    re.IGNORECASE,
)

# Weight multipliers
INVASIVE_WEIGHT = 2.0
COLONIZATION_WEIGHT = 0.5
HOUSEKEEPING_WEIGHT = 1.0


def classify_protein(header: str) -> tuple[str, float, str]:
    """Classify a GAS protein by infection phase from its FASTA header.

    Returns (phase, weight, reason).
    """
    if _INVASIVE_PATTERN.search(header):
        match = _INVASIVE_PATTERN.search(header)
        return "invasive", INVASIVE_WEIGHT, match.group(0)  # type: ignore[union-attr]
    if _COLONIZATION_PATTERN.search(header):
        match = _COLONIZATION_PATTERN.search(header)
        return "colonization", COLONIZATION_WEIGHT, match.group(0)  # type: ignore[union-attr]
    return "housekeeping", HOUSEKEEPING_WEIGHT, ""


def build_phase_annotation(proteome_dir: Path) -> pd.DataFrame:
    """Parse all GAS FASTA files and classify each protein by infection phase.

    Returns DataFrame with columns: protein_id, serotype, phase, weight, reason, description.
    """
    rows: list[dict] = []
    combined = proteome_dir / "gas_combined.fasta"

    if not combined.exists():
        raise FileNotFoundError(f"Combined proteome not found: {combined}")

    with open(combined) as f:
        for line in f:
            if not line.startswith(">"):
                continue
            header = line.strip()
            # Extract protein ID (first field after >)
            protein_id = header[1:].split()[0]
            # Extract description
            description = header[1:].split(None, 1)[1] if " " in header[1:] else ""

            # Determine serotype from header
            serotype = _extract_serotype(header)

            phase, weight, reason = classify_protein(header)
            rows.append({
                "protein_id": protein_id,
                "serotype": serotype,
                "phase": phase,
                "weight": weight,
                "match_reason": reason,
                "description": description,
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Phase classification: %d invasive, %d colonization, %d housekeeping (of %d total)",
        (df["phase"] == "invasive").sum(),
        (df["phase"] == "colonization").sum(),
        (df["phase"] == "housekeeping").sum(),
        len(df),
    )
    return df


def _extract_serotype(header: str) -> str:
    """Extract GAS serotype from FASTA header (e.g., 'STRP1' -> 'M1')."""
    # Match patterns like STRP1, STRP3, STRPC (M12), STRPZ (M49)
    strain_map = {
        "STRP1": "M1",
        "STRP3": "M3",
        "STRPC": "M12",
        "STRPZ": "M49",
    }
    for code, serotype in strain_map.items():
        if code in header:
            return serotype

    # Check for organism description
    m = re.search(r"serotype M(\d+)", header)
    if m:
        return f"M{m.group(1)}"

    # UniParc entries — try taxonomy
    if "OX=301447" in header:
        return "M1"
    if "OX=198466" in header:
        return "M3"
    if "OX=370551" in header or "OX=370553" in header:
        return "M12"
    if "OX=392612" in header:
        return "M5"
    if "OX=186103" in header:
        return "M18"
    if "OX=471876" in header:
        return "M49"

    # UniParc IDs without strain info
    if header.startswith(">UPI"):
        return "unknown"

    return "unknown"


def apply_phase_weights(hits_df: pd.DataFrame, phase_df: pd.DataFrame) -> pd.DataFrame:
    """Apply phase-based weights to mimicry screening hits.

    Joins hits with phase annotations on the GAS protein ID (qseqid column)
    and adds phase_weight, phase, and weighted_bitscore columns.
    """
    # Build lookup from phase annotations (drop duplicates keeping first)
    deduped = phase_df.drop_duplicates(subset="protein_id", keep="first")
    phase_lookup = dict(
        zip(deduped["protein_id"], zip(deduped["phase"], deduped["weight"]))
    )

    phases = []
    weights = []
    for _, row in hits_df.iterrows():
        protein_id = row["qseqid"]
        phase, weight = phase_lookup.get(protein_id, ("housekeeping", HOUSEKEEPING_WEIGHT))
        phases.append(phase)
        weights.append(weight)

    result = hits_df.copy()
    result["phase"] = phases
    result["phase_weight"] = weights
    result["weighted_bitscore"] = result["bitscore"] * result["phase_weight"]

    # Re-sort by weighted bitscore
    result = result.sort_values("weighted_bitscore", ascending=False)

    logger.info(
        "Applied phase weights: %d invasive hits (2x), %d colonization (0.5x), %d housekeeping (1x)",
        (result["phase"] == "invasive").sum(),
        (result["phase"] == "colonization").sum(),
        (result["phase"] == "housekeeping").sum(),
    )
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="FasBCAX phase weighting for GAS molecular mimicry hits",
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.dest.mkdir(parents=True, exist_ok=True)

    # 1. Build phase annotation for entire GAS proteome
    logger.info("Building phase annotation from GAS proteomes...")
    phase_df = build_phase_annotation(PROTEOME_DIR)

    # Save phase annotation CSV
    annotation_path = args.dest / "gas_phase_annotations.csv"
    phase_df.to_csv(annotation_path, index=False)
    logger.info("Phase annotations saved: %s (%d proteins)", annotation_path.name, len(phase_df))

    # Report Mrp status
    mrp_hits = phase_df[
        phase_df["description"].str.contains("M-related|mrp|fcrA", case=False, na=False)
    ]
    if mrp_hits.empty:
        logger.warning(
            "Mrp (M-related protein) NOT found in proteomes. "
            "This is expected for M1/SF370 which lacks mrp. "
            "Check M3/M12/M18 proteomes for mrp presence."
        )
    else:
        logger.info("Mrp found: %d entries across serotypes", len(mrp_hits))

    # Report streptokinase status
    ska_hits = phase_df[
        phase_df["description"].str.contains("streptokinase|GN=ska", case=False, na=False)
    ]
    logger.info("Streptokinase: %d entries across serotypes", len(ska_hits))

    # 2. Apply weights to filtered mimicry hits
    filtered_path = SCREEN_DIR / "hits_filtered.tsv"
    if filtered_path.exists():
        logger.info("Applying phase weights to mimicry hits...")
        hits_df = pd.read_csv(filtered_path, sep="\t")
        weighted = apply_phase_weights(hits_df, phase_df)

        weighted_path = args.dest / "hits_weighted.tsv"
        weighted.to_csv(weighted_path, sep="\t", index=False)
        logger.info("Weighted hits saved: %s (%d rows)", weighted_path.name, len(weighted))

        # Summary of top weighted hits
        summary_path = args.dest / "weighting_summary.txt"
        with open(summary_path, "w") as f:
            f.write("FasBCAX Phase Weighting Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total proteome proteins classified: {len(phase_df)}\n")
            f.write(f"  Invasive-phase: {(phase_df['phase'] == 'invasive').sum()}\n")
            f.write(f"  Colonization-phase: {(phase_df['phase'] == 'colonization').sum()}\n")
            f.write(f"  Housekeeping: {(phase_df['phase'] == 'housekeeping').sum()}\n")
            f.write(f"\nStreptokinase entries: {len(ska_hits)}\n")
            f.write(f"Mrp entries: {len(mrp_hits)}\n")
            f.write(f"\nWeighted mimicry hits: {len(weighted)}\n")

            for phase in ["invasive", "colonization", "housekeeping"]:
                subset = weighted[weighted["phase"] == phase]
                if not subset.empty:
                    f.write(f"\n--- {phase.upper()} phase hits ({len(subset)}) ---\n")
                    for _, row in subset.head(10).iterrows():
                        gene = str(row.get("human_gene", "?")) if pd.notna(row.get("human_gene")) else "?"
                        f.write(
                            f"  {str(row['qseqid'])[:40]:40s} -> {gene:10s} "
                            f"pident={row['pident']:.1f}% "
                            f"bitscore={row['bitscore']:.0f} "
                            f"weighted={row['weighted_bitscore']:.0f}\n"
                        )

        logger.info("Summary: %s", summary_path.name)
    else:
        logger.warning("No filtered hits found at %s — run mimicry_screen.py first", filtered_path)

    logger.info("Done. Phase weighting results in %s", args.dest)


if __name__ == "__main__":
    main()
