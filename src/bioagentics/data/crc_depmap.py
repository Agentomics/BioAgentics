"""Filter DepMap to CRC cell lines and annotate with driver mutations.

Produces an annotated cell line table with oncogenic driver status,
KRAS allele type, BRAF V600E status, and MSI classification.

Usage:
    uv run python -m bioagentics.data.crc_depmap
    uv run python -m bioagentics.data.crc_depmap --dest output/crc-kras-dependencies/crc_cell_lines_classified.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_model_metadata, load_depmap_mutations
from bioagentics.data.crc_common import (
    DRIVER_GENES,
    classify_kras_allele,
    classify_msi_status,
)

DEFAULT_DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
DEFAULT_DEST = REPO_ROOT / "output" / "crc-kras-dependencies" / "crc_cell_lines_classified.csv"


def annotate_crc_lines(depmap_dir: str | Path) -> pd.DataFrame:
    """Filter DepMap to CRC lines and annotate with driver mutations.

    Returns DataFrame indexed by ModelID with columns:
    - Cell line metadata (name, subtype, etc.)
    - Per-driver mutation status (bool) and protein change
    - BRAF_V600E specific flag
    - KRAS allele classification
    - MSI status
    """
    depmap_dir = Path(depmap_dir)
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    muts = load_depmap_mutations(depmap_dir / "OmicsSomaticMutations.csv")

    # Filter to CRC: Colorectal Adenocarcinoma
    crc = meta[meta["OncotreePrimaryDisease"] == "Colorectal Adenocarcinoma"].copy()
    crc_ids = set(crc.index)

    # Filter mutations to CRC lines and driver genes, HIGH/MODERATE impact
    driver_muts = muts[
        (muts["ModelID"].isin(crc_ids))
        & (muts["HugoSymbol"].isin(DRIVER_GENES))
        & (muts["VepImpact"].isin(["HIGH", "MODERATE"]))
    ].copy()

    # Annotate each driver gene as mutated/WT per cell line
    for gene in DRIVER_GENES:
        gene_muts = driver_muts[driver_muts["HugoSymbol"] == gene]
        mutated_lines = set(gene_muts["ModelID"])
        crc[f"{gene}_mutated"] = crc.index.isin(mutated_lines)

        pc_map = gene_muts.groupby("ModelID")["ProteinChange"].apply(
            lambda x: ";".join(x.dropna().unique())
        )
        crc[f"{gene}_protein_change"] = crc.index.map(pc_map).fillna("")

    # BRAF V600E specific flag (mutually exclusive with KRAS in CRC)
    braf_muts = driver_muts[driver_muts["HugoSymbol"] == "BRAF"]
    v600e_lines = set(
        braf_muts[braf_muts["ProteinChange"] == "p.V600E"]["ModelID"]
    )
    crc["BRAF_V600E"] = crc.index.isin(v600e_lines)

    # Classify KRAS allele type
    kras_muts = driver_muts[driver_muts["HugoSymbol"] == "KRAS"]
    kras_alleles = kras_muts.groupby("ModelID")["ProteinChange"].apply(list)

    def get_kras_allele(model_id: str) -> str:
        if model_id in kras_alleles.index:
            return classify_kras_allele(kras_alleles[model_id])
        return "WT"

    crc["KRAS_allele"] = [get_kras_allele(mid) for mid in crc.index]

    # MSI status from curated list (no metadata column available)
    crc["MSI_status"] = crc["StrippedCellLineName"].apply(
        lambda name: classify_msi_status(name)
    )

    # Select output columns
    keep_cols = [
        "CellLineName", "StrippedCellLineName", "OncotreeSubtype", "OncotreeCode",
        "Sex", "PrimaryOrMetastasis",
    ]
    driver_cols = []
    for gene in DRIVER_GENES:
        driver_cols.extend([f"{gene}_mutated", f"{gene}_protein_change"])
    keep_cols.extend(driver_cols)
    keep_cols.extend(["BRAF_V600E", "KRAS_allele", "MSI_status"])

    keep_cols = [c for c in keep_cols if c in crc.columns]
    return crc[keep_cols]


def _print_qc(df: pd.DataFrame) -> None:
    """Print QC summary statistics."""
    print(f"\nCRC cell lines: {len(df)}")
    print(f"\nOncotree subtypes:")
    print(df["OncotreeSubtype"].value_counts().to_string())

    kras_mut = df[df["KRAS_mutated"]]
    print(f"\nKRAS-mutant lines: {len(kras_mut)} / {len(df)}")
    print(f"\nKRAS allele distribution:")
    print(df["KRAS_allele"].value_counts().to_string())

    print(f"\nBRAF V600E lines: {df['BRAF_V600E'].sum()}")

    # Check BRAF V600E / KRAS mutual exclusivity
    both = df[df["BRAF_V600E"] & df["KRAS_mutated"]]
    if len(both) == 0:
        print("BRAF V600E / KRAS mutual exclusivity: CONFIRMED")
    else:
        print(f"WARNING: {len(both)} lines have both BRAF V600E and KRAS mutation")

    print(f"\nMSI status:")
    print(df["MSI_status"].value_counts().to_string())

    print(f"\nCo-mutation rates in KRAS-mutant lines:")
    for gene in ["APC", "TP53", "PIK3CA", "SMAD4"]:
        col = f"{gene}_mutated"
        if col in kras_mut.columns:
            rate = kras_mut[col].mean() * 100
            print(f"  {gene}: {rate:.0f}%")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Filter DepMap to CRC lines and annotate with driver mutations",
    )
    parser.add_argument(
        "--depmap-dir",
        type=Path,
        default=DEFAULT_DEPMAP_DIR,
        help=f"DepMap data directory (default: {DEFAULT_DEPMAP_DIR})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Output CSV path",
    )
    args = parser.parse_args(argv)

    print("Filtering DepMap to CRC cell lines...")
    df = annotate_crc_lines(args.depmap_dir)

    _print_qc(df)

    args.dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.dest)
    print(f"\nSaved to {args.dest}")


if __name__ == "__main__":
    main()
