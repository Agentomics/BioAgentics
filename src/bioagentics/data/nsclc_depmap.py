"""Filter DepMap to NSCLC cell lines and annotate with driver mutations.

Produces an annotated cell line table with oncogenic driver status,
KRAS allele type, and molecular subtype classification (KP/KL/KOnly/KRAS-WT).

Usage:
    uv run python -m bioagentics.data.nsclc_depmap
    uv run python -m bioagentics.data.nsclc_depmap --dest data/depmap/25q3/nsclc_cell_lines_annotated.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bioagentics.data.gene_ids import load_depmap_model_metadata, load_depmap_mutations
from bioagentics.data.nsclc_common import (
    DRIVER_GENES,
    classify_kras_allele,
    classify_molecular_subtype,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"


def annotate_nsclc_lines(depmap_dir: str | Path) -> pd.DataFrame:
    """Filter DepMap to NSCLC lines and annotate with driver mutations.

    Returns DataFrame indexed by ModelID with columns:
    - Cell line metadata (name, subtype, etc.)
    - Per-driver mutation status (bool) and protein change
    - KRAS allele classification
    - Molecular subtype (KP, KL, KOnly, KRAS-WT)
    """
    depmap_dir = Path(depmap_dir)
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")
    muts = load_depmap_mutations(depmap_dir / "OmicsSomaticMutations.csv")

    # Filter to NSCLC
    nsclc = meta[meta["OncotreePrimaryDisease"] == "Non-Small Cell Lung Cancer"].copy()
    nsclc_ids = set(nsclc.index)

    # Filter mutations to NSCLC lines and driver genes, HIGH/MODERATE impact
    driver_muts = muts[
        (muts["ModelID"].isin(nsclc_ids))
        & (muts["HugoSymbol"].isin(DRIVER_GENES))
        & (muts["VepImpact"].isin(["HIGH", "MODERATE"]))
    ].copy()

    # Annotate each driver gene as mutated/WT per cell line
    for gene in DRIVER_GENES:
        gene_muts = driver_muts[driver_muts["HugoSymbol"] == gene]
        mutated_lines = set(gene_muts["ModelID"])
        nsclc[f"{gene}_mutated"] = nsclc.index.isin(mutated_lines)

        # Store protein change(s) for reference
        pc_map = gene_muts.groupby("ModelID")["ProteinChange"].apply(
            lambda x: ";".join(x.dropna().unique())
        )
        nsclc[f"{gene}_protein_change"] = nsclc.index.map(pc_map).fillna("")

    # Classify KRAS allele type
    kras_muts = driver_muts[driver_muts["HugoSymbol"] == "KRAS"]
    kras_alleles = kras_muts.groupby("ModelID")["ProteinChange"].apply(list)

    def get_kras_allele(model_id: str) -> str:
        if model_id in kras_alleles.index:
            return classify_kras_allele(kras_alleles[model_id])
        return "WT"

    nsclc["KRAS_allele"] = [get_kras_allele(mid) for mid in nsclc.index]

    # Classify molecular subtypes
    nsclc["molecular_subtype"] = nsclc.apply(classify_molecular_subtype, axis=1)

    # Select output columns
    keep_cols = [
        "CellLineName", "StrippedCellLineName", "OncotreeSubtype", "OncotreeCode",
        "Sex", "PrimaryOrMetastasis",
    ]
    driver_cols = []
    for gene in DRIVER_GENES:
        driver_cols.extend([f"{gene}_mutated", f"{gene}_protein_change"])
    keep_cols.extend(driver_cols)
    keep_cols.extend(["KRAS_allele", "molecular_subtype"])

    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in nsclc.columns]
    return nsclc[keep_cols]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Filter DepMap to NSCLC lines and annotate with driver mutations",
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
        default=DEFAULT_DEPMAP_DIR / "nsclc_cell_lines_annotated.csv",
        help="Output CSV path",
    )
    args = parser.parse_args(argv)

    print("Filtering DepMap to NSCLC cell lines...")
    df = annotate_nsclc_lines(args.depmap_dir)

    print(f"\nNSCLC cell lines: {len(df)}")
    print(f"\nOncotree subtypes:")
    print(df["OncotreeSubtype"].value_counts().to_string())
    print(f"\nMolecular subtypes:")
    print(df["molecular_subtype"].value_counts().to_string())
    print(f"\nKRAS alleles (among KRAS-mutant):")
    kras_mut = df[df["KRAS_mutated"]]
    print(kras_mut["KRAS_allele"].value_counts().to_string())

    df.to_csv(args.dest)
    print(f"\nSaved to {args.dest}")


if __name__ == "__main__":
    main()
