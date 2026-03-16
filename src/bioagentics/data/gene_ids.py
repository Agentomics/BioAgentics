"""Gene ID harmonization for DepMap and TCGA datasets.

Provides loaders that normalize all gene identifiers to HUGO symbols,
enabling cross-dataset analyses.

DepMap matrices use "SYMBOL (ENTREZ_ID)" column headers.
TCGA RNA-seq uses versioned Ensembl IDs (ENSG*.##) with gene_name as HUGO symbol.
TCGA MAF files use Hugo_Symbol directly.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pandas as pd

DEPMAP_GENE_RE = re.compile(r"^(.+?)\s+\((\d+)\)$")


def parse_depmap_gene_col(col: str) -> tuple[str, int | None]:
    """Parse DepMap column header 'SYMBOL (ENTREZ_ID)' into (symbol, entrez_id)."""
    m = DEPMAP_GENE_RE.match(col)
    if m:
        return m.group(1), int(m.group(2))
    return col, None


def load_depmap_matrix(path: str | Path) -> pd.DataFrame:
    """Load a DepMap gene-by-sample matrix with HUGO symbol columns.

    Reads CSVs where first column is model ID (ACH-*) and gene columns
    use "SYMBOL (ENTREZ)" format. Returns DataFrame indexed by ModelID
    with HUGO symbol column names.

    Duplicate symbols (rare) are resolved by keeping the first occurrence.
    """
    df = pd.read_csv(path, index_col=0)

    # Expression files have extra metadata columns before gene columns
    meta_cols = ["SequencingID", "ModelID", "IsDefaultEntryForModel",
                 "ModelConditionID", "IsDefaultEntryForMC"]
    present_meta = [c for c in meta_cols if c in df.columns]

    if present_meta:
        # Use ModelID as index if available, filter to default entries
        if "IsDefaultEntryForModel" in df.columns:
            df = df[df["IsDefaultEntryForModel"] == "Yes"].copy()
        if "ModelID" in df.columns:
            df = df.set_index("ModelID")
        df = df.drop(columns=[c for c in present_meta if c in df.columns], errors="ignore")

    # Parse gene columns
    rename = {}
    for col in df.columns:
        symbol, _ = parse_depmap_gene_col(col)
        rename[col] = symbol

    df = df.rename(columns=rename)

    # Drop duplicate gene symbols (keep first)
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    return df


def load_depmap_mutations(path: str | Path) -> pd.DataFrame:
    """Load DepMap somatic mutations (long format).

    Returns DataFrame with key columns: ModelID, HugoSymbol, ProteinChange,
    VariantType, MolecularConsequence, etc.
    """
    cols = [
        "ModelID", "HugoSymbol", "EnsemblGeneID", "Chrom", "Pos",
        "Ref", "Alt", "VariantType", "VariantInfo", "ProteinChange",
        "MolecularConsequence", "VepImpact",
    ]
    df = pd.read_csv(path, usecols=lambda c: c in cols)
    return df


def load_depmap_model_metadata(path: str | Path) -> pd.DataFrame:
    """Load DepMap Model.csv cell line metadata."""
    df = pd.read_csv(path, index_col="ModelID")
    return df


def load_tcga_expression_sample(path: str | Path) -> pd.Series:
    """Load a single TCGA STAR-Counts file, returning TPM values keyed by HUGO symbol.

    Skips the first 4 summary rows (N_unmapped, N_multimapping, etc.) and
    the comment header line.
    """
    df = pd.read_csv(
        path, sep="\t", comment="#",
        usecols=["gene_id", "gene_name", "gene_type", "tpm_unstranded"],
    )
    # Keep protein-coding genes only
    df = df[df["gene_type"] == "protein_coding"].copy()
    # Use HUGO symbol; drop duplicates (keep first)
    df = df.drop_duplicates(subset="gene_name", keep="first")
    return df.set_index("gene_name")["tpm_unstranded"]


def load_tcga_expression_matrix(expr_dir: str | Path) -> pd.DataFrame:
    """Load all TCGA STAR-Counts files from a directory into a gene x sample matrix.

    Returns DataFrame with HUGO symbol index and file-UUID columns.
    Uses TPM values (tpm_unstranded), protein-coding genes only.
    """
    expr_dir = Path(expr_dir)
    files = sorted(expr_dir.glob("*.tsv"))
    if not files:
        raise FileNotFoundError(f"No .tsv files found in {expr_dir}")

    series_list = []
    sample_ids = []

    for f in files:
        # Extract UUID from filename
        uuid = f.stem.split(".")[0]
        s = load_tcga_expression_sample(f)
        series_list.append(s)
        sample_ids.append(uuid)

    df = pd.DataFrame(series_list, index=sample_ids)
    df.index.name = "sample_id"
    return df


def load_tcga_mutations(mut_dir: str | Path) -> pd.DataFrame:
    """Load all TCGA MAF files from a directory into a single DataFrame.

    Reads gzipped MAF files. Returns long-format DataFrame with key columns:
    Hugo_Symbol, Entrez_Gene_Id, Variant_Classification, etc.
    """
    mut_dir = Path(mut_dir)
    files = sorted(mut_dir.glob("*.maf.gz"))
    if not files:
        raise FileNotFoundError(f"No .maf.gz files found in {mut_dir}")

    key_cols = [
        "Hugo_Symbol", "Entrez_Gene_Id", "Chromosome", "Start_Position",
        "End_Position", "Variant_Classification", "Variant_Type",
        "Reference_Allele", "Tumor_Seq_Allele2", "Tumor_Sample_Barcode",
        "HGVSp_Short",
    ]

    frames = []
    for f in files:
        with gzip.open(f, "rt") as fh:
            # Skip comment lines
            df = pd.read_csv(fh, sep="\t", comment="#", usecols=lambda c: c in key_cols)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def load_tcga_clinical(clinical_path: str | Path) -> pd.DataFrame:
    """Load TCGA clinical JSON (from GDC cases API) into a flat DataFrame."""
    import json

    with open(clinical_path) as f:
        cases = json.load(f)

    rows = []
    for case in cases:
        row = {
            "case_id": case.get("case_id"),
            "submitter_id": case.get("submitter_id"),
        }
        demo = case.get("demographic", {})
        if isinstance(demo, list):
            demo = demo[0] if demo else {}
        row["gender"] = demo.get("gender")
        row["vital_status"] = demo.get("vital_status")
        row["days_to_death"] = demo.get("days_to_death")
        row["race"] = demo.get("race")
        row["ethnicity"] = demo.get("ethnicity")

        diag = case.get("diagnoses", [])
        if isinstance(diag, list) and diag:
            d = diag[0]
            row["primary_diagnosis"] = d.get("primary_diagnosis")
            row["tumor_stage"] = d.get("tumor_stage")
            row["days_to_last_follow_up"] = d.get("days_to_last_follow_up")
            row["age_at_diagnosis"] = d.get("age_at_diagnosis")
            row["tissue_or_organ_of_origin"] = d.get("tissue_or_organ_of_origin")

        rows.append(row)

    return pd.DataFrame(rows)


def find_common_genes(*gene_lists: list[str] | pd.Index) -> list[str]:
    """Find genes common to all provided gene lists/indices."""
    sets = [set(g) for g in gene_lists]
    common = sets[0]
    for s in sets[1:]:
        common &= s
    return sorted(common)
