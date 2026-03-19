"""Multi-modal SNP-to-gene mapping for TS GWAS loci.

Maps GWAS SNPs to genes using three complementary approaches:
1. Positional mapping (within configurable window of gene boundaries)
2. eQTL mapping (GTEx v8 brain-tissue eQTLs)
3. Chromatin interaction mapping (Hi-C brain data)

Results are integrated to produce a unified SNP-gene mapping with
evidence scores across modalities.

Usage:
    python -m bioagentics.tourettes.ts_gwas_functional_annotation.snp_to_gene \
        --gwas data/tourettes/ts-gwas-functional-annotation/gwas/tsaicg_cleaned.tsv \
        --gene-annot data/tourettes/ts-gwas-functional-annotation/gene_annotations/genes.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.tourettes.ts_gwas_functional_annotation.config import (
    EQTL_DIR,
    EQTL_FDR_THRESHOLD,
    GENE_ANNOT_DIR,
    GTEX_BRAIN_TISSUES,
    HIC_DIR,
    MAPPING_DIR,
    POSITIONAL_WINDOW_KB,
    TS_CANDIDATE_GENES,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


@dataclass
class GeneMapping:
    """A single SNP-to-gene mapping with evidence from multiple modalities."""

    snp: str
    gene: str
    chr: int
    snp_bp: int
    gene_start: int
    gene_end: int
    distance_kb: float
    positional: bool = False
    eqtl: bool = False
    eqtl_tissues: list[str] = field(default_factory=list)
    eqtl_best_p: float = 1.0
    hic: bool = False
    hic_tissues: list[str] = field(default_factory=list)
    n_evidence: int = 0
    is_known_candidate: bool = False

    def __post_init__(self):
        self.n_evidence = sum([self.positional, self.eqtl, self.hic])
        self.is_known_candidate = self.gene in TS_CANDIDATE_GENES


# ---------------------------------------------------------------------------
# Gene annotations
# ---------------------------------------------------------------------------


def load_gene_annotations(path: Path | None = None) -> pd.DataFrame:
    """Load gene annotations (GENE, CHR, START, STOP).

    Supports NCBI gene_info format, Ensembl GTF-derived TSV, and generic TSV.
    """
    if path is None:
        path = GENE_ANNOT_DIR / "genes.tsv"

    if not path.exists():
        logger.warning("Gene annotation file not found: %s", path)
        return pd.DataFrame(columns=["GENE", "CHR", "START", "STOP"])

    df = pd.read_csv(path, sep="\t")

    col_map = {
        "SYMBOL": "GENE", "gene_symbol": "GENE", "gene_name": "GENE",
        "Gene": "GENE", "symbol": "GENE", "GENE_SYMBOL": "GENE",
        "CHROMOSOME": "CHR", "chr": "CHR", "chrom": "CHR", "#chrom": "CHR",
        "START_POS": "START", "start": "START", "chromStart": "START",
        "tx_start": "START", "txStart": "START",
        "STOP_POS": "STOP", "stop": "STOP", "end": "STOP", "END": "STOP",
        "chromEnd": "STOP", "tx_end": "STOP", "txEnd": "STOP",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    required = {"GENE", "CHR", "START", "STOP"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.error("Gene annotation missing columns: %s", missing)
        return pd.DataFrame(columns=["GENE", "CHR", "START", "STOP"])

    # Clean chromosome
    df["CHR"] = df["CHR"].astype(str).str.replace("chr", "", case=False)
    df["CHR"] = pd.to_numeric(df["CHR"], errors="coerce")
    df = df.dropna(subset=["CHR"])
    df["CHR"] = df["CHR"].astype(int)
    df = df[df["CHR"].between(1, 22)]

    # Deduplicate genes (keep longest transcript span)
    df["_span"] = df["STOP"] - df["START"]
    df = df.sort_values("_span", ascending=False).drop_duplicates(
        subset=["GENE", "CHR"], keep="first"
    )
    df = df.drop(columns=["_span"])

    logger.info("Loaded %d gene annotations", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1. Positional mapping
# ---------------------------------------------------------------------------


def positional_mapping(
    gwas_df: pd.DataFrame,
    gene_annot: pd.DataFrame,
    window_kb: int = POSITIONAL_WINDOW_KB,
) -> list[GeneMapping]:
    """Map SNPs to nearby genes within a positional window.

    Parameters
    ----------
    gwas_df : pd.DataFrame
        GWAS with SNP, CHR, BP, P columns.
    gene_annot : pd.DataFrame
        Gene annotations with GENE, CHR, START, STOP columns.
    window_kb : int
        Window size in kb around gene boundaries.

    Returns
    -------
    List of GeneMapping objects with positional evidence.
    """
    if gene_annot.empty:
        logger.warning("No gene annotations available for positional mapping")
        return []

    if not {"SNP", "CHR", "BP"}.issubset(gwas_df.columns):
        logger.warning("GWAS missing CHR/BP columns for positional mapping")
        return []

    window = window_kb * 1000
    mappings = []

    for chr_val in gwas_df["CHR"].unique():
        chr_snps = gwas_df[gwas_df["CHR"] == chr_val]
        chr_genes = gene_annot[gene_annot["CHR"] == chr_val]

        if chr_genes.empty:
            continue

        for _, snp_row in chr_snps.iterrows():
            snp_bp = int(snp_row["BP"])
            gene_start_ext = chr_genes["START"] - window
            gene_stop_ext = chr_genes["STOP"] + window

            nearby = chr_genes[
                (snp_bp >= gene_start_ext) & (snp_bp <= gene_stop_ext)
            ]

            for _, gene_row in nearby.iterrows():
                gene_start = int(gene_row["START"])
                gene_end = int(gene_row["STOP"])

                if snp_bp < gene_start:
                    dist = (gene_start - snp_bp) / 1000.0
                elif snp_bp > gene_end:
                    dist = (snp_bp - gene_end) / 1000.0
                else:
                    dist = 0.0  # within gene body

                mappings.append(GeneMapping(
                    snp=str(snp_row["SNP"]),
                    gene=str(gene_row["GENE"]),
                    chr=int(chr_val),
                    snp_bp=snp_bp,
                    gene_start=gene_start,
                    gene_end=gene_end,
                    distance_kb=dist,
                    positional=True,
                ))

    logger.info(
        "Positional mapping: %d SNP-gene pairs (%d unique genes)",
        len(mappings),
        len({m.gene for m in mappings}),
    )
    return mappings


# ---------------------------------------------------------------------------
# 2. eQTL mapping
# ---------------------------------------------------------------------------


def load_eqtl_data(
    eqtl_dir: Path | None = None,
    tissues: list[str] | None = None,
    fdr_threshold: float = EQTL_FDR_THRESHOLD,
) -> pd.DataFrame:
    """Load GTEx v8 significant eQTL pairs for brain tissues.

    Expects files named like:
        Brain_Caudate_basal_ganglia.v8.signif_variant_gene_pairs.txt.gz

    Standardized columns: SNP, GENE, TISSUE, PVAL, FDR, BETA, SE
    """
    if eqtl_dir is None:
        eqtl_dir = EQTL_DIR
    if tissues is None:
        tissues = GTEX_BRAIN_TISSUES

    all_eqtls = []
    for tissue in tissues:
        patterns = [
            f"{tissue}.v8.signif_variant_gene_pairs.txt.gz",
            f"{tissue}.v8.signif_variant_gene_pairs.txt",
            f"{tissue}.signif_pairs.tsv.gz",
            f"{tissue}.signif_pairs.tsv",
        ]

        eqtl_path = None
        for pat in patterns:
            candidate = eqtl_dir / pat
            if candidate.exists():
                eqtl_path = candidate
                break

        if eqtl_path is None:
            continue

        try:
            df = pd.read_csv(eqtl_path, sep="\t")
        except Exception as e:
            logger.warning("Failed to load eQTL file %s: %s", eqtl_path, e)
            continue

        col_map = {
            "variant_id": "SNP", "rs_id_dbSNP151_GRCh38p7": "RSID",
            "gene_id": "GENE_ID", "gene_name": "GENE",
            "pval_nominal": "PVAL", "pval_beta": "PVAL",
            "slope": "BETA", "slope_se": "SE",
            "qval": "FDR", "qvalue": "FDR",
        }
        df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
        df["TISSUE"] = tissue

        if "FDR" in df.columns:
            df = df[df["FDR"] <= fdr_threshold]

        all_eqtls.append(df)

    if not all_eqtls:
        logger.warning("No eQTL data found in %s", eqtl_dir)
        return pd.DataFrame()

    result = pd.concat(all_eqtls, ignore_index=True)
    logger.info("Loaded %d significant eQTL pairs across %d tissues",
                len(result), len(all_eqtls))
    return result


def eqtl_mapping(
    gwas_df: pd.DataFrame,
    eqtl_df: pd.DataFrame | None = None,
    eqtl_dir: Path | None = None,
) -> list[GeneMapping]:
    """Map GWAS SNPs to genes via brain eQTL associations.

    Parameters
    ----------
    gwas_df : pd.DataFrame
        GWAS with SNP column.
    eqtl_df : pd.DataFrame | None
        Pre-loaded eQTL data. If None, loaded from eqtl_dir.
    eqtl_dir : Path | None
        Directory containing eQTL files.

    Returns
    -------
    List of GeneMapping objects with eQTL evidence.
    """
    if eqtl_df is None:
        eqtl_df = load_eqtl_data(eqtl_dir)

    if eqtl_df.empty:
        logger.warning("No eQTL data available for mapping")
        return []

    # Match by SNP identifier
    gwas_snps = set(gwas_df["SNP"].astype(str))

    # Identify SNP column in eQTL data
    snp_col = "SNP" if "SNP" in eqtl_df.columns else "RSID"
    if snp_col not in eqtl_df.columns:
        logger.warning("eQTL data has no SNP/RSID column")
        return []

    matched = eqtl_df[eqtl_df[snp_col].astype(str).isin(gwas_snps)]

    if matched.empty:
        logger.info("No GWAS SNPs found in eQTL data")
        return []

    # Get gene column
    gene_col = "GENE" if "GENE" in matched.columns else "GENE_ID"

    # Group by SNP-gene pair, collect tissues
    mappings = []
    grouped = matched.groupby([snp_col, gene_col])

    for (snp, gene), group in grouped:
        tissues = group["TISSUE"].unique().tolist() if "TISSUE" in group.columns else []
        best_p = float(group["PVAL"].min()) if "PVAL" in group.columns else 1.0

        # Get positional info from GWAS if available
        gwas_match = gwas_df[gwas_df["SNP"] == snp]
        chr_val = int(gwas_match["CHR"].iloc[0]) if "CHR" in gwas_match.columns and len(gwas_match) > 0 else 0
        snp_bp = int(gwas_match["BP"].iloc[0]) if "BP" in gwas_match.columns and len(gwas_match) > 0 else 0

        mappings.append(GeneMapping(
            snp=str(snp),
            gene=str(gene),
            chr=chr_val,
            snp_bp=snp_bp,
            gene_start=0,
            gene_end=0,
            distance_kb=0.0,
            eqtl=True,
            eqtl_tissues=tissues,
            eqtl_best_p=best_p,
        ))

    logger.info(
        "eQTL mapping: %d SNP-gene pairs (%d unique genes, %d tissues)",
        len(mappings),
        len({m.gene for m in mappings}),
        len({t for m in mappings for t in m.eqtl_tissues}),
    )
    return mappings


# ---------------------------------------------------------------------------
# 3. Chromatin interaction (Hi-C) mapping
# ---------------------------------------------------------------------------


def load_hic_data(hic_dir: Path | None = None) -> pd.DataFrame:
    """Load brain-specific Hi-C chromatin interaction data.

    Expects TSV files with columns: CHR1, START1, END1, CHR2, START2, END2, TISSUE
    or BED-like format from PsychENCODE/ENCODE.
    """
    if hic_dir is None:
        hic_dir = HIC_DIR

    if not hic_dir.exists():
        logger.warning("Hi-C directory not found: %s", hic_dir)
        return pd.DataFrame()

    all_hic = []
    for f in sorted(hic_dir.glob("*.tsv*")):
        try:
            df = pd.read_csv(f, sep="\t")
            col_map = {
                "chr1": "CHR1", "start1": "START1", "end1": "END1",
                "chr2": "CHR2", "start2": "START2", "end2": "END2",
                "tissue": "TISSUE", "cell_type": "TISSUE",
            }
            df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
            all_hic.append(df)
        except Exception as e:
            logger.warning("Failed to load Hi-C file %s: %s", f, e)

    if not all_hic:
        return pd.DataFrame()

    result = pd.concat(all_hic, ignore_index=True)
    logger.info("Loaded %d Hi-C interactions", len(result))
    return result


def hic_mapping(
    gwas_df: pd.DataFrame,
    gene_annot: pd.DataFrame,
    hic_df: pd.DataFrame | None = None,
    hic_dir: Path | None = None,
) -> list[GeneMapping]:
    """Map SNPs to distal genes via chromatin interactions.

    Parameters
    ----------
    gwas_df : pd.DataFrame
        GWAS with SNP, CHR, BP columns.
    gene_annot : pd.DataFrame
        Gene annotations for identifying gene partners.
    hic_df : pd.DataFrame | None
        Pre-loaded Hi-C data.
    hic_dir : Path | None
        Directory containing Hi-C files.

    Returns
    -------
    List of GeneMapping objects with Hi-C evidence.
    """
    if hic_df is None:
        hic_df = load_hic_data(hic_dir)

    if hic_df.empty:
        logger.warning("No Hi-C data available for mapping")
        return []

    if not {"SNP", "CHR", "BP"}.issubset(gwas_df.columns):
        logger.warning("GWAS missing CHR/BP for Hi-C mapping")
        return []

    required_hic = {"CHR1", "START1", "END1", "CHR2", "START2", "END2"}
    if not required_hic.issubset(hic_df.columns):
        logger.warning("Hi-C data missing required columns: %s",
                       required_hic - set(hic_df.columns))
        return []

    mappings = []

    for _, snp_row in gwas_df.iterrows():
        snp_chr = int(snp_row["CHR"])
        snp_bp = int(snp_row["BP"])

        # Find Hi-C interactions where anchor 1 overlaps the SNP
        anchor1_match = hic_df[
            (hic_df["CHR1"] == snp_chr) &
            (hic_df["START1"] <= snp_bp) &
            (hic_df["END1"] >= snp_bp)
        ]

        # For each matched interaction, find genes at anchor 2
        for _, hic_row in anchor1_match.iterrows():
            chr2 = int(hic_row["CHR2"])
            start2 = int(hic_row["START2"])
            end2 = int(hic_row["END2"])

            # Find genes overlapping anchor 2
            genes_at_anchor = gene_annot[
                (gene_annot["CHR"] == chr2) &
                (gene_annot["START"] <= end2) &
                (gene_annot["STOP"] >= start2)
            ]

            tissues = [str(hic_row["TISSUE"])] if "TISSUE" in hic_row.index else []

            for _, gene_row in genes_at_anchor.iterrows():
                dist = abs(snp_bp - (int(gene_row["START"]) + int(gene_row["STOP"])) / 2) / 1000.0

                mappings.append(GeneMapping(
                    snp=str(snp_row["SNP"]),
                    gene=str(gene_row["GENE"]),
                    chr=snp_chr,
                    snp_bp=snp_bp,
                    gene_start=int(gene_row["START"]),
                    gene_end=int(gene_row["STOP"]),
                    distance_kb=dist,
                    hic=True,
                    hic_tissues=tissues,
                ))

    logger.info(
        "Hi-C mapping: %d SNP-gene pairs (%d unique genes)",
        len(mappings),
        len({m.gene for m in mappings}),
    )
    return mappings


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


def integrate_mappings(
    positional_maps: list[GeneMapping],
    eqtl_maps: list[GeneMapping],
    hic_maps: list[GeneMapping],
) -> pd.DataFrame:
    """Integrate multi-modal SNP-to-gene mappings.

    Merges evidence from positional, eQTL, and Hi-C mapping into a
    unified table with evidence counts per SNP-gene pair.

    Returns
    -------
    DataFrame with one row per unique SNP-gene pair, columns:
    SNP, GENE, CHR, BP, DISTANCE_KB, POSITIONAL, EQTL, EQTL_TISSUES,
    EQTL_BEST_P, HIC, HIC_TISSUES, N_EVIDENCE, IS_CANDIDATE
    """
    # Collect all mappings indexed by (SNP, GENE)
    merged: dict[tuple[str, str], dict] = {}

    for m in positional_maps:
        key = (m.snp, m.gene)
        if key not in merged:
            merged[key] = _init_row(m)
        merged[key]["POSITIONAL"] = True
        merged[key]["DISTANCE_KB"] = min(merged[key]["DISTANCE_KB"], m.distance_kb)

    for m in eqtl_maps:
        key = (m.snp, m.gene)
        if key not in merged:
            merged[key] = _init_row(m)
        merged[key]["EQTL"] = True
        merged[key]["EQTL_TISSUES"] = list(set(
            merged[key]["EQTL_TISSUES"] + m.eqtl_tissues
        ))
        merged[key]["EQTL_BEST_P"] = min(merged[key]["EQTL_BEST_P"], m.eqtl_best_p)

    for m in hic_maps:
        key = (m.snp, m.gene)
        if key not in merged:
            merged[key] = _init_row(m)
        merged[key]["HIC"] = True
        merged[key]["HIC_TISSUES"] = list(set(
            merged[key]["HIC_TISSUES"] + m.hic_tissues
        ))

    # Compute evidence count
    for row in merged.values():
        row["N_EVIDENCE"] = sum([row["POSITIONAL"], row["EQTL"], row["HIC"]])
        row["IS_CANDIDATE"] = row["GENE"] in TS_CANDIDATE_GENES
        # Flatten tissue lists to strings for TSV output
        row["EQTL_TISSUES"] = ";".join(row["EQTL_TISSUES"])
        row["HIC_TISSUES"] = ";".join(row["HIC_TISSUES"])

    if not merged:
        return pd.DataFrame()

    df = pd.DataFrame(merged.values())
    df = df.sort_values(["N_EVIDENCE", "EQTL_BEST_P"], ascending=[False, True])

    logger.info(
        "Integrated: %d SNP-gene pairs, %d unique genes, "
        "%d with multi-modal evidence, %d known candidates",
        len(df),
        df["GENE"].nunique(),
        (df["N_EVIDENCE"] >= 2).sum(),
        df["IS_CANDIDATE"].sum(),
    )

    return df.reset_index(drop=True)


def _init_row(m: GeneMapping) -> dict:
    """Initialize a merged mapping row from a GeneMapping."""
    return {
        "SNP": m.snp,
        "GENE": m.gene,
        "CHR": m.chr,
        "BP": m.snp_bp,
        "GENE_START": m.gene_start,
        "GENE_END": m.gene_end,
        "DISTANCE_KB": m.distance_kb,
        "POSITIONAL": False,
        "EQTL": False,
        "EQTL_TISSUES": [],
        "EQTL_BEST_P": 1.0,
        "HIC": False,
        "HIC_TISSUES": [],
        "N_EVIDENCE": 0,
        "IS_CANDIDATE": False,
    }


def run_snp_to_gene(
    gwas_path: Path,
    gene_annot_path: Path | None = None,
    eqtl_dir: Path | None = None,
    hic_dir: Path | None = None,
    output_dir: Path | None = None,
    window_kb: int = POSITIONAL_WINDOW_KB,
) -> pd.DataFrame:
    """Run the full multi-modal SNP-to-gene mapping pipeline.

    Parameters
    ----------
    gwas_path : Path
        Cleaned GWAS summary statistics file.
    gene_annot_path : Path | None
        Gene annotation file.
    eqtl_dir : Path | None
        GTEx eQTL data directory.
    hic_dir : Path | None
        Hi-C data directory.
    output_dir : Path | None
        Output directory.
    window_kb : int
        Positional mapping window in kb.

    Returns
    -------
    Integrated SNP-to-gene mapping DataFrame.
    """
    ensure_dirs()
    if output_dir is None:
        output_dir = MAPPING_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load GWAS
    gwas_df = pd.read_csv(gwas_path, sep="\t")
    logger.info("Loaded %d GWAS SNPs", len(gwas_df))

    # Load gene annotations
    gene_annot = load_gene_annotations(gene_annot_path)

    # 1. Positional mapping
    pos_maps = positional_mapping(gwas_df, gene_annot, window_kb)

    # 2. eQTL mapping
    eqtl_maps = eqtl_mapping(gwas_df, eqtl_dir=eqtl_dir)

    # 3. Hi-C mapping
    hic_maps = hic_mapping(gwas_df, gene_annot, hic_dir=hic_dir)

    # Integrate
    integrated = integrate_mappings(pos_maps, eqtl_maps, hic_maps)

    # Write outputs
    if not integrated.empty:
        out_path = output_dir / "snp_to_gene_integrated.tsv"
        integrated.to_csv(out_path, sep="\t", index=False, float_format="%.6g")
        logger.info("Wrote integrated mappings to %s", out_path)

        # Write per-modality summaries
        for modality, maps in [("positional", pos_maps), ("eqtl", eqtl_maps), ("hic", hic_maps)]:
            if maps:
                rows = [{
                    "SNP": m.snp, "GENE": m.gene, "CHR": m.chr,
                    "BP": m.snp_bp, "DISTANCE_KB": m.distance_kb,
                } for m in maps]
                mod_df = pd.DataFrame(rows)
                mod_path = output_dir / f"snp_to_gene_{modality}.tsv"
                mod_df.to_csv(mod_path, sep="\t", index=False, float_format="%.6g")

    return integrated


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-modal SNP-to-gene mapping for TS GWAS."
    )
    parser.add_argument(
        "--gwas", type=Path, required=True,
        help="Cleaned GWAS summary statistics file.",
    )
    parser.add_argument(
        "--gene-annot", type=Path, default=None,
        help="Gene annotation file (TSV: GENE, CHR, START, STOP).",
    )
    parser.add_argument(
        "--eqtl-dir", type=Path, default=None,
        help="GTEx eQTL data directory.",
    )
    parser.add_argument(
        "--hic-dir", type=Path, default=None,
        help="Hi-C chromatin interaction data directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--window-kb", type=int, default=POSITIONAL_WINDOW_KB,
        help=f"Positional mapping window in kb (default: {POSITIONAL_WINDOW_KB}).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.gwas.exists():
        logger.error("GWAS file not found: %s", args.gwas)
        sys.exit(1)

    run_snp_to_gene(
        gwas_path=args.gwas,
        gene_annot_path=args.gene_annot,
        eqtl_dir=args.eqtl_dir,
        hic_dir=args.hic_dir,
        output_dir=args.output_dir,
        window_kb=args.window_kb,
    )


if __name__ == "__main__":
    main()
