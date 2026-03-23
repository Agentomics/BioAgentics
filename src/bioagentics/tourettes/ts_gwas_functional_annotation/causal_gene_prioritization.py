"""Causal gene prioritization for TS GWAS loci.

Integrates all evidence layers (positional, eQTL, Hi-C, pathway, literature)
to score and rank candidate causal genes per GWAS locus. Each evidence type
contributes points to a total priority score per gene.

Scoring system:
  1. Positional mapping: within 10kb of gene = 1 point
  2. eQTL evidence: significant brain eQTL = 2 points per tissue (capped at 6)
  3. Chromatin interaction: Hi-C loop = 2 points
  4. Pathway convergence: gene in convergent pathway = 1 point per pathway (capped at 3)
  5. Prior literature: known TS/neuropsych gene = 1 point
  6. MAGMA gene-based hit flag (BCL11B, NDFIP2, RBM26)

Usage:
    python -m bioagentics.tourettes.ts_gwas_functional_annotation.causal_gene_prioritization \
        --snp-to-gene output/.../snp_to_gene/snp_to_gene_integrated.tsv \
        --gene-results output/.../magma_results/gene_results.tsv
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
    ENRICHMENT_DIR,
    MAGMA_DIR,
    MAPPING_DIR,
    OUTPUT_DIR,
    TS_CANDIDATE_GENES,
    TS_DE_NOVO_GENES,
    ensure_dirs,
)

logger = logging.getLogger(__name__)

# MAGMA gene-based hits from TS GWAS (flagged in task description)
MAGMA_GENE_HITS = ["BCL11B", "NDFIP2", "RBM26"]

# Scoring weights
SCORE_POSITIONAL = 1
SCORE_EQTL_PER_TISSUE = 2
SCORE_EQTL_CAP = 6
SCORE_HIC = 2
SCORE_PATHWAY_PER_HIT = 1
SCORE_PATHWAY_CAP = 3
SCORE_LITERATURE = 1

# Known TS / neuropsych genes (union of candidate + de novo lists)
KNOWN_TS_GENES = set(TS_CANDIDATE_GENES) | set(TS_DE_NOVO_GENES)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GeneEvidence:
    """Evidence profile for a single gene."""

    gene: str
    chr: int = 0
    locus_id: int = 0
    positional: bool = False
    distance_kb: float = float("inf")
    eqtl: bool = False
    eqtl_n_tissues: int = 0
    eqtl_tissues: list[str] = field(default_factory=list)
    eqtl_best_p: float = 1.0
    hic: bool = False
    hic_tissues: list[str] = field(default_factory=list)
    n_convergent_pathways: int = 0
    convergent_pathways: list[str] = field(default_factory=list)
    is_known_ts_gene: bool = False
    is_magma_hit: bool = False
    magma_gene_p: float = 1.0
    total_score: float = 0.0
    rank_in_locus: int = 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_gene_score(ev: GeneEvidence) -> float:
    """Compute the priority score for a gene based on its evidence profile."""
    score = 0.0

    # 1. Positional
    if ev.positional:
        score += SCORE_POSITIONAL

    # 2. eQTL (2 points per tissue, capped at 6)
    if ev.eqtl:
        score += min(ev.eqtl_n_tissues * SCORE_EQTL_PER_TISSUE, SCORE_EQTL_CAP)

    # 3. Hi-C
    if ev.hic:
        score += SCORE_HIC

    # 4. Pathway convergence (1 per pathway, capped at 3)
    score += min(ev.n_convergent_pathways * SCORE_PATHWAY_PER_HIT, SCORE_PATHWAY_CAP)

    # 5. Literature
    if ev.is_known_ts_gene:
        score += SCORE_LITERATURE

    return score


# ---------------------------------------------------------------------------
# Locus assignment
# ---------------------------------------------------------------------------


def assign_genes_to_loci(
    snp_gene_df: pd.DataFrame,
    gene_results_df: pd.DataFrame | None = None,
    merge_distance_kb: int = 1000,
) -> dict[int, list[str]]:
    """Assign genes to loci based on chromosomal proximity.

    Groups genes from the integrated SNP-to-gene mapping by chromosome and
    proximity, returning a dict of locus_id -> gene list.

    Parameters
    ----------
    snp_gene_df : pd.DataFrame
        Integrated SNP-to-gene mapping (from snp_to_gene module).
    gene_results_df : pd.DataFrame | None
        Optional gene-level results with P-values for ordering.
    merge_distance_kb : int
        Maximum distance (kb) to merge genes into one locus.
    """
    if snp_gene_df.empty:
        return {}

    # Get unique genes with their positions
    gene_col = "GENE"
    chr_col = "CHR"

    if gene_col not in snp_gene_df.columns or chr_col not in snp_gene_df.columns:
        logger.warning("SNP-gene mapping missing GENE/CHR columns")
        return {}

    # Use gene midpoint for positioning
    genes = snp_gene_df.drop_duplicates(subset=[gene_col]).copy()

    # Compute gene midpoint from GENE_START/GENE_END if available, else from BP
    if "GENE_START" in genes.columns and "GENE_END" in genes.columns:
        genes["_midpoint"] = (genes["GENE_START"] + genes["GENE_END"]) / 2
    elif "BP" in genes.columns:
        genes["_midpoint"] = genes["BP"]
    else:
        logger.warning("No positional columns for locus assignment")
        return {}

    genes = genes.sort_values([chr_col, "_midpoint"]).reset_index(drop=True)
    merge_dist = merge_distance_kb * 1000

    loci: dict[int, list[str]] = {}
    locus_id = 0

    for _, chr_df in genes.groupby(chr_col):
        chr_df = chr_df.sort_values("_midpoint").reset_index(drop=True)
        current_genes: list[str] = []
        current_end: float = 0

        for _, row in chr_df.iterrows():
            midpoint = row["_midpoint"]
            if not current_genes:
                current_genes = [row[gene_col]]
                current_end = midpoint
            elif midpoint - current_end <= merge_dist:
                current_genes.append(row[gene_col])
                current_end = midpoint
            else:
                locus_id += 1
                loci[locus_id] = list(current_genes)
                current_genes = [row[gene_col]]
                current_end = midpoint

        if current_genes:
            locus_id += 1
            loci[locus_id] = list(current_genes)

    logger.info("Assigned %d genes to %d loci",
                sum(len(g) for g in loci.values()), len(loci))
    return loci


# ---------------------------------------------------------------------------
# Evidence integration
# ---------------------------------------------------------------------------


def build_gene_evidence(
    snp_gene_df: pd.DataFrame,
    gene_results_df: pd.DataFrame | None = None,
    convergence_df: pd.DataFrame | None = None,
    loci: dict[int, list[str]] | None = None,
) -> list[GeneEvidence]:
    """Build evidence profiles for all genes in the integrated mapping.

    Parameters
    ----------
    snp_gene_df : pd.DataFrame
        Integrated SNP-to-gene mapping with columns:
        GENE, CHR, POSITIONAL, EQTL, EQTL_TISSUES, EQTL_BEST_P, HIC, HIC_TISSUES, etc.
    gene_results_df : pd.DataFrame | None
        Gene-level results from MAGMA analysis (GENE, P columns).
    convergence_df : pd.DataFrame | None
        Pathway convergence results (PATHWAY, CONVERGENCE_FDR, etc.).
    loci : dict[int, list[str]] | None
        Pre-computed locus assignments. If None, computed from snp_gene_df.
    """
    if snp_gene_df.empty:
        return []

    # Compute loci if needed
    if loci is None:
        loci = assign_genes_to_loci(snp_gene_df, gene_results_df)

    # Build gene->locus lookup
    gene_to_locus: dict[str, int] = {}
    for lid, genes in loci.items():
        for g in genes:
            gene_to_locus[g] = lid

    # Build convergent pathway lookup: gene -> list of pathway names
    gene_to_pathways: dict[str, list[str]] = {}
    if convergence_df is not None and not convergence_df.empty:
        sig_pathways = _extract_convergent_pathways(convergence_df)
        gene_to_pathways = sig_pathways

    # Build gene-level P-value lookup
    gene_p: dict[str, float] = {}
    if gene_results_df is not None and not gene_results_df.empty:
        if "GENE" in gene_results_df.columns and "P" in gene_results_df.columns:
            for _, row in gene_results_df.iterrows():
                gene_p[str(row["GENE"])] = float(row["P"])

    # Aggregate evidence per gene (a gene may map via multiple SNPs)
    gene_evidence: dict[str, GeneEvidence] = {}

    for _, row in snp_gene_df.iterrows():
        gene = str(row.get("GENE", ""))
        if not gene:
            continue

        if gene not in gene_evidence:
            gene_evidence[gene] = GeneEvidence(
                gene=gene,
                chr=int(row.get("CHR", 0)),
                locus_id=gene_to_locus.get(gene, 0),
                is_known_ts_gene=gene in KNOWN_TS_GENES,
                is_magma_hit=gene in MAGMA_GENE_HITS,
                magma_gene_p=gene_p.get(gene, 1.0),
            )

        ev = gene_evidence[gene]

        # Positional
        if row.get("POSITIONAL", False):
            ev.positional = True
            dist = float(row.get("DISTANCE_KB", float("inf")))
            ev.distance_kb = min(ev.distance_kb, dist)

        # eQTL
        if row.get("EQTL", False):
            ev.eqtl = True
            tissues_str = str(row.get("EQTL_TISSUES", ""))
            if tissues_str and tissues_str != "nan":
                new_tissues = [t.strip() for t in tissues_str.split(";") if t.strip()]
                ev.eqtl_tissues = list(set(ev.eqtl_tissues + new_tissues))
            best_p = row.get("EQTL_BEST_P", 1.0)
            if pd.notna(best_p):
                ev.eqtl_best_p = min(ev.eqtl_best_p, float(best_p))

        # Hi-C
        if row.get("HIC", False):
            ev.hic = True
            hic_str = str(row.get("HIC_TISSUES", ""))
            if hic_str and hic_str != "nan":
                new_hic = [t.strip() for t in hic_str.split(";") if t.strip()]
                ev.hic_tissues = list(set(ev.hic_tissues + new_hic))

    # Update eQTL tissue counts and pathway info
    for gene, ev in gene_evidence.items():
        ev.eqtl_n_tissues = len(ev.eqtl_tissues)
        ev.convergent_pathways = gene_to_pathways.get(gene, [])
        ev.n_convergent_pathways = len(ev.convergent_pathways)
        ev.total_score = compute_gene_score(ev)

    result = list(gene_evidence.values())
    logger.info("Built evidence for %d genes", len(result))
    return result


def _extract_convergent_pathways(
    convergence_df: pd.DataFrame,
    fdr_threshold: float = 0.05,
) -> dict[str, list[str]]:
    """Extract gene->pathway mappings from convergence results.

    Reads the pathway convergence results and returns a dict of
    gene -> list of convergent pathway names that gene belongs to.

    The convergence_df is expected to have a column indicating which
    genes contributed to each pathway's convergence. We use a pragmatic
    approach: pathways passing FDR threshold with genes listed in
    a GENES or TOP_GENES column.
    """
    gene_to_pathways: dict[str, list[str]] = {}

    if "CONVERGENCE_FDR" not in convergence_df.columns:
        return gene_to_pathways

    sig = convergence_df[convergence_df["CONVERGENCE_FDR"] < fdr_threshold]

    # Look for gene columns
    gene_col = None
    for col in ["GENES", "TOP_GENES", "PATHWAY_GENES"]:
        if col in sig.columns:
            gene_col = col
            break

    if gene_col is None:
        # If no gene column, just return pathway names mapped to empty
        return gene_to_pathways

    for _, row in sig.iterrows():
        pathway = str(row.get("PATHWAY", ""))
        genes_str = str(row.get(gene_col, ""))
        if genes_str and genes_str != "nan":
            genes = [g.strip() for g in genes_str.split(";") if g.strip()]
            for g in genes:
                gene_to_pathways.setdefault(g, []).append(pathway)

    return gene_to_pathways


# ---------------------------------------------------------------------------
# Ranking and output
# ---------------------------------------------------------------------------


def rank_genes_per_locus(
    evidence: list[GeneEvidence],
) -> list[GeneEvidence]:
    """Rank genes within each locus by total score, breaking ties by MAGMA P.

    Modifies rank_in_locus on each GeneEvidence and returns the full list
    sorted by locus then rank.
    """
    from collections import defaultdict

    locus_groups: dict[int, list[GeneEvidence]] = defaultdict(list)
    for ev in evidence:
        locus_groups[ev.locus_id].append(ev)

    ranked: list[GeneEvidence] = []
    for lid in sorted(locus_groups.keys()):
        genes = locus_groups[lid]
        # Sort by score descending, then MAGMA P ascending
        genes.sort(key=lambda e: (-e.total_score, e.magma_gene_p))
        for i, ev in enumerate(genes):
            ev.rank_in_locus = i + 1
        ranked.extend(genes)

    return ranked


def evidence_to_dataframe(evidence: list[GeneEvidence]) -> pd.DataFrame:
    """Convert gene evidence list to a DataFrame for output."""
    rows = []
    for ev in evidence:
        rows.append({
            "GENE": ev.gene,
            "CHR": ev.chr,
            "LOCUS_ID": ev.locus_id,
            "RANK_IN_LOCUS": ev.rank_in_locus,
            "TOTAL_SCORE": ev.total_score,
            "POSITIONAL": ev.positional,
            "DISTANCE_KB": ev.distance_kb if ev.distance_kb != float("inf") else np.nan,
            "EQTL": ev.eqtl,
            "EQTL_N_TISSUES": ev.eqtl_n_tissues,
            "EQTL_TISSUES": ";".join(ev.eqtl_tissues),
            "EQTL_BEST_P": ev.eqtl_best_p if ev.eqtl_best_p < 1.0 else np.nan,
            "HIC": ev.hic,
            "HIC_TISSUES": ";".join(ev.hic_tissues),
            "N_CONVERGENT_PATHWAYS": ev.n_convergent_pathways,
            "CONVERGENT_PATHWAYS": ";".join(ev.convergent_pathways),
            "IS_KNOWN_TS_GENE": ev.is_known_ts_gene,
            "IS_MAGMA_HIT": ev.is_magma_hit,
            "MAGMA_GENE_P": ev.magma_gene_p if ev.magma_gene_p < 1.0 else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    return df


def write_prioritization_results(
    evidence: list[GeneEvidence],
    output_path: Path,
) -> Path:
    """Write ranked gene prioritization results to TSV."""
    df = evidence_to_dataframe(evidence)
    if df.empty:
        logger.warning("No gene evidence to write")
        return output_path

    df.to_csv(output_path, sep="\t", index=False, float_format="%.6g")
    logger.info("Wrote %d gene prioritization results to %s", len(df), output_path)
    return output_path


def write_prioritization_summary(
    evidence: list[GeneEvidence],
    output_dir: Path,
) -> Path:
    """Write a human-readable prioritization summary."""
    lines = ["# Causal Gene Prioritization Summary\n"]

    n_loci = len({ev.locus_id for ev in evidence})
    lines.append(f"- Total genes scored: {len(evidence)}")
    lines.append(f"- Loci: {n_loci}")

    # Top genes overall
    top = sorted(evidence, key=lambda e: (-e.total_score, e.magma_gene_p))[:20]

    lines.append(f"\n## Top 20 Prioritized Genes\n")
    lines.append(
        "| Rank | Gene | Locus | Score | Pos | eQTL(t) | Hi-C | Pathways | Lit | MAGMA |"
    )
    lines.append(
        "|------|------|-------|-------|-----|---------|------|----------|-----|-------|"
    )
    for i, ev in enumerate(top, 1):
        magma_flag = "***" if ev.is_magma_hit else ""
        lit_flag = "Y" if ev.is_known_ts_gene else ""
        lines.append(
            f"| {i} | {ev.gene}{magma_flag} | {ev.locus_id} | "
            f"{ev.total_score:.0f} | "
            f"{'Y' if ev.positional else ''} | "
            f"{'Y(' + str(ev.eqtl_n_tissues) + ')' if ev.eqtl else ''} | "
            f"{'Y' if ev.hic else ''} | "
            f"{ev.n_convergent_pathways} | "
            f"{lit_flag} | "
            f"{ev.magma_gene_p:.2e}{magma_flag} |"
        )

    # MAGMA hits
    magma_hits = [ev for ev in evidence if ev.is_magma_hit]
    if magma_hits:
        lines.append(f"\n## MAGMA Gene-Based Hits\n")
        for ev in magma_hits:
            lines.append(
                f"- **{ev.gene}** (locus {ev.locus_id}): "
                f"score={ev.total_score:.0f}, P={ev.magma_gene_p:.2e}"
            )

    # Evidence distribution
    lines.append(f"\n## Evidence Distribution\n")
    n_pos = sum(1 for ev in evidence if ev.positional)
    n_eqtl = sum(1 for ev in evidence if ev.eqtl)
    n_hic = sum(1 for ev in evidence if ev.hic)
    n_pw = sum(1 for ev in evidence if ev.n_convergent_pathways > 0)
    n_lit = sum(1 for ev in evidence if ev.is_known_ts_gene)
    lines.append(f"- Positional: {n_pos}/{len(evidence)}")
    lines.append(f"- eQTL: {n_eqtl}/{len(evidence)}")
    lines.append(f"- Hi-C: {n_hic}/{len(evidence)}")
    lines.append(f"- Pathway convergence: {n_pw}/{len(evidence)}")
    lines.append(f"- Literature: {n_lit}/{len(evidence)}")

    path = output_dir / "causal_gene_prioritization_summary.md"
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote prioritization summary to %s", path)
    return path


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_causal_gene_prioritization(
    snp_gene_path: Path,
    gene_results_path: Path | None = None,
    convergence_path: Path | None = None,
    output_dir: Path | None = None,
    merge_distance_kb: int = 1000,
) -> list[GeneEvidence]:
    """Run the full causal gene prioritization pipeline.

    Parameters
    ----------
    snp_gene_path : Path
        Integrated SNP-to-gene mapping TSV.
    gene_results_path : Path | None
        Gene-level MAGMA results TSV.
    convergence_path : Path | None
        Pathway convergence results TSV.
    output_dir : Path | None
        Output directory. Defaults to OUTPUT_DIR.
    merge_distance_kb : int
        Distance for merging genes into loci.
    """
    ensure_dirs()
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SNP-to-gene mapping
    snp_gene_df = pd.read_csv(snp_gene_path, sep="\t")
    logger.info("Loaded %d SNP-gene mappings", len(snp_gene_df))

    # Load gene results
    gene_results_df = None
    if gene_results_path is None:
        default_path = MAGMA_DIR / "gene_results.tsv"
        if default_path.exists():
            gene_results_path = default_path
    if gene_results_path is not None and gene_results_path.exists():
        gene_results_df = pd.read_csv(gene_results_path, sep="\t")
        logger.info("Loaded %d gene results", len(gene_results_df))

    # Load convergence results
    convergence_df = None
    if convergence_path is None:
        default_path = ENRICHMENT_DIR / "pathway_convergence.tsv"
        if default_path.exists():
            convergence_path = default_path
    if convergence_path is not None and convergence_path.exists():
        convergence_df = pd.read_csv(convergence_path, sep="\t")
        logger.info("Loaded %d convergence results", len(convergence_df))

    # Assign genes to loci
    loci = assign_genes_to_loci(snp_gene_df, gene_results_df, merge_distance_kb)

    # Build evidence
    evidence = build_gene_evidence(snp_gene_df, gene_results_df, convergence_df, loci)

    # Rank
    evidence = rank_genes_per_locus(evidence)

    # Write outputs
    out_path = output_dir / "causal_gene_prioritization.tsv"
    write_prioritization_results(evidence, out_path)
    write_prioritization_summary(evidence, output_dir)

    # Log top results
    top5 = sorted(evidence, key=lambda e: (-e.total_score, e.magma_gene_p))[:5]
    for ev in top5:
        logger.info(
            "Top gene: %s (locus %d, score %.0f, MAGMA P=%.2e%s)",
            ev.gene, ev.locus_id, ev.total_score, ev.magma_gene_p,
            " [MAGMA HIT]" if ev.is_magma_hit else "",
        )

    return evidence


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Causal gene prioritization for TS GWAS loci.",
    )
    parser.add_argument(
        "--snp-to-gene", type=Path, required=True,
        help="Integrated SNP-to-gene mapping TSV.",
    )
    parser.add_argument(
        "--gene-results", type=Path, default=None,
        help="Gene-level MAGMA results TSV.",
    )
    parser.add_argument(
        "--convergence", type=Path, default=None,
        help="Pathway convergence results TSV.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--merge-distance-kb", type=int, default=1000,
        help="Merge distance in kb for loci (default: 1000).",
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

    if not args.snp_to_gene.exists():
        logger.error("SNP-to-gene file not found: %s", args.snp_to_gene)
        sys.exit(1)

    run_causal_gene_prioritization(
        snp_gene_path=args.snp_to_gene,
        gene_results_path=args.gene_results,
        convergence_path=args.convergence,
        output_dir=args.output_dir,
        merge_distance_kb=args.merge_distance_kb,
    )


if __name__ == "__main__":
    main()
