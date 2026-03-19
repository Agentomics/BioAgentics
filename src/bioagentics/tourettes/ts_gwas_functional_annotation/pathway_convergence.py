"""Pathway convergence analysis for TS GWAS.

Identifies biological pathways enriched across multiple independent GWAS loci,
testing whether convergence exceeds chance. Focuses on TS-relevant domains:
dopaminergic signaling, synaptic adhesion, axon guidance, and neurodevelopment.

Steps:
  1. Define independent loci from gene-level results (proximity-based merging)
  2. Map pathways to loci via their constituent genes
  3. Test convergence per pathway (hypergeometric: more loci than expected)
  4. Group pathways into biological domains and score domain-level convergence
  5. FDR correction and output

Usage:
    python -m bioagentics.tourettes.ts_gwas_functional_annotation.pathway_convergence \
        --gene-results output/tourettes/ts-gwas-functional-annotation/magma_results/gene_results.tsv \
        --gene-set-results output/tourettes/ts-gwas-functional-annotation/magma_results/gene_set_results.tsv
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.tourettes.ts_gwas_functional_annotation.config import (
    ENRICHMENT_DIR,
    MAGMA_DIR,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Biological domain definitions
# ---------------------------------------------------------------------------

BIOLOGICAL_DOMAINS: dict[str, list[str]] = {
    "dopaminergic_signaling": [
        "dopamine", "drd1", "drd2", "catecholamine", "monoamine",
        "tyrosine_hydroxylase", "reward",
    ],
    "synaptic_adhesion": [
        "synap", "adhesion", "neurexin", "neuroligin", "nrxn", "nlgn",
        "contactin", "slitrk", "cell_junction",
    ],
    "axon_guidance": [
        "axon", "guidance", "semaphorin", "sema", "netrin", "ntn",
        "robo", "slit", "ephrin", "plexin",
    ],
    "neurodevelopment": [
        "neurodevelop", "neurogenesis", "neuron_differentiation",
        "brain_development", "cortical", "forebrain", "telencephal",
    ],
    "gaba_glutamate": [
        "gaba", "glutamate", "glutamater", "gabaerg", "inhibitory_synap",
        "excitatory_synap",
    ],
    "ion_channel": [
        "ion_channel", "potassium", "calcium_channel", "sodium_channel",
        "voltage_gated",
    ],
    "immune_neuroinflammation": [
        "immune", "inflammat", "microgli", "complement", "cytokine",
        "interleukin",
    ],
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Locus:
    """An independent GWAS locus defined by grouped genes."""

    locus_id: int
    chr: int
    start: int
    stop: int
    genes: list[str]
    lead_gene: str
    lead_gene_p: float


@dataclass
class PathwayConvergenceResult:
    """Convergence result for a single pathway across loci."""

    pathway: str
    source: str
    n_loci_total: int
    n_loci_hit: int
    loci_hit: list[int]
    genes_per_locus: dict[int, list[str]]
    n_pathway_genes_in_data: int
    convergence_p: float
    convergence_fdr: float = 1.0
    domain: str = ""
    enrichment_p: float = 1.0
    enrichment_fdr: float = 1.0


@dataclass
class DomainConvergenceResult:
    """Domain-level convergence summary."""

    domain: str
    n_pathways: int
    n_pathways_convergent: int
    median_loci_hit: float
    top_pathways: list[str]
    fisher_p: float
    fisher_fdr: float = 1.0


# ---------------------------------------------------------------------------
# Locus definition
# ---------------------------------------------------------------------------


def define_loci(
    gene_results_df: pd.DataFrame,
    p_threshold: float = 0.05,
    merge_distance_kb: int = 1000,
) -> list[Locus]:
    """Define independent loci by grouping significant genes by proximity.

    Genes with P < p_threshold are sorted by position and merged into loci
    if within merge_distance_kb of each other on the same chromosome.

    Parameters
    ----------
    gene_results_df : pd.DataFrame
        Gene-level results with columns: GENE, CHR, START, STOP, P.
    p_threshold : float
        P-value threshold for including genes (default: 0.05).
    merge_distance_kb : int
        Maximum distance (kb) between genes to merge into one locus.
    """
    required = {"GENE", "CHR", "START", "STOP", "P"}
    if not required.issubset(gene_results_df.columns):
        missing = required - set(gene_results_df.columns)
        logger.warning("Missing columns for locus definition: %s", missing)
        return []

    sig = gene_results_df[gene_results_df["P"] < p_threshold].copy()
    if sig.empty:
        logger.info("No genes at P < %g — no loci defined", p_threshold)
        return []

    sig = sig.sort_values(["CHR", "START"]).reset_index(drop=True)
    merge_dist = merge_distance_kb * 1000

    loci: list[Locus] = []
    locus_id = 0

    for chr_val, chr_df in sig.groupby("CHR"):
        chr_df = chr_df.sort_values("START").reset_index(drop=True)
        current_genes: list[str] = []
        current_start = 0
        current_stop = 0

        for _, row in chr_df.iterrows():
            if not current_genes:
                current_genes = [row["GENE"]]
                current_start = int(row["START"])
                current_stop = int(row["STOP"])
            elif int(row["START"]) - current_stop <= merge_dist:
                current_genes.append(row["GENE"])
                current_stop = max(current_stop, int(row["STOP"]))
            else:
                locus_id += 1
                loci.append(_make_locus(
                    locus_id, int(chr_val), current_start, current_stop,
                    current_genes, chr_df,
                ))
                current_genes = [row["GENE"]]
                current_start = int(row["START"])
                current_stop = int(row["STOP"])

        if current_genes:
            locus_id += 1
            loci.append(_make_locus(
                locus_id, int(chr_val), current_start, current_stop,
                current_genes, chr_df,
            ))

    logger.info("Defined %d independent loci from %d significant genes",
                len(loci), len(sig))
    return loci


def _make_locus(
    locus_id: int,
    chr_val: int,
    start: int,
    stop: int,
    genes: list[str],
    chr_df: pd.DataFrame,
) -> Locus:
    """Create a Locus from a group of genes."""
    gene_rows = chr_df[chr_df["GENE"].isin(genes)]
    lead_idx = gene_rows["P"].idxmin()
    lead_gene = str(gene_rows.loc[lead_idx, "GENE"])
    lead_p = float(gene_rows.loc[lead_idx, "P"])
    return Locus(
        locus_id=locus_id,
        chr=chr_val,
        start=start,
        stop=stop,
        genes=list(genes),
        lead_gene=lead_gene,
        lead_gene_p=lead_p,
    )


# ---------------------------------------------------------------------------
# Pathway-locus mapping and convergence testing
# ---------------------------------------------------------------------------


def _classify_domain(pathway_name: str) -> str:
    """Assign a pathway to a biological domain based on keyword matching."""
    name_lower = pathway_name.lower().replace("-", "_").replace(" ", "_")
    for domain, keywords in BIOLOGICAL_DOMAINS.items():
        for kw in keywords:
            if kw in name_lower:
                return domain
    return "other"


def pathway_convergence_analysis(
    loci: list[Locus],
    gene_set_results_df: pd.DataFrame,
    gene_sets: dict[str, tuple[str, list[str]]],
    enrichment_p_threshold: float = 0.05,
) -> list[PathwayConvergenceResult]:
    """Test pathway convergence across independent loci.

    For each enriched pathway, count how many loci contain at least one
    gene from that pathway. Tests significance via hypergeometric test.

    Parameters
    ----------
    loci : list[Locus]
        Independent loci from define_loci.
    gene_set_results_df : pd.DataFrame
        MAGMA gene-set results (GENE_SET, SOURCE, P, FDR_Q, TOP_GENES).
    gene_sets : dict
        Full gene-set definitions: name -> (source, gene_list).
    enrichment_p_threshold : float
        Only test convergence for pathways with enrichment P below this.
    """
    if not loci:
        logger.warning("No loci provided — skipping convergence analysis")
        return []

    n_loci = len(loci)
    all_locus_genes = set()
    for loc in loci:
        all_locus_genes.update(loc.genes)

    # Build locus index: gene -> set of locus IDs
    gene_to_loci: dict[str, set[int]] = {}
    for loc in loci:
        for g in loc.genes:
            gene_to_loci.setdefault(g, set()).add(loc.locus_id)

    # Filter to enriched pathways
    if gene_set_results_df.empty:
        logger.info("No gene-set results — using all gene sets")
        enriched_names = set(gene_sets.keys())
    else:
        required_cols = {"GENE_SET", "P"}
        if not required_cols.issubset(gene_set_results_df.columns):
            logger.warning("Gene-set results missing columns: %s",
                           required_cols - set(gene_set_results_df.columns))
            enriched_names = set(gene_sets.keys())
        else:
            mask = gene_set_results_df["P"] < enrichment_p_threshold
            enriched_names = set(gene_set_results_df.loc[mask, "GENE_SET"])

    results: list[PathwayConvergenceResult] = []

    for gs_name, (source, gs_genes) in gene_sets.items():
        if gs_name not in enriched_names:
            continue

        gs_gene_set = set(gs_genes)
        genes_in_data = gs_gene_set & all_locus_genes

        if len(genes_in_data) < 2:
            continue

        # Which loci are hit by this pathway?
        loci_hit: dict[int, list[str]] = {}
        for g in genes_in_data:
            for lid in gene_to_loci.get(g, set()):
                loci_hit.setdefault(lid, []).append(g)

        n_hit = len(loci_hit)
        if n_hit < 1:
            continue

        # Hypergeometric test: P(X >= n_hit)
        # Population: n_loci loci, K loci that could be hit by pathway genes
        # We use a simple model: each locus has some genes, pathway has some genes
        # P(locus hit) = 1 - P(none of locus's genes in pathway)
        # For the hypergeometric: total genes in loci, pathway genes in loci,
        # we test if n_hit loci is more than expected
        # Approximation: under null, P(locus hit) = n_pathway_genes_in_data / n_all_locus_genes
        n_all = len(all_locus_genes)
        n_pw = len(genes_in_data)
        if n_all > 0 and n_pw > 0:
            # Expected fraction of loci hit under random gene placement
            # For each locus, P(at least one gene in pathway) ≈ 1-(1-n_pw/n_all)^n_genes_in_locus
            # Use binomial test as a tractable approximation
            avg_genes_per_locus = n_all / n_loci if n_loci > 0 else 1
            p_hit = 1 - (1 - n_pw / n_all) ** avg_genes_per_locus
            p_hit = min(max(p_hit, 1e-10), 1 - 1e-10)
            conv_p = float(stats.binom.sf(n_hit - 1, n_loci, p_hit))
        else:
            conv_p = 1.0

        # Get enrichment P from gene-set results
        enrich_p = 1.0
        enrich_fdr = 1.0
        if not gene_set_results_df.empty and "GENE_SET" in gene_set_results_df.columns:
            match = gene_set_results_df[gene_set_results_df["GENE_SET"] == gs_name]
            if not match.empty:
                enrich_p = float(match.iloc[0].get("P", 1.0))
                enrich_fdr = float(match.iloc[0].get("FDR_Q", 1.0))

        results.append(PathwayConvergenceResult(
            pathway=gs_name,
            source=source,
            n_loci_total=n_loci,
            n_loci_hit=n_hit,
            loci_hit=sorted(loci_hit.keys()),
            genes_per_locus=loci_hit,
            n_pathway_genes_in_data=n_pw,
            convergence_p=conv_p,
            domain=_classify_domain(gs_name),
            enrichment_p=enrich_p,
            enrichment_fdr=enrich_fdr,
        ))

    # FDR correction (Benjamini-Hochberg)
    results.sort(key=lambda r: r.convergence_p)
    n_tests = len(results)
    for i, r in enumerate(results):
        r.convergence_fdr = min(r.convergence_p * n_tests / (i + 1), 1.0)

    # Ensure monotonicity
    for i in range(n_tests - 2, -1, -1):
        results[i].convergence_fdr = min(
            results[i].convergence_fdr, results[i + 1].convergence_fdr,
        )

    n_sig = sum(1 for r in results if r.convergence_fdr < 0.05)
    logger.info(
        "Pathway convergence: %d pathways tested, %d convergent (FDR < 0.05)",
        n_tests, n_sig,
    )
    return results


# ---------------------------------------------------------------------------
# Domain-level convergence
# ---------------------------------------------------------------------------


def domain_convergence(
    pathway_results: list[PathwayConvergenceResult],
    convergence_fdr_threshold: float = 0.05,
) -> list[DomainConvergenceResult]:
    """Summarize convergence at the biological domain level.

    Uses Fisher's method to combine convergence P-values within each domain.
    """
    if not pathway_results:
        return []

    domain_pathways: dict[str, list[PathwayConvergenceResult]] = {}
    for r in pathway_results:
        domain_pathways.setdefault(r.domain, []).append(r)

    results: list[DomainConvergenceResult] = []
    for domain, pw_list in domain_pathways.items():
        n_pw = len(pw_list)
        n_conv = sum(1 for r in pw_list if r.convergence_fdr < convergence_fdr_threshold)
        loci_hits = [r.n_loci_hit for r in pw_list]
        median_hit = float(np.median(loci_hits)) if loci_hits else 0.0

        # Fisher's combined P-value
        p_values = [r.convergence_p for r in pw_list]
        p_values = [max(p, 1e-300) for p in p_values]  # avoid log(0)
        chi2_stat = -2 * sum(np.log(p) for p in p_values)
        fisher_p = float(stats.chi2.sf(chi2_stat, 2 * n_pw))

        # Top pathways by convergence
        sorted_pw = sorted(pw_list, key=lambda r: r.convergence_p)
        top = [r.pathway for r in sorted_pw[:5]]

        results.append(DomainConvergenceResult(
            domain=domain,
            n_pathways=n_pw,
            n_pathways_convergent=n_conv,
            median_loci_hit=median_hit,
            top_pathways=top,
            fisher_p=fisher_p,
        ))

    # FDR on domain P-values
    results.sort(key=lambda r: r.fisher_p)
    n_tests = len(results)
    for i, r in enumerate(results):
        r.fisher_fdr = min(r.fisher_p * n_tests / (i + 1), 1.0)
    for i in range(n_tests - 2, -1, -1):
        results[i].fisher_fdr = min(results[i].fisher_fdr, results[i + 1].fisher_fdr)

    return results


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_gene_results(path: Path) -> pd.DataFrame:
    """Load gene-level results TSV."""
    df = pd.read_csv(path, sep="\t")
    logger.info("Loaded %d gene results from %s", len(df), path)
    return df


def load_gene_set_results(path: Path) -> pd.DataFrame:
    """Load gene-set enrichment results TSV."""
    df = pd.read_csv(path, sep="\t")
    logger.info("Loaded %d gene-set results from %s", len(df), path)
    return df


def write_convergence_results(
    results: list[PathwayConvergenceResult],
    output_dir: Path,
) -> Path:
    """Write pathway convergence results to TSV."""
    rows = [{
        "PATHWAY": r.pathway,
        "SOURCE": r.source,
        "DOMAIN": r.domain,
        "N_LOCI_TOTAL": r.n_loci_total,
        "N_LOCI_HIT": r.n_loci_hit,
        "LOCI_HIT": ";".join(str(x) for x in r.loci_hit),
        "N_PATHWAY_GENES": r.n_pathway_genes_in_data,
        "CONVERGENCE_P": r.convergence_p,
        "CONVERGENCE_FDR": r.convergence_fdr,
        "ENRICHMENT_P": r.enrichment_p,
        "ENRICHMENT_FDR": r.enrichment_fdr,
    } for r in results]

    df = pd.DataFrame(rows)
    path = output_dir / "pathway_convergence.tsv"
    df.to_csv(path, sep="\t", index=False, float_format="%.6g")
    logger.info("Wrote %d convergence results to %s", len(rows), path)
    return path


def write_domain_results(
    results: list[DomainConvergenceResult],
    output_dir: Path,
) -> Path:
    """Write domain convergence summary to TSV."""
    rows = [{
        "DOMAIN": r.domain,
        "N_PATHWAYS": r.n_pathways,
        "N_CONVERGENT": r.n_pathways_convergent,
        "MEDIAN_LOCI_HIT": r.median_loci_hit,
        "FISHER_P": r.fisher_p,
        "FISHER_FDR": r.fisher_fdr,
        "TOP_PATHWAYS": ";".join(r.top_pathways),
    } for r in results]

    df = pd.DataFrame(rows)
    path = output_dir / "domain_convergence.tsv"
    df.to_csv(path, sep="\t", index=False, float_format="%.6g")
    logger.info("Wrote %d domain results to %s", len(rows), path)
    return path


def write_convergence_summary(
    pathway_results: list[PathwayConvergenceResult],
    domain_results: list[DomainConvergenceResult],
    loci: list[Locus],
    output_dir: Path,
) -> Path:
    """Write a human-readable convergence summary."""
    lines = ["# Pathway Convergence Summary\n"]

    lines.append(f"## Loci\n")
    lines.append(f"- Independent loci: {len(loci)}")
    total_genes = sum(len(loc.genes) for loc in loci)
    lines.append(f"- Total genes across loci: {total_genes}\n")

    if loci:
        lines.append("| Locus | CHR | Region | N Genes | Lead Gene | Lead P |")
        lines.append("|-------|-----|--------|---------|-----------|--------|")
        for loc in sorted(loci, key=lambda x: x.lead_gene_p):
            lines.append(
                f"| {loc.locus_id} | {loc.chr} | "
                f"{loc.start:,}-{loc.stop:,} | {len(loc.genes)} | "
                f"{loc.lead_gene} | {loc.lead_gene_p:.2e} |"
            )

    lines.append(f"\n## Pathway Convergence\n")
    sig_pw = [r for r in pathway_results if r.convergence_fdr < 0.05]
    sug_pw = [r for r in pathway_results if r.convergence_fdr < 0.25]
    lines.append(f"- Pathways tested: {len(pathway_results)}")
    lines.append(f"- Significant (FDR < 0.05): {len(sig_pw)}")
    lines.append(f"- Suggestive (FDR < 0.25): {len(sug_pw)}\n")

    if pathway_results:
        lines.append("| Pathway | Domain | Loci Hit | Conv P | Conv FDR | Enrich P |")
        lines.append("|---------|--------|----------|--------|----------|----------|")
        for r in pathway_results[:20]:
            lines.append(
                f"| {r.pathway} | {r.domain} | "
                f"{r.n_loci_hit}/{r.n_loci_total} | "
                f"{r.convergence_p:.2e} | {r.convergence_fdr:.3f} | "
                f"{r.enrichment_p:.2e} |"
            )

    lines.append(f"\n## Domain Summary\n")
    if domain_results:
        lines.append("| Domain | Pathways | Convergent | Median Loci | Fisher P | FDR |")
        lines.append("|--------|----------|------------|-------------|----------|-----|")
        for d in domain_results:
            lines.append(
                f"| {d.domain} | {d.n_pathways} | {d.n_pathways_convergent} | "
                f"{d.median_loci_hit:.1f} | {d.fisher_p:.2e} | {d.fisher_fdr:.3f} |"
            )

    path = output_dir / "convergence_summary.md"
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote convergence summary to %s", path)
    return path


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_pathway_convergence(
    gene_results_path: Path,
    gene_set_results_path: Path | None = None,
    gene_sets_dir: Path | None = None,
    output_dir: Path | None = None,
    p_threshold: float = 0.05,
    merge_distance_kb: int = 1000,
    enrichment_p_threshold: float = 0.05,
) -> tuple[list[PathwayConvergenceResult], list[DomainConvergenceResult]]:
    """Run full pathway convergence pipeline.

    Parameters
    ----------
    gene_results_path : Path
        Gene-level results TSV from MAGMA analysis.
    gene_set_results_path : Path | None
        Gene-set results TSV. Defaults to magma_results dir.
    gene_sets_dir : Path | None
        Directory with GMT files. Defaults to project gene_sets dir.
    output_dir : Path | None
        Output directory. Defaults to enrichment dir.
    p_threshold : float
        P-value threshold for defining loci.
    merge_distance_kb : int
        Distance for merging genes into loci.
    enrichment_p_threshold : float
        P-value threshold for selecting enriched pathways.
    """
    ensure_dirs()
    if output_dir is None:
        output_dir = ENRICHMENT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load gene results
    gene_results_df = load_gene_results(gene_results_path)

    # Load gene-set results
    if gene_set_results_path is None:
        default_gs_path = MAGMA_DIR / "gene_set_results.tsv"
        if default_gs_path.exists():
            gene_set_results_path = default_gs_path

    gs_results_df = pd.DataFrame()
    if gene_set_results_path is not None and gene_set_results_path.exists():
        gs_results_df = load_gene_set_results(gene_set_results_path)

    # Load gene-set definitions
    from bioagentics.tourettes.ts_gwas_functional_annotation.magma_analysis import (
        load_all_gene_sets,
    )
    gene_sets = load_all_gene_sets(gene_sets_dir)

    # Step 1: Define loci
    loci = define_loci(gene_results_df, p_threshold, merge_distance_kb)

    # Step 2: Pathway convergence
    pw_results = pathway_convergence_analysis(
        loci, gs_results_df, gene_sets, enrichment_p_threshold,
    )

    # Step 3: Domain convergence
    dom_results = domain_convergence(pw_results)

    # Step 4: Write outputs
    if pw_results:
        write_convergence_results(pw_results, output_dir)
    if dom_results:
        write_domain_results(dom_results, output_dir)
    write_convergence_summary(pw_results, dom_results, loci, output_dir)

    return pw_results, dom_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Pathway convergence analysis for TS GWAS.",
    )
    parser.add_argument(
        "--gene-results", type=Path, required=True,
        help="Gene-level results TSV from MAGMA analysis.",
    )
    parser.add_argument(
        "--gene-set-results", type=Path, default=None,
        help="Gene-set enrichment results TSV.",
    )
    parser.add_argument(
        "--gene-sets-dir", type=Path, default=None,
        help="Directory containing GMT gene-set files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--p-threshold", type=float, default=0.05,
        help="P-value threshold for locus definition (default: 0.05).",
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

    if not args.gene_results.exists():
        logger.error("Gene results file not found: %s", args.gene_results)
        sys.exit(1)

    run_pathway_convergence(
        gene_results_path=args.gene_results,
        gene_set_results_path=args.gene_set_results,
        gene_sets_dir=args.gene_sets_dir,
        output_dir=args.output_dir,
        p_threshold=args.p_threshold,
        merge_distance_kb=args.merge_distance_kb,
    )


if __name__ == "__main__":
    main()
