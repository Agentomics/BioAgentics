"""Cross-disorder pathway comparison for TS GWAS.

Compares TS-enriched pathways with OCD, ADHD, and ASD GWAS results using
the Grotzinger et al. (Nature 2025) 5-factor genomic model and 238
pleiotropic loci. Identifies TS-specific pathway signatures distinct from
shared psychiatric pathways.

Steps:
  1. Map TS loci overlap with cross-disorder pleiotropic loci
  2. Compare pathway enrichment across disorders and genomic factors
  3. Classify pathways as TS-specific, compulsive-factor shared, or broadly shared
  4. Generate cross-disorder enrichment matrix for heatmap visualization

Usage:
    python -m bioagentics.tourettes.ts_gwas_functional_annotation.cross_disorder \
        --ts-enrichment output/.../enrichment/pathway_convergence.tsv \
        --cross-disorder-dir data/.../cross_disorder/
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
    CROSS_DISORDER_DIR,
    DATA_DIR,
    ENRICHMENT_DIR,
    ensure_dirs,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grotzinger et al. (Nature 2025) factor model definitions
# ---------------------------------------------------------------------------

DISORDERS: list[str] = [
    "TS", "OCD", "AN", "ADHD", "ASD",
    "SCZ", "BPD", "MDD", "ANX", "AUD",
]

GENOMIC_FACTORS: dict[str, list[str]] = {
    "compulsive": ["OCD", "AN", "TS"],
    "neurodevelopmental": ["ADHD", "ASD"],
    "psychotic": ["SCZ", "BPD"],
    "internalizing": ["MDD", "ANX"],
    "substance_use": ["AUD"],
}

FACTOR_DESCRIPTIONS: dict[str, str] = {
    "compulsive": "Compulsive disorders (OCD, AN, TS)",
    "neurodevelopmental": "Neurodevelopmental disorders (ADHD, ASD)",
    "psychotic": "Psychotic spectrum (SCZ, BPD)",
    "internalizing": "Internalizing disorders (MDD, ANX)",
    "substance_use": "Substance use disorders (AUD)",
}

DISORDER_FULL_NAMES: dict[str, str] = {
    "TS": "Tourette Syndrome",
    "OCD": "Obsessive-Compulsive Disorder",
    "AN": "Anorexia Nervosa",
    "ADHD": "Attention-Deficit/Hyperactivity Disorder",
    "ASD": "Autism Spectrum Disorder",
    "SCZ": "Schizophrenia",
    "BPD": "Bipolar Disorder",
    "MDD": "Major Depressive Disorder",
    "ANX": "Anxiety Disorders",
    "AUD": "Alcohol Use Disorder",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PleioLocus:
    """A pleiotropic locus from cross-disorder GWAS."""

    locus_id: int
    chr: int
    start: int
    stop: int
    lead_snp: str
    disorders: list[str]
    factors: list[str]
    n_disorders: int


@dataclass
class LocusOverlap:
    """Overlap between a TS locus and a pleiotropic locus."""

    ts_locus_id: int
    pleio_locus_id: int
    chr: int
    overlap_start: int
    overlap_stop: int
    ts_genes: list[str]
    shared_disorders: list[str]
    shared_factors: list[str]


@dataclass
class CrossDisorderPathway:
    """Pathway enrichment comparison across disorders."""

    pathway: str
    source: str
    ts_p: float
    ts_fdr: float
    disorder_p: dict[str, float]
    disorder_fdr: dict[str, float]
    factor_min_p: dict[str, float]
    classification: str  # ts_specific, compulsive_shared, broadly_shared
    specificity_score: float
    n_disorders_enriched: int


@dataclass
class CrossDisorderSummary:
    """Summary of cross-disorder comparison results."""

    n_ts_pathways: int
    n_ts_specific: int
    n_compulsive_shared: int
    n_broadly_shared: int
    n_pleio_loci_overlap: int
    factor_enrichment_counts: dict[str, int]


# ---------------------------------------------------------------------------
# Pleiotropic locus overlap
# ---------------------------------------------------------------------------


def load_pleiotropic_loci(path: Path) -> list[PleioLocus]:
    """Load pleiotropic loci from TSV.

    Expected columns: LOCUS_ID, CHR, START, STOP, LEAD_SNP, DISORDERS, FACTORS.
    DISORDERS and FACTORS are semicolon-delimited.
    """
    df = pd.read_csv(path, sep="\t")
    required = {"LOCUS_ID", "CHR", "START", "STOP"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        logger.warning("Pleiotropic loci file missing columns: %s", missing)
        return []

    loci = []
    for _, row in df.iterrows():
        disorders_str = str(row.get("DISORDERS", ""))
        disorders = [d.strip() for d in disorders_str.split(";") if d.strip()]
        factors_str = str(row.get("FACTORS", ""))
        factors = [f.strip() for f in factors_str.split(";") if f.strip()]

        loci.append(PleioLocus(
            locus_id=int(row["LOCUS_ID"]),
            chr=int(row["CHR"]),
            start=int(row["START"]),
            stop=int(row["STOP"]),
            lead_snp=str(row.get("LEAD_SNP", "")),
            disorders=disorders,
            factors=factors,
            n_disorders=len(disorders),
        ))

    logger.info("Loaded %d pleiotropic loci from %s", len(loci), path)
    return loci


def compute_locus_overlaps(
    ts_loci: list[dict],
    pleio_loci: list[PleioLocus],
    window_kb: int = 0,
) -> list[LocusOverlap]:
    """Find genomic overlaps between TS loci and pleiotropic loci.

    Parameters
    ----------
    ts_loci : list[dict]
        TS loci with keys: locus_id, chr, start, stop, genes.
    pleio_loci : list[PleioLocus]
        Pleiotropic loci from cross-disorder GWAS.
    window_kb : int
        Extension window in kb around each locus for overlap detection.
    """
    if not ts_loci or not pleio_loci:
        return []

    window = window_kb * 1000
    overlaps = []

    # Index pleiotropic loci by chromosome
    pleio_by_chr: dict[int, list[PleioLocus]] = {}
    for pl in pleio_loci:
        pleio_by_chr.setdefault(pl.chr, []).append(pl)

    for ts in ts_loci:
        ts_chr = ts["chr"]
        ts_start = ts["start"] - window
        ts_stop = ts["stop"] + window

        for pl in pleio_by_chr.get(ts_chr, []):
            # Check overlap
            ol_start = max(ts_start, pl.start)
            ol_stop = min(ts_stop, pl.stop)
            if ol_start <= ol_stop:
                # Determine which factors are involved
                factors = _disorders_to_factors(pl.disorders)
                overlaps.append(LocusOverlap(
                    ts_locus_id=ts["locus_id"],
                    pleio_locus_id=pl.locus_id,
                    chr=ts_chr,
                    overlap_start=ol_start,
                    overlap_stop=ol_stop,
                    ts_genes=list(ts.get("genes", [])),
                    shared_disorders=pl.disorders,
                    shared_factors=factors,
                ))

    logger.info(
        "Found %d overlaps between %d TS loci and %d pleiotropic loci",
        len(overlaps), len(ts_loci), len(pleio_loci),
    )
    return overlaps


def _disorders_to_factors(disorders: list[str]) -> list[str]:
    """Map a list of disorders to their genomic factors."""
    factors = set()
    for factor, factor_disorders in GENOMIC_FACTORS.items():
        if any(d in factor_disorders for d in disorders):
            factors.add(factor)
    return sorted(factors)


# ---------------------------------------------------------------------------
# Cross-disorder pathway enrichment comparison
# ---------------------------------------------------------------------------


def load_disorder_enrichment(
    enrichment_dir: Path,
    disorders: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Load pathway enrichment results for multiple disorders.

    Expects files named {disorder}_pathway_enrichment.tsv in the directory,
    each with columns: PATHWAY, SOURCE, P, FDR.
    """
    if disorders is None:
        disorders = DISORDERS

    results: dict[str, pd.DataFrame] = {}
    for disorder in disorders:
        fname = f"{disorder.lower()}_pathway_enrichment.tsv"
        path = enrichment_dir / fname
        if path.exists():
            df = pd.read_csv(path, sep="\t")
            results[disorder] = df
            logger.info("Loaded %d pathways for %s", len(df), disorder)
        else:
            logger.debug("No enrichment file for %s at %s", disorder, path)

    return results


def compare_pathway_enrichment(
    ts_enrichment: pd.DataFrame,
    disorder_enrichments: dict[str, pd.DataFrame],
    p_threshold: float = 0.05,
    fdr_threshold: float = 0.25,
) -> list[CrossDisorderPathway]:
    """Compare pathway enrichment between TS and other disorders.

    Classifies each TS-enriched pathway as:
    - ts_specific: enriched in TS but not in any other disorder (P < threshold)
    - compulsive_shared: enriched in TS and >= 1 compulsive-factor disorder
    - broadly_shared: enriched in TS and disorders across >= 2 factors

    Parameters
    ----------
    ts_enrichment : pd.DataFrame
        TS pathway enrichment with columns: PATHWAY, SOURCE, P, FDR
        (or CONVERGENCE_P, CONVERGENCE_FDR from convergence analysis).
    disorder_enrichments : dict[str, pd.DataFrame]
        Enrichment results per disorder.
    p_threshold : float
        P-value threshold for considering a pathway enriched in a disorder.
    fdr_threshold : float
        FDR threshold for TS pathways to include in comparison.
    """
    if ts_enrichment.empty:
        return []

    # Normalize TS enrichment column names
    ts_df = _normalize_enrichment_df(ts_enrichment)
    if ts_df.empty:
        return []

    # Filter to suggestive TS pathways
    ts_sig = ts_df[ts_df["FDR"] < fdr_threshold].copy()
    if ts_sig.empty:
        # Fall back to top pathways by P
        ts_sig = ts_df.nsmallest(min(50, len(ts_df)), "P")

    results: list[CrossDisorderPathway] = []

    for _, ts_row in ts_sig.iterrows():
        pathway = str(ts_row["PATHWAY"])
        source = str(ts_row.get("SOURCE", ""))
        ts_p = float(ts_row["P"])
        ts_fdr = float(ts_row["FDR"])

        # Collect P-values from other disorders
        disorder_p: dict[str, float] = {"TS": ts_p}
        disorder_fdr: dict[str, float] = {"TS": ts_fdr}

        for disorder, d_df in disorder_enrichments.items():
            d_norm = _normalize_enrichment_df(d_df)
            match = d_norm[d_norm["PATHWAY"] == pathway]
            if not match.empty:
                disorder_p[disorder] = float(match.iloc[0]["P"])
                disorder_fdr[disorder] = float(match.iloc[0]["FDR"])
            else:
                disorder_p[disorder] = 1.0
                disorder_fdr[disorder] = 1.0

        # Compute factor-level min P
        factor_min_p: dict[str, float] = {}
        for factor, factor_disorders in GENOMIC_FACTORS.items():
            factor_ps = [disorder_p.get(d, 1.0) for d in factor_disorders]
            factor_min_p[factor] = min(factor_ps) if factor_ps else 1.0

        # Count enriched disorders (excluding TS)
        enriched_disorders = [
            d for d, p in disorder_p.items()
            if d != "TS" and p < p_threshold
        ]
        n_enriched = len(enriched_disorders)

        # Classify pathway
        classification = _classify_pathway(
            disorder_p, p_threshold,
        )

        # Specificity score: -log10(TS_P) * (1 - fraction of other disorders enriched)
        n_other = len(disorder_p) - 1  # exclude TS
        frac_shared = n_enriched / max(n_other, 1)
        specificity = -np.log10(max(ts_p, 1e-300)) * (1 - frac_shared)

        results.append(CrossDisorderPathway(
            pathway=pathway,
            source=source,
            ts_p=ts_p,
            ts_fdr=ts_fdr,
            disorder_p=disorder_p,
            disorder_fdr=disorder_fdr,
            factor_min_p=factor_min_p,
            classification=classification,
            specificity_score=float(specificity),
            n_disorders_enriched=n_enriched,
        ))

    results.sort(key=lambda r: r.specificity_score, reverse=True)
    n_spec = sum(1 for r in results if r.classification == "ts_specific")
    n_comp = sum(1 for r in results if r.classification == "compulsive_shared")
    n_broad = sum(1 for r in results if r.classification == "broadly_shared")
    logger.info(
        "Cross-disorder comparison: %d pathways — %d TS-specific, "
        "%d compulsive-shared, %d broadly-shared",
        len(results), n_spec, n_comp, n_broad,
    )
    return results


def _normalize_enrichment_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize enrichment DataFrame columns to PATHWAY, P, FDR, SOURCE."""
    out = df.copy()

    # Handle convergence-format columns
    col_map = {
        "CONVERGENCE_P": "P",
        "CONVERGENCE_FDR": "FDR",
        "ENRICHMENT_P": "P",
        "ENRICHMENT_FDR": "FDR",
        "GENE_SET": "PATHWAY",
        "FDR_Q": "FDR",
    }

    for old, new in col_map.items():
        if old in out.columns and new not in out.columns:
            out[new] = out[old]

    required = {"PATHWAY", "P"}
    if not required.issubset(out.columns):
        logger.warning("Cannot normalize enrichment df: missing %s",
                       required - set(out.columns))
        return pd.DataFrame()

    if "FDR" not in out.columns:
        out["FDR"] = out["P"]  # fallback: use raw P
    if "SOURCE" not in out.columns:
        out["SOURCE"] = ""

    return out


def _classify_pathway(
    disorder_p: dict[str, float],
    p_threshold: float,
) -> str:
    """Classify a pathway based on cross-disorder enrichment pattern."""
    enriched_disorders = {
        d for d, p in disorder_p.items() if p < p_threshold
    }

    if not enriched_disorders or enriched_disorders == {"TS"}:
        return "ts_specific"

    # Check which factors have enriched disorders
    enriched_factors = set()
    for factor, factor_disorders in GENOMIC_FACTORS.items():
        if any(d in enriched_disorders for d in factor_disorders):
            enriched_factors.add(factor)

    if enriched_factors == {"compulsive"}:
        return "compulsive_shared"

    if len(enriched_factors) >= 2:
        return "broadly_shared"

    # Enriched in TS + one non-compulsive factor only
    return "compulsive_shared" if "compulsive" in enriched_factors else "broadly_shared"


# ---------------------------------------------------------------------------
# Enrichment matrix for heatmap visualization
# ---------------------------------------------------------------------------


def build_enrichment_matrix(
    results: list[CrossDisorderPathway],
    disorders: list[str] | None = None,
    use_neglog10: bool = True,
) -> pd.DataFrame:
    """Build pathway x disorder enrichment matrix.

    Returns a DataFrame with pathways as rows and disorders as columns.
    Values are -log10(P) if use_neglog10=True, else raw P-values.
    """
    if not results:
        return pd.DataFrame()

    if disorders is None:
        all_disorders: set[str] = set()
        for r in results:
            all_disorders.update(r.disorder_p.keys())
        disorders = sorted(all_disorders)

    rows = []
    for r in results:
        row: dict[str, object] = {
            "PATHWAY": r.pathway,
            "SOURCE": r.source,
            "CLASSIFICATION": r.classification,
        }
        for d in disorders:
            p = r.disorder_p.get(d, 1.0)
            if use_neglog10:
                row[d] = -np.log10(max(p, 1e-300))
            else:
                row[d] = p
        rows.append(row)

    return pd.DataFrame(rows)


def build_factor_matrix(
    results: list[CrossDisorderPathway],
    use_neglog10: bool = True,
) -> pd.DataFrame:
    """Build pathway x factor enrichment matrix.

    Values are -log10(min P within factor).
    """
    if not results:
        return pd.DataFrame()

    factors = list(GENOMIC_FACTORS.keys())
    rows = []
    for r in results:
        row: dict[str, object] = {
            "PATHWAY": r.pathway,
            "CLASSIFICATION": r.classification,
        }
        for f in factors:
            p = r.factor_min_p.get(f, 1.0)
            if use_neglog10:
                row[f] = -np.log10(max(p, 1e-300))
            else:
                row[f] = p
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------


def summarize_cross_disorder(
    results: list[CrossDisorderPathway],
    overlaps: list[LocusOverlap] | None = None,
) -> CrossDisorderSummary:
    """Compute summary statistics for cross-disorder comparison."""
    n_total = len(results)
    n_spec = sum(1 for r in results if r.classification == "ts_specific")
    n_comp = sum(1 for r in results if r.classification == "compulsive_shared")
    n_broad = sum(1 for r in results if r.classification == "broadly_shared")

    factor_counts: dict[str, int] = {}
    for factor in GENOMIC_FACTORS:
        factor_counts[factor] = sum(
            1 for r in results
            if r.factor_min_p.get(factor, 1.0) < 0.05
        )

    return CrossDisorderSummary(
        n_ts_pathways=n_total,
        n_ts_specific=n_spec,
        n_compulsive_shared=n_comp,
        n_broadly_shared=n_broad,
        n_pleio_loci_overlap=len(overlaps) if overlaps else 0,
        factor_enrichment_counts=factor_counts,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_cross_disorder_results(
    results: list[CrossDisorderPathway],
    output_dir: Path,
) -> Path:
    """Write cross-disorder pathway comparison results to TSV."""
    rows = []
    for r in results:
        row = {
            "PATHWAY": r.pathway,
            "SOURCE": r.source,
            "TS_P": r.ts_p,
            "TS_FDR": r.ts_fdr,
            "CLASSIFICATION": r.classification,
            "SPECIFICITY_SCORE": r.specificity_score,
            "N_DISORDERS_ENRICHED": r.n_disorders_enriched,
        }
        # Add per-disorder P-values
        for d in sorted(r.disorder_p.keys()):
            if d != "TS":
                row[f"{d}_P"] = r.disorder_p[d]
        # Add per-factor min P
        for f in GENOMIC_FACTORS:
            row[f"FACTOR_{f}_P"] = r.factor_min_p.get(f, 1.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    path = output_dir / "cross_disorder_pathways.tsv"
    df.to_csv(path, sep="\t", index=False, float_format="%.6g")
    logger.info("Wrote %d cross-disorder results to %s", len(rows), path)
    return path


def write_enrichment_heatmap(
    results: list[CrossDisorderPathway],
    output_dir: Path,
) -> Path:
    """Write enrichment matrix TSV for heatmap visualization."""
    matrix = build_enrichment_matrix(results)
    path = output_dir / "enrichment_heatmap.tsv"
    matrix.to_csv(path, sep="\t", index=False, float_format="%.4f")
    logger.info("Wrote enrichment heatmap (%d x %d) to %s",
                len(matrix), len(matrix.columns), path)
    return path


def write_locus_overlaps(
    overlaps: list[LocusOverlap],
    output_dir: Path,
) -> Path:
    """Write locus overlap results to TSV."""
    rows = [{
        "TS_LOCUS_ID": o.ts_locus_id,
        "PLEIO_LOCUS_ID": o.pleio_locus_id,
        "CHR": o.chr,
        "OVERLAP_START": o.overlap_start,
        "OVERLAP_STOP": o.overlap_stop,
        "TS_GENES": ";".join(o.ts_genes),
        "SHARED_DISORDERS": ";".join(o.shared_disorders),
        "SHARED_FACTORS": ";".join(o.shared_factors),
    } for o in overlaps]

    df = pd.DataFrame(rows)
    path = output_dir / "locus_overlaps.tsv"
    df.to_csv(path, sep="\t", index=False)
    logger.info("Wrote %d locus overlaps to %s", len(rows), path)
    return path


def write_cross_disorder_summary(
    results: list[CrossDisorderPathway],
    overlaps: list[LocusOverlap],
    summary: CrossDisorderSummary,
    output_dir: Path,
) -> Path:
    """Write a human-readable cross-disorder comparison summary."""
    lines = ["# Cross-Disorder Pathway Comparison Summary\n"]

    lines.append("## Grotzinger et al. Factor Model\n")
    for factor, desc in FACTOR_DESCRIPTIONS.items():
        lines.append(f"- **{factor}**: {desc}")

    lines.append(f"\n## Locus Overlap\n")
    lines.append(f"- Pleiotropic locus overlaps: {summary.n_pleio_loci_overlap}")
    if overlaps:
        lines.append("")
        lines.append("| TS Locus | Pleio Locus | CHR | Shared Disorders | Factors |")
        lines.append("|----------|-------------|-----|------------------|---------|")
        for o in overlaps[:20]:
            lines.append(
                f"| {o.ts_locus_id} | {o.pleio_locus_id} | {o.chr} | "
                f"{', '.join(o.shared_disorders)} | {', '.join(o.shared_factors)} |"
            )

    lines.append(f"\n## Pathway Classification\n")
    lines.append(f"- Total TS pathways analyzed: {summary.n_ts_pathways}")
    lines.append(f"- TS-specific: {summary.n_ts_specific}")
    lines.append(f"- Compulsive-factor shared: {summary.n_compulsive_shared}")
    lines.append(f"- Broadly shared: {summary.n_broadly_shared}")

    lines.append(f"\n## Factor Enrichment Counts\n")
    for factor, count in summary.factor_enrichment_counts.items():
        lines.append(f"- {factor}: {count} pathways enriched")

    # TS-specific pathways
    ts_specific = [r for r in results if r.classification == "ts_specific"]
    if ts_specific:
        lines.append(f"\n## Top TS-Specific Pathways\n")
        lines.append("| Pathway | Source | TS P | Specificity |")
        lines.append("|---------|--------|------|-------------|")
        for r in ts_specific[:15]:
            lines.append(
                f"| {r.pathway} | {r.source} | {r.ts_p:.2e} | "
                f"{r.specificity_score:.2f} |"
            )

    # Compulsive-shared pathways
    comp_shared = [r for r in results if r.classification == "compulsive_shared"]
    if comp_shared:
        lines.append(f"\n## Compulsive-Factor Shared Pathways\n")
        lines.append("| Pathway | TS P | OCD P | AN P | Factor P |")
        lines.append("|---------|------|-------|------|----------|")
        for r in sorted(comp_shared, key=lambda x: x.ts_p)[:15]:
            ocd_p = r.disorder_p.get("OCD", 1.0)
            an_p = r.disorder_p.get("AN", 1.0)
            comp_p = r.factor_min_p.get("compulsive", 1.0)
            lines.append(
                f"| {r.pathway} | {r.ts_p:.2e} | {ocd_p:.2e} | "
                f"{an_p:.2e} | {comp_p:.2e} |"
            )

    path = output_dir / "cross_disorder_summary.md"
    path.write_text("\n".join(lines) + "\n")
    logger.info("Wrote cross-disorder summary to %s", path)
    return path


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_cross_disorder_comparison(
    ts_enrichment_path: Path,
    cross_disorder_dir: Path | None = None,
    pleiotropic_loci_path: Path | None = None,
    ts_loci: list[dict] | None = None,
    output_dir: Path | None = None,
    p_threshold: float = 0.05,
    fdr_threshold: float = 0.25,
) -> tuple[list[CrossDisorderPathway], list[LocusOverlap]]:
    """Run full cross-disorder comparison pipeline.

    Parameters
    ----------
    ts_enrichment_path : Path
        TS pathway enrichment results (TSV).
    cross_disorder_dir : Path | None
        Directory with per-disorder enrichment files. Defaults to data dir.
    pleiotropic_loci_path : Path | None
        TSV of pleiotropic loci. If None, locus overlap is skipped.
    ts_loci : list[dict] | None
        TS loci for overlap analysis (locus_id, chr, start, stop, genes).
    output_dir : Path | None
        Output directory.
    p_threshold : float
        P-value threshold for enrichment in a disorder.
    fdr_threshold : float
        FDR threshold for selecting TS pathways.
    """
    ensure_dirs()
    if output_dir is None:
        output_dir = CROSS_DISORDER_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load TS enrichment
    ts_df = pd.read_csv(ts_enrichment_path, sep="\t")
    logger.info("Loaded %d TS enrichment results from %s",
                len(ts_df), ts_enrichment_path)

    # Load cross-disorder enrichments
    if cross_disorder_dir is None:
        cross_disorder_dir = DATA_DIR / "cross_disorder"
    disorder_enrichments = load_disorder_enrichment(cross_disorder_dir)

    # Pathway comparison
    cd_results = compare_pathway_enrichment(
        ts_df, disorder_enrichments, p_threshold, fdr_threshold,
    )

    # Locus overlap
    overlaps: list[LocusOverlap] = []
    if pleiotropic_loci_path is not None and pleiotropic_loci_path.exists():
        pleio_loci = load_pleiotropic_loci(pleiotropic_loci_path)
        if ts_loci is not None:
            overlaps = compute_locus_overlaps(ts_loci, pleio_loci)

    # Summary
    summary = summarize_cross_disorder(cd_results, overlaps)

    # Write outputs
    if cd_results:
        write_cross_disorder_results(cd_results, output_dir)
        write_enrichment_heatmap(cd_results, output_dir)
    if overlaps:
        write_locus_overlaps(overlaps, output_dir)
    write_cross_disorder_summary(cd_results, overlaps, summary, output_dir)

    return cd_results, overlaps


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Cross-disorder pathway comparison for TS GWAS.",
    )
    parser.add_argument(
        "--ts-enrichment", type=Path, required=True,
        help="TS pathway enrichment results TSV.",
    )
    parser.add_argument(
        "--cross-disorder-dir", type=Path, default=None,
        help="Directory with per-disorder enrichment TSV files.",
    )
    parser.add_argument(
        "--pleiotropic-loci", type=Path, default=None,
        help="Pleiotropic loci TSV from Grotzinger et al.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--p-threshold", type=float, default=0.05,
        help="P-value threshold for enrichment (default: 0.05).",
    )
    parser.add_argument(
        "--fdr-threshold", type=float, default=0.25,
        help="FDR threshold for TS pathway selection (default: 0.25).",
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

    if not args.ts_enrichment.exists():
        logger.error("TS enrichment file not found: %s", args.ts_enrichment)
        sys.exit(1)

    run_cross_disorder_comparison(
        ts_enrichment_path=args.ts_enrichment,
        cross_disorder_dir=args.cross_disorder_dir,
        pleiotropic_loci_path=args.pleiotropic_loci,
        output_dir=args.output_dir,
        p_threshold=args.p_threshold,
        fdr_threshold=args.fdr_threshold,
    )


if __name__ == "__main__":
    main()
