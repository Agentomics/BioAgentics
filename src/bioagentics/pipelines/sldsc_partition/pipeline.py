"""Stratified LDSC (S-LDSC) pipeline for functional annotation partitioning.

Partitions genetic correlations between Tourette syndrome and psychiatric
comorbidities by functional annotation categories (coding, regulatory,
brain-expressed, CSTC circuit) using stratified LD score regression
(Finucane et al. 2015).

Extends the base LDSC pipeline with annotation-stratified regression to
identify which genomic functional categories drive TS-OCD vs. TS-ADHD
genetic correlations.
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

from bioagentics.config import REPO_ROOT
from bioagentics.pipelines.ldsc_correlation.pipeline import (
    load_sumstats,
    munge_sumstats,
)

logger = logging.getLogger(__name__)


@dataclass
class AnnotationEnrichment:
    """Enrichment result for a single annotation category."""

    annotation: str
    coefficient: float
    coefficient_se: float
    enrichment: float
    enrichment_p: float
    prop_snps: float
    prop_h2: float


@dataclass
class PartitionedResult:
    """Result of S-LDSC partitioned analysis for a trait pair."""

    trait1: str
    trait2: str
    annotations: list[AnnotationEnrichment] = field(default_factory=list)
    total_h2: float = 0.0
    n_snps: int = 0


def load_annotations(path: Path) -> pd.DataFrame:
    """Load functional annotation matrix.

    Expects a file or directory with columns: SNP, then one binary (0/1) column
    per annotation category. If path is a directory, reads all .annot[.gz] files
    and merges on SNP.

    Standard baseline-LD v2.2 annotations include 97 categories such as:
    Coding, Conserved, CTCF, DHS, Enhancer, H3K27ac, H3K4me1, H3K4me3,
    Intron, Promoter, TSS, UTR, etc.
    """
    if path.is_dir():
        frames = []
        for f in sorted(path.glob("*.annot*")):
            if f.name.endswith((".annot", ".annot.gz", ".annot.tsv")):
                df = pd.read_csv(f, sep="\t")
                frames.append(df)
        if not frames:
            raise FileNotFoundError(f"No .annot files found in {path}")
        # Concatenate per-chromosome files
        return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["SNP"])
    else:
        return pd.read_csv(path, sep="\t")


def load_stratified_ld_scores(path: Path) -> pd.DataFrame:
    """Load annotation-stratified LD scores.

    Expects columns: SNP, then L2_<annotation> for each category.
    If path is a directory, reads per-chromosome files and concatenates.
    """
    if path.is_dir():
        frames = []
        for f in sorted(path.glob("*.l2.ldscore*")):
            frames.append(pd.read_csv(f, sep="\t"))
        if not frames:
            raise FileNotFoundError(f"No stratified LD score files in {path}")
        ldscore = pd.concat(frames, ignore_index=True)
    else:
        ldscore = pd.read_csv(path, sep="\t")

    ldscore.columns = ldscore.columns.str.strip()
    if "SNP" not in ldscore.columns:
        raise ValueError("Stratified LD score file must contain SNP column")

    ldscore.drop_duplicates(subset=["SNP"], inplace=True)
    logger.info("Loaded stratified LD scores: %d SNPs, %d columns", len(ldscore), len(ldscore.columns))
    return ldscore


def sldsc_regression(
    sumstats: pd.DataFrame,
    stratified_ld: pd.DataFrame,
    annotation_cols: list[str],
    trait_name: str = "trait",
) -> PartitionedResult:
    """Run S-LDSC partitioned heritability regression.

    The model is: E[chi2_j] = N * sum_c(tau_c * l_{j,c}) / M + 1
    where tau_c is the per-SNP heritability contribution of annotation c,
    and l_{j,c} is the stratified LD score for SNP j and annotation c.

    Parameters
    ----------
    sumstats : pd.DataFrame
        Munged summary statistics with SNP, Z, N columns.
    stratified_ld : pd.DataFrame
        Stratified LD scores with SNP column plus L2_<annot> columns.
    annotation_cols : list[str]
        Names of annotation columns in stratified_ld to use.
    trait_name : str
        Label for the trait.

    Returns
    -------
    PartitionedResult with per-annotation enrichment estimates.
    """
    # Merge sumstats with stratified LD scores
    ld_cols = [f"L2_{c}" for c in annotation_cols]
    available_ld = [c for c in ld_cols if c in stratified_ld.columns]
    if not available_ld:
        raise ValueError(
            f"No matching L2_* columns found. Available: {[c for c in stratified_ld.columns if c.startswith('L2_')]}"
        )

    merged = sumstats[["SNP", "Z", "N"]].merge(
        stratified_ld[["SNP"] + available_ld], on="SNP"
    )

    n_snps = len(merged)
    if n_snps < 200:
        logger.warning("Only %d SNPs for S-LDSC — results unreliable", n_snps)

    chi2 = merged["Z"].values ** 2
    n_vals = merged["N"].values
    m = n_snps

    # Build design matrix: X_j = [N * l_{j,c} / M for each c] + [1]
    X_parts = []
    for col in available_ld:
        ld_c = merged[col].values
        X_parts.append((n_vals * ld_c / m).reshape(-1, 1))
    X_parts.append(np.ones((n_snps, 1)))  # intercept
    X = np.hstack(X_parts)

    # Total LD score for weights
    total_ld = np.zeros(n_snps)
    for col in available_ld:
        total_ld += merged[col].values
    total_ld = np.maximum(total_ld, 1.0)

    # Element-wise weighting to avoid N×N dense diagonal matrices (OOM on real GWAS data)
    w = 1.0 / (total_ld**2)

    # WLS regression
    XtW = X.T * w  # broadcast w across columns, equivalent to X.T @ diag(w)
    XtWX = XtW @ X
    XtWy = XtW @ chi2

    try:
        beta = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        logger.error("Singular matrix in S-LDSC regression for %s", trait_name)
        return PartitionedResult(trait1=trait_name, trait2="partitioned", n_snps=n_snps)

    # tau_c values (per-SNP heritability for each annotation)
    taus = beta[:-1]

    # Standard errors via sandwich estimator
    # meat = X' @ diag(w) @ diag(r^2) @ diag(w) @ X = X' @ diag(w^2 * r^2) @ X
    residuals = chi2 - X @ beta
    wr2 = (w * residuals) ** 2
    meat = (X.T * wr2) @ X
    try:
        bread = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError:
        bread = np.zeros_like(XtWX)

    cov = bread @ meat @ bread
    tau_ses = np.sqrt(np.maximum(np.diag(cov)[:-1], 0))

    # Compute enrichment: prop_h2 / prop_snps
    # prop_snps_c = (number of SNPs in annotation c) / M
    # h2_c = tau_c * M_c (total heritability from category c)
    total_h2 = float(np.sum(taus * m))  # approximate total h2

    enrichments = []
    for i, col in enumerate(available_ld):
        annot_name = col.replace("L2_", "")
        ld_c = merged[col].values

        # Proportion of SNPs with non-zero LD in this category
        prop_snps = float(np.mean(ld_c > 0))
        if prop_snps == 0:
            prop_snps = 1.0 / n_snps

        # Proportion of h2 from this category
        h2_c = float(taus[i] * np.sum(ld_c > 0))
        prop_h2 = h2_c / total_h2 if total_h2 != 0 else 0.0

        # Enrichment = prop_h2 / prop_snps
        enrichment = prop_h2 / prop_snps if prop_snps > 0 else 0.0

        # P-value for tau_c != 0 (one-sided: tau_c > 0 for enrichment)
        if tau_ses[i] > 0:
            z_stat = taus[i] / tau_ses[i]
            enrich_p = float(stats.norm.sf(z_stat))
        else:
            enrich_p = np.nan

        enrichments.append(
            AnnotationEnrichment(
                annotation=annot_name,
                coefficient=float(taus[i]),
                coefficient_se=float(tau_ses[i]),
                enrichment=enrichment,
                enrichment_p=enrich_p,
                prop_snps=prop_snps,
                prop_h2=prop_h2,
            )
        )

    return PartitionedResult(
        trait1=trait_name,
        trait2="partitioned",
        annotations=enrichments,
        total_h2=total_h2,
        n_snps=n_snps,
    )


def compute_partitioned_correlations(
    sumstats_dir: Path,
    stratified_ld: pd.DataFrame,
    annotation_cols: list[str],
    traits: list[str] | None = None,
) -> list[PartitionedResult]:
    """Run S-LDSC for all traits in a directory.

    Parameters
    ----------
    sumstats_dir : Path
        Directory with GWAS summary stats files.
    stratified_ld : pd.DataFrame
        Stratified LD scores.
    annotation_cols : list[str]
        Annotation categories to partition by.
    traits : list[str] | None
        Specific traits to analyze. If None, analyzes all found files.

    Returns
    -------
    List of PartitionedResult, one per trait.
    """
    suffixes = (".sumstats.gz", ".sumstats", ".tsv", ".txt", ".csv")
    files: dict[str, Path] = {}
    for f in sumstats_dir.iterdir():
        if f.is_file():
            name = f.name
            for suf in suffixes:
                if name.endswith(suf):
                    trait = name[: -len(suf)]
                    files[trait] = f
                    break

    if traits:
        files = {k: v for k, v in files.items() if k in traits}

    logger.info("Running S-LDSC for %d traits: %s", len(files), sorted(files.keys()))

    results = []
    for trait_name, path in sorted(files.items()):
        logger.info("S-LDSC partitioning for %s...", trait_name)
        df = munge_sumstats(load_sumstats(path), trait_name)
        result = sldsc_regression(df, stratified_ld, annotation_cols, trait_name)
        results.append(result)

        sig = [a for a in result.annotations if a.enrichment_p < 0.05]
        logger.info(
            "  %s: %d/%d annotations enriched (p<0.05), total h2=%.4f",
            trait_name,
            len(sig),
            len(result.annotations),
            result.total_h2,
        )

    return results


def results_to_dataframe(results: list[PartitionedResult]) -> pd.DataFrame:
    """Convert S-LDSC results to a tidy DataFrame."""
    rows = []
    for r in results:
        for a in r.annotations:
            rows.append(
                {
                    "trait": r.trait1,
                    "annotation": a.annotation,
                    "tau": a.coefficient,
                    "tau_se": a.coefficient_se,
                    "enrichment": a.enrichment,
                    "enrichment_p": a.enrichment_p,
                    "prop_snps": a.prop_snps,
                    "prop_h2": a.prop_h2,
                    "total_h2": r.total_h2,
                    "n_snps": r.n_snps,
                }
            )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the S-LDSC partitioning pipeline."""
    parser = argparse.ArgumentParser(
        description="Partition TS genetic correlations by functional annotation using S-LDSC."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing GWAS summary statistics.",
    )
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Directory containing annotation files (.annot) and stratified LD scores.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "tourettes" / "ts-comorbidity-genetic-architecture" / "phase1",
        help="Output directory.",
    )
    parser.add_argument(
        "--traits",
        nargs="*",
        default=None,
        help="Specific traits to analyze (default: all).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.input_dir.exists():
        logger.error("Input directory does not exist: %s", args.input_dir)
        sys.exit(1)

    annotations_dir = args.annotations_dir
    if not annotations_dir.exists():
        logger.error("Annotations directory does not exist: %s", annotations_dir)
        sys.exit(1)

    # Load stratified LD scores
    ld_path = annotations_dir / "ld_scores"
    if not ld_path.exists():
        ld_path = annotations_dir  # try the directory itself
    stratified_ld = load_stratified_ld_scores(ld_path)

    # Discover annotation columns from LD score column names
    annotation_cols = [
        c.replace("L2_", "") for c in stratified_ld.columns if c.startswith("L2_")
    ]
    if not annotation_cols:
        logger.error("No L2_* annotation columns found in stratified LD scores")
        sys.exit(1)

    logger.info("Found %d annotation categories: %s", len(annotation_cols), annotation_cols)

    # Run pipeline
    results = compute_partitioned_correlations(
        args.input_dir, stratified_ld, annotation_cols, args.traits
    )

    if not results:
        logger.warning("No results computed.")
        sys.exit(0)

    # Write output
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    df = results_to_dataframe(results)
    out_path = output_dir / "sldsc_enrichment.tsv"
    df.to_csv(out_path, sep="\t", index=False, float_format="%.6f")
    logger.info("Wrote enrichment results to %s", out_path)


if __name__ == "__main__":
    main()
