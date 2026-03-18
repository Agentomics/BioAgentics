"""Stratified PRS construction pipeline for factor-specific polygenic risk scores.

Constructs polygenic risk scores (PRS) stratified by genomic SEM factor
decomposition. Uses clump-and-threshold (C+T) methodology to build
three stratified PRS from factor-specific GWAS:
  1. TS-OCD shared PRS (compulsive factor SNP weights)
  2. TS-ADHD shared PRS (neurodevelopmental factor SNP weights)
  3. TS-specific PRS (residual GWAS weights)

Compares variance explained (R²) of stratified PRS vs. unstratified
aggregate TS PRS to validate factor decomposition.

Pure-Python implementation: LD clumping via greedy correlation-based
pruning (analogous to PRSice-2 / PLINK --clump), PRS scoring via
weighted allele dosage sums.
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

logger = logging.getLogger(__name__)

# Default P-value thresholds for C+T
DEFAULT_P_THRESHOLDS = [5e-8, 1e-5, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0]

# Default LD clumping parameters
DEFAULT_CLUMP_R2 = 0.1
DEFAULT_CLUMP_KB = 250


@dataclass
class PRSWeights:
    """SNP weights for a single PRS stratum at a given P-value threshold."""

    stratum: str
    p_threshold: float
    snp_ids: list[str]
    weights: np.ndarray  # effect sizes (beta or Z)
    effect_alleles: list[str]
    n_snps: int = 0

    def __post_init__(self) -> None:
        self.n_snps = len(self.snp_ids)


@dataclass
class PRSResult:
    """Result of PRS scoring for one stratum at one threshold."""

    stratum: str
    p_threshold: float
    n_snps: int
    r_squared: float
    r_squared_se: float
    p_value: float
    mean_score: float
    sd_score: float


@dataclass
class StratifiedPRSComparison:
    """Comparison of stratified vs. aggregate PRS performance."""

    strata_results: list[PRSResult]
    aggregate_result: PRSResult | None
    combined_r_squared: float  # R² from all strata jointly
    combined_p_value: float
    r_squared_improvement: float  # combined - aggregate


def load_factor_gwas(path: Path) -> pd.DataFrame:
    """Load factor-specific GWAS summary statistics.

    Expected columns: SNP, A1, A2, BETA (or Z), SE, P, N
    Handles both tab-delimited and space-delimited formats.
    """
    sep = "\t" if path.suffix in (".tsv", ".tab") else None

    try:
        df = pd.read_csv(path, sep=sep, engine="python")
    except Exception:
        logger.error("Failed to load GWAS file: %s", path)
        raise

    # Standardize column names
    col_map = {
        "RSID": "SNP",
        "rsid": "SNP",
        "MarkerName": "SNP",
        "MARKERNAME": "SNP",
        "EFFECT_ALLELE": "A1",
        "ALT": "A1",
        "OTHER_ALLELE": "A2",
        "REF": "A2",
        "EFFECT": "BETA",
        "OR": "BETA",
        "B": "BETA",
        "STDERR": "SE",
        "SE_BETA": "SE",
        "PVAL": "P",
        "P_VALUE": "P",
        "PVALUE": "P",
        "ZSCORE": "Z",
        "NMISS": "N",
        "NEFF": "N",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # Derive BETA from Z/SE if needed
    if "BETA" not in df.columns and "Z" in df.columns:
        if "N" in df.columns:
            df["BETA"] = df["Z"] / np.sqrt(df["N"])
        else:
            df["BETA"] = df["Z"]

    # Derive Z from BETA/SE if needed
    if "Z" not in df.columns and "BETA" in df.columns and "SE" in df.columns:
        mask = df["SE"] > 0
        df.loc[mask, "Z"] = df.loc[mask, "BETA"] / df.loc[mask, "SE"]

    # Derive P from Z if needed
    if "P" not in df.columns and "Z" in df.columns:
        df["P"] = 2 * stats.norm.sf(np.abs(df["Z"]))

    required = {"SNP", "A1", "BETA", "P"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns after standardization: {missing}")

    # Basic QC
    df = df.dropna(subset=["SNP", "BETA", "P"])
    df = df.drop_duplicates(subset=["SNP"], keep="first")
    df = df[df["P"] > 0]  # remove invalid P-values

    return df.reset_index(drop=True)


def ld_clump(
    gwas_df: pd.DataFrame,
    ld_matrix: pd.DataFrame | None = None,
    r2_threshold: float = DEFAULT_CLUMP_R2,
    kb_window: int = DEFAULT_CLUMP_KB,
) -> pd.DataFrame:
    """Greedy LD clumping to select independent SNPs.

    Sorts SNPs by P-value, then greedily removes SNPs in LD (r² > threshold)
    within a genomic window.

    Parameters
    ----------
    gwas_df : pd.DataFrame
        GWAS results with SNP, BETA, P columns. Optionally CHR, BP for
        position-based windowing.
    ld_matrix : pd.DataFrame | None
        Pairwise LD (r²) matrix indexed by SNP ID. If None, uses position-
        based proxy (SNPs within kb_window are assumed correlated).
    r2_threshold : float
        LD r² threshold for clumping (default: 0.1).
    kb_window : int
        Clumping window in kilobases (default: 250).

    Returns
    -------
    DataFrame with clumped (independent) SNPs only.
    """
    df = gwas_df.sort_values("P").reset_index(drop=True)
    keep = np.ones(len(df), dtype=bool)

    if ld_matrix is not None:
        # Use actual LD matrix
        snp_to_idx = {s: i for i, s in enumerate(df["SNP"])}
        for i in range(len(df)):
            if not keep[i]:
                continue
            snp_i = df.loc[i, "SNP"]
            for j in range(i + 1, len(df)):
                if not keep[j]:
                    continue
                snp_j = df.loc[j, "SNP"]
                if snp_i in ld_matrix.index and snp_j in ld_matrix.columns:
                    r2 = ld_matrix.loc[snp_i, snp_j] ** 2
                    if r2 > r2_threshold:
                        keep[j] = False
    elif "CHR" in df.columns and "BP" in df.columns:
        # Position-based windowing
        for i in range(len(df)):
            if not keep[i]:
                continue
            chr_i = df.loc[i, "CHR"]
            bp_i = df.loc[i, "BP"]
            for j in range(i + 1, len(df)):
                if not keep[j]:
                    continue
                if df.loc[j, "CHR"] == chr_i:
                    dist_kb = abs(df.loc[j, "BP"] - bp_i) / 1000
                    if dist_kb < kb_window:
                        keep[j] = False
    else:
        # No LD info: keep all (user responsibility to provide LD data)
        logger.warning("No LD information available — skipping clumping")

    n_removed = int((~keep).sum())
    logger.info(
        "LD clumping: %d → %d SNPs (removed %d, r²>%.2f)",
        len(df),
        int(keep.sum()),
        n_removed,
        r2_threshold,
    )
    return df[keep].reset_index(drop=True)


def threshold_snps(
    gwas_df: pd.DataFrame,
    p_threshold: float,
) -> pd.DataFrame:
    """Select SNPs below a P-value threshold."""
    mask = gwas_df["P"] <= p_threshold
    return gwas_df[mask].reset_index(drop=True)


def compute_prs_weights(
    gwas_df: pd.DataFrame,
    stratum: str,
    p_thresholds: list[float] | None = None,
    ld_matrix: pd.DataFrame | None = None,
    clump_r2: float = DEFAULT_CLUMP_R2,
    clump_kb: int = DEFAULT_CLUMP_KB,
) -> list[PRSWeights]:
    """Compute PRS weights at multiple P-value thresholds after LD clumping.

    Parameters
    ----------
    gwas_df : pd.DataFrame
        Factor-specific GWAS results.
    stratum : str
        Name of PRS stratum (e.g., "compulsive", "neurodevelopmental").
    p_thresholds : list[float] | None
        P-value thresholds. Defaults to DEFAULT_P_THRESHOLDS.
    ld_matrix : pd.DataFrame | None
        LD matrix for clumping.
    clump_r2 : float
        LD r² threshold for clumping.
    clump_kb : int
        Clumping window in kb.

    Returns
    -------
    List of PRSWeights, one per threshold.
    """
    if p_thresholds is None:
        p_thresholds = DEFAULT_P_THRESHOLDS

    # Clump first (clumping is threshold-independent)
    clumped = ld_clump(gwas_df, ld_matrix, clump_r2, clump_kb)

    weights_list = []
    for pt in sorted(p_thresholds):
        selected = threshold_snps(clumped, pt)
        if len(selected) == 0:
            logger.debug("No SNPs at P<%g for stratum %s", pt, stratum)
            continue

        a1 = selected["A1"].tolist() if "A1" in selected.columns else ["A"] * len(selected)

        weights_list.append(
            PRSWeights(
                stratum=stratum,
                p_threshold=pt,
                snp_ids=selected["SNP"].tolist(),
                weights=selected["BETA"].values.copy(),
                effect_alleles=a1,
            )
        )
        logger.info(
            "Stratum %s, P<%g: %d SNPs selected",
            stratum,
            pt,
            len(selected),
        )

    return weights_list


def score_individuals(
    genotypes: pd.DataFrame,
    weights: PRSWeights,
) -> np.ndarray:
    """Compute PRS for each individual given genotype dosages and weights.

    Parameters
    ----------
    genotypes : pd.DataFrame
        Allele dosage matrix: rows = individuals, columns = SNP IDs.
        Values are dosages of effect allele (0, 1, 2 or continuous for imputed).
    weights : PRSWeights
        SNP weights from compute_prs_weights.

    Returns
    -------
    Array of PRS values, one per individual.
    """
    # Find overlapping SNPs
    shared = [s for s in weights.snp_ids if s in genotypes.columns]
    if not shared:
        logger.warning("No overlapping SNPs between weights and genotypes for %s", weights.stratum)
        return np.zeros(len(genotypes))

    # Align weights to shared SNPs
    weight_idx = [weights.snp_ids.index(s) for s in shared]
    w = weights.weights[weight_idx]

    # Score: sum(dosage_j * weight_j) for each individual
    dosage = genotypes[shared].values
    scores = dosage @ w

    if len(shared) < len(weights.snp_ids):
        logger.warning(
            "Only %d/%d SNPs found in genotypes for %s",
            len(shared),
            len(weights.snp_ids),
            weights.stratum,
        )

    return scores


def evaluate_prs(
    scores: np.ndarray,
    phenotype: np.ndarray,
    covariates: np.ndarray | None = None,
    stratum: str = "",
    p_threshold: float = 0.0,
) -> PRSResult:
    """Evaluate PRS predictive performance via incremental R².

    Parameters
    ----------
    scores : np.ndarray
        PRS values for each individual.
    phenotype : np.ndarray
        Phenotype values (binary or continuous).
    covariates : np.ndarray | None
        Covariate matrix (e.g., PCs, age, sex). If provided, computes
        incremental R² above covariates.
    stratum : str
        Label for the PRS stratum.
    p_threshold : float
        P-value threshold used for this PRS.

    Returns
    -------
    PRSResult with R², P-value, and descriptive statistics.
    """
    valid = np.isfinite(scores) & np.isfinite(phenotype)
    if covariates is not None:
        valid &= np.all(np.isfinite(covariates), axis=1)

    scores_v = scores[valid]
    pheno_v = phenotype[valid]
    n = len(scores_v)

    if n < 10:
        logger.warning("Too few valid samples (%d) for PRS evaluation", n)
        return PRSResult(
            stratum=stratum,
            p_threshold=p_threshold,
            n_snps=0,
            r_squared=0.0,
            r_squared_se=0.0,
            p_value=1.0,
            mean_score=0.0,
            sd_score=0.0,
        )

    if covariates is not None:
        cov_v = covariates[valid]
        # Incremental R²: R²(cov+PRS) - R²(cov)
        # Full model
        X_full = np.column_stack([cov_v, scores_v])
        X_full = np.column_stack([np.ones(n), X_full])
        try:
            beta_full = np.linalg.lstsq(X_full, pheno_v, rcond=None)[0]
            resid_full = pheno_v - X_full @ beta_full
            ss_res_full = float(np.sum(resid_full**2))
        except np.linalg.LinAlgError:
            ss_res_full = float(np.sum((pheno_v - pheno_v.mean()) ** 2))

        # Covariate-only model
        X_cov = np.column_stack([np.ones(n), cov_v])
        try:
            beta_cov = np.linalg.lstsq(X_cov, pheno_v, rcond=None)[0]
            resid_cov = pheno_v - X_cov @ beta_cov
            ss_res_cov = float(np.sum(resid_cov**2))
        except np.linalg.LinAlgError:
            ss_res_cov = float(np.sum((pheno_v - pheno_v.mean()) ** 2))

        ss_tot = float(np.sum((pheno_v - pheno_v.mean()) ** 2))
        r2_full = 1 - ss_res_full / ss_tot if ss_tot > 0 else 0
        r2_cov = 1 - ss_res_cov / ss_tot if ss_tot > 0 else 0
        r2 = max(r2_full - r2_cov, 0)
    else:
        # Simple correlation-based R²
        corr = np.corrcoef(scores_v, pheno_v)[0, 1]
        r2 = corr**2

    # P-value from F-test for R² significance
    if r2 > 0 and n > 2:
        f_stat = r2 * (n - 2) / max(1 - r2, 1e-15)
        pval = float(stats.f.sf(f_stat, 1, n - 2))
    else:
        pval = 1.0

    # Bootstrap SE of R² (fast approximation)
    r2_se = np.sqrt(4 * r2 * (1 - r2) ** 2 / max(n - 3, 1)) if n > 3 else 0.0

    return PRSResult(
        stratum=stratum,
        p_threshold=p_threshold,
        n_snps=len(scores_v),
        r_squared=float(r2),
        r_squared_se=float(r2_se),
        p_value=pval,
        mean_score=float(np.mean(scores_v)),
        sd_score=float(np.std(scores_v)),
    )


def compare_stratified_vs_aggregate(
    strata_scores: dict[str, np.ndarray],
    aggregate_scores: np.ndarray | None,
    phenotype: np.ndarray,
    covariates: np.ndarray | None = None,
    strata_p_thresholds: dict[str, float] | None = None,
) -> StratifiedPRSComparison:
    """Compare stratified PRS (joint model) vs. aggregate PRS.

    Parameters
    ----------
    strata_scores : dict[str, np.ndarray]
        PRS scores per stratum (stratum_name -> scores).
    aggregate_scores : np.ndarray | None
        Unstratified aggregate PRS scores.
    phenotype : np.ndarray
        Phenotype values.
    covariates : np.ndarray | None
        Covariates for incremental R².
    strata_p_thresholds : dict[str, float] | None
        P-value threshold used for each stratum.

    Returns
    -------
    StratifiedPRSComparison with per-stratum and joint R².
    """
    p_thresholds = strata_p_thresholds or {}

    # Per-stratum R²
    strata_results = []
    for name, scores in strata_scores.items():
        pt = p_thresholds.get(name, 0.0)
        result = evaluate_prs(scores, phenotype, covariates, name, pt)
        strata_results.append(result)

    # Combined R² (all strata as predictors jointly)
    valid = np.isfinite(phenotype)
    all_scores = []
    for scores in strata_scores.values():
        valid &= np.isfinite(scores)
        all_scores.append(scores)

    n = int(valid.sum())
    combined_r2 = 0.0
    combined_pval = 1.0

    if n > 10 and all_scores:
        pheno_v = phenotype[valid]
        X = np.column_stack([s[valid] for s in all_scores])

        if covariates is not None:
            cov_v = covariates[valid]
            X_full = np.column_stack([np.ones(n), cov_v, X])
            X_cov = np.column_stack([np.ones(n), cov_v])
        else:
            X_full = np.column_stack([np.ones(n), X])
            X_cov = np.ones((n, 1))

        try:
            beta_full = np.linalg.lstsq(X_full, pheno_v, rcond=None)[0]
            ss_res_full = float(np.sum((pheno_v - X_full @ beta_full) ** 2))
            beta_cov = np.linalg.lstsq(X_cov, pheno_v, rcond=None)[0]
            ss_res_cov = float(np.sum((pheno_v - X_cov @ beta_cov) ** 2))
            ss_tot = float(np.sum((pheno_v - pheno_v.mean()) ** 2))

            r2_full = 1 - ss_res_full / ss_tot if ss_tot > 0 else 0
            r2_cov = 1 - ss_res_cov / ss_tot if ss_tot > 0 else 0
            combined_r2 = max(r2_full - r2_cov, 0)

            # F-test for joint significance
            k = len(all_scores)
            df1 = k
            df2 = n - X_full.shape[1]
            if df2 > 0 and ss_res_full > 0:
                f_stat = ((ss_res_cov - ss_res_full) / df1) / (ss_res_full / df2)
                combined_pval = float(stats.f.sf(max(f_stat, 0), df1, df2))
        except np.linalg.LinAlgError:
            pass

    # Aggregate R²
    aggregate_result = None
    if aggregate_scores is not None:
        aggregate_result = evaluate_prs(
            aggregate_scores, phenotype, covariates, "aggregate", 0.0
        )

    improvement = combined_r2 - (aggregate_result.r_squared if aggregate_result else 0)

    return StratifiedPRSComparison(
        strata_results=strata_results,
        aggregate_result=aggregate_result,
        combined_r_squared=combined_r2,
        combined_p_value=combined_pval,
        r_squared_improvement=improvement,
    )


def run_stratified_prs(
    factor_gwas_dir: Path,
    target_genotypes: Path | None = None,
    output_dir: Path | None = None,
    ld_matrix_path: Path | None = None,
    p_thresholds: list[float] | None = None,
    clump_r2: float = DEFAULT_CLUMP_R2,
    clump_kb: int = DEFAULT_CLUMP_KB,
) -> StratifiedPRSComparison | None:
    """Run the full stratified PRS pipeline.

    Parameters
    ----------
    factor_gwas_dir : Path
        Directory containing factor-specific GWAS files.
        Expected files: compulsive.tsv, neurodevelopmental.tsv, ts_specific.tsv
        Optional: aggregate.tsv (unstratified TS GWAS)
    target_genotypes : Path | None
        Path to target genotype file (TSV: rows=individuals, cols=SNPs).
    output_dir : Path | None
        Output directory.
    ld_matrix_path : Path | None
        Path to LD matrix file.
    p_thresholds : list[float] | None
        P-value thresholds for C+T.
    clump_r2 : float
        LD r² clumping threshold.
    clump_kb : int
        Clumping window in kb.

    Returns
    -------
    StratifiedPRSComparison if target genotypes provided, else None.
    """
    if output_dir is None:
        output_dir = REPO_ROOT / "output" / "tourettes" / "ts-comorbidity-genetic-architecture" / "phase3"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover factor GWAS files
    strata_names = {
        "compulsive": ["compulsive.tsv", "compulsive_factor.tsv"],
        "neurodevelopmental": ["neurodevelopmental.tsv", "neurodevelopmental_factor.tsv"],
        "ts_specific": ["ts_specific.tsv", "ts_residual.tsv", "residual.tsv"],
    }

    ld_matrix = None
    if ld_matrix_path and ld_matrix_path.exists():
        logger.info("Loading LD matrix from %s", ld_matrix_path)
        ld_matrix = pd.read_csv(ld_matrix_path, sep="\t", index_col=0)

    # Compute weights for each stratum
    all_weights: dict[str, list[PRSWeights]] = {}

    for stratum, filenames in strata_names.items():
        gwas_path = None
        for fn in filenames:
            candidate = factor_gwas_dir / fn
            if candidate.exists():
                gwas_path = candidate
                break

        if gwas_path is None:
            logger.warning("No GWAS file found for stratum %s in %s", stratum, factor_gwas_dir)
            continue

        logger.info("Loading %s GWAS from %s", stratum, gwas_path)
        gwas_df = load_factor_gwas(gwas_path)
        weights = compute_prs_weights(
            gwas_df, stratum, p_thresholds, ld_matrix, clump_r2, clump_kb
        )
        all_weights[stratum] = weights

        # Write PRS weights files
        for w in weights:
            wt_df = pd.DataFrame({
                "SNP": w.snp_ids,
                "A1": w.effect_alleles,
                "WEIGHT": w.weights,
            })
            safe_pt = f"{w.p_threshold:.0e}".replace("+", "")
            fname = f"prs_weights_{stratum}_p{safe_pt}.tsv"
            wt_df.to_csv(output_dir / fname, sep="\t", index=False, float_format="%.6f")

    if not all_weights:
        logger.error("No factor GWAS files found in %s", factor_gwas_dir)
        sys.exit(1)

    logger.info("Computed PRS weights for %d strata", len(all_weights))

    # If target genotypes provided, score and evaluate
    comparison = None
    if target_genotypes and target_genotypes.exists():
        logger.info("Loading target genotypes from %s", target_genotypes)
        geno_df = pd.read_csv(target_genotypes, sep="\t", index_col=0)

        # Expect a phenotype column or separate file
        phenotype = None
        pheno_path = target_genotypes.parent / "phenotype.tsv"
        if pheno_path.exists():
            pheno_df = pd.read_csv(pheno_path, sep="\t", index_col=0)
            if "PHENO" in pheno_df.columns:
                phenotype = pheno_df["PHENO"].values
        elif "PHENO" in geno_df.columns:
            phenotype = geno_df.pop("PHENO").values

        if phenotype is None:
            logger.warning("No phenotype data — skipping R² evaluation")
        else:
            # Score at best threshold per stratum (most SNPs passing)
            best_scores: dict[str, np.ndarray] = {}
            best_thresholds: dict[str, float] = {}

            for stratum, weights_list in all_weights.items():
                if not weights_list:
                    continue
                # Use the threshold with most SNPs as default
                best_w = max(weights_list, key=lambda w: w.n_snps)
                scores = score_individuals(geno_df, best_w)
                best_scores[stratum] = scores
                best_thresholds[stratum] = best_w.p_threshold

                # Write per-individual scores
                score_df = pd.DataFrame({
                    "IID": geno_df.index,
                    f"PRS_{stratum}": scores,
                })
                score_df.to_csv(
                    output_dir / f"scores_{stratum}.tsv",
                    sep="\t",
                    index=False,
                    float_format="%.6f",
                )

            # Aggregate PRS (if available)
            aggregate_scores = None
            agg_path = None
            for fn in ["aggregate.tsv", "ts_aggregate.tsv"]:
                candidate = factor_gwas_dir / fn
                if candidate.exists():
                    agg_path = candidate
                    break

            if agg_path:
                agg_gwas = load_factor_gwas(agg_path)
                agg_weights = compute_prs_weights(
                    agg_gwas, "aggregate", p_thresholds, ld_matrix, clump_r2, clump_kb
                )
                if agg_weights:
                    best_agg = max(agg_weights, key=lambda w: w.n_snps)
                    aggregate_scores = score_individuals(geno_df, best_agg)

            # Compare
            comparison = compare_stratified_vs_aggregate(
                best_scores, aggregate_scores, phenotype, strata_p_thresholds=best_thresholds
            )

            # Write comparison results
            _write_comparison(comparison, output_dir)

    # Write summary of weights across all thresholds
    _write_weights_summary(all_weights, output_dir)

    return comparison


def _write_weights_summary(
    all_weights: dict[str, list[PRSWeights]],
    output_dir: Path,
) -> None:
    """Write a summary table of PRS weights across strata and thresholds."""
    rows = []
    for stratum, weights_list in all_weights.items():
        for w in weights_list:
            rows.append({
                "stratum": w.stratum,
                "p_threshold": w.p_threshold,
                "n_snps": w.n_snps,
                "mean_abs_weight": float(np.mean(np.abs(w.weights))) if w.n_snps > 0 else 0,
                "max_abs_weight": float(np.max(np.abs(w.weights))) if w.n_snps > 0 else 0,
            })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "prs_weights_summary.tsv", sep="\t", index=False, float_format="%.6f")
        logger.info("Wrote PRS weights summary to %s", output_dir / "prs_weights_summary.tsv")


def _write_comparison(
    comparison: StratifiedPRSComparison,
    output_dir: Path,
) -> None:
    """Write PRS comparison results to output files."""
    # Per-stratum results
    rows = []
    for r in comparison.strata_results:
        rows.append({
            "stratum": r.stratum,
            "p_threshold": r.p_threshold,
            "n_snps": r.n_snps,
            "r_squared": r.r_squared,
            "r_squared_se": r.r_squared_se,
            "p_value": r.p_value,
            "mean_score": r.mean_score,
            "sd_score": r.sd_score,
        })

    if comparison.aggregate_result:
        r = comparison.aggregate_result
        rows.append({
            "stratum": "aggregate",
            "p_threshold": r.p_threshold,
            "n_snps": r.n_snps,
            "r_squared": r.r_squared,
            "r_squared_se": r.r_squared_se,
            "p_value": r.p_value,
            "mean_score": r.mean_score,
            "sd_score": r.sd_score,
        })

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "prs_r_squared.tsv", sep="\t", index=False, float_format="%.6f")

    # Summary comparison
    summary = {
        "combined_r_squared": comparison.combined_r_squared,
        "combined_p_value": comparison.combined_p_value,
        "aggregate_r_squared": comparison.aggregate_result.r_squared if comparison.aggregate_result else None,
        "r_squared_improvement": comparison.r_squared_improvement,
        "n_strata": len(comparison.strata_results),
    }
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / "prs_comparison_summary.tsv", sep="\t", index=False, float_format="%.6f")
    logger.info(
        "Stratified PRS R²=%.4f vs aggregate R²=%.4f (improvement: %.4f)",
        comparison.combined_r_squared,
        comparison.aggregate_result.r_squared if comparison.aggregate_result else 0,
        comparison.r_squared_improvement,
    )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the stratified PRS pipeline."""
    parser = argparse.ArgumentParser(
        description="Construct stratified PRS from factor-specific GWAS results."
    )
    parser.add_argument(
        "--factor-gwas-dir",
        type=Path,
        required=True,
        help="Directory containing factor-specific GWAS files (compulsive.tsv, etc.).",
    )
    parser.add_argument(
        "--target-genotypes",
        type=Path,
        default=None,
        help="Path to target genotype dosage file (TSV: rows=individuals, cols=SNPs).",
    )
    parser.add_argument(
        "--ld-matrix",
        type=Path,
        default=None,
        help="Path to LD matrix file for clumping (TSV, SNP x SNP).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "tourettes" / "ts-comorbidity-genetic-architecture" / "phase3",
        help="Output directory.",
    )
    parser.add_argument(
        "--clump-r2",
        type=float,
        default=DEFAULT_CLUMP_R2,
        help=f"LD r² threshold for clumping (default: {DEFAULT_CLUMP_R2}).",
    )
    parser.add_argument(
        "--clump-kb",
        type=int,
        default=DEFAULT_CLUMP_KB,
        help=f"Clumping window in kb (default: {DEFAULT_CLUMP_KB}).",
    )
    parser.add_argument(
        "--p-thresholds",
        type=float,
        nargs="*",
        default=None,
        help="P-value thresholds for C+T (default: 5e-8 1e-5 1e-3 0.01 0.05 0.1 0.5 1.0).",
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

    if not args.factor_gwas_dir.exists():
        logger.error("Factor GWAS directory not found: %s", args.factor_gwas_dir)
        sys.exit(1)

    run_stratified_prs(
        factor_gwas_dir=args.factor_gwas_dir,
        target_genotypes=args.target_genotypes,
        output_dir=args.output_dir,
        ld_matrix_path=args.ld_matrix,
        p_thresholds=args.p_thresholds,
        clump_r2=args.clump_r2,
        clump_kb=args.clump_kb,
    )


if __name__ == "__main__":
    main()
