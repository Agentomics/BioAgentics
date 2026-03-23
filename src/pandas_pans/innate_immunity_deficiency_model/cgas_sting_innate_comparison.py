"""Compare innate immune profiles in patients with cGAS-STING pathway variants.

Task 105: Compare innate immune gene expression in patients with TREX1/SAMHD1
(cGAS-STING pathway) variants vs those without.

Since per-patient WES genotypes are not available, this uses a group-level proxy:
PANS pre-IVIG patients (Vettiatil 2026 cohort has pathogenic TREX1/SAMHD1 variants)
vs healthy controls. Tests cGAS-STING pathway gene expression and interferon-
stimulated gene (ISG) output as functional readout.

Data sources:
  - innate_adaptive_ratio_bulk.csv (module-level cGAS-STING sums)
  - innate_adaptive_ratio_scrna.csv (module-level cGAS-STING sums)
  - GSE278678_pans_ivig_counts.csv.gz (per-gene bulk counts for cGAS-STING genes)

Usage::

    uv run python -m pandas_pans.innate_immunity_deficiency_model.cgas_sting_innate_comparison
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import DATA_DIR, REPO_ROOT
from bioagentics.pandas_pans.innate_immunity_modules import (
    CGAS_STING_GENES,
    INNATE_MODULES,
    get_all_innate_genes,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "innate-immunity-deficiency-model"
BULK_PATH = DATA_DIR / "pandas_pans" / "ivig-mechanism-single-cell-analysis" / "bulk" / "GSE278678_pans_ivig_counts.csv.gz"


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation (effect size for Mann-Whitney U)."""
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def _assign_condition(sample: str) -> str:
    """Map sample name to condition."""
    if sample.startswith("Control"):
        return "control"
    elif sample.startswith("PreIVIG"):
        return "pre"
    elif sample.startswith("PostIVIG"):
        return "post"
    elif sample.startswith("Secondpost"):
        return "second_post"
    return "unknown"


def test_cgas_module_level() -> pd.DataFrame:
    """Test cGAS-STING module expression: PANS pre-IVIG vs controls.

    Uses per-sample module sums from the innate/adaptive ratio data.
    """
    results: list[dict] = []

    for source_name, filename in [("bulk", "innate_adaptive_ratio_bulk.csv"),
                                   ("scrna_pseudobulk", "innate_adaptive_ratio_scrna.csv")]:
        path = OUTPUT_DIR / filename
        if not path.exists():
            continue

        df = pd.read_csv(path)
        pre = df[df["condition"] == "pre"]["innate_cgas_sting_pathway"].values
        ctrl = df[df["condition"] == "control"]["innate_cgas_sting_pathway"].values

        if len(pre) < 2 or len(ctrl) < 2:
            continue

        u_stat, p_val = stats.mannwhitneyu(pre, ctrl, alternative="two-sided")
        r_bs = _rank_biserial(u_stat, len(pre), len(ctrl))

        results.append({
            "data_source": source_name,
            "module": "cgas_sting_pathway",
            "comparison": "pre_vs_control",
            "n_pans": len(pre),
            "n_control": len(ctrl),
            "mean_pans": float(np.mean(pre)),
            "mean_control": float(np.mean(ctrl)),
            "fold_change": float(np.mean(pre) / np.mean(ctrl)) if np.mean(ctrl) > 0 else np.nan,
            "U_statistic": float(u_stat),
            "p_value": float(p_val),
            "rank_biserial_r": float(r_bs),
            "direction": "pans_higher" if np.mean(pre) > np.mean(ctrl) else "pans_lower",
        })

    return pd.DataFrame(results)


def test_per_gene_cgas_bulk() -> pd.DataFrame:
    """Test per-gene cGAS-STING pathway expression in bulk RNA-seq.

    Extracts cGAS-STING genes from the bulk count matrix and compares
    PANS pre-IVIG vs controls.
    """
    if not BULK_PATH.exists():
        logger.warning("Bulk counts file not found: %s", BULK_PATH)
        return pd.DataFrame()

    df = pd.read_csv(BULK_PATH)
    sample_cols = [c for c in df.columns if c not in ("Unnamed: 0", "ENTREZID", "SYMBOL")]

    results: list[dict] = []

    for gene in CGAS_STING_GENES:
        gene_row = df[df["SYMBOL"] == gene]
        if gene_row.empty:
            continue

        # Extract per-sample expression
        pre_vals = []
        ctrl_vals = []
        for sample in sample_cols:
            val = float(gene_row[sample].iloc[0])
            condition = _assign_condition(sample)
            if condition == "pre":
                pre_vals.append(val)
            elif condition == "control":
                ctrl_vals.append(val)

        pre = np.array(pre_vals)
        ctrl = np.array(ctrl_vals)

        if len(pre) < 2 or len(ctrl) < 2:
            continue

        u_stat, p_val = stats.mannwhitneyu(pre, ctrl, alternative="two-sided")
        r_bs = _rank_biserial(u_stat, len(pre), len(ctrl))

        results.append({
            "gene": gene,
            "data_source": "bulk",
            "n_pans": len(pre),
            "n_control": len(ctrl),
            "mean_pans": float(np.mean(pre)),
            "mean_control": float(np.mean(ctrl)),
            "fold_change": float(np.mean(pre) / np.mean(ctrl)) if np.mean(ctrl) > 0 else np.nan,
            "U_statistic": float(u_stat),
            "p_value": float(p_val),
            "rank_biserial_r": float(r_bs),
            "direction": "pans_higher" if np.mean(pre) > np.mean(ctrl) else "pans_lower",
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty and len(result_df) > 1:
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(result_df["p_value"].values, method="fdr_bh")
        result_df["fdr"] = fdr

    return result_df


def test_innate_module_comparison() -> pd.DataFrame:
    """Compare all innate modules between PANS and controls.

    Tests whether cGAS-STING pathway shows a distinct pattern compared
    to other innate modules (e.g., lectin complement, trained immunity).
    """
    results: list[dict] = []

    for source_name, filename in [("bulk", "innate_adaptive_ratio_bulk.csv"),
                                   ("scrna_pseudobulk", "innate_adaptive_ratio_scrna.csv")]:
        path = OUTPUT_DIR / filename
        if not path.exists():
            continue

        df = pd.read_csv(path)

        for mod_name in INNATE_MODULES:
            col = f"innate_{mod_name}"
            if col not in df.columns:
                continue

            pre = df[df["condition"] == "pre"][col].values
            ctrl = df[df["condition"] == "control"][col].values

            if len(pre) < 2 or len(ctrl) < 2:
                continue

            u_stat, p_val = stats.mannwhitneyu(pre, ctrl, alternative="two-sided")
            r_bs = _rank_biserial(u_stat, len(pre), len(ctrl))

            results.append({
                "data_source": source_name,
                "module": mod_name,
                "n_pans": len(pre),
                "n_control": len(ctrl),
                "mean_pans": float(np.mean(pre)),
                "mean_control": float(np.mean(ctrl)),
                "fold_change": float(np.mean(pre) / np.mean(ctrl)) if np.mean(ctrl) > 0 else np.nan,
                "U_statistic": float(u_stat),
                "p_value": float(p_val),
                "rank_biserial_r": float(r_bs),
                "direction": "pans_higher" if np.mean(pre) > np.mean(ctrl) else "pans_lower",
            })

    result_df = pd.DataFrame(results)
    if not result_df.empty:
        # FDR correction within each data source
        for source in result_df["data_source"].unique():
            mask = result_df["data_source"] == source
            p_vals = result_df.loc[mask, "p_value"].values
            if len(p_vals) > 1:
                from statsmodels.stats.multitest import multipletests
                _, fdr, _, _ = multipletests(p_vals, method="fdr_bh")
                result_df.loc[mask, "fdr"] = fdr

    return result_df


def run_cgas_sting_comparison() -> dict[str, Path]:
    """Run full cGAS-STING innate profile comparison pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # 1. Module-level cGAS-STING test
    logger.info("=== cGAS-STING module-level comparison ===")
    module_results = test_cgas_module_level()
    if not module_results.empty:
        path = OUTPUT_DIR / "cgas_sting_module_assoc.csv"
        module_results.to_csv(path, index=False)
        outputs["module_assoc"] = path
        logger.info("Module-level results:\n%s",
                     module_results[["data_source", "mean_pans", "mean_control",
                                     "fold_change", "p_value", "rank_biserial_r",
                                     "direction"]].to_string(index=False))

    # 2. Per-gene bulk test
    logger.info("\n=== Per-gene cGAS-STING bulk expression ===")
    gene_results = test_per_gene_cgas_bulk()
    if not gene_results.empty:
        path = OUTPUT_DIR / "cgas_sting_gene_bulk_assoc.csv"
        gene_results.to_csv(path, index=False)
        outputs["gene_bulk_assoc"] = path
        logger.info("Per-gene bulk results:\n%s",
                     gene_results[["gene", "mean_pans", "mean_control",
                                   "fold_change", "p_value",
                                   "direction"]].to_string(index=False))

    # 3. All innate modules comparison
    logger.info("\n=== All innate modules: PANS vs controls ===")
    all_modules = test_innate_module_comparison()
    if not all_modules.empty:
        path = OUTPUT_DIR / "innate_module_pans_vs_control.csv"
        all_modules.to_csv(path, index=False)
        outputs["all_modules"] = path
        logger.info("All modules (bulk):\n%s",
                     all_modules[all_modules["data_source"] == "bulk"][
                         ["module", "fold_change", "p_value", "fdr",
                          "direction"]].to_string(index=False))
        logger.info("\nAll modules (scRNA):\n%s",
                     all_modules[all_modules["data_source"] == "scrna_pseudobulk"][
                         ["module", "fold_change", "p_value", "fdr",
                          "direction"]].to_string(index=False))

    # 4. Summary
    logger.info("\n=== Summary ===")
    logger.info("TREX1 and SAMHD1 carry pathogenic variants in Vettiatil 2026 "
                "PANS cohort. Loss-of-function in these genes activates "
                "cGAS-STING, causing constitutive type I IFN production. "
                "Expression differences in this pathway between PANS and "
                "controls may reflect downstream effects of these variants.")

    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_cgas_sting_comparison()
