"""Test lectin complement variant to innate immune expression association.

Task 104: Do patients from a variant-enriched PANS cohort show altered lectin
complement pathway expression compared to controls?

Since per-patient WES genotypes are not available (Vettiatil 2026 supplementary
data not yet parsed), this uses a group-level proxy: PANS pre-IVIG patients
(known to be enriched for MBL2/MASP1/MASP2 variants, FDR=1.12e-5) vs healthy
controls, testing for differences in lectin complement gene expression.

Data sources:
  - lectin_complement_scrna_expression.csv (per-gene, per-cell-type, per-patient)
  - lectin_complement_bulk_expression.csv (per-gene, per-sample)
  - innate_adaptive_ratio_scrna.csv (module-level sums)
  - innate_adaptive_ratio_bulk.csv (module-level sums)

Usage::

    uv run python -m pandas_pans.innate_immunity_deficiency_model.lectin_variant_expression_assoc
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.pandas_pans.innate_immunity_modules import LECTIN_COMPLEMENT_GENES

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "innate-immunity-deficiency-model"


def _rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Compute rank-biserial correlation (effect size for Mann-Whitney U)."""
    return 1.0 - (2.0 * u_stat) / (n1 * n2)


def test_module_level_association() -> pd.DataFrame:
    """Test lectin complement module expression: PANS pre-IVIG vs controls.

    Uses per-sample module sums from the innate/adaptive ratio pipeline.
    Tests both bulk and scRNA-seq data.
    """
    results: list[dict] = []

    # --- Bulk ---
    bulk_path = OUTPUT_DIR / "innate_adaptive_ratio_bulk.csv"
    if bulk_path.exists():
        bulk = pd.read_csv(bulk_path)
        pre = bulk[bulk["condition"] == "pre"]["innate_lectin_complement"].values
        ctrl = bulk[bulk["condition"] == "control"]["innate_lectin_complement"].values

        if len(pre) >= 2 and len(ctrl) >= 2:
            u_stat, p_val = stats.mannwhitneyu(pre, ctrl, alternative="two-sided")
            r_bs = _rank_biserial(u_stat, len(pre), len(ctrl))
            results.append({
                "data_source": "bulk",
                "module": "lectin_complement",
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

    # --- scRNA-seq ---
    scrna_path = OUTPUT_DIR / "innate_adaptive_ratio_scrna.csv"
    if scrna_path.exists():
        scrna = pd.read_csv(scrna_path)
        pre = scrna[scrna["condition"] == "pre"]["innate_lectin_complement"].values
        ctrl = scrna[scrna["condition"] == "control"]["innate_lectin_complement"].values

        if len(pre) >= 2 and len(ctrl) >= 2:
            u_stat, p_val = stats.mannwhitneyu(pre, ctrl, alternative="two-sided")
            r_bs = _rank_biserial(u_stat, len(pre), len(ctrl))
            results.append({
                "data_source": "scrna_pseudobulk",
                "module": "lectin_complement",
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


def test_per_gene_bulk_association() -> pd.DataFrame:
    """Test per-gene lectin complement expression in bulk: PANS pre vs controls."""
    bulk_path = OUTPUT_DIR / "lectin_complement_bulk_expression.csv"
    if not bulk_path.exists():
        logger.warning("Bulk expression file not found")
        return pd.DataFrame()

    bulk = pd.read_csv(bulk_path)
    results: list[dict] = []

    for gene in LECTIN_COMPLEMENT_GENES:
        gene_data = bulk[bulk["gene"] == gene]
        if gene_data.empty:
            continue

        pre = gene_data[gene_data["condition"] == "pre"]["raw_count"].values
        ctrl = gene_data[gene_data["condition"] == "control"]["raw_count"].values

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

    return pd.DataFrame(results)


def test_per_gene_scrna_association() -> pd.DataFrame:
    """Test per-gene lectin complement expression in scRNA-seq per cell type.

    Uses per-patient, per-cell-type mean expression from the scRNA-seq
    extraction output.
    """
    scrna_path = OUTPUT_DIR / "lectin_complement_scrna_expression.csv"
    if not scrna_path.exists():
        logger.warning("scRNA expression file not found")
        return pd.DataFrame()

    scrna = pd.read_csv(scrna_path)
    results: list[dict] = []

    for gene in LECTIN_COMPLEMENT_GENES:
        gene_data = scrna[scrna["gene"] == gene]
        if gene_data.empty:
            continue

        for cell_type in gene_data["cell_type"].unique():
            ct_data = gene_data[gene_data["cell_type"] == cell_type]
            pre = ct_data[ct_data["condition"] == "pre"]["mean_expr"].values
            ctrl = ct_data[ct_data["condition"] == "control"]["mean_expr"].values

            if len(pre) < 2 or len(ctrl) < 2:
                continue

            # Skip if all zeros
            if np.all(pre == 0) and np.all(ctrl == 0):
                continue

            u_stat, p_val = stats.mannwhitneyu(pre, ctrl, alternative="two-sided")
            r_bs = _rank_biserial(u_stat, len(pre), len(ctrl))

            results.append({
                "gene": gene,
                "cell_type": cell_type,
                "data_source": "scrna",
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
        # FDR correction across all scRNA tests
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(result_df["p_value"].values, method="fdr_bh")
        result_df["fdr"] = fdr

    return result_df


def run_lectin_variant_association() -> dict[str, Path]:
    """Run full lectin complement variant-expression association pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # 1. Module-level test
    logger.info("=== Module-level lectin complement association ===")
    module_results = test_module_level_association()
    if not module_results.empty:
        path = OUTPUT_DIR / "lectin_variant_module_assoc.csv"
        module_results.to_csv(path, index=False)
        outputs["module_assoc"] = path
        logger.info("Module-level results:\n%s",
                     module_results[["data_source", "mean_pans", "mean_control",
                                     "fold_change", "p_value", "rank_biserial_r",
                                     "direction"]].to_string(index=False))

    # 2. Per-gene bulk test
    logger.info("\n=== Per-gene bulk association ===")
    bulk_gene_results = test_per_gene_bulk_association()
    if not bulk_gene_results.empty:
        path = OUTPUT_DIR / "lectin_variant_gene_bulk_assoc.csv"
        bulk_gene_results.to_csv(path, index=False)
        outputs["gene_bulk_assoc"] = path
        logger.info("Per-gene bulk results:\n%s",
                     bulk_gene_results[["gene", "mean_pans", "mean_control",
                                        "fold_change", "p_value",
                                        "direction"]].to_string(index=False))

    # 3. Per-gene scRNA-seq test
    logger.info("\n=== Per-gene scRNA-seq association (by cell type) ===")
    scrna_gene_results = test_per_gene_scrna_association()
    if not scrna_gene_results.empty:
        path = OUTPUT_DIR / "lectin_variant_gene_scrna_assoc.csv"
        scrna_gene_results.to_csv(path, index=False)
        outputs["gene_scrna_assoc"] = path

        # Show significant results
        sig = scrna_gene_results[scrna_gene_results["fdr"] < 0.1]
        if not sig.empty:
            logger.info("Significant (FDR<0.1) scRNA results:\n%s",
                         sig[["gene", "cell_type", "mean_pans", "mean_control",
                              "fold_change", "p_value", "fdr",
                              "direction"]].to_string(index=False))
        else:
            logger.info("No FDR<0.1 results (n_patients small, expected)")
            top5 = scrna_gene_results.nsmallest(5, "p_value")
            logger.info("Top 5 by p-value:\n%s",
                         top5[["gene", "cell_type", "mean_pans", "mean_control",
                               "fold_change", "p_value",
                               "direction"]].to_string(index=False))

    # 4. Summary
    logger.info("\n=== Summary ===")
    logger.info("NOTE: This is a group-level proxy analysis. PANS patients are "
                "enriched for MBL2/MASP1/MASP2 variants (Vettiatil 2026, "
                "FDR=1.12e-5). Per-patient genotype-expression linkage requires "
                "individual WES data not yet available.")

    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_lectin_variant_association()
