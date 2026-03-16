"""Positive control validation for predicted NSCLC dependencies.

Validates predictions against known NSCLC biology: oncogene addiction
(KRAS, EGFR, ALK, ROS1, MET) and KL-subtype pathway annotations
informed by clinical trial outcomes.

Usage:
    from bioagentics.models.validation import validate_positive_controls
    report = validate_positive_controls(dep_matrix, subtypes, mutations)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# Oncogene-addiction positive controls: gene -> mutation column name
POSITIVE_CONTROLS = {
    "KRAS": "KRAS_mutated",
    "EGFR": "EGFR_mutated",
    "ALK": "ALK_mutated",
    "ROS1": "ROS1_mutated",
    "MET": "MET_mutated",
}

# KL-subtype pathway annotations (based on clinical evidence)
KL_POSITIVE_PATHWAYS = {
    "dual_ICB": {
        "genes": ["CTLA4", "CD80", "CD86"],
        "evidence": "POSEIDON 5-year: HR 0.57 for STK11-mutant with tremelimumab+durva+chemo",
    },
    "myeloid_markers": {
        "genes": ["CD14", "CD68", "ITGAM", "NOS2"],
        "evidence": "KL tumors show distinct myeloid-enriched TME",
    },
    "cd4_signaling": {
        "genes": ["CD4", "CD3E", "CD3D", "CD3G"],
        "evidence": "CD4 T-cell dysfunction in STK11-mutant NSCLC",
    },
}

KL_NEGATIVE_CONTROLS = {
    "mTOR_pathway": {
        "genes": ["MTOR", "RPS6KB1", "EIF4EBP1", "AKT1"],
        "flag": "clinically refuted - vistusertib Phase II failure",
    },
    "ATR_pathway": {
        "genes": ["ATR", "CHEK1", "ATRIP"],
        "flag": "biomarker-dependent, clinically unvalidated - LATIFY Phase III failure (ceralasertib + durvalumab, Dec 2025)",
    },
}

AUC_THRESHOLD = 0.75


@dataclass
class ValidationReport:
    """Results of positive control validation."""

    control_aucs: dict = field(default_factory=dict)       # gene -> AUC
    controls_passed: list[str] = field(default_factory=list)
    controls_failed: list[str] = field(default_factory=list)
    controls_skipped: list[str] = field(default_factory=list)
    kl_pathway_flags: list[dict] = field(default_factory=list)
    auc_threshold: float = AUC_THRESHOLD
    n_controls_tested: int = 0
    n_controls_passed: int = 0


def validate_positive_controls(
    dep_matrix: pd.DataFrame,
    subtypes: pd.Series,
    mutations: pd.DataFrame,
    auc_threshold: float = AUC_THRESHOLD,
) -> ValidationReport:
    """Validate predicted dependencies against known NSCLC biology.

    Parameters
    ----------
    dep_matrix : DataFrame (patients x genes)
        Predicted dependency scores.
    subtypes : Series
        Molecular subtype per patient.
    mutations : DataFrame
        Patient mutation data with bool columns (e.g. KRAS_mutated).

    Returns
    -------
    ValidationReport with AUC per control, pass/fail, and KL pathway flags.
    """
    report = ValidationReport(auc_threshold=auc_threshold)
    common = dep_matrix.index.intersection(mutations.index)

    # 1. Oncogene addiction controls
    for gene, mut_col in POSITIVE_CONTROLS.items():
        if gene not in dep_matrix.columns:
            logger.warning("Control gene %s not in dependency matrix — skipped", gene)
            report.controls_skipped.append(gene)
            continue

        if mut_col not in mutations.columns:
            logger.warning("Mutation column %s not found — skipped", mut_col)
            report.controls_skipped.append(gene)
            continue

        y_true = mutations.loc[common, mut_col].astype(int).values
        y_score = -dep_matrix.loc[common, gene].values  # more negative = more dependent

        # Need both classes present
        if y_true.sum() < 3 or (len(y_true) - y_true.sum()) < 3:
            logger.warning("Control %s: too few mutant/WT patients — skipped", gene)
            report.controls_skipped.append(gene)
            continue

        auc = roc_auc_score(y_true, y_score)
        report.control_aucs[gene] = float(auc)

        if auc >= auc_threshold:
            report.controls_passed.append(gene)
        else:
            report.controls_failed.append(gene)

    report.n_controls_tested = len(report.control_aucs)
    report.n_controls_passed = len(report.controls_passed)

    # 2. KL pathway flags
    kl_mask = subtypes.loc[common] == "KL"
    non_kl_mask = ~kl_mask

    if kl_mask.sum() >= 5:
        _flag_kl_pathways(dep_matrix.loc[common], kl_mask, non_kl_mask, report)
    else:
        logger.warning("Too few KL patients (%d) for pathway analysis", kl_mask.sum())

    logger.info(
        "Validation: %d/%d controls passed (AUC >= %.2f)",
        report.n_controls_passed, report.n_controls_tested, auc_threshold,
    )

    return report


def _flag_kl_pathways(
    dep: pd.DataFrame,
    kl_mask: pd.Series,
    non_kl_mask: pd.Series,
    report: ValidationReport,
) -> None:
    """Check if KL-relevant pathway genes appear as KL-specific dependencies."""
    dep_genes = set(dep.columns)

    # Positive pathways
    for pathway, info in KL_POSITIVE_PATHWAYS.items():
        found_genes = [g for g in info["genes"] if g in dep_genes]
        for gene in found_genes:
            kl_dep = dep.loc[kl_mask, gene].mean()
            non_kl_dep = dep.loc[non_kl_mask, gene].mean()
            report.kl_pathway_flags.append({
                "gene": gene,
                "pathway": pathway,
                "type": "positive_control",
                "kl_mean_dep": float(kl_dep),
                "non_kl_mean_dep": float(non_kl_dep),
                "kl_more_dependent": bool(kl_dep < non_kl_dep),  # more negative = more dependent
                "evidence": info["evidence"],
            })

    # Negative controls (known clinical failures)
    for pathway, info in KL_NEGATIVE_CONTROLS.items():
        found_genes = [g for g in info["genes"] if g in dep_genes]
        for gene in found_genes:
            kl_dep = dep.loc[kl_mask, gene].mean()
            non_kl_dep = dep.loc[non_kl_mask, gene].mean()
            is_kl_specific = kl_dep < non_kl_dep  # more negative = more dependent
            flag_entry = {
                "gene": gene,
                "pathway": pathway,
                "type": "negative_control",
                "kl_mean_dep": float(kl_dep),
                "non_kl_mean_dep": float(non_kl_dep),
                "kl_more_dependent": bool(is_kl_specific),
                "clinical_flag": info["flag"],
            }
            if is_kl_specific:
                flag_entry["warning"] = f"KL-specific dependency detected but {info['flag']}"
            report.kl_pathway_flags.append(flag_entry)


def save_validation_report(report: ValidationReport, results_dir: str | Path) -> Path:
    """Save validation report to JSON."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    out = {
        "auc_threshold": report.auc_threshold,
        "n_controls_tested": report.n_controls_tested,
        "n_controls_passed": report.n_controls_passed,
        "control_aucs": report.control_aucs,
        "controls_passed": report.controls_passed,
        "controls_failed": report.controls_failed,
        "controls_skipped": report.controls_skipped,
        "kl_pathway_flags": report.kl_pathway_flags,
    }

    out_path = results_dir / "validation_report.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    logger.info("Saved validation report to %s", out_path)
    return out_path
