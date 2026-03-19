"""Stage-stratified evaluation and comparison analysis for CRC detection panel.

Evaluates classifier performance separately by CRC stage (I-IV) using TCGA
methylation data (which has stage annotations). Compares combined panel vs
methylation-only vs protein-only vs single best marker (SEPT9).

Output:
    output/diagnostics/crc-liquid-biopsy-panel/stage_stratified_results.parquet
    output/diagnostics/crc-liquid-biopsy-panel/comparison_table.csv
    output/diagnostics/crc-liquid-biopsy-panel/roc_curves.png

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.stage_evaluation [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "crc-liquid-biopsy-panel"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_tcga_with_stages(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TCGA methylation + clinical data with stage annotations.

    Returns (methylation matrix [CpGs x samples], clinical with stage info).
    """
    meth = pd.read_parquet(data_dir / "tcga_methylation.parquet")
    clinical = pd.read_parquet(data_dir / "tcga_clinical.parquet")
    clinical["stage_numeric"] = pd.to_numeric(clinical["stage_numeric"], errors="coerce")
    clinical = clinical.dropna(subset=["stage_numeric"])
    clinical["stage_numeric"] = clinical["stage_numeric"].astype(int)
    return meth, clinical


def _load_methylation_signatures(output_dir: Path) -> list[str]:
    """Load the discovered methylation signature CpGs."""
    sig_path = output_dir / "methylation_signatures.parquet"
    sigs = pd.read_parquet(sig_path)
    return sigs.index.tolist()


def _load_protein_data(data_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Load protein features and labels from GSE164191.

    Returns (feature_matrix, labels, feature_names).
    """
    comp = pd.read_parquet(output_dir / "protein_complementarity_analysis.parquet")
    expr = pd.read_parquet(data_dir / "gse164191_protein_biomarkers.parquet")
    meta = pd.read_parquet(data_dir / "gse164191_metadata.parquet")

    sig_proteins = comp[comp["p_value"] < 0.05]
    if sig_proteins.empty:
        sig_proteins = comp.head(10)

    probe_ids = sig_proteins["probe_id"].tolist()
    probe_to_gene = dict(zip(sig_proteins["probe_id"], sig_proteins.index))

    available = [p for p in probe_ids if p in expr.index]
    samples = meta.index.tolist()
    X = expr.loc[available, samples].T
    X.columns = [f"prot_{probe_to_gene.get(c, c)}" for c in X.columns]

    y = (meta.loc[samples, "condition"] == "CRC").astype(int).values
    valid = ~X.isna().any(axis=1).values
    X = X.loc[valid]
    y = y[valid]

    return X, y, X.columns.tolist()


def _load_cfdna_features(data_dir: Path, output_dir: Path, n_top: int = 50) -> tuple[pd.DataFrame, np.ndarray]:
    """Load cfDNA methylation features for methylation-only evaluation.

    Returns (feature_matrix, labels).
    """
    validated = pd.read_parquet(output_dir / "cfdna_validated_markers.parquet")
    top_cpgs = validated.head(n_top).index.tolist()

    meth = pd.read_parquet(data_dir / "gse149282_cfdna_methylation.parquet")
    meta = pd.read_parquet(data_dir / "gse149282_metadata.parquet")

    available = [c for c in top_cpgs if c in meth.index]
    X = meth.loc[available].T
    X.columns = [f"meth_{c}" for c in X.columns]
    X["condition"] = meta.loc[X.index, "condition"]

    y = (X["condition"] == "CRC").astype(int).values
    X = X.drop(columns=["condition"])

    valid = ~X.isna().any(axis=1).values
    X = X.loc[valid]
    y = y[valid]

    return X, y


# ---------------------------------------------------------------------------
# Stage-stratified evaluation on TCGA methylation data
# ---------------------------------------------------------------------------

def evaluate_by_stage(
    data_dir: Path,
    output_dir: Path,
    n_sig_cpgs: int = 200,
) -> pd.DataFrame:
    """Evaluate methylation signature performance by CRC stage using TCGA data.

    Trains a classifier on TCGA methylation data and reports per-stage sensitivity.
    """
    meth, clinical = _load_tcga_with_stages(data_dir)
    sig_cpgs = _load_methylation_signatures(output_dir)

    # Match methylation samples to clinical via 12-char TCGA case prefix
    meth_samples = list(meth.columns)
    # Build case_id -> meth_sample mapping (truncate to 12-char prefix)
    prefix_to_meth = {}
    for s in meth_samples:
        prefix = s[:12]
        prefix_to_meth.setdefault(prefix, []).append(s)

    clinical_with_meth = []
    for case_id, row in clinical.iterrows():
        case_prefix = str(case_id)[:12]
        if case_prefix in prefix_to_meth:
            # Use first tumor sample (01A suffix)
            matched = prefix_to_meth[case_prefix]
            tumor = [s for s in matched if "-01" in s[12:]]
            sid = tumor[0] if tumor else matched[0]
            clinical_with_meth.append({
                "sample_id": sid,
                "case_id": case_id,
                "stage": row["stage"],
                "stage_numeric": row["stage_numeric"],
                "condition": "CRC",
            })

    sample_info = pd.DataFrame(clinical_with_meth)
    if sample_info.empty:
        logger.warning("No TCGA samples matched between methylation and clinical data")
        return pd.DataFrame()

    logger.info("Matched %d TCGA samples with stage info", len(sample_info))
    logger.info("Stage distribution: %s", sample_info["stage_numeric"].value_counts().sort_index().to_dict())

    # Build feature matrix from signature CpGs
    available_cpgs = [c for c in sig_cpgs[:n_sig_cpgs] if c in meth.index]
    matched_samples = sample_info["sample_id"].tolist()

    # Normal samples: TCGA uses -11 for solid tissue normal
    all_samples = meth.columns.tolist()
    matched_set = set(matched_samples)
    normal_samples = [s for s in all_samples if "-11" in s[12:] and s not in matched_set]
    if not normal_samples:
        normal_samples = [s for s in all_samples if s not in matched_set]
    if not normal_samples:
        logger.warning("No normal samples found in TCGA methylation data")
        return pd.DataFrame()

    # Filter CpGs with >10% NaN, then median-impute remaining
    all_used_samples = matched_samples + normal_samples
    raw = meth.loc[available_cpgs, all_used_samples].T.astype(float)
    nan_rate = raw.isna().mean()
    good_cpgs = nan_rate[nan_rate < 0.10].index.tolist()
    raw = raw[good_cpgs]
    # Median impute per CpG
    for col in raw.columns:
        median_val = raw[col].median()
        raw[col] = raw[col].fillna(median_val)

    X = raw.values
    y = np.array([1] * len(matched_samples) + [0] * len(normal_samples))
    stages = list(sample_info["stage_numeric"].values) + [0] * len(normal_samples)
    stages = np.array(stages)

    logger.info("After NaN filter: %d CpGs retained (from %d)", len(good_cpgs), len(available_cpgs))

    logger.info("Feature matrix: %d samples x %d CpGs (%d CRC, %d normal)",
                X.shape[0], X.shape[1], y.sum(), (1 - y).sum())

    # 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_probs = np.zeros(len(y))
    all_preds = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
        clf.fit(X_train, y[train_idx])

        all_probs[test_idx] = clf.predict_proba(X_test)[:, 1]

    # Find threshold at 95% specificity
    fpr, tpr, thresholds = roc_curve(y, all_probs)
    spec_idx = np.searchsorted(1 - fpr[::-1], 0.95)
    thresh_95 = thresholds[::-1][min(spec_idx, len(thresholds) - 1)]
    all_preds = (all_probs >= thresh_95).astype(int)

    overall_auc = roc_auc_score(y, all_probs)

    # Per-stage results
    results = []
    for stage in sorted(set(stages)):
        mask = stages == stage
        if mask.sum() == 0:
            continue

        stage_label = {0: "Normal", 1: "Stage I", 2: "Stage II", 3: "Stage III", 4: "Stage IV"}.get(
            stage, f"Stage {stage}"
        )
        n = int(mask.sum())
        n_positive = int(y[mask].sum())
        n_negative = n - n_positive

        if len(np.unique(y[mask])) >= 2:
            stage_auc = roc_auc_score(y[mask], all_probs[mask])
        else:
            stage_auc = np.nan

        # Sensitivity = TP / (TP + FN)
        tp = int(((all_preds[mask] == 1) & (y[mask] == 1)).sum())
        fn = int(((all_preds[mask] == 0) & (y[mask] == 1)).sum())
        tn = int(((all_preds[mask] == 0) & (y[mask] == 0)).sum())
        fp = int(((all_preds[mask] == 1) & (y[mask] == 0)).sum())

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        results.append({
            "stage": stage_label,
            "stage_numeric": int(stage),
            "n_samples": n,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "auc": float(stage_auc) if not np.isnan(stage_auc) else None,
            "sensitivity_at_95spec": float(sensitivity) if not np.isnan(sensitivity) else None,
            "specificity": float(specificity) if not np.isnan(specificity) else None,
            "tp": tp, "fn": fn, "tn": tn, "fp": fp,
        })

    results.append({
        "stage": "Overall",
        "stage_numeric": -1,
        "n_samples": len(y),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "auc": float(overall_auc),
        "sensitivity_at_95spec": float(tpr[::-1][min(spec_idx, len(tpr) - 1)]),
        "specificity": 0.95,
        "tp": int(((all_preds == 1) & (y == 1)).sum()),
        "fn": int(((all_preds == 0) & (y == 1)).sum()),
        "tn": int(((all_preds == 0) & (y == 0)).sum()),
        "fp": int(((all_preds == 1) & (y == 0)).sum()),
    })

    df = pd.DataFrame(results)
    logger.info("Stage-stratified results:\n%s", df.to_string(index=False))
    return df


# ---------------------------------------------------------------------------
# Panel comparison analysis
# ---------------------------------------------------------------------------

def _cv_evaluate(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """Cross-validated evaluation returning AUC + sensitivity@95spec."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    probs = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
        clf.fit(X_tr, y[train_idx])
        probs[test_idx] = clf.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)
    spec_idx = np.searchsorted(1 - fpr[::-1], 0.95)
    sens = tpr[::-1][min(spec_idx, len(tpr) - 1)]

    # Confusion matrix at Youden threshold
    j = tpr - fpr
    best_idx = np.argmax(j)
    _, _, thresholds = roc_curve(y, probs)
    thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    preds = (probs >= thresh).astype(int)
    cm = confusion_matrix(y, preds)

    return {
        "auc": float(auc),
        "sensitivity_at_95spec": float(sens),
        "sensitivity_youden": float(tpr[best_idx]),
        "specificity_youden": float(1 - fpr[best_idx]),
        "confusion_matrix": cm.tolist(),
        "roc_fpr": fpr.tolist(),
        "roc_tpr": tpr.tolist(),
    }


def compare_panels(data_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Compare combined panel vs methylation-only vs protein-only vs SEPT9.

    Returns (comparison_table, roc_data_dict).
    """
    # Load optimized panel config
    with open(output_dir / "optimized_panel.json") as f:
        panel_config = json.load(f)

    optimal_features = panel_config["optimal_panel"]["features"]
    prot_features_list = [f for f in optimal_features if f.startswith("prot_")]

    # --- Protein-based evaluations (GSE164191) ---
    prot_X, prot_y, all_prot_names = _load_protein_data(data_dir, output_dir)

    # 1. Combined panel (all selected protein features)
    combined_cols = [c for c in prot_features_list if c in prot_X.columns]
    combined_result = _cv_evaluate(prot_X[combined_cols].values, prot_y)
    combined_result["panel"] = "Combined (protein)"
    combined_result["n_features"] = len(combined_cols)
    combined_result["cost_usd"] = panel_config["optimal_panel"]["estimated_cost_usd"]

    # 2. Protein-only (all significant proteins)
    all_prot_result = _cv_evaluate(prot_X.values, prot_y)
    all_prot_result["panel"] = "All proteins"
    all_prot_result["n_features"] = len(all_prot_names)
    all_prot_result["cost_usd"] = len(all_prot_names) * 15.0

    # 3. SEPT9 alone
    sept9_col = [c for c in prot_X.columns if "SEPT9" in c]
    if sept9_col:
        sept9_result = _cv_evaluate(prot_X[sept9_col].values, prot_y)
    else:
        sept9_result = {"auc": np.nan, "sensitivity_at_95spec": np.nan,
                        "sensitivity_youden": np.nan, "specificity_youden": np.nan,
                        "confusion_matrix": [], "roc_fpr": [], "roc_tpr": []}
    sept9_result["panel"] = "SEPT9 alone"
    sept9_result["n_features"] = 1
    sept9_result["cost_usd"] = 15.0

    # --- Methylation-only evaluation (GSE149282) ---
    meth_X, meth_y = _load_cfdna_features(data_dir, output_dir)
    meth_result = _cv_evaluate(meth_X.values, meth_y)
    meth_result["panel"] = "Methylation only (cfDNA)"
    meth_result["n_features"] = meth_X.shape[1]
    meth_result["cost_usd"] = meth_X.shape[1] * 75.0

    # Build comparison table
    rows = []
    roc_data = {}
    for r in [combined_result, all_prot_result, sept9_result, meth_result]:
        rows.append({
            "Panel": r["panel"],
            "N_features": r["n_features"],
            "AUC": r.get("auc"),
            "Sensitivity_at_95spec": r.get("sensitivity_at_95spec"),
            "Sensitivity_Youden": r.get("sensitivity_youden"),
            "Specificity_Youden": r.get("specificity_youden"),
            "Cost_USD": r.get("cost_usd"),
        })
        if r.get("roc_fpr"):
            roc_data[r["panel"]] = {"fpr": r["roc_fpr"], "tpr": r["roc_tpr"]}

    comparison = pd.DataFrame(rows)

    # Check success criterion: combined >= single-modality + 5% AUC
    combined_auc = combined_result.get("auc", 0)
    sept9_auc = sept9_result.get("auc", 0) if not np.isnan(sept9_result.get("auc", np.nan)) else 0
    delta = combined_auc - sept9_auc
    logger.info("Combined vs SEPT9: AUC delta = %.3f (target >= 0.05)", delta)

    return comparison, roc_data


# ---------------------------------------------------------------------------
# ROC curve plotting
# ---------------------------------------------------------------------------

def plot_roc_curves(roc_data: dict, output_dir: Path) -> Path:
    """Generate ROC curves comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))

    colors = {"Combined (protein)": "#2196F3", "All proteins": "#4CAF50",
              "SEPT9 alone": "#FF9800", "Methylation only (cfDNA)": "#9C27B0"}

    for panel_name, data in roc_data.items():
        fpr, tpr = data["fpr"], data["tpr"]
        auc_val = np.trapezoid(tpr, fpr)
        color = colors.get(panel_name, "#666666")
        ax.plot(fpr, tpr, label=f"{panel_name} (AUC={auc_val:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.axvline(x=0.05, color="red", linestyle=":", alpha=0.5, label="95% specificity")
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("CRC Detection: Panel Comparison ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.2)

    output_path = output_dir / "roc_curves.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curves to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_stage_evaluation(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict:
    """Run stage-stratified evaluation and comparison analysis."""
    results_path = output_dir / "stage_stratified_results.parquet"
    comparison_path = output_dir / "comparison_table.csv"

    if results_path.exists() and comparison_path.exists() and not force:
        logger.info("Loading cached stage evaluation results")
        return {
            "stage_results": pd.read_parquet(results_path),
            "comparison": pd.read_csv(comparison_path),
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage-stratified evaluation
    logger.info("=== Stage-Stratified Evaluation (TCGA methylation) ===")
    stage_results = evaluate_by_stage(data_dir, output_dir)
    if not stage_results.empty:
        stage_results.to_parquet(results_path)
        logger.info("Saved stage results to %s", results_path)

    # Panel comparison
    logger.info("=== Panel Comparison Analysis ===")
    comparison, roc_data = compare_panels(data_dir, output_dir)
    comparison.to_csv(comparison_path, index=False)
    logger.info("Saved comparison table to %s", comparison_path)

    # ROC curves
    if roc_data:
        plot_roc_curves(roc_data, output_dir)

    return {
        "stage_results": stage_results,
        "comparison": comparison,
    }


def main():
    parser = argparse.ArgumentParser(description="Stage-stratified evaluation")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_stage_evaluation(args.data_dir, args.output_dir, force=args.force)

    if not result["stage_results"].empty:
        print("\n=== Stage-Stratified Results ===")
        cols = ["stage", "n_samples", "auc", "sensitivity_at_95spec", "specificity"]
        available = [c for c in cols if c in result["stage_results"].columns]
        print(result["stage_results"][available].to_string(index=False))

    print("\n=== Panel Comparison ===")
    print(result["comparison"].to_string(index=False))


if __name__ == "__main__":
    main()
