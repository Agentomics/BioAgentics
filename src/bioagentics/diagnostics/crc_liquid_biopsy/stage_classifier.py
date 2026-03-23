"""Stage-specific feature selection and staging classifier for CRC.

Detection-optimized CpGs (tumor vs normal) have near-random staging
power (AUC ~0.52). This module selects CpGs that discriminate Early
(Stage I-II) vs Advanced (Stage III-IV) CRC, then builds a staging
classifier intended to run after a positive detection result.

Output:
    output/diagnostics/crc-liquid-biopsy-panel/stage_classifier_results.json

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.stage_classifier [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "crc-liquid-biopsy-panel"

# Candidate genes from prior analysis (task 1088)
STAGE_CANDIDATE_GENES = ["ADHFE1", "MDFI", "C1orf70", "GRASP", "IRX5"]


def load_staged_tumor_data(
    data_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TCGA methylation data for tumor samples with stage annotations.

    Returns (methylation [CpGs x tumor_samples], sample_info DataFrame).
    """
    meth = pd.read_parquet(data_dir / "tcga_methylation.parquet")
    clinical = pd.read_parquet(data_dir / "tcga_clinical.parquet")
    clinical["stage_numeric"] = pd.to_numeric(clinical["stage_numeric"], errors="coerce")
    clinical = clinical.dropna(subset=["stage_numeric"])
    clinical["stage_numeric"] = clinical["stage_numeric"].astype(int)

    # Match tumor samples (01A) to clinical cases via 12-char prefix
    prefix_to_sample = {}
    for s in meth.columns:
        parts = s.split("-")
        if len(parts) >= 4 and parts[3][:2] == "01":
            prefix = s[:12]
            prefix_to_sample[prefix] = s

    rows = []
    for case_id, row in clinical.iterrows():
        prefix = str(case_id)[:12]
        if prefix in prefix_to_sample:
            rows.append({
                "sample_id": prefix_to_sample[prefix],
                "stage": int(row["stage_numeric"]),
                "stage_group": "early" if row["stage_numeric"] <= 2 else "advanced",
            })

    info = pd.DataFrame(rows)
    logger.info("Staged tumor samples: %d (early=%d, advanced=%d)",
                len(info),
                (info["stage_group"] == "early").sum(),
                (info["stage_group"] == "advanced").sum())
    logger.info("Per stage: %s", info["stage"].value_counts().sort_index().to_dict())

    tumor_meth = meth[info["sample_id"].tolist()]
    return tumor_meth, info


def discover_stage_cpgs(
    meth: pd.DataFrame,
    sample_info: pd.DataFrame,
    q_threshold: float = 0.05,
    min_delta: float = 0.05,
    max_cpgs: int = 500,
) -> pd.DataFrame:
    """Differential methylation between Early and Advanced CRC.

    Processes in chunks to stay within memory constraints.
    Returns DataFrame of significant CpGs sorted by effect size.
    """
    early_ids = sample_info[sample_info["stage_group"] == "early"]["sample_id"].tolist()
    adv_ids = sample_info[sample_info["stage_group"] == "advanced"]["sample_id"].tolist()

    early_data = meth[early_ids]
    adv_data = meth[adv_ids]

    # Filter CpGs with sufficient data in both groups
    min_early = len(early_ids) // 2
    min_adv = len(adv_ids) // 2
    valid = (early_data.notna().sum(axis=1) >= min_early) & (adv_data.notna().sum(axis=1) >= min_adv)
    early_valid = early_data.loc[valid]
    adv_valid = adv_data.loc[valid]

    logger.info("CpGs with sufficient data for stage comparison: %d", valid.sum())

    mean_early = early_valid.mean(axis=1)
    mean_adv = adv_valid.mean(axis=1)
    delta = mean_adv - mean_early
    abs_delta = delta.abs()

    # Pre-filter by effect size to reduce multiple testing burden
    prefilter = abs_delta >= min_delta
    logger.info("CpGs passing min delta-beta (%.2f): %d", min_delta, prefilter.sum())

    cpg_ids = early_valid.index[prefilter].tolist()
    early_filt = early_valid.loc[prefilter]
    adv_filt = adv_valid.loc[prefilter]

    # Wilcoxon tests in chunks
    p_values = []
    chunk_size = 10000
    for start in range(0, len(cpg_ids), chunk_size):
        end = min(start + chunk_size, len(cpg_ids))
        for cpg in cpg_ids[start:end]:
            e_vals = early_filt.loc[cpg].dropna().values
            a_vals = adv_filt.loc[cpg].dropna().values
            if len(e_vals) < 5 or len(a_vals) < 5:
                p_values.append(np.nan)
                continue
            _, p = mannwhitneyu(e_vals, a_vals, alternative="two-sided")
            p_values.append(p)
        if end % 50000 == 0:
            logger.info("  ...processed %d / %d CpGs", end, len(cpg_ids))

    result = pd.DataFrame({
        "delta_beta_stage": delta.loc[prefilter].values,
        "abs_delta_stage": abs_delta.loc[prefilter].values,
        "mean_early": mean_early.loc[prefilter].values,
        "mean_advanced": mean_adv.loc[prefilter].values,
        "p_value": p_values,
    }, index=cpg_ids)
    result.index.name = "cpg_id"

    # FDR correction
    valid_p = result["p_value"].dropna()
    if len(valid_p) > 0:
        ranked = valid_p.rank()
        n_tests = len(valid_p)
        result.loc[valid_p.index, "q_value"] = valid_p * n_tests / ranked
        result["q_value"] = result["q_value"].clip(upper=1.0)

    sig = result[(result["q_value"] <= q_threshold)].copy()
    sig = sig.sort_values("abs_delta_stage", ascending=False)

    logger.info("Stage-discriminative CpGs (q <= %.2f): %d", q_threshold, len(sig))
    return sig.head(max_cpgs)


def annotate_stage_cpgs(stage_cpgs: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Annotate stage CpGs with gene names from 450K manifest."""
    manifest_path = data_dir / "illumina_450k_manifest.csv"
    if not manifest_path.exists():
        logger.warning("No manifest found; skipping gene annotation")
        stage_cpgs["gene_name"] = None
        return stage_cpgs

    manifest = pd.read_csv(manifest_path, usecols=["IlmnID", "UCSC_RefGene_Name"])
    manifest = manifest.set_index("IlmnID")
    common = stage_cpgs.index.intersection(manifest.index)
    stage_cpgs["gene_name"] = None
    if len(common) > 0:
        stage_cpgs.loc[common, "gene_name"] = manifest.loc[common, "UCSC_RefGene_Name"]
    logger.info("Annotated %d / %d stage CpGs with gene names", len(common), len(stage_cpgs))

    # Check candidate genes
    for gene in STAGE_CANDIDATE_GENES:
        mask = stage_cpgs["gene_name"].fillna("").str.contains(gene, case=False)
        if mask.any():
            logger.info("  %s: %d CpGs in stage set", gene, mask.sum())
    return stage_cpgs


def train_staging_classifier(
    meth: pd.DataFrame,
    sample_info: pd.DataFrame,
    stage_cpgs: pd.DataFrame,
    n_features: int = 50,
) -> dict:
    """Train Early vs Advanced staging classifier with cross-validation.

    Returns performance metrics dict.
    """
    top_cpgs = stage_cpgs.head(n_features).index.tolist()
    available = [c for c in top_cpgs if c in meth.index]
    samples = sample_info["sample_id"].tolist()

    X = meth.loc[available, samples].T.values.astype(float)
    y = (sample_info["stage_group"] == "advanced").astype(int).values

    # Handle NaN: median impute per feature
    for j in range(X.shape[1]):
        col = X[:, j]
        mask = np.isnan(col)
        if mask.any():
            median = np.nanmedian(col)
            X[mask, j] = median

    # Remove zero-variance features
    variances = np.var(X, axis=0)
    keep = variances > 1e-10
    X = X[:, keep]
    kept_cpgs = [available[i] for i in range(len(available)) if keep[i]]

    logger.info("Staging feature matrix: %d samples x %d CpGs", X.shape[0], X.shape[1])

    # 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probs = np.zeros(len(y))

    for train_idx, test_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])

        model = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 10).tolist(),
            penalty="l2", solver="lbfgs",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring="roc_auc", max_iter=5000, random_state=42,
        )
        model.fit(X_tr, y[train_idx])
        probs[test_idx] = model.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y, probs)
    fpr, tpr, _ = roc_curve(y, probs)

    idx90 = np.searchsorted(1 - fpr[::-1], 0.90)
    sens_90 = tpr[::-1][min(idx90, len(tpr) - 1)]
    idx80 = np.searchsorted(1 - fpr[::-1], 0.80)
    sens_80 = tpr[::-1][min(idx80, len(tpr) - 1)]

    # Per-stage accuracy
    thresh = 0.5
    preds = (probs >= thresh).astype(int)
    per_stage = {}
    for stage in sorted(sample_info["stage"].unique()):
        mask = sample_info["stage"].values == stage
        n = int(mask.sum())
        correct = int((preds[mask] == y[mask]).sum())
        per_stage[f"stage_{stage}"] = {
            "n": n,
            "accuracy": float(correct / n) if n > 0 else 0,
            "advanced_pred_rate": float(preds[mask].mean()),
        }

    logger.info("Staging classifier: AUC=%.3f, sens@90spec=%.3f, sens@80spec=%.3f",
                auc, sens_90, sens_80)

    return {
        "auc": float(auc),
        "sensitivity_at_90spec": float(sens_90),
        "sensitivity_at_80spec": float(sens_80),
        "n_features": len(kept_cpgs),
        "n_samples": len(y),
        "n_early": int((y == 0).sum()),
        "n_advanced": int((y == 1).sum()),
        "per_stage": per_stage,
        "method": "LogisticRegressionCV L2, 5-fold CV",
        "top_features": kept_cpgs[:10],
    }


def run_stage_classifier(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict:
    """Run stage-specific feature selection and classifier pipeline."""
    output_path = output_dir / "stage_classifier_results.json"
    if output_path.exists() and not force:
        logger.info("Loading cached results from %s", output_path)
        with open(output_path) as f:
            return json.load(f)

    meth, sample_info = load_staged_tumor_data(data_dir)
    stage_cpgs = discover_stage_cpgs(meth, sample_info)

    if stage_cpgs.empty:
        logger.warning("No stage-discriminative CpGs found")
        results = {"error": "No stage-discriminative CpGs found"}
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        return results

    stage_cpgs = annotate_stage_cpgs(stage_cpgs, data_dir)

    # Save stage CpG discovery results
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_cpgs.to_parquet(output_dir / "stage_discriminative_cpgs.parquet")

    # Train staging classifier
    classifier_results = train_staging_classifier(meth, sample_info, stage_cpgs)

    # Baseline comparison: detection CpGs for staging
    sigs = pd.read_parquet(output_dir / "methylation_signatures.parquet")
    baseline_cpgs = sigs.head(50).index.tolist()
    baseline_available = [c for c in baseline_cpgs if c in meth.index]
    samples = sample_info["sample_id"].tolist()
    X_base = meth.loc[baseline_available, samples].T.values.astype(float)
    y_base = (sample_info["stage_group"] == "advanced").astype(int).values
    for j in range(X_base.shape[1]):
        col = X_base[:, j]
        mask = np.isnan(col)
        if mask.any():
            X_base[mask, j] = np.nanmedian(col)
    variances = np.var(X_base, axis=0)
    keep = variances > 1e-10
    X_base = X_base[:, keep]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    base_probs = np.zeros(len(y_base))
    for train_idx, test_idx in cv.split(X_base, y_base):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_base[train_idx])
        X_te = scaler.transform(X_base[test_idx])
        clf = LogisticRegressionCV(
            Cs=np.logspace(-3, 2, 10).tolist(),
            penalty="l2", solver="lbfgs",
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
            scoring="roc_auc", max_iter=5000, random_state=42,
        )
        clf.fit(X_tr, y_base[train_idx])
        base_probs[test_idx] = clf.predict_proba(X_te)[:, 1]
    baseline_auc = roc_auc_score(y_base, base_probs)

    # Top genes in stage CpGs
    gene_counts = stage_cpgs["gene_name"].dropna().str.split(";").explode().value_counts()

    results = {
        "stage_discriminative_cpgs": {
            "total_significant": len(stage_cpgs),
            "top_genes": gene_counts.head(20).to_dict() if len(gene_counts) > 0 else {},
            "candidate_gene_hits": {
                g: int(stage_cpgs["gene_name"].fillna("").str.contains(g, case=False).sum())
                for g in STAGE_CANDIDATE_GENES
            },
        },
        "staging_classifier": classifier_results,
        "baseline_detection_cpgs_for_staging": {
            "auc": float(baseline_auc),
            "note": "Detection-optimized CpGs used for staging (expected ~0.5)",
        },
        "improvement": {
            "delta_auc": float(classifier_results["auc"] - baseline_auc),
            "stage_optimized_auc": float(classifier_results["auc"]),
            "detection_optimized_auc": float(baseline_auc),
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", output_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Stage-specific CRC classifier")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_stage_classifier(args.data_dir, args.output_dir, force=args.force)

    if "error" in results:
        print(f"Error: {results['error']}")
        return

    disc = results["stage_discriminative_cpgs"]
    print(f"\n=== Stage-Discriminative CpGs ===")
    print(f"Significant CpGs: {disc['total_significant']}")
    if disc["top_genes"]:
        print(f"Top genes: {', '.join(list(disc['top_genes'].keys())[:10])}")
    print(f"Candidate gene hits: {disc['candidate_gene_hits']}")

    clf = results["staging_classifier"]
    print(f"\n=== Staging Classifier (Early vs Advanced) ===")
    print(f"AUC: {clf['auc']:.3f}")
    print(f"Sensitivity at 90% spec: {clf['sensitivity_at_90spec']:.3f}")
    print(f"Sensitivity at 80% spec: {clf['sensitivity_at_80spec']:.3f}")
    print(f"Samples: {clf['n_early']} early, {clf['n_advanced']} advanced")
    for stage, info in clf["per_stage"].items():
        print(f"  {stage}: n={info['n']}, accuracy={info['accuracy']:.3f}, adv_pred_rate={info['advanced_pred_rate']:.3f}")

    imp = results["improvement"]
    print(f"\n=== Improvement over Detection CpGs ===")
    print(f"Stage-optimized AUC: {imp['stage_optimized_auc']:.3f}")
    print(f"Detection-optimized AUC: {imp['detection_optimized_auc']:.3f}")
    print(f"Delta: {imp['delta_auc']:+.3f}")


if __name__ == "__main__":
    main()
