#!/usr/bin/env python3
"""Phase 3.1: Test epithelial reserve score for stricture progression in RISK cohort.

Applies the epithelial reserve score to GSE93624 (RISK cohort, 245 samples, 27 progressors)
and tests association with complication progression. Compares to existing CPA3/EMT model.

Note: PGC and BPIFB1 are missing from GSE93624. Uses 4/6 available genes (CPO, GAS1, CASP6, SNX3).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

RISK_DIR = Path("data/crohns/cd-stricture-risk-prediction/processed")
OUTPUT_DIR = Path("output/crohns/cd-epithelial-reserve-treatment-stratification/phase3")

FULL_RESERVE_GENES = ["PGC", "BPIFB1", "CPO", "GAS1", "CASP6", "SNX3"]
FIBROSIS_GENES = ["SERPINE1", "GREM1", "CPA3", "MMP2", "MMP9", "COL1A1", "TGFB1", "VIM", "CDH2"]


def load_risk_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load RISK cohort expression and phenotype data."""
    pheno = pd.read_csv(RISK_DIR / "gse93624_phenotype.tsv", sep="\t")
    expr = pd.read_csv(RISK_DIR / "gse93624_expression.tsv.gz", sep="\t")
    expr = expr.set_index("gene")
    return expr, pheno


def compute_score(
    expression: pd.DataFrame, genes: list[str], score_name: str
) -> pd.DataFrame:
    """Compute mean z-score for a gene panel across all samples."""
    available = [g for g in genes if g in expression.index]
    missing = set(genes) - set(available)
    if missing:
        print(f"  Warning: {score_name} missing genes: {missing}")
    if not available:
        print(f"  ERROR: no genes available for {score_name}")
        return pd.DataFrame()

    # Z-score each gene across samples
    gene_expr = expression.loc[available]
    z_scores = gene_expr.apply(lambda row: stats.zscore(row.values), axis=1, result_type="expand")
    z_scores.columns = expression.columns
    score = z_scores.mean(axis=0)

    records = [{"gsm_id": sid, score_name: float(score[sid])} for sid in expression.columns]
    return pd.DataFrame(records)


def cross_validate(
    df: pd.DataFrame, feature_col: str, n_splits: int = 5, n_repeats: int = 20
) -> dict:
    """Stratified cross-validation with logistic regression."""
    X = df[[feature_col]].values
    y = df["progressed"].values
    scaler = StandardScaler()

    all_aucs = []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)
        for train_idx, test_idx in skf.split(X, y):
            X_train = scaler.fit_transform(X[train_idx])
            X_test = scaler.transform(X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]

            if len(np.unique(y_test)) > 1:
                all_aucs.append(roc_auc_score(y_test, y_prob))

    # Full-data fit for ROC curve and coefficient
    X_scaled = scaler.fit_transform(X)
    clf_full = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    clf_full.fit(X_scaled, y)
    y_prob_full = clf_full.predict_proba(X_scaled)[:, 1]
    full_auc = roc_auc_score(y, y_prob_full)

    return {
        "cv_mean_auc": round(float(np.mean(all_aucs)), 4),
        "cv_std_auc": round(float(np.std(all_aucs)), 4),
        "cv_median_auc": round(float(np.median(all_aucs)), 4),
        "cv_95ci": [round(float(np.percentile(all_aucs, 2.5)), 4),
                    round(float(np.percentile(all_aucs, 97.5)), 4)],
        "full_data_auc": round(full_auc, 4),
        "coef": round(float(clf_full.coef_[0, 0]), 4),
        "n_folds": len(all_aucs),
        "y_true": y.tolist(),
        "y_prob": y_prob_full.tolist(),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading RISK cohort data...")
    expression, pheno = load_risk_data()
    print(f"  Expression: {expression.shape[0]} genes x {expression.shape[1]} samples")
    print(f"  Phenotype: {len(pheno)} samples")

    # Filter to CD patients only
    cd_pheno = pheno[pheno["is_cd"] == True].copy()
    cd_samples = [s for s in cd_pheno["gsm_id"] if s in expression.columns]
    cd_pheno = cd_pheno[cd_pheno["gsm_id"].isin(cd_samples)].copy()
    cd_pheno["progressed"] = cd_pheno["complication_progressed"].astype(int)
    expression = expression[cd_samples]
    print(f"  CD patients: {len(cd_pheno)} ({cd_pheno['progressed'].sum()} progressors)")

    # Check gene availability
    available_reserve = [g for g in FULL_RESERVE_GENES if g in expression.index]
    available_fibrosis = [g for g in FIBROSIS_GENES if g in expression.index]
    print(f"\n  Reserve genes available: {available_reserve} ({len(available_reserve)}/{len(FULL_RESERVE_GENES)})")
    print(f"  Fibrosis genes available: {available_fibrosis} ({len(available_fibrosis)}/{len(FIBROSIS_GENES)})")

    # Compute scores
    print("\nComputing scores...")
    reserve_scores = compute_score(expression, FULL_RESERVE_GENES, "reserve_score")
    fibrosis_scores = compute_score(expression, FIBROSIS_GENES, "fibrosis_score")

    # Merge
    df = cd_pheno[["gsm_id", "progressed", "gender", "age_at_diagnosis", "tissue"]].copy()
    df = df.merge(reserve_scores, on="gsm_id", how="left")
    df = df.merge(fibrosis_scores, on="gsm_id", how="left")
    df = df.dropna(subset=["reserve_score", "fibrosis_score"])
    print(f"  Merged dataset: {len(df)} samples")

    # Descriptive stats
    print("\nDescriptive statistics:")
    for score_col in ["reserve_score", "fibrosis_score"]:
        prog = df[df["progressed"] == 1][score_col]
        non_prog = df[df["progressed"] == 0][score_col]
        t_stat, t_p = stats.ttest_ind(prog, non_prog)
        mwu_stat, mwu_p = stats.mannwhitneyu(prog, non_prog, alternative="two-sided")
        print(f"  {score_col}:")
        print(f"    Progressors:     mean={prog.mean():.4f}, std={prog.std():.4f}, n={len(prog)}")
        print(f"    Non-progressors: mean={non_prog.mean():.4f}, std={non_prog.std():.4f}, n={len(non_prog)}")
        print(f"    t-test: t={t_stat:.4f}, p={t_p:.6f}")
        print(f"    MWU: U={mwu_stat:.0f}, p={mwu_p:.6f}")

    # Cross-validation
    print("\n=== Cross-Validation Results ===")
    results = {}
    for feat, label in [
        ("reserve_score", "Epithelial Reserve (4-gene)"),
        ("fibrosis_score", "Fibrosis/EMT"),
    ]:
        res = cross_validate(df, feat)
        results[feat] = res
        print(f"  {label}: CV AUC={res['cv_mean_auc']:.4f} ± {res['cv_std_auc']:.4f}, "
              f"full-data AUC={res['full_data_auc']:.4f}, coef={res['coef']}")

    # Correlation between scores
    r, p = stats.spearmanr(df["reserve_score"], df["fibrosis_score"])
    print(f"\n  Reserve-Fibrosis correlation: rho={r:.4f}, p={p:.6f}")

    # Interaction model (reserve × fibrosis)
    print("\n=== Interaction Model ===")
    df["reserve_x_fibrosis"] = df["reserve_score"] * df["fibrosis_score"]
    scaler = StandardScaler()
    X_interact = scaler.fit_transform(df[["reserve_score", "fibrosis_score", "reserve_x_fibrosis"]].values)
    y = df["progressed"].values
    clf_interact = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
    clf_interact.fit(X_interact, y)
    y_prob_interact = clf_interact.predict_proba(X_interact)[:, 1]
    interact_auc = roc_auc_score(y, y_prob_interact)
    print(f"  Interaction model full-data AUC: {interact_auc:.4f}")
    print(f"  Coefficients: reserve={clf_interact.coef_[0, 0]:.4f}, "
          f"fibrosis={clf_interact.coef_[0, 1]:.4f}, "
          f"interaction={clf_interact.coef_[0, 2]:.4f}")

    # Success criterion
    reserve_t_p = stats.ttest_ind(
        df[df["progressed"] == 1]["reserve_score"],
        df[df["progressed"] == 0]["reserve_score"]
    ).pvalue
    criterion_met = reserve_t_p < 0.05
    print(f"\n  Success criterion (p<0.05): {'MET' if criterion_met else 'NOT MET'} (p={reserve_t_p:.6f})")

    # Compare to existing CPA3/EMT model
    existing_auc = 0.608  # from cd-stricture-risk-prediction transcriptomic model
    print(f"  Existing CPA3/EMT model AUC: {existing_auc}")
    print(f"  Reserve score AUC: {results['reserve_score']['cv_mean_auc']}")

    # Save
    print("\nSaving results...")
    df.to_csv(OUTPUT_DIR / "risk_cohort_scores.csv", index=False)

    # Clean results for JSON
    summary_results = {}
    for k, v in results.items():
        clean = {kk: vv for kk, vv in v.items() if kk not in ("y_true", "y_prob")}
        summary_results[k] = clean

    summary = {
        "phase": "3.1",
        "description": "Epithelial reserve score for stricture progression in RISK cohort (GSE93624)",
        "dataset": "GSE93624 (RISK cohort)",
        "n_cd_patients": len(df),
        "n_progressors": int(df["progressed"].sum()),
        "genes_available": available_reserve,
        "genes_missing": list(set(FULL_RESERVE_GENES) - set(available_reserve)),
        "cv_results": summary_results,
        "interaction_model_auc": round(interact_auc, 4),
        "reserve_fibrosis_correlation": {"spearman_r": round(float(r), 4), "p_value": round(float(p), 6)},
        "success_criterion": {
            "test": "reserve score associated with progression (p<0.05)",
            "met": bool(criterion_met),
            "p_value": round(float(reserve_t_p), 6),
        },
        "comparison_to_existing": {
            "existing_cpa3_emt_auc": existing_auc,
            "reserve_cv_auc": results["reserve_score"]["cv_mean_auc"],
        },
    }
    with open(OUTPUT_DIR / "phase3_1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    print("Generating plots...")

    # ROC curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, feat, title in [
        (axes[0], "reserve_score", f"Reserve (4-gene)\nCV AUC={results['reserve_score']['cv_mean_auc']:.3f}"),
        (axes[1], "fibrosis_score", f"Fibrosis/EMT\nCV AUC={results['fibrosis_score']['cv_mean_auc']:.3f}"),
    ]:
        y_true = np.array(results[feat]["y_true"])
        y_prob = np.array(results[feat]["y_prob"])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, color="steelblue", lw=2)
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(title)
        ax.set_aspect("equal")
    fig.suptitle("Stricture Progression Prediction (RISK Cohort)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "risk_roc_curves.png", dpi=150)
    plt.close(fig)

    # Score distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, title in [
        (axes[0], "reserve_score", "Epithelial Reserve Score"),
        (axes[1], "fibrosis_score", "Fibrosis/EMT Score"),
    ]:
        prog = df[df["progressed"] == 1][col]
        non_prog = df[df["progressed"] == 0][col]
        ax.hist(non_prog, bins=20, alpha=0.6, color="steelblue", label=f"Non-prog (n={len(non_prog)})", edgecolor="white")
        ax.hist(prog, bins=20, alpha=0.6, color="salmon", label=f"Progressor (n={len(prog)})", edgecolor="white")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "risk_score_distributions.png", dpi=150)
    plt.close(fig)

    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
