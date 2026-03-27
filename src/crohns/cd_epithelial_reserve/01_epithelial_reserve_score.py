#!/usr/bin/env python3
"""Phase 1.1: Epithelial reserve score as anti-TNF response predictor.

Computes a 6-gene epithelial reserve score (PGC, BPIFB1, CPO, GAS1, CASP6, SNX3)
and tests it as a single-feature logistic regression predictor of anti-TNF response
using leave-one-study-out cross-validation (LOSO-CV).
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

# Paths
PROCESSED_DIR = Path("output/crohns/anti-tnf-response-prediction/processed")
OUTPUT_DIR = Path("output/crohns/cd-epithelial-reserve-treatment-stratification")

# Epithelial reserve signature genes (from corrected anti-TNF SHAP analysis)
EPITHELIAL_GENES = ["PGC", "BPIFB1", "CPO", "GAS1", "CASP6", "SNX3"]
STUDIES = ["GSE16879", "GSE12251", "GSE73661"]


def load_expression_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge expression matrices from all anti-TNF cohorts."""
    metadata = pd.read_csv(PROCESSED_DIR / "combined_metadata.csv")
    metadata = metadata[metadata["study"].isin(STUDIES)].copy()

    frames = []
    for study in STUDIES:
        expr = pd.read_csv(PROCESSED_DIR / f"{study}_expression.csv")
        expr = expr.set_index("gene_symbol")
        # Keep only samples in metadata
        study_samples = metadata[metadata["study"] == study]["sample_id"].tolist()
        expr = expr[[c for c in study_samples if c in expr.columns]]
        frames.append(expr)

    # Align on common genes
    common_genes = frames[0].index
    for f in frames[1:]:
        common_genes = common_genes.intersection(f.index)
    aligned = [f.loc[common_genes] for f in frames]
    expression = pd.concat(aligned, axis=1)

    return expression, metadata


def compute_epithelial_reserve_score(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    genes: list[str] | None = None,
) -> pd.DataFrame:
    """Compute epithelial reserve score as mean z-score of signature genes.

    Z-scores are computed within each study to avoid batch effects.
    """
    if genes is None:
        genes = EPITHELIAL_GENES

    available = [g for g in genes if g in expression.index]
    if len(available) < len(genes):
        missing = set(genes) - set(available)
        print(f"Warning: missing genes: {missing}")

    records = []
    for study in STUDIES:
        study_samples = metadata[metadata["study"] == study]["sample_id"].tolist()
        study_samples = [s for s in study_samples if s in expression.columns]
        if not study_samples:
            continue

        study_expr = expression.loc[available, study_samples]
        # Z-score within study (gene-wise)
        z_scores = study_expr.apply(lambda row: stats.zscore(row.values), axis=1, result_type="expand")
        z_scores.columns = study_samples
        # Mean z-score across signature genes per sample
        score = z_scores.mean(axis=0)
        for sid in study_samples:
            records.append({"sample_id": sid, "study": study, "epithelial_reserve_score": float(score[sid])})

    scores_df = pd.DataFrame(records)
    scores_df = scores_df.merge(
        metadata[["sample_id", "response_status"]], on="sample_id", how="left"
    )
    scores_df["response_binary"] = (scores_df["response_status"] == "responder").astype(int)
    return scores_df


def loso_cv(scores_df: pd.DataFrame) -> dict:
    """Leave-one-study-out cross-validation with logistic regression."""
    studies = scores_df["study"].unique()
    all_y_true = []
    all_y_prob = []
    per_study_results = []

    for held_out in studies:
        train = scores_df[scores_df["study"] != held_out].copy()
        test = scores_df[scores_df["study"] == held_out].copy()

        X_train = train[["epithelial_reserve_score"]].values
        y_train = train["response_binary"].values
        X_test = test[["epithelial_reserve_score"]].values
        y_test = test["response_binary"].values

        clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        fold_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
        per_study_results.append({
            "held_out_study": held_out,
            "n_test": len(y_test),
            "n_responders": int(y_test.sum()),
            "n_non_responders": int(len(y_test) - y_test.sum()),
            "auc": round(fold_auc, 4),
            "coef": round(float(clf.coef_[0, 0]), 4),
            "intercept": round(float(clf.intercept_[0]), 4),
        })

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    overall_auc = roc_auc_score(all_y_true, all_y_prob)

    # Bootstrap 95% CI for AUC
    rng = np.random.RandomState(42)
    n_boot = 2000
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(all_y_true), size=len(all_y_true), replace=True)
        if len(np.unique(all_y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(all_y_true[idx], all_y_prob[idx]))
    ci_low = np.percentile(boot_aucs, 2.5)
    ci_high = np.percentile(boot_aucs, 97.5)

    return {
        "overall_auc": round(overall_auc, 4),
        "auc_95ci_low": round(ci_low, 4),
        "auc_95ci_high": round(ci_high, 4),
        "n_samples": len(all_y_true),
        "n_responders": int(all_y_true.sum()),
        "n_non_responders": int(len(all_y_true) - all_y_true.sum()),
        "per_study": per_study_results,
        "y_true": all_y_true,
        "y_prob": all_y_prob,
    }


def confound_check(scores_df: pd.DataFrame, expression: pd.DataFrame) -> dict:
    """Check whether epithelial reserve score correlates with inflammation markers."""
    # Potential confounders available in expression data
    inflammation_markers = {
        "CRP": "CRP",
        "S100A8": "S100A8",  # calprotectin subunit A
        "S100A9": "S100A9",  # calprotectin subunit B
        "TNF": "TNF",
        "IL6": "IL6",
        "IL1B": "IL1B",
        "CXCL8": "CXCL8",  # IL-8
    }

    results = {}
    for label, gene in inflammation_markers.items():
        if gene not in expression.index:
            results[label] = {"available": False}
            continue

        corr_records = []
        for _, row in scores_df.iterrows():
            sid = row["sample_id"]
            if sid in expression.columns:
                corr_records.append({
                    "epithelial_score": row["epithelial_reserve_score"],
                    "marker_value": expression.loc[gene, sid],
                })
        if not corr_records:
            results[label] = {"available": False}
            continue

        cdf = pd.DataFrame(corr_records)
        r, p = stats.spearmanr(cdf["epithelial_score"], cdf["marker_value"])
        results[label] = {
            "available": True,
            "spearman_r": round(float(r), 4),
            "p_value": round(float(p), 6),
            "n": len(cdf),
        }

    return results


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, auc: float, ci: tuple, out: Path) -> None:
    """Plot ROC curve with AUC and CI."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {auc:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Epithelial Reserve Score: Anti-TNF Response (LOSO-CV)")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_score_distribution(scores_df: pd.DataFrame, out: Path) -> None:
    """Plot epithelial reserve score distribution by response status and study."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # By response status
    for status, color in [("responder", "steelblue"), ("non_responder", "salmon")]:
        vals = scores_df[scores_df["response_status"] == status]["epithelial_reserve_score"]
        axes[0].hist(vals, bins=15, alpha=0.6, color=color, label=status, edgecolor="white")
    axes[0].set_xlabel("Epithelial Reserve Score")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Score Distribution by Response")
    axes[0].legend()

    # Box plot by study + response
    studies = scores_df["study"].unique()
    positions = []
    labels = []
    data_groups = []
    for i, study in enumerate(studies):
        for j, status in enumerate(["responder", "non_responder"]):
            vals = scores_df[
                (scores_df["study"] == study) & (scores_df["response_status"] == status)
            ]["epithelial_reserve_score"].values
            data_groups.append(vals)
            positions.append(i * 3 + j)
            labels.append(f"{study}\n{status[:4]}")

    bp = axes[1].boxplot(data_groups, positions=positions, widths=0.8, patch_artist=True)
    colors = ["steelblue", "salmon"] * len(studies)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    axes[1].set_ylabel("Epithelial Reserve Score")
    axes[1].set_title("Score by Study and Response")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_gene_heatmap(expression: pd.DataFrame, scores_df: pd.DataFrame, out: Path) -> None:
    """Heatmap of the 6 signature genes across samples ordered by score."""
    scores_df = scores_df.sort_values("epithelial_reserve_score")
    samples = [s for s in scores_df["sample_id"] if s in expression.columns]
    genes = [g for g in EPITHELIAL_GENES if g in expression.index]

    mat = expression.loc[genes, samples].values
    # Z-score per gene across all samples for visualization
    mat_z = (mat - mat.mean(axis=1, keepdims=True)) / (mat.std(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(mat_z, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3)
    ax.set_yticks(range(len(genes)))
    ax.set_yticklabels(genes)
    ax.set_xlabel("Samples (ordered by epithelial reserve score)")
    ax.set_title("Epithelial Reserve Genes Expression (z-score)")
    ax.set_xticks([])

    # Color bar for response
    resp_colors = []
    for s in samples:
        row = scores_df[scores_df["sample_id"] == s]
        if not row.empty and row.iloc[0]["response_status"] == "responder":
            resp_colors.append("steelblue")
        else:
            resp_colors.append("salmon")
    ax2 = fig.add_axes([ax.get_position().x0, 0.02, ax.get_position().width, 0.03])
    for i, c in enumerate(resp_colors):
        ax2.axvspan(i - 0.5, i + 0.5, color=c)
    ax2.set_xlim(-0.5, len(samples) - 0.5)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_xlabel("Blue=Responder, Red=Non-responder", fontsize=8)

    fig.colorbar(im, ax=ax, shrink=0.7, label="Z-score")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("Loading expression data...")
    expression, metadata = load_expression_data()
    print(f"  Expression: {expression.shape[0]} genes x {expression.shape[1]} samples")
    print(f"  Metadata: {len(metadata)} samples")

    print("Computing epithelial reserve score...")
    scores_df = compute_epithelial_reserve_score(expression, metadata)
    print(f"  Scores computed for {len(scores_df)} samples")

    # Descriptive stats
    for status in ["responder", "non_responder"]:
        vals = scores_df[scores_df["response_status"] == status]["epithelial_reserve_score"]
        print(f"  {status}: mean={vals.mean():.3f}, std={vals.std():.3f}, n={len(vals)}")
    t_stat, t_pval = stats.ttest_ind(
        scores_df[scores_df["response_binary"] == 1]["epithelial_reserve_score"],
        scores_df[scores_df["response_binary"] == 0]["epithelial_reserve_score"],
    )
    mwu_stat, mwu_pval = stats.mannwhitneyu(
        scores_df[scores_df["response_binary"] == 1]["epithelial_reserve_score"],
        scores_df[scores_df["response_binary"] == 0]["epithelial_reserve_score"],
        alternative="two-sided",
    )
    print(f"  Responder vs Non-responder: t={t_stat:.3f}, p={t_pval:.4f} (t-test)")
    print(f"  Mann-Whitney U: U={mwu_stat:.0f}, p={mwu_pval:.4f}")

    print("Running LOSO-CV...")
    results = loso_cv(scores_df)
    print(f"  Overall AUC: {results['overall_auc']:.4f} [{results['auc_95ci_low']:.4f}-{results['auc_95ci_high']:.4f}]")
    for fold in results["per_study"]:
        print(f"    {fold['held_out_study']}: AUC={fold['auc']:.4f} (n={fold['n_test']})")

    target_met = results["overall_auc"] >= 0.65
    print(f"  Target AUC > 0.65: {'MET' if target_met else 'NOT MET'}")

    print("Running confound check...")
    confounds = confound_check(scores_df, expression)
    for marker, res in confounds.items():
        if res.get("available"):
            sig = "*" if res["p_value"] < 0.05 else ""
            print(f"  {marker}: rho={res['spearman_r']:.3f}, p={res['p_value']:.4f}{sig}")
        else:
            print(f"  {marker}: not available")

    # Save results
    print("Saving results...")
    scores_df.to_csv(OUTPUT_DIR / "epithelial_reserve_scores.csv", index=False)

    summary = {
        "phase": "1.1",
        "description": "Epithelial reserve score (6-gene) as anti-TNF response predictor",
        "genes": EPITHELIAL_GENES,
        "scoring_method": "mean z-score (within-study normalization)",
        "cv_method": "leave-one-study-out",
        "classifier": "logistic regression (single feature)",
        "overall_auc": results["overall_auc"],
        "auc_95ci": [results["auc_95ci_low"], results["auc_95ci_high"]],
        "target_auc": 0.65,
        "target_met": target_met,
        "n_samples": results["n_samples"],
        "n_responders": results["n_responders"],
        "n_non_responders": results["n_non_responders"],
        "per_study_results": results["per_study"],
        "group_comparison": {
            "t_test": {"statistic": round(float(t_stat), 4), "p_value": round(float(t_pval), 6)},
            "mann_whitney_u": {"statistic": round(float(mwu_stat), 4), "p_value": round(float(mwu_pval), 6)},
        },
        "confound_check": confounds,
    }
    with open(OUTPUT_DIR / "phase1_1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    print("Generating plots...")
    plot_roc(
        results["y_true"],
        results["y_prob"],
        results["overall_auc"],
        (results["auc_95ci_low"], results["auc_95ci_high"]),
        figures_dir / "roc_epithelial_reserve_loso.png",
    )
    plot_score_distribution(scores_df, figures_dir / "score_distribution.png")
    plot_gene_heatmap(expression, scores_df, figures_dir / "gene_heatmap.png")

    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
