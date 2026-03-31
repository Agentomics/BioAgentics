#!/usr/bin/env python3
"""Phase 2.4: FAE burden scoring and reserve vs identity hypothesis testing.

Tests two competing hypotheses for anti-TNF non-response:
  H1 (reserve): Low epithelial reserve genes → poor barrier → non-response
  H2 (identity): High FAE burden → biologic-unresolved inflammation → non-response

Uses existing bulk expression data from GSE16879, GSE12251, GSE73661.
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

PROCESSED_DIR = Path("output/crohns/anti-tnf-response-prediction/processed")
OUTPUT_DIR = Path("output/crohns/cd-epithelial-reserve-treatment-stratification/phase2")
STUDIES = ["GSE16879", "GSE12251", "GSE73661"]

# Gene panels
FAE_GENES = ["GP2", "CCL20", "TNFAIP2", "MARCKSL1"]
RESERVE_GENES = ["PGC", "BPIFB1", "CPO", "GAS1", "CASP6", "SNX3"]
RESERVE_CORE = ["PGC", "BPIFB1", "CPO", "GAS1"]  # core 4 from task description


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load expression and metadata from all anti-TNF cohorts."""
    metadata = pd.read_csv(PROCESSED_DIR / "combined_metadata.csv")
    metadata = metadata[metadata["study"].isin(STUDIES)].copy()

    frames = []
    for study in STUDIES:
        expr = pd.read_csv(PROCESSED_DIR / f"{study}_expression.csv")
        expr = expr.set_index("gene_symbol")
        study_samples = metadata[metadata["study"] == study]["sample_id"].tolist()
        expr = expr[[c for c in study_samples if c in expr.columns]]
        frames.append(expr)

    common_genes = frames[0].index
    for f in frames[1:]:
        common_genes = common_genes.intersection(f.index)
    aligned = [f.loc[common_genes] for f in frames]
    expression = pd.concat(aligned, axis=1)
    return expression, metadata


def compute_score(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    genes: list[str],
    score_name: str,
) -> pd.DataFrame:
    """Compute mean z-score (within-study) for a gene panel."""
    available = [g for g in genes if g in expression.index]
    missing = set(genes) - set(available)
    if missing:
        print(f"  Warning: missing genes for {score_name}: {missing}")

    records = []
    for study in STUDIES:
        study_samples = metadata[metadata["study"] == study]["sample_id"].tolist()
        study_samples = [s for s in study_samples if s in expression.columns]
        if not study_samples:
            continue

        study_expr = expression.loc[available, study_samples]
        z_scores = study_expr.apply(
            lambda row: stats.zscore(row.values), axis=1, result_type="expand"
        )
        z_scores.columns = study_samples
        score = z_scores.mean(axis=0)
        for sid in study_samples:
            records.append({"sample_id": sid, "study": study, score_name: float(score[sid])})

    return pd.DataFrame(records)


def loso_cv_single_feature(
    df: pd.DataFrame, feature_col: str
) -> dict:
    """LOSO-CV with logistic regression on a single feature."""
    studies = df["study"].unique()
    all_y_true, all_y_prob = [], []
    per_study = []

    for held_out in studies:
        train = df[df["study"] != held_out]
        test = df[df["study"] == held_out]

        X_train = train[[feature_col]].values
        y_train = train["response_binary"].values
        X_test = test[[feature_col]].values
        y_test = test["response_binary"].values

        clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        fold_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
        per_study.append({
            "held_out_study": held_out,
            "n_test": len(y_test),
            "auc": round(fold_auc, 4),
            "coef": round(float(clf.coef_[0, 0]), 4),
        })
        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    overall_auc = roc_auc_score(all_y_true, all_y_prob)

    # Bootstrap 95% CI
    rng = np.random.RandomState(42)
    boot_aucs = []
    for _ in range(2000):
        idx = rng.choice(len(all_y_true), size=len(all_y_true), replace=True)
        if len(np.unique(all_y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(all_y_true[idx], all_y_prob[idx]))

    return {
        "overall_auc": round(overall_auc, 4),
        "auc_95ci": [round(np.percentile(boot_aucs, 2.5), 4), round(np.percentile(boot_aucs, 97.5), 4)],
        "per_study": per_study,
        "y_true": all_y_true,
        "y_prob": all_y_prob,
    }


def loso_cv_combined(
    df: pd.DataFrame, feature_cols: list[str]
) -> dict:
    """LOSO-CV with logistic regression on multiple features."""
    studies = df["study"].unique()
    all_y_true, all_y_prob = [], []
    per_study = []

    for held_out in studies:
        train = df[df["study"] != held_out]
        test = df[df["study"] == held_out]

        X_train = train[feature_cols].values
        y_train = train["response_binary"].values
        X_test = test[feature_cols].values
        y_test = test["response_binary"].values

        clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        fold_auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else float("nan")
        per_study.append({
            "held_out_study": held_out,
            "n_test": len(y_test),
            "auc": round(fold_auc, 4),
        })
        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    overall_auc = roc_auc_score(all_y_true, all_y_prob)

    rng = np.random.RandomState(42)
    boot_aucs = []
    for _ in range(2000):
        idx = rng.choice(len(all_y_true), size=len(all_y_true), replace=True)
        if len(np.unique(all_y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(all_y_true[idx], all_y_prob[idx]))

    return {
        "overall_auc": round(overall_auc, 4),
        "auc_95ci": [round(np.percentile(boot_aucs, 2.5), 4), round(np.percentile(boot_aucs, 97.5), 4)],
        "per_study": per_study,
        "y_true": all_y_true,
        "y_prob": all_y_prob,
    }


def region_analysis(
    df: pd.DataFrame, metadata: pd.DataFrame
) -> dict:
    """Analyze FAE/reserve scores by biopsy region (ileal vs colonic)."""
    # Merge tissue type info
    tissue_map = metadata.set_index("sample_id")["tissue_type"].to_dict()
    df = df.copy()
    df["tissue"] = df["sample_id"].map(tissue_map)
    # Normalize tissue labels
    df["region"] = df["tissue"].apply(
        lambda x: "ileum" if "ileum" in str(x).lower() or "ileal" in str(x).lower()
        else "colon"
    )

    results = {"region_counts": df.groupby(["study", "region"]).size().to_dict()}
    # Convert tuple keys to string for JSON serialization
    results["region_counts"] = {f"{k[0]}_{k[1]}": int(v) for k, v in results["region_counts"].items()}

    # Per-region group comparisons
    for region in ["ileum", "colon"]:
        subset = df[df["region"] == region]
        if len(subset) < 5 or subset["response_binary"].nunique() < 2:
            results[f"{region}_analysis"] = {"n": len(subset), "skip": "insufficient samples"}
            continue

        region_res = {"n": len(subset)}
        for score_col in ["fae_burden_score", "reserve_score"]:
            resp = subset[subset["response_binary"] == 1][score_col]
            non_resp = subset[subset["response_binary"] == 0][score_col]
            if len(resp) >= 2 and len(non_resp) >= 2:
                t_stat, t_p = stats.ttest_ind(resp, non_resp)
                mwu_stat, mwu_p = stats.mannwhitneyu(resp, non_resp, alternative="two-sided")
                region_res[score_col] = {
                    "resp_mean": round(float(resp.mean()), 4),
                    "non_resp_mean": round(float(non_resp.mean()), 4),
                    "diff": round(float(resp.mean() - non_resp.mean()), 4),
                    "t_p": round(float(t_p), 6),
                    "mwu_p": round(float(mwu_p), 6),
                    "n_resp": len(resp),
                    "n_non_resp": len(non_resp),
                }
        results[f"{region}_analysis"] = region_res

    return results


def correlation_matrix(
    expression: pd.DataFrame, metadata: pd.DataFrame
) -> dict:
    """Cross-reference FAE and reserve gene expression (Spearman correlations)."""
    all_genes = FAE_GENES + RESERVE_CORE
    available = [g for g in all_genes if g in expression.index]

    # Within-study z-score then concatenate
    z_frames = []
    for study in STUDIES:
        study_samples = metadata[metadata["study"] == study]["sample_id"].tolist()
        study_samples = [s for s in study_samples if s in expression.columns]
        study_expr = expression.loc[available, study_samples]
        z_scores = study_expr.apply(
            lambda row: stats.zscore(row.values), axis=1, result_type="expand"
        )
        z_scores.columns = study_samples
        z_frames.append(z_scores)

    z_all = pd.concat(z_frames, axis=1)

    # Spearman correlation matrix
    corr_results = {}
    for g1 in FAE_GENES:
        if g1 not in z_all.index:
            continue
        for g2 in RESERVE_CORE:
            if g2 not in z_all.index:
                continue
            r, p = stats.spearmanr(z_all.loc[g1].values, z_all.loc[g2].values)
            corr_results[f"{g1}_vs_{g2}"] = {
                "spearman_r": round(float(r), 4),
                "p_value": round(float(p), 6),
            }

    return corr_results


def plot_hypothesis_comparison(
    fae_res: dict, reserve_res: dict, combined_res: dict, out: Path
) -> None:
    """Bar chart comparing H1, H2, combined AUCs per study and overall."""
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = ["Overall"] + [r["held_out_study"] for r in reserve_res["per_study"]]
    reserve_aucs = [reserve_res["overall_auc"]] + [r["auc"] for r in reserve_res["per_study"]]
    fae_aucs = [fae_res["overall_auc"]] + [r["auc"] for r in fae_res["per_study"]]
    combined_aucs = [combined_res["overall_auc"]] + [r["auc"] for r in combined_res["per_study"]]

    x = np.arange(len(labels))
    w = 0.25
    ax.bar(x - w, reserve_aucs, w, label="H1: Reserve", color="steelblue", alpha=0.8)
    ax.bar(x, fae_aucs, w, label="H2: FAE Identity", color="coral", alpha=0.8)
    ax.bar(x + w, combined_aucs, w, label="Combined", color="seagreen", alpha=0.8)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Held-out Study (LOSO-CV)")
    ax.set_title("H1 (Reserve) vs H2 (FAE Identity) vs Combined\nAnti-TNF Response Prediction")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, 1.05)

    for i, (r, f, c) in enumerate(zip(reserve_aucs, fae_aucs, combined_aucs)):
        ax.text(i - w, r + 0.02, f"{r:.3f}", ha="center", fontsize=7)
        ax.text(i, f + 0.02, f"{f:.3f}", ha="center", fontsize=7)
        ax.text(i + w, c + 0.02, f"{c:.3f}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(corr_results: dict, out: Path) -> None:
    """Heatmap of FAE vs reserve gene correlations."""
    fae_genes_avail = sorted(set(k.split("_vs_")[0] for k in corr_results))
    res_genes_avail = sorted(set(k.split("_vs_")[1] for k in corr_results))

    mat = np.zeros((len(fae_genes_avail), len(res_genes_avail)))
    for i, fg in enumerate(fae_genes_avail):
        for j, rg in enumerate(res_genes_avail):
            key = f"{fg}_vs_{rg}"
            mat[i, j] = corr_results[key]["spearman_r"] if key in corr_results else 0

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(res_genes_avail)))
    ax.set_xticklabels(res_genes_avail, rotation=45, ha="right")
    ax.set_yticks(range(len(fae_genes_avail)))
    ax.set_yticklabels(fae_genes_avail)
    ax.set_title("FAE vs Reserve Gene Correlations (Spearman)")

    for i in range(len(fae_genes_avail)):
        for j in range(len(res_genes_avail)):
            key = f"{fae_genes_avail[i]}_vs_{res_genes_avail[j]}"
            p = corr_results.get(key, {}).get("p_value", 1)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            ax.text(j, i, f"{mat[i, j]:.2f}{sig}", ha="center", va="center", fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8, label="Spearman r")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_scores_by_region(df: pd.DataFrame, out: Path) -> None:
    """Box plots of FAE and reserve scores by region and response."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, score_col, title in [
        (axes[0], "reserve_score", "Epithelial Reserve Score"),
        (axes[1], "fae_burden_score", "FAE Burden Score"),
    ]:
        groups = []
        labels = []
        colors = []
        for region in ["ileum", "colon"]:
            for resp in [1, 0]:
                subset = df[(df["region"] == region) & (df["response_binary"] == resp)]
                groups.append(subset[score_col].values)
                labels.append(f"{region}\n{'R' if resp else 'NR'}")
                colors.append("steelblue" if resp else "salmon")

        bp = ax.boxplot(groups, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_xticklabels(labels)
        ax.set_ylabel(score_col)
        ax.set_title(title)
        ax.axhline(0, color="gray", ls="--", lw=0.5)

    fig.suptitle("Scores by Biopsy Region and Response", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    expression, metadata = load_data()
    print(f"  {expression.shape[0]} genes x {expression.shape[1]} samples")

    # Step 1: Compute scores
    print("Computing FAE burden score...")
    fae_scores = compute_score(expression, metadata, FAE_GENES, "fae_burden_score")
    print("Computing reserve score...")
    reserve_scores = compute_score(expression, metadata, RESERVE_GENES, "reserve_score")

    # Merge scores with response info
    resp_info = metadata[["sample_id", "response_status"]].copy()
    resp_info["response_binary"] = (resp_info["response_status"] == "responder").astype(int)

    df = fae_scores.merge(reserve_scores[["sample_id", "reserve_score"]], on="sample_id")
    df = df.merge(resp_info, on="sample_id", how="left")
    print(f"  {len(df)} samples with both scores")

    # Descriptive stats
    print("\nDescriptive statistics:")
    for score_col in ["fae_burden_score", "reserve_score"]:
        for status in ["responder", "non_responder"]:
            vals = df[df["response_status"] == status][score_col]
            print(f"  {score_col} [{status}]: mean={vals.mean():.4f}, std={vals.std():.4f}, n={len(vals)}")

    # Step 2: LOSO-CV head-to-head
    print("\n--- H1 (Reserve) LOSO-CV ---")
    reserve_cv = loso_cv_single_feature(df, "reserve_score")
    print(f"  AUC={reserve_cv['overall_auc']:.4f} {reserve_cv['auc_95ci']}")
    for r in reserve_cv["per_study"]:
        print(f"    {r['held_out_study']}: AUC={r['auc']:.4f}, coef={r['coef']}")

    print("\n--- H2 (FAE Identity) LOSO-CV ---")
    fae_cv = loso_cv_single_feature(df, "fae_burden_score")
    print(f"  AUC={fae_cv['overall_auc']:.4f} {fae_cv['auc_95ci']}")
    for r in fae_cv["per_study"]:
        print(f"    {r['held_out_study']}: AUC={r['auc']:.4f}, coef={r['coef']}")

    print("\n--- Combined (Reserve + FAE) LOSO-CV ---")
    combined_cv = loso_cv_combined(df, ["reserve_score", "fae_burden_score"])
    print(f"  AUC={combined_cv['overall_auc']:.4f} {combined_cv['auc_95ci']}")
    for r in combined_cv["per_study"]:
        print(f"    {r['held_out_study']}: AUC={r['auc']:.4f}")

    # Step 3: Region analysis
    print("\n--- Biopsy Region Analysis ---")
    tissue_map = metadata.set_index("sample_id")["tissue_type"].to_dict()
    df["tissue"] = df["sample_id"].map(tissue_map)
    df["region"] = df["tissue"].apply(
        lambda x: "ileum" if "ileum" in str(x).lower() or "ileal" in str(x).lower()
        else "colon"
    )
    print(f"  Region distribution: {df['region'].value_counts().to_dict()}")
    print(f"  Per study:")
    for study in STUDIES:
        study_df = df[df["study"] == study]
        print(f"    {study}: {study_df['region'].value_counts().to_dict()}")

    region_res = region_analysis(df, metadata)
    for region in ["ileum", "colon"]:
        key = f"{region}_analysis"
        if key in region_res:
            ra = region_res[key]
            print(f"\n  {region.upper()} (n={ra['n']}):")
            for score_col in ["fae_burden_score", "reserve_score"]:
                if score_col in ra:
                    s = ra[score_col]
                    print(f"    {score_col}: resp={s['resp_mean']:.4f} vs non_resp={s['non_resp_mean']:.4f}, "
                          f"diff={s['diff']:.4f}, t_p={s['t_p']:.4f}, mwu_p={s['mwu_p']:.4f}")

    # Step 4: Cross-reference correlations
    print("\n--- FAE vs Reserve Gene Correlations ---")
    corr_res = correlation_matrix(expression, metadata)
    for pair, vals in sorted(corr_res.items()):
        sig = "*" if vals["p_value"] < 0.05 else ""
        print(f"  {pair}: rho={vals['spearman_r']:.4f}, p={vals['p_value']:.6f}{sig}")

    # Step 5: Determine winner
    delta = fae_cv["overall_auc"] - reserve_cv["overall_auc"]
    if delta > 0.05:
        winner = "H2 (FAE Identity)"
    elif delta < -0.05:
        winner = "H1 (Reserve)"
    else:
        winner = "Neither hypothesis clearly dominant (delta < 0.05 AUC)"
    print(f"\n  Verdict: {winner} (delta AUC = {delta:+.4f})")

    # Save outputs
    print("\nSaving results...")
    df.to_csv(OUTPUT_DIR / "fae_reserve_scores.csv", index=False)

    # Remove numpy arrays for JSON serialization
    for res in [reserve_cv, fae_cv, combined_cv]:
        res.pop("y_true", None)
        res.pop("y_prob", None)

    summary = {
        "phase": "2.4",
        "description": "FAE burden vs epithelial reserve hypothesis comparison",
        "fae_genes": FAE_GENES,
        "reserve_genes": RESERVE_GENES,
        "h1_reserve_loso_cv": reserve_cv,
        "h2_fae_identity_loso_cv": fae_cv,
        "combined_loso_cv": combined_cv,
        "delta_auc_fae_minus_reserve": round(delta, 4),
        "verdict": winner,
        "region_analysis": region_res,
        "gene_correlations": corr_res,
        "tissue_composition": {
            "GSE16879": {"colon": 19, "ileum": 18},
            "GSE12251": {"colon": 23},
            "GSE73661": {"colon": 23},
        },
        "key_finding_gse73661": "All colonic biopsies — FAE (ileum-specific) pathway irrelevant here. "
        "Direction reversal in Phase 1 NOT explained by FAE dominance; must reflect "
        "colonic-specific biology where reserve genes have different significance.",
    }
    with open(OUTPUT_DIR / "phase2_4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    # Reload y_true/y_prob for plots
    reserve_cv2 = loso_cv_single_feature(df, "reserve_score")
    fae_cv2 = loso_cv_single_feature(df, "fae_burden_score")
    combined_cv2 = loso_cv_combined(df, ["reserve_score", "fae_burden_score"])

    plot_hypothesis_comparison(fae_cv, reserve_cv, combined_cv, OUTPUT_DIR / "hypothesis_comparison.png")
    plot_correlation_heatmap(corr_res, OUTPUT_DIR / "fae_reserve_correlation.png")
    plot_scores_by_region(df, OUTPUT_DIR / "scores_by_region.png")

    # ROC curves for both hypotheses
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, res, title in [
        (axes[0], reserve_cv2, f"H1: Reserve (AUC={reserve_cv2['overall_auc']:.3f})"),
        (axes[1], fae_cv2, f"H2: FAE Identity (AUC={fae_cv2['overall_auc']:.3f})"),
        (axes[2], combined_cv2, f"Combined (AUC={combined_cv2['overall_auc']:.3f})"),
    ]:
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, lw=2, color="steelblue")
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=0.8)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(title)
        ax.set_aspect("equal")
    fig.suptitle("ROC Curves: Reserve vs FAE Identity (LOSO-CV)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_h1_vs_h2.png", dpi=150)
    plt.close(fig)

    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
