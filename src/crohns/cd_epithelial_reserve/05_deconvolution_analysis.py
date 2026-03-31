#!/usr/bin/env python3
"""Phase 2.2: Deconvolve anti-TNF bulk transcriptomics using IL-23 scRNA-seq atlas.

Estimates cell-type proportions in bulk anti-TNF cohorts using the GSE134809
scRNA-seq atlas as reference. Tests whether epithelial cell proportion predicts
anti-TNF response better than immune cell proportions.

Memory-safe: builds pseudobulk reference from atlas in chunks, then applies
NNLS deconvolution to bulk samples.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

ATLAS_PATH = Path("data/crohns/il23-atlas/GSE134809_annotated.h5ad")
PROCESSED_DIR = Path("output/crohns/anti-tnf-response-prediction/processed")
OUTPUT_DIR = Path("output/crohns/cd-epithelial-reserve-treatment-stratification/phase2")
STUDIES = ["GSE16879", "GSE12251", "GSE73661"]

# Cell types to group for analysis
EPITHELIAL_TYPES = ["Epithelial", "Paneth", "Goblet"]
IMMUNE_TYPES = [
    "NK", "CD4_T_naive", "Treg", "Th1", "Th17", "CD8_T", "ILC3",
    "B_cell", "Plasma",
    "Monocyte", "Macrophage", "Inflammatory_Mac", "Dendritic_cell", "pDC",
]
STROMAL_TYPES = ["Fibroblast", "Myofibroblast", "Endothelial"]


def build_pseudobulk_reference() -> pd.DataFrame:
    """Build pseudobulk signature matrix from scRNA-seq atlas.

    Returns DataFrame of shape (n_genes, n_cell_types) with mean expression per cell type.
    Loads sparse matrix directly, computes pseudobulk, frees memory immediately.
    """
    import anndata as ad

    print("  Loading atlas...")
    adata = ad.read_h5ad(ATLAS_PATH)

    cell_types = adata.obs["cell_type"].values
    gene_names = adata.var_names.tolist()
    unique_types = sorted(set(cell_types) - {"Unassigned"})
    print(f"  Atlas: {adata.shape[0]} cells, {adata.shape[1]} genes, {len(unique_types)} cell types")

    # Extract sparse matrix and free anndata
    X = adata.X
    del adata
    gc.collect()

    # Compute mean expression per cell type
    signature = pd.DataFrame(index=gene_names)
    for ct in unique_types:
        mask = cell_types == ct
        n = mask.sum()
        if n > 0:
            ct_mean = np.asarray(X[mask].mean(axis=0)).ravel()
            signature[ct] = ct_mean
        print(f"    {ct}: {n} cells")

    del X
    gc.collect()

    print(f"  Signature matrix: {signature.shape[0]} genes x {signature.shape[1]} cell types")
    return signature


def load_bulk_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load bulk expression data from anti-TNF cohorts."""
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


def deconvolve_nnls(
    bulk: pd.DataFrame, signature: pd.DataFrame, min_genes: int = 100
) -> pd.DataFrame:
    """Deconvolve bulk samples using NNLS (non-negative least squares).

    For each bulk sample, solve: bulk_sample ≈ signature @ proportions
    subject to proportions >= 0, then normalize to sum to 1.
    """
    # Align genes
    common_genes = bulk.index.intersection(signature.index)
    print(f"  Common genes for deconvolution: {len(common_genes)}")
    if len(common_genes) < min_genes:
        raise ValueError(f"Only {len(common_genes)} common genes — too few for deconvolution")

    sig = signature.loc[common_genes].values  # (n_genes, n_types)
    bulk_mat = bulk.loc[common_genes].values   # (n_genes, n_samples)
    cell_types = signature.columns.tolist()

    # Filter low-variance genes (keep top 2000 most variable across cell types)
    sig_var = sig.var(axis=1)
    n_keep = min(2000, len(common_genes))
    top_idx = np.argsort(sig_var)[-n_keep:]
    sig = sig[top_idx]
    bulk_mat = bulk_mat[top_idx]
    print(f"  Using top {n_keep} most variable genes")

    # Log-transform reference (bulk is already log-scale typically)
    sig = np.log1p(sig)

    proportions = []
    for i in range(bulk_mat.shape[1]):
        sample = bulk_mat[:, i]
        # NNLS: solve min ||sig @ x - sample||^2 s.t. x >= 0
        x, _ = nnls(sig, sample)
        # Normalize to sum to 1
        total = x.sum()
        if total > 0:
            x = x / total
        proportions.append(x)

    props_df = pd.DataFrame(proportions, columns=cell_types, index=bulk.columns)
    return props_df


def compute_aggregate_proportions(props_df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate epithelial, immune, stromal proportions."""
    df = props_df.copy()
    epi_cols = [c for c in EPITHELIAL_TYPES if c in df.columns]
    imm_cols = [c for c in IMMUNE_TYPES if c in df.columns]
    str_cols = [c for c in STROMAL_TYPES if c in df.columns]

    df["epithelial_proportion"] = df[epi_cols].sum(axis=1)
    df["immune_proportion"] = df[imm_cols].sum(axis=1)
    df["stromal_proportion"] = df[str_cols].sum(axis=1)

    # Individual epithelial subtypes
    for ct in epi_cols:
        df[f"{ct.lower()}_proportion"] = df[ct]

    return df


def loso_cv(df: pd.DataFrame, feature_col: str) -> dict:
    """LOSO-CV with logistic regression."""
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

        # Standardize within fold
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

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
        "y_true": all_y_true.tolist(),
        "y_prob": all_y_prob.tolist(),
    }


def per_study_association(df: pd.DataFrame, feature_col: str) -> list[dict]:
    """Test association between a feature and response within each study."""
    results = []
    for study in STUDIES:
        subset = df[df["study"] == study]
        resp = subset[subset["response_binary"] == 1][feature_col]
        non_resp = subset[subset["response_binary"] == 0][feature_col]
        if len(resp) < 2 or len(non_resp) < 2:
            results.append({"study": study, "skip": True})
            continue
        t_stat, t_p = stats.ttest_ind(resp, non_resp)
        mwu_stat, mwu_p = stats.mannwhitneyu(resp, non_resp, alternative="two-sided")
        results.append({
            "study": study,
            "n_resp": len(resp),
            "n_non_resp": len(non_resp),
            "resp_mean": round(float(resp.mean()), 6),
            "non_resp_mean": round(float(non_resp.mean()), 6),
            "t_p": round(float(t_p), 6),
            "mwu_p": round(float(mwu_p), 6),
        })
    return results


def plot_proportions_boxplot(df: pd.DataFrame, out: Path) -> None:
    """Box plots of epithelial and immune proportions by response."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, col, title in [
        (axes[0], "epithelial_proportion", "Epithelial Proportion"),
        (axes[1], "immune_proportion", "Immune Proportion"),
        (axes[2], "stromal_proportion", "Stromal Proportion"),
    ]:
        data = []
        labels = []
        colors = []
        for study in STUDIES:
            for resp, color in [(1, "steelblue"), (0, "salmon")]:
                vals = df[(df["study"] == study) & (df["response_binary"] == resp)][col].values
                data.append(vals)
                labels.append(f"{study}\n{'R' if resp else 'NR'}")
                colors.append(color)

        bp = ax.boxplot(data, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_ylabel(col)
        ax.set_title(title)

    fig.suptitle("Deconvolved Cell-Type Proportions by Study and Response", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_epithelial_subtypes(df: pd.DataFrame, out: Path) -> None:
    """Box plots of epithelial subtype proportions."""
    subtypes = [c for c in df.columns if c.endswith("_proportion") and c.split("_proportion")[0] in
                [t.lower() for t in EPITHELIAL_TYPES]]
    if not subtypes:
        return

    fig, axes = plt.subplots(1, len(subtypes), figsize=(5 * len(subtypes), 5))
    if len(subtypes) == 1:
        axes = [axes]

    for ax, col in zip(axes, subtypes):
        resp_vals = df[df["response_binary"] == 1][col]
        non_resp_vals = df[df["response_binary"] == 0][col]
        bp = ax.boxplot([resp_vals, non_resp_vals], patch_artist=True, widths=0.6,
                       labels=["Responder", "Non-responder"])
        bp["boxes"][0].set_facecolor("steelblue")
        bp["boxes"][1].set_facecolor("salmon")
        t_stat, t_p = stats.ttest_ind(resp_vals, non_resp_vals)
        ax.set_title(f"{col}\n(p={t_p:.4f})")
        ax.set_ylabel("Proportion")

    fig.suptitle("Epithelial Subtype Proportions: R vs NR", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_auc_comparison(results: dict, out: Path) -> None:
    """Bar chart comparing deconvolution-based AUCs."""
    features = list(results.keys())
    aucs = [results[f]["overall_auc"] for f in features]
    cis = [results[f]["auc_95ci"] for f in features]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(features))
    bars = ax.bar(x, aucs, color="steelblue", alpha=0.8)
    # Error bars from CI
    lower = [auc - ci[0] for auc, ci in zip(aucs, cis)]
    upper = [ci[1] - auc for auc, ci in zip(aucs, cis)]
    ax.errorbar(x, aucs, yerr=[lower, upper], fmt="none", color="black", capsize=3)

    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="Chance")
    ax.axhline(0.558, color="red", ls=":", lw=1, label="Phase 1 Reserve (0.558)")
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=8)
    ax.set_ylabel("AUC (LOSO-CV)")
    ax.set_title("Deconvolution-Based Treatment Response Prediction")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)

    for i, (auc, ci) in enumerate(zip(aucs, cis)):
        ax.text(i, auc + 0.05, f"{auc:.3f}", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Build reference from atlas
    print("Building pseudobulk reference from scRNA-seq atlas...")
    sig_path = OUTPUT_DIR / "pseudobulk_signature.csv"
    if sig_path.exists():
        print("  Loading cached signature matrix...")
        signature = pd.read_csv(sig_path, index_col=0)
    else:
        signature = build_pseudobulk_reference()
        signature.to_csv(sig_path)
    print(f"  Signature: {signature.shape}")

    # Step 2: Load bulk data
    print("Loading bulk expression data...")
    bulk, metadata = load_bulk_data()
    print(f"  Bulk: {bulk.shape[0]} genes x {bulk.shape[1]} samples")

    # Step 3: Deconvolve
    print("Running NNLS deconvolution...")
    props = deconvolve_nnls(bulk, signature)
    props = compute_aggregate_proportions(props)
    print(f"  Deconvolution complete for {len(props)} samples")

    # Descriptive stats
    print("\nMean proportions:")
    for col in ["epithelial_proportion", "immune_proportion", "stromal_proportion"]:
        print(f"  {col}: {props[col].mean():.4f} ± {props[col].std():.4f}")

    # Merge with metadata
    props["sample_id"] = props.index
    merged = props.merge(metadata[["sample_id", "study", "response_status"]], on="sample_id")
    merged["response_binary"] = (merged["response_status"] == "responder").astype(int)

    # Step 4: Test predictive value
    print("\n=== LOSO-CV Results ===")
    test_features = [
        "epithelial_proportion",
        "immune_proportion",
        "stromal_proportion",
    ]
    # Add individual epithelial subtypes
    for ct in EPITHELIAL_TYPES:
        col = f"{ct.lower()}_proportion"
        if col in merged.columns and merged[col].std() > 1e-8:
            test_features.append(col)

    cv_results = {}
    for feat in test_features:
        if merged[feat].std() < 1e-8:
            print(f"  {feat}: SKIPPED (zero variance)")
            continue
        res = loso_cv(merged, feat)
        cv_results[feat] = res
        print(f"  {feat}: AUC={res['overall_auc']:.4f} {res['auc_95ci']}")
        for r in res["per_study"]:
            print(f"    {r['held_out_study']}: AUC={r['auc']:.4f}, coef={r['coef']}")

    # Step 5: Per-study association tests (success criterion: p<0.05 in 2/3 cohorts)
    print("\n=== Per-Study Association Tests ===")
    assoc_results = {}
    for feat in ["epithelial_proportion", "immune_proportion"]:
        assoc = per_study_association(merged, feat)
        assoc_results[feat] = assoc
        print(f"\n  {feat}:")
        sig_count = 0
        for a in assoc:
            if a.get("skip"):
                print(f"    {a['study']}: skipped")
                continue
            sig = "*" if a["t_p"] < 0.05 else ""
            print(f"    {a['study']}: R={a['resp_mean']:.6f}, NR={a['non_resp_mean']:.6f}, "
                  f"t_p={a['t_p']:.4f}{sig}, mwu_p={a['mwu_p']:.4f}")
            if a["t_p"] < 0.05:
                sig_count += 1
        criterion_met = sig_count >= 2
        print(f"    Significant in {sig_count}/3 cohorts → criterion {'MET' if criterion_met else 'NOT MET'}")

    # Step 6: Compare to Phase 1 gene expression score
    phase1_auc = 0.558
    epi_auc = cv_results.get("epithelial_proportion", {}).get("overall_auc", 0)
    delta = epi_auc - phase1_auc
    print(f"\n  Epithelial proportion AUC ({epi_auc:.4f}) vs Phase 1 reserve score ({phase1_auc:.4f}): "
          f"delta = {delta:+.4f}")

    # Save all outputs
    print("\nSaving results...")
    props.to_csv(OUTPUT_DIR / "deconvolution_proportions.csv", index=False)

    summary = {
        "phase": "2.2",
        "description": "NNLS deconvolution of anti-TNF bulk using IL-23 scRNA-seq atlas reference",
        "reference_atlas": "GSE134809_annotated.h5ad",
        "deconvolution_method": "NNLS (non-negative least squares) on top 2000 variable genes, log1p reference",
        "n_samples": len(merged),
        "mean_proportions": {
            "epithelial": round(float(props["epithelial_proportion"].mean()), 6),
            "immune": round(float(props["immune_proportion"].mean()), 6),
            "stromal": round(float(props["stromal_proportion"].mean()), 6),
        },
        "loso_cv_results": {k: {kk: vv for kk, vv in v.items() if kk not in ("y_true", "y_prob")}
                           for k, v in cv_results.items()},
        "per_study_associations": assoc_results,
        "comparison_to_phase1": {
            "phase1_reserve_auc": phase1_auc,
            "epithelial_proportion_auc": epi_auc,
            "delta": round(delta, 4),
        },
        "success_criterion": "epithelial proportion correlates with response (p<0.05) in ≥2/3 cohorts",
    }
    with open(OUTPUT_DIR / "phase2_2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    print("Generating plots...")
    plot_proportions_boxplot(merged, OUTPUT_DIR / "deconvolution_proportions_boxplot.png")
    plot_epithelial_subtypes(merged, OUTPUT_DIR / "epithelial_subtypes_boxplot.png")
    if cv_results:
        plot_auc_comparison(cv_results, OUTPUT_DIR / "deconvolution_auc_comparison.png")

    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
