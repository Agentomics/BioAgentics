#!/usr/bin/env python3
"""Phase 1.3: Expanded epithelial defense panel vs 6-gene signature.

Adds Paneth cell markers (DEFA5, DEFA6, LYZ, REG3A) and goblet cell markers
(MUC2, TFF3, CLCA1) to the original 6-gene signature. Compares 6-gene,
13-gene, and univariate-filtered subsets via LOSO-CV.
"""

from __future__ import annotations

import importlib.util
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

# Import Phase 1.1 module
_spec = importlib.util.spec_from_file_location(
    "phase1_1",
    Path(__file__).parent / "01_epithelial_reserve_score.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

EPITHELIAL_GENES = _mod.EPITHELIAL_GENES
STUDIES = _mod.STUDIES
load_expression_data = _mod.load_expression_data

OUTPUT_DIR = Path("output/crohns/cd-epithelial-reserve-treatment-stratification")

# Additional markers
PANETH_MARKERS = ["DEFA5", "DEFA6", "LYZ", "REG3A"]
GOBLET_MARKERS = ["MUC2", "TFF3", "CLCA1"]
EXPANDED_GENES = EPITHELIAL_GENES + PANETH_MARKERS + GOBLET_MARKERS


def compute_score_for_genes(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    genes: list[str],
) -> pd.DataFrame:
    """Compute mean z-score for given gene list (within-study normalization)."""
    available = [g for g in genes if g in expression.index]
    records = []
    for study in STUDIES:
        study_samples = metadata[metadata["study"] == study]["sample_id"].tolist()
        study_samples = [s for s in study_samples if s in expression.columns]
        if not study_samples:
            continue

        study_expr = expression.loc[available, study_samples]
        z_scores = study_expr.apply(lambda row: stats.zscore(row.values), axis=1, result_type="expand")
        z_scores.columns = study_samples
        score = z_scores.mean(axis=0)
        for sid in study_samples:
            records.append({"sample_id": sid, "study": study, "score": float(score[sid])})

    df = pd.DataFrame(records)
    df = df.merge(metadata[["sample_id", "response_status"]], on="sample_id", how="left")
    df["response_binary"] = (df["response_status"] == "responder").astype(int)
    return df


def loso_cv_single(df: pd.DataFrame) -> tuple[float, list[float], np.ndarray, np.ndarray]:
    """LOSO-CV with logistic regression. Returns overall AUC, bootstrap CIs, y_true, y_prob."""
    all_y_true = []
    all_y_prob = []

    for held_out in df["study"].unique():
        train = df[df["study"] != held_out]
        test = df[df["study"] == held_out]

        clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(train[["score"]].values, train["response_binary"].values)
        y_prob = clf.predict_proba(test[["score"]].values)[:, 1]

        all_y_true.extend(test["response_binary"].values.tolist())
        all_y_prob.extend(y_prob.tolist())

    y_true = np.array(all_y_true)
    y_prob = np.array(all_y_prob)
    auc = roc_auc_score(y_true, y_prob)

    rng = np.random.RandomState(42)
    boot_aucs = []
    for _ in range(2000):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    ci = [round(np.percentile(boot_aucs, 2.5), 4), round(np.percentile(boot_aucs, 97.5), 4)]
    return round(float(auc), 4), ci, y_true, y_prob


def univariate_filter(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    candidate_genes: list[str],
    top_k: int = 6,
) -> list[str]:
    """Select top genes by univariate association with response (Mann-Whitney U).

    Uses all samples pooled (not LOSO) for filtering. This is a discovery step.
    """
    available = [g for g in candidate_genes if g in expression.index]
    samples = [s for s in metadata["sample_id"] if s in expression.columns]
    resp = metadata.set_index("sample_id").loc[samples, "response_status"]

    resp_samples = resp[resp == "responder"].index.tolist()
    nonresp_samples = resp[resp == "non_responder"].index.tolist()

    gene_scores = []
    for gene in available:
        vals_r = expression.loc[gene, resp_samples].values.astype(float)
        vals_nr = expression.loc[gene, nonresp_samples].values.astype(float)
        try:
            _, p = stats.mannwhitneyu(vals_r, vals_nr, alternative="two-sided")
        except ValueError:
            p = 1.0
        gene_scores.append((gene, p))

    gene_scores.sort(key=lambda x: x[1])
    selected = [g for g, _ in gene_scores[:top_k]]
    return selected


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("Loading expression data...")
    expression, metadata = load_expression_data()

    # Define panels to compare
    panels = {
        "6-gene (original)": EPITHELIAL_GENES,
        "13-gene (expanded)": EXPANDED_GENES,
        "Paneth only (4)": PANETH_MARKERS,
        "Goblet only (3)": GOBLET_MARKERS,
    }

    # Univariate-filtered subset from the 13-gene pool
    print("Running univariate gene filtering...")
    filtered_genes = univariate_filter(expression, metadata, EXPANDED_GENES, top_k=6)
    panels["Filtered top-6"] = filtered_genes
    print(f"  Filtered genes: {filtered_genes}")

    # Evaluate each panel
    results = {}
    for name, genes in panels.items():
        print(f"\nEvaluating: {name} ({len(genes)} genes: {genes})")
        df = compute_score_for_genes(expression, metadata, genes)
        auc, ci, y_true, y_prob = loso_cv_single(df)
        results[name] = {
            "genes": genes,
            "n_genes": len(genes),
            "auc": auc,
            "auc_95ci": ci,
            "y_true": y_true,
            "y_prob": y_prob,
        }
        print(f"  AUC: {auc} [{ci[0]}-{ci[1]}]")

    # Individual gene AUCs
    print("\nIndividual gene AUCs:")
    gene_aucs = {}
    for gene in EXPANDED_GENES:
        if gene not in expression.index:
            continue
        df = compute_score_for_genes(expression, metadata, [gene])
        auc, ci, _, _ = loso_cv_single(df)
        gene_aucs[gene] = {"auc": auc, "auc_95ci": ci}
        marker = "**" if auc >= 0.60 else ""
        print(f"  {gene}: AUC={auc} {marker}")

    # Determine best panel
    best_name = max(results, key=lambda n: results[n]["auc"])
    print(f"\nBest panel: {best_name} (AUC={results[best_name]['auc']})")
    expanded_better = results["13-gene (expanded)"]["auc"] > results["6-gene (original)"]["auc"]
    print(f"Expanded panel improves over original: {expanded_better}")

    # Save results
    summary = {
        "phase": "1.3",
        "description": "Expanded epithelial defense panel comparison",
        "panels": {
            name: {"genes": r["genes"], "n_genes": r["n_genes"], "auc": r["auc"], "auc_95ci": r["auc_95ci"]}
            for name, r in results.items()
        },
        "individual_gene_aucs": gene_aucs,
        "univariate_filtered_genes": filtered_genes,
        "best_panel": best_name,
        "expanded_improves_over_original": expanded_better,
    }
    with open(OUTPUT_DIR / "phase1_3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot: panel comparison
    print("Generating plots...")
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(results.keys())
    aucs = [results[n]["auc"] for n in names]
    ci_lows = [results[n]["auc_95ci"][0] for n in names]
    ci_highs = [results[n]["auc_95ci"][1] for n in names]
    err_low = [a - cl for a, cl in zip(aucs, ci_lows)]
    err_high = [ch - a for a, ch in zip(aucs, ci_highs)]

    colors = ["steelblue" if "original" in n else "mediumseagreen" for n in names]
    x = range(len(names))
    bars = ax.bar(x, aucs, color=colors, edgecolor="white", width=0.6)
    ax.errorbar(x, aucs, yerr=[err_low, err_high], fmt="none", color="black", capsize=5)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1)
    ax.axhline(0.65, color="green", linestyle="--", lw=1, alpha=0.4, label="Target 0.65")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax.set_ylabel("AUC (LOSO-CV)")
    ax.set_title("Epithelial Defense Panels: Anti-TNF Response Prediction")
    ax.set_ylim(0, 1)
    for bar, a in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{a:.3f}", ha="center", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "panel_comparison.png", dpi=150)
    plt.close(fig)

    # Plot: individual gene AUCs
    sorted_genes = sorted(gene_aucs.items(), key=lambda x: x[1]["auc"], reverse=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    gnames = [g for g, _ in sorted_genes]
    gaucs = [v["auc"] for _, v in sorted_genes]
    gcolors = ["steelblue" if g in EPITHELIAL_GENES else "mediumseagreen" if g in PANETH_MARKERS else "coral" for g in gnames]
    bars = ax.bar(range(len(gnames)), gaucs, color=gcolors, edgecolor="white")
    ax.axhline(0.5, color="gray", linestyle="--", lw=1)
    ax.set_xticks(range(len(gnames)))
    ax.set_xticklabels(gnames, fontsize=8, rotation=45, ha="right")
    ax.set_ylabel("AUC (LOSO-CV)")
    ax.set_title("Individual Gene AUCs for Anti-TNF Response\n(Blue=Original, Green=Paneth, Coral=Goblet)")
    ax.set_ylim(0, 1)
    for bar, a in zip(bars, gaucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{a:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "individual_gene_aucs.png", dpi=150)
    plt.close(fig)

    # ROC overlay for panels
    fig, ax = plt.subplots(figsize=(7, 7))
    cmap = {"6-gene (original)": "steelblue", "13-gene (expanded)": "green", "Paneth only (4)": "orange", "Goblet only (3)": "coral", "Filtered top-6": "purple"}
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, color=cmap.get(name, "gray"), lw=2, label=f"{name} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison: Epithelial Defense Panels")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(figures_dir / "roc_panel_comparison.png", dpi=150)
    plt.close(fig)

    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
