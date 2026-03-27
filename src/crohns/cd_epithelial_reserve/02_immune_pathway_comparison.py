#!/usr/bin/env python3
"""Phase 1.2: Head-to-head comparison of epithelial vs immune pathway scores.

Builds immune pathway scores (TNF, IL-23/Th17, integrin/trafficking) and compares
their predictive performance against the epithelial reserve score for anti-TNF
response using LOSO-CV with DeLong test for AUC comparison.
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

import importlib.util

# Import Phase 1.1 module (numbered filename requires importlib)
_spec = importlib.util.spec_from_file_location(
    "phase1_1",
    Path(__file__).parent / "01_epithelial_reserve_score.py",
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

EPITHELIAL_GENES = _mod.EPITHELIAL_GENES
STUDIES = _mod.STUDIES
compute_epithelial_reserve_score = _mod.compute_epithelial_reserve_score
load_expression_data = _mod.load_expression_data

OUTPUT_DIR = Path("output/crohns/cd-epithelial-reserve-treatment-stratification")

# Immune pathway gene sets
PATHWAY_GENE_SETS = {
    "tnf_signaling": [
        "TNFRSF1A", "TNFRSF1B", "NFKB1", "NFKB2", "RELA", "TRAF2", "TRADD",
        "RIPK1", "BIRC2", "BIRC3", "TNFAIP3", "IKBKG", "MAP3K7", "CHUK", "IKBKB",
    ],
    "il23_th17": [
        "IL23R", "IL23A", "IL12RB1", "RORC", "IL17A", "IL17F", "IL22", "IL21",
        "STAT3", "CCR6", "IL6ST", "TYK2", "JAK2",
    ],
    "integrin_trafficking": [
        "ITGA4", "ITGB7", "MADCAM1", "VCAM1", "ICAM1", "SELL", "SELP",
        "CCL25", "CCR9", "GPR15", "ITGAE", "ITGB1",
    ],
}


def compute_pathway_score(
    expression: pd.DataFrame,
    metadata: pd.DataFrame,
    genes: list[str],
    pathway_name: str,
) -> pd.DataFrame:
    """Compute a pathway score as mean z-score (within-study)."""
    available = [g for g in genes if g in expression.index]
    if len(available) < len(genes):
        missing = set(genes) - set(available)
        print(f"  Warning: {pathway_name} missing genes: {missing}")

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
            records.append({
                "sample_id": sid,
                "study": study,
                f"{pathway_name}_score": float(score[sid]),
            })

    return pd.DataFrame(records)


def loso_cv_single_feature(
    df: pd.DataFrame, feature_col: str
) -> tuple[float, np.ndarray, np.ndarray]:
    """LOSO-CV with logistic regression on a single feature. Returns AUC, y_true, y_prob."""
    all_y_true = []
    all_y_prob = []

    for held_out in df["study"].unique():
        train = df[df["study"] != held_out]
        test = df[df["study"] == held_out]

        X_train = train[[feature_col]].values
        y_train = train["response_binary"].values
        X_test = test[[feature_col]].values
        y_test = test["response_binary"].values

        clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

    y_true = np.array(all_y_true)
    y_prob = np.array(all_y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return auc, y_true, y_prob


def delong_test(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray) -> dict:
    """DeLong test for comparing two AUCs on the same data.

    Implements the fast DeLong algorithm (Sun & Xu, 2014).
    """
    n1 = int(y_true.sum())
    n0 = len(y_true) - n1

    idx_pos = np.where(y_true == 1)[0]
    idx_neg = np.where(y_true == 0)[0]

    # Placement values for model A
    scores_a_pos = y_prob_a[idx_pos]
    scores_a_neg = y_prob_a[idx_neg]
    # For each positive, compute fraction of negatives it beats
    v_a_pos = np.array([np.mean(scores_a_pos[i] > scores_a_neg) + 0.5 * np.mean(scores_a_pos[i] == scores_a_neg) for i in range(n1)])
    v_a_neg = np.array([np.mean(scores_a_pos > scores_a_neg[j]) + 0.5 * np.mean(scores_a_pos == scores_a_neg[j]) for j in range(n0)])

    # Placement values for model B
    scores_b_pos = y_prob_b[idx_pos]
    scores_b_neg = y_prob_b[idx_neg]
    v_b_pos = np.array([np.mean(scores_b_pos[i] > scores_b_neg) + 0.5 * np.mean(scores_b_pos[i] == scores_b_neg) for i in range(n1)])
    v_b_neg = np.array([np.mean(scores_b_pos > scores_b_neg[j]) + 0.5 * np.mean(scores_b_pos == scores_b_neg[j]) for j in range(n0)])

    auc_a = np.mean(v_a_pos)
    auc_b = np.mean(v_b_pos)

    # Covariance matrix of (AUC_A, AUC_B)
    s10 = np.cov(np.column_stack([v_a_pos, v_b_pos]), rowvar=False)
    s01 = np.cov(np.column_stack([v_a_neg, v_b_neg]), rowvar=False)

    s = s10 / n1 + s01 / n0
    diff = auc_a - auc_b

    # Variance of the difference
    var_diff = s[0, 0] + s[1, 1] - 2 * s[0, 1]

    if var_diff <= 0:
        return {"auc_a": round(auc_a, 4), "auc_b": round(auc_b, 4), "diff": round(diff, 4), "z": 0.0, "p_value": 1.0}

    z = diff / np.sqrt(var_diff)
    p_value = 2 * stats.norm.sf(abs(z))

    return {
        "auc_a": round(float(auc_a), 4),
        "auc_b": round(float(auc_b), 4),
        "diff": round(float(diff), 4),
        "z": round(float(z), 4),
        "p_value": round(float(p_value), 6),
    }


def plot_comparison(pathway_results: dict, out: Path) -> None:
    """Bar chart comparing AUCs across pathway scores."""
    names = list(pathway_results.keys())
    aucs = [pathway_results[n]["auc"] for n in names]
    ci_lows = [pathway_results[n]["auc_95ci"][0] for n in names]
    ci_highs = [pathway_results[n]["auc_95ci"][1] for n in names]
    errors_low = [a - cl for a, cl in zip(aucs, ci_lows)]
    errors_high = [ch - a for a, ch in zip(aucs, ci_highs)]

    colors = ["steelblue" if n == "epithelial_reserve" else "salmon" for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(names))
    bars = ax.bar(x, aucs, color=colors, edgecolor="white", width=0.6)
    ax.errorbar(x, aucs, yerr=[errors_low, errors_high], fmt="none", color="black", capsize=5)
    ax.axhline(y=0.5, color="gray", linestyle="--", lw=1, label="Chance")
    ax.axhline(y=0.65, color="green", linestyle="--", lw=1, alpha=0.5, label="Target (0.65)")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("AUC (LOSO-CV)")
    ax.set_title("Epithelial Reserve vs Immune Pathway Scores\nAnti-TNF Response Prediction")
    ax.set_ylim(0, 1)
    ax.legend()

    for bar, auc_val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{auc_val:.3f}", ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc_comparison(pathway_results: dict, out: Path) -> None:
    """Overlay ROC curves for all pathway scores."""
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = {"epithelial_reserve": "steelblue", "tnf_signaling": "red", "il23_th17": "orange", "integrin_trafficking": "purple"}

    for name, res in pathway_results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_prob"])
        ax.plot(fpr, tpr, color=colors.get(name, "gray"), lw=2, label=f"{name} (AUC={res['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Comparison: Epithelial vs Immune Pathway Scores")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(exist_ok=True)

    print("Loading expression data...")
    expression, metadata = load_expression_data()

    # Compute epithelial reserve score
    print("Computing epithelial reserve score...")
    epi_df = compute_epithelial_reserve_score(expression, metadata)

    # Compute immune pathway scores
    pathway_dfs = {"epithelial_reserve": epi_df}
    for pathway_name, genes in PATHWAY_GENE_SETS.items():
        print(f"Computing {pathway_name} score ({len(genes)} genes)...")
        pdf = compute_pathway_score(expression, metadata, genes, pathway_name)
        pdf = pdf.merge(metadata[["sample_id", "response_status"]], on="sample_id", how="left")
        pdf["response_binary"] = (pdf["response_status"] == "responder").astype(int)
        pathway_dfs[pathway_name] = pdf

    # Run LOSO-CV for each pathway
    pathway_results = {}
    for name, pdf in pathway_dfs.items():
        score_col = "epithelial_reserve_score" if name == "epithelial_reserve" else f"{name}_score"
        auc, y_true, y_prob = loso_cv_single_feature(pdf, score_col)

        # Bootstrap CI
        rng = np.random.RandomState(42)
        boot_aucs = []
        for _ in range(2000):
            idx = rng.choice(len(y_true), size=len(y_true), replace=True)
            if len(np.unique(y_true[idx])) < 2:
                continue
            boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

        pathway_results[name] = {
            "auc": round(auc, 4),
            "auc_95ci": [round(np.percentile(boot_aucs, 2.5), 4), round(np.percentile(boot_aucs, 97.5), 4)],
            "n_genes": len(EPITHELIAL_GENES) if name == "epithelial_reserve" else len(PATHWAY_GENE_SETS[name]),
            "y_true": y_true,
            "y_prob": y_prob,
        }
        print(f"  {name}: AUC={auc:.4f} [{pathway_results[name]['auc_95ci'][0]:.4f}-{pathway_results[name]['auc_95ci'][1]:.4f}]")

    # DeLong tests: epithelial vs each immune pathway
    print("\nDeLong tests (epithelial vs immune):")
    epi_res = pathway_results["epithelial_reserve"]
    delong_results = {}
    for name in PATHWAY_GENE_SETS:
        imm_res = pathway_results[name]
        dl = delong_test(epi_res["y_true"], epi_res["y_prob"], imm_res["y_prob"])
        delong_results[f"epithelial_vs_{name}"] = dl
        sig = "*" if dl["p_value"] < 0.05 else ""
        print(f"  vs {name}: diff={dl['diff']:+.4f}, z={dl['z']:.3f}, p={dl['p_value']:.4f}{sig}")

    # Check target: epithelial outperforms by > 0.05 AUC
    epi_auc = pathway_results["epithelial_reserve"]["auc"]
    outperforms = all(epi_auc - pathway_results[n]["auc"] > 0.05 for n in PATHWAY_GENE_SETS)
    print(f"\nTarget (epithelial > immune by >0.05 AUC): {'MET' if outperforms else 'NOT MET'}")

    # Save results
    print("Saving results...")
    summary = {
        "phase": "1.2",
        "description": "Head-to-head comparison of epithelial vs immune pathway scores",
        "cv_method": "leave-one-study-out",
        "classifier": "logistic regression (single feature)",
        "pathway_aucs": {
            name: {"auc": r["auc"], "auc_95ci": r["auc_95ci"], "n_genes": r["n_genes"]}
            for name, r in pathway_results.items()
        },
        "delong_tests": delong_results,
        "target_outperforms_by_005": outperforms,
        "gene_sets": {
            "epithelial_reserve": EPITHELIAL_GENES,
            **PATHWAY_GENE_SETS,
        },
    }
    with open(OUTPUT_DIR / "phase1_2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    print("Generating plots...")
    plot_comparison(pathway_results, figures_dir / "auc_comparison_epithelial_vs_immune.png")
    plot_roc_comparison(pathway_results, figures_dir / "roc_comparison_epithelial_vs_immune.png")

    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
