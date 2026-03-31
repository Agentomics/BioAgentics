"""Phase 4b addendum: 4-gene CDK2 response signature across DepMap lines.

Computes a CDK2 response signature from CCND1, CCNE1, RB1, CDKN2A expression
(NatComm 2025 method), overlays on CDKN2A Phase 4b MTAP-stratified PRISM
results, and identifies lines predicted to benefit from CDK2+CDK4/6i combination.

Logic:
  CDK2_score = z(CCNE1) + z(CCND1) - z(RB1) - z(CDKN2A)
  High score → high cyclin activation + low tumor suppressor → predicted CDK2i responder

Usage:
    uv run python -m cancer.cdkn2a_pancancer_dependency_atlas.04b_cdk2_response_signature
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase1"
PHASE4B_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase4b_mtap_stratified"

EXPR_FILE = DEPMAP_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
PHASE1_CLASSIF = PHASE1_DIR / "cdkn2a_classification.csv"

SIGNATURE_GENES = ["CCND1", "CCNE1", "RB1", "CDKN2A"]
# Positive weights: high expression drives CDK2 dependency
# Negative weights: low expression drives CDK2 dependency
WEIGHTS = {"CCNE1": 1.0, "CCND1": 1.0, "RB1": -1.0, "CDKN2A": -1.0}

# CDK2 responder threshold: top quartile of CDK2 score
CDK2_RESPONDER_QUANTILE = 0.75


def main() -> None:
    print("=" * 80)
    print("CDK2 Response Signature: 4-gene weighted composite")
    print("=" * 80)

    # 1. Load expression data using shared loader (handles metadata columns + ModelID index)
    full_expr = load_depmap_matrix(EXPR_FILE)
    missing = [g for g in SIGNATURE_GENES if g not in full_expr.columns]
    if missing:
        raise ValueError(f"Missing genes in expression data: {missing}")
    expr = full_expr[SIGNATURE_GENES].copy()
    del full_expr
    print(f"\nExpression data loaded: {len(expr)} cell lines, {len(expr.columns)} genes")
    print(f"Genes: {list(expr.columns)}")

    # 2. Z-score normalize each gene
    expr_z = expr.apply(stats.zscore, nan_policy="omit")

    # 3. Compute CDK2 response score
    cdk2_score = sum(WEIGHTS[g] * expr_z[g] for g in SIGNATURE_GENES)
    cdk2_score.name = "CDK2_response_score"
    print(f"\nCDK2 score computed: mean={cdk2_score.mean():.3f}, std={cdk2_score.std():.3f}")

    # 4. Load Phase 1 classifications
    classif = pd.read_csv(PHASE1_CLASSIF)
    classif = classif.set_index("ModelID")

    # 5. Merge CDK2 score with classification
    merged = classif.join(cdk2_score, how="inner")
    merged = merged.join(expr[["CCNE1", "CCND1", "RB1"]], how="left", rsuffix="_expr")
    print(f"Merged with Phase 1: {len(merged)} cell lines")

    # 6. Define CDK2 responder threshold
    threshold = cdk2_score.quantile(CDK2_RESPONDER_QUANTILE)
    merged["CDK2_predicted_responder"] = merged["CDK2_response_score"] > threshold
    print(f"CDK2 responder threshold (Q{CDK2_RESPONDER_QUANTILE:.0%}): {threshold:.3f}")

    # 7. Focus on CDKN2A-deleted, MTAP-intact, RB1-intact lines (Phase 4b cohort)
    cohort = merged[
        (merged["CDKN2A_status"] == "deleted")
        & (~merged["MTAP_co_deleted"])
        & (merged["RB1_status"] == "intact")
    ].copy()
    print(f"\nPhase 4b cohort (CDKN2A-del/MTAP-intact/RB1-intact): {len(cohort)} lines")

    n_responders = cohort["CDK2_predicted_responder"].sum()
    print(f"Predicted CDK2i responders: {n_responders}/{len(cohort)} ({n_responders/len(cohort)*100:.1f}%)")

    # 8. Check overlap with CCNE1 amplification
    ccne1_amp = cohort["CCNE1_amplified"].sum()
    both = (cohort["CDK2_predicted_responder"] & cohort["CCNE1_amplified"]).sum()
    cdk2_only = (cohort["CDK2_predicted_responder"] & ~cohort["CCNE1_amplified"]).sum()
    ccne1_only = (~cohort["CDK2_predicted_responder"] & cohort["CCNE1_amplified"]).sum()

    print(f"\nCCNE1 amplified in cohort: {ccne1_amp}/{len(cohort)}")
    print(f"CDK2-responder AND CCNE1-amp: {both}")
    print(f"CDK2-responder but NOT CCNE1-amp: {cdk2_only}")
    print(f"CCNE1-amp but NOT CDK2-responder: {ccne1_only}")

    # 9. Save outputs
    # CDK2 scores for all lines
    score_df = merged[["OncotreeLineage", "CDKN2A_status", "RB1_status",
                        "MTAP_co_deleted", "CCNE1_amplified",
                        "CDK2_response_score", "CDK2_predicted_responder"]].copy()
    score_df = score_df.sort_values("CDK2_response_score", ascending=False)
    score_path = PHASE4B_DIR / "cdk2_response_scores.csv"
    score_df.to_csv(score_path)
    print(f"\nSaved CDK2 scores for {len(score_df)} lines to {score_path}")

    # Phase 4b cohort CDK2 responders
    responders = cohort[cohort["CDK2_predicted_responder"]].sort_values(
        "CDK2_response_score", ascending=False
    )
    responder_cols = ["OncotreeLineage", "CCNE1_amplified", "TP53_status",
                      "CDK2_response_score", "CCNE1", "CCND1", "RB1"]
    resp_path = PHASE4B_DIR / "cdk2_predicted_responders.csv"
    responders[responder_cols].to_csv(resp_path)
    print(f"Saved {len(responders)} predicted CDK2i responders to {resp_path}")

    # 10. Summary statistics by lineage
    print("\n" + "=" * 80)
    print("CDK2 RESPONDERS BY LINEAGE (CDKN2A-del/MTAP-intact/RB1-intact cohort)")
    print("=" * 80)
    lineage_stats = cohort.groupby("OncotreeLineage").agg(
        total=("CDK2_predicted_responder", "size"),
        responders=("CDK2_predicted_responder", "sum"),
        mean_score=("CDK2_response_score", "mean"),
    )
    lineage_stats["pct"] = (lineage_stats["responders"] / lineage_stats["total"] * 100).round(1)
    lineage_stats = lineage_stats[lineage_stats["total"] >= 3].sort_values("pct", ascending=False)

    for _, row in lineage_stats.iterrows():
        print(f"  {row.name}: {int(row['responders'])}/{int(row['total'])} ({row['pct']:.0f}%), mean score={row['mean_score']:.2f}")

    # 11. Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: CDK2 score distribution by CDKN2A status
    ax = axes[0]
    for status, color in [("deleted", "#e74c3c"), ("intact", "#2ecc71")]:
        subset = merged[merged["CDKN2A_status"] == status]["CDK2_response_score"].dropna()
        ax.hist(subset, bins=50, alpha=0.6, label=f"CDKN2A {status} (n={len(subset)})",
                color=color, density=True)
    ax.axvline(threshold, color="black", linestyle="--", label=f"Q75 threshold ({threshold:.2f})")
    ax.set_xlabel("CDK2 Response Score")
    ax.set_ylabel("Density")
    ax.set_title("CDK2 Response Score by CDKN2A Status")
    ax.legend(fontsize=8)

    # Panel B: CDK2 score vs CCNE1 expression in Phase 4b cohort
    ax = axes[1]
    for amp, color, label in [(True, "#e74c3c", "CCNE1-amp"), (False, "#3498db", "CCNE1-normal")]:
        subset = cohort[cohort["CCNE1_amplified"] == amp]
        ax.scatter(subset["CCNE1"], subset["CDK2_response_score"],
                   c=color, alpha=0.6, s=20, label=f"{label} (n={len(subset)})")
    ax.axhline(threshold, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("CCNE1 Expression (log2 TPM+1)")
    ax.set_ylabel("CDK2 Response Score")
    ax.set_title("CDK2 Score vs CCNE1 in Phase 4b Cohort")
    ax.legend(fontsize=8)

    plt.tight_layout()
    fig_path = PHASE4B_DIR / "cdk2_response_signature.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure to {fig_path}")


if __name__ == "__main__":
    main()
