"""Evaluation and benchmarking module for anti-TNF response classifier.

Generates comprehensive evaluation: per-study AUC, ROC curves, risk
stratification plots, and benchmarks against published signatures and models.

Usage:
    uv run python -m bioagentics.data.anti_tnf.evaluation
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_DIR = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "classifier"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "evaluation"

# Published benchmark AUCs for comparison
BENCHMARKS = {
    "adalimumab_ML_clinical": {
        "auc": 0.935,
        "source": "PLOS One 2025, doi:10.1371/journal.pone.0331447",
        "features": "Clinical/lab (calprotectin, CRP, hemoglobin)",
        "outcome": "48-week response",
    },
    "EPIC_CD_blood_methylation": {
        "auc": 0.25,
        "source": "Lancet GH 2025, PMID 40614748",
        "features": "Blood DNA methylation (adalimumab)",
        "outcome": "Primary non-response",
    },
    "TabNet_multimodal": {
        "auc": 0.858,
        "source": "J Crohns Colitis 2026, PMID 41298328",
        "features": "Histopathology + CT enterography + clinical labs",
        "outcome": "IFX primary non-response",
    },
    "Porto_5gene_mucosal": {
        "auc": 0.88,
        "source": "PubMed 40819280, 2025",
        "features": "TREM1, IL23R, CCL7, IL17F, YES1 (mucosal)",
        "outcome": "Anti-TNF response (pediatric CD)",
    },
}


def load_classifier_results(classifier_dir: Path) -> dict[str, pd.DataFrame]:
    """Load per-study metrics and aggregate metrics for all trained models."""
    agg = pd.read_csv(classifier_dir / "aggregate_metrics.csv")
    per_study = {}
    for _, row in agg.iterrows():
        model = row["model"]
        ps_path = classifier_dir / f"{model}_per_study.csv"
        if ps_path.exists():
            per_study[model] = pd.read_csv(ps_path)
    return {"aggregate": agg, "per_study": per_study}


def benchmark_comparison(our_results: pd.DataFrame) -> pd.DataFrame:
    """Compare our classifier AUCs against published benchmarks."""
    records = []

    for _, row in our_results.iterrows():
        records.append({
            "model": f"Ours ({row['model']})",
            "auc": row["auc"],
            "source": "This study (LOSO-CV, within-fold preprocessing)",
            "features": "Mucosal transcriptomics",
            "outcome": "Anti-TNF response (CD, LOSO-CV)",
        })

    for name, info in BENCHMARKS.items():
        records.append({
            "model": name,
            "auc": info["auc"],
            "source": info["source"],
            "features": info["features"],
            "outcome": info["outcome"],
        })

    return pd.DataFrame(records).sort_values("auc", ascending=False)


def plot_benchmark_comparison(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart comparing our AUC vs published benchmarks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["#2ca02c" if "Ours" in str(m) else "#1f77b4"
              for m in comparison["model"]]

    df = comparison.sort_values("auc", ascending=True)
    ax.barh(range(len(df)), df["auc"].values, color=colors, alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["model"], fontsize=9)
    ax.set_xlabel("AUC")
    ax.set_title("Classifier Performance vs Published Benchmarks")
    ax.axvline(0.75, ls="--", c="gray", alpha=0.5, label="Success threshold (0.75)")
    ax.axvline(0.5, ls=":", c="gray", alpha=0.3, label="Random (0.50)")

    # Add value labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["auc"] + 0.01, i, f"{row['auc']:.3f}", va="center", fontsize=8)

    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "benchmark_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved benchmark comparison plot")


def plot_per_study_heatmap(per_study: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Heatmap of per-study AUC for each model."""
    if not per_study:
        return

    # Build matrix: models x studies
    models = sorted(per_study.keys())
    studies = sorted(per_study[models[0]]["study"].unique()) if models else []

    matrix = np.full((len(models), len(studies)), np.nan)
    for i, model in enumerate(models):
        ps = per_study[model].set_index("study")
        for j, study in enumerate(studies):
            if study in ps.index:
                matrix[i, j] = ps.loc[study, "auc"]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(studies)))
    ax.set_xticklabels(studies, fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    # Add value text
    for i in range(len(models)):
        for j in range(len(studies)):
            val = matrix[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if val < 0.4 or val > 0.8 else "black")

    fig.colorbar(im, ax=ax, label="AUC")
    ax.set_title("Per-Study AUC by Model (LOSO-CV)")
    fig.tight_layout()
    fig.savefig(output_dir / "per_study_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved per-study heatmap")


def generate_evaluation_report(
    results: dict,
    comparison: pd.DataFrame,
    output_dir: Path,
) -> str:
    """Generate text evaluation report."""
    agg = results["aggregate"]
    best = agg.loc[agg["auc"].idxmax()]

    lines = [
        "# Anti-TNF Response Classifier Evaluation Report",
        "",
        "## Pipeline: Within-Fold LOSO-CV (Leakage-Free)",
        "",
        "## Best Model",
        f"- Model: {best['model']}",
        f"- Aggregate AUC: {best['auc']:.3f}",
        f"- Balanced Accuracy: {best['balanced_accuracy']:.3f}",
        f"- Sensitivity: {best['sensitivity']:.3f}",
        f"- Specificity: {best['specificity']:.3f}",
        "",
        "## All Models",
    ]

    for _, row in agg.iterrows():
        lines.append(f"- {row['model']}: AUC={row['auc']:.3f}, BA={row['balanced_accuracy']:.3f}")

    lines.extend(["", "## Per-Study Performance"])
    for model, ps in results["per_study"].items():
        lines.append(f"\n### {model}")
        for _, row in ps.iterrows():
            lines.append(
                f"- {row['study']}: AUC={row['auc']:.3f} "
                f"(n={int(row['n_samples'])}, R={int(row['n_responder'])}, NR={int(row['n_non_responder'])})"
            )

    lines.extend(["", "## Benchmark Comparison"])
    for _, row in comparison.iterrows():
        marker = " <<<" if "Ours" in str(row["model"]) else ""
        lines.append(f"- {row['model']}: AUC={row['auc']:.3f}{marker}")

    success = best["auc"] > 0.75
    lines.extend([
        "",
        "## Success Criteria",
        f"- LOSO AUC > 0.75: {'PASS' if success else 'FAIL'} ({best['auc']:.3f})",
    ])

    report = "\n".join(lines)
    report_path = output_dir / "evaluation_report.md"
    report_path.write_text(report)
    logger.info("Saved evaluation report to %s", report_path)
    return report


def run_evaluation(
    classifier_dir: Path = DEFAULT_CLASSIFIER_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
) -> None:
    """Run the full evaluation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_classifier_results(classifier_dir)
    logger.info("Loaded results for %d models", len(results["per_study"]))

    # Benchmark comparison
    comparison = benchmark_comparison(results["aggregate"])
    comparison.to_csv(output_dir / "benchmark_comparison.csv", index=False)
    plot_benchmark_comparison(comparison, output_dir)

    # Per-study heatmap
    plot_per_study_heatmap(results["per_study"], output_dir)

    # Report
    report = generate_evaluation_report(results, comparison, output_dir)

    # Log summary
    best = results["aggregate"].loc[results["aggregate"]["auc"].idxmax()]
    logger.info(
        "Evaluation complete. Best model: %s (AUC=%.3f). Success: %s",
        best["model"], best["auc"], "YES" if best["auc"] > 0.75 else "NO",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate anti-TNF response classifier against benchmarks"
    )
    parser.add_argument("--classifier-dir", type=Path, default=DEFAULT_CLASSIFIER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_evaluation(classifier_dir=args.classifier_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
