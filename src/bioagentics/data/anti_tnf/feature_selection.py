"""Feature selection pipeline for anti-TNF response prediction.

Identifies a compact gene signature (10-30 genes) using stability selection,
recursive feature elimination, and evaluation of candidate gene sets from literature.

Usage:
    uv run python -m bioagentics.data.anti_tnf.feature_selection
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
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_BATCH_DIR = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "batch_correction"
DEFAULT_DE_DIR = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "differential_expression"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "feature_selection"

# Literature-based candidate gene sets
CANDIDATE_SETS = {
    "Porto_5gene": ["TREM1", "IL23R", "CCL7", "IL17F", "YES1"],
    "OSMR_pathway": ["OSMR", "OSM", "IL6ST"],
    "ECM_fibroblast": ["MMP2", "COL1A1", "CD81"],
    "RORC_JNK": ["RORC", "MAP2K4", "MAP2K7", "MAPK8", "MAPK9"],
    "NAMPT_NAD": ["NAMPT", "NMNAT1", "NMNAT2", "NAPRT"],
}


def load_data(
    batch_dir: Path,
    de_dir: Path,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load batch-corrected expression and DE results.

    Returns (X: samples x genes, y: response labels, de_results).
    """
    expr = pd.read_csv(batch_dir / "expression_combat.csv", index_col=0)
    metadata = pd.read_csv(batch_dir / "metadata.csv")
    de = pd.read_csv(de_dir / "de_results.csv")

    # Align
    meta = metadata.set_index("sample_id").loc[expr.columns]
    y = (meta["response_status"] == "non_responder").astype(int)
    y.name = "non_responder"

    # Transpose: samples x genes
    X = expr.T

    return X, y, de


def stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_iterations: int = 200,
    subsample_fraction: float = 0.7,
    alpha: float = 0.01,
    random_state: int = 42,
) -> pd.Series:
    """Stability selection using randomized L1 logistic regression.

    Returns selection frequency for each gene (0-1).
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y)
    n_sub = int(n_samples * subsample_fraction)
    gene_counts = np.zeros(X.shape[1])

    scaler = StandardScaler()

    for i in range(n_iterations):
        # Random subsample
        idx = rng.choice(n_samples, n_sub, replace=False)
        X_sub = scaler.fit_transform(X.iloc[idx].values)
        y_sub = y.iloc[idx].values

        # Randomized L1 with random penalty scaling
        scale = rng.uniform(0.5, 1.5, X.shape[1])
        X_scaled = X_sub * scale

        model = LogisticRegression(
            penalty="l1", C=1.0 / alpha, solver="liblinear",
            max_iter=1000, random_state=rng.randint(1e6),
        )
        model.fit(X_scaled, y_sub)

        selected = np.abs(model.coef_[0]) > 0
        gene_counts += selected

    freq = pd.Series(gene_counts / n_iterations, index=X.columns, name="stability_freq")
    return freq.sort_values(ascending=False)


def rfe_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 30,
    estimator_type: str = "logistic",
) -> list[str]:
    """Recursive feature elimination.

    Returns list of selected gene names.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)

    if estimator_type == "logistic":
        estimator = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000, random_state=42,
        )
    else:
        estimator = RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=-1,
        )

    rfe = RFE(estimator, n_features_to_select=n_features, step=0.1)
    rfe.fit(X_scaled, y.values)

    selected = X.columns[rfe.support_].tolist()
    logger.info("RFE (%s): selected %d features", estimator_type, len(selected))
    return selected


def evaluate_candidates(
    de: pd.DataFrame,
    stability_freq: pd.Series,
    all_genes: set[str],
) -> pd.DataFrame:
    """Evaluate literature candidate gene sets against DE and stability results."""
    records = []
    de_indexed = de.set_index("gene")

    for set_name, genes in CANDIDATE_SETS.items():
        for gene in genes:
            present = gene in all_genes
            if present and gene in de_indexed.index:
                row = de_indexed.loc[gene]
                freq = stability_freq.get(gene, 0.0)
                records.append({
                    "candidate_set": set_name,
                    "gene": gene,
                    "present": True,
                    "logFC": row["logFC"],
                    "adj_p_value": row["adj_p_value"],
                    "de_significant": row["adj_p_value"] < 0.05,
                    "stability_freq": freq,
                })
            else:
                records.append({
                    "candidate_set": set_name,
                    "gene": gene,
                    "present": present,
                    "logFC": np.nan,
                    "adj_p_value": np.nan,
                    "de_significant": False,
                    "stability_freq": stability_freq.get(gene, 0.0),
                })

    return pd.DataFrame(records)


def combine_and_rank(
    stability_freq: pd.Series,
    rfe_lr_genes: list[str],
    rfe_rf_genes: list[str],
    de: pd.DataFrame,
    max_genes: int = 30,
) -> pd.DataFrame:
    """Combine evidence from multiple methods to produce a final ranked gene list."""
    all_genes = set(stability_freq.index)
    de_indexed = de.set_index("gene")

    # Forced candidate genes from literature
    forced_candidates = set()
    for genes in CANDIDATE_SETS.values():
        forced_candidates.update(g for g in genes if g in all_genes)

    records = []
    for gene in all_genes:
        freq = stability_freq.get(gene, 0)
        in_rfe_lr = gene in rfe_lr_genes
        in_rfe_rf = gene in rfe_rf_genes
        is_candidate = gene in forced_candidates

        de_row = de_indexed.loc[gene] if gene in de_indexed.index else None
        adj_p = de_row["adj_p_value"] if de_row is not None else 1.0
        logfc = de_row["logFC"] if de_row is not None else 0.0

        # Composite score: stability + RFE + DE significance + candidate bonus
        score = freq * 2
        if in_rfe_lr:
            score += 0.5
        if in_rfe_rf:
            score += 0.5
        if adj_p < 0.05:
            score += 0.3
        if adj_p < 0.01:
            score += 0.2
        if is_candidate:
            score += 0.3

        records.append({
            "gene": gene,
            "stability_freq": freq,
            "rfe_logistic": in_rfe_lr,
            "rfe_rf": in_rfe_rf,
            "literature_candidate": is_candidate,
            "adj_p_value": adj_p,
            "logFC": logfc,
            "composite_score": score,
        })

    ranked = pd.DataFrame(records).sort_values("composite_score", ascending=False)
    ranked = ranked.reset_index(drop=True)
    ranked["selected"] = ranked.index < max_genes

    return ranked


def plot_selection_results(ranked: pd.DataFrame, output_dir: Path) -> None:
    """Plot selection frequency and composite scores for top genes."""
    top = ranked.head(40)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Stability frequency
    colors = ["#2ca02c" if s else "#1f77b4" for s in top["selected"]]
    axes[0].barh(range(len(top)), top["stability_freq"].values, color=colors)
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels(top["gene"].values, fontsize=8)
    axes[0].set_xlabel("Stability Selection Frequency")
    axes[0].set_title("Top 40 Genes: Stability Frequency")
    axes[0].invert_yaxis()

    # Composite score
    axes[1].barh(range(len(top)), top["composite_score"].values, color=colors)
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels(top["gene"].values, fontsize=8)
    axes[1].set_xlabel("Composite Score")
    axes[1].set_title("Top 40 Genes: Composite Score")
    axes[1].invert_yaxis()

    fig.suptitle("Feature Selection Results", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "feature_selection_results.png", dpi=150)
    plt.close(fig)
    logger.info("Saved feature selection plot")


def run_feature_selection(
    batch_dir: Path = DEFAULT_BATCH_DIR,
    de_dir: Path = DEFAULT_DE_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
    max_genes: int = 30,
) -> pd.DataFrame:
    """Run the full feature selection pipeline.

    Returns the ranked gene list with selection status.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    X, y, de = load_data(batch_dir, de_dir)
    logger.info("Loaded: %d samples x %d genes", *X.shape)

    # Pre-filter to DE-significant or candidate genes for computational efficiency
    de_sig_genes = set(de[de["adj_p_value"] < 0.1]["gene"])
    candidate_genes = set()
    for genes in CANDIDATE_SETS.values():
        candidate_genes.update(genes)
    keep_genes = (de_sig_genes | candidate_genes) & set(X.columns)
    X_filtered = X[sorted(keep_genes)]
    logger.info("Pre-filtered to %d genes (DE adj.p<0.1 + candidates)", X_filtered.shape[1])

    # Stability selection
    logger.info("Running stability selection...")
    stability_freq = stability_selection(X_filtered, y)

    # RFE with logistic regression
    logger.info("Running RFE (logistic regression)...")
    rfe_lr = rfe_selection(X_filtered, y, n_features=max_genes, estimator_type="logistic")

    # RFE with random forest
    logger.info("Running RFE (random forest)...")
    rfe_rf = rfe_selection(X_filtered, y, n_features=max_genes, estimator_type="rf")

    # Evaluate candidate gene sets
    candidates = evaluate_candidates(de, stability_freq, set(X.columns))
    candidates.to_csv(output_dir / "candidate_evaluation.csv", index=False)
    logger.info("Candidate evaluation saved")

    # Combine and rank
    ranked = combine_and_rank(stability_freq, rfe_lr, rfe_rf, de, max_genes=max_genes)
    ranked.to_csv(output_dir / "ranked_genes.csv", index=False)
    logger.info("Ranked gene list saved: %d selected", ranked["selected"].sum())

    # Selected signature
    signature = ranked[ranked["selected"]]["gene"].tolist()
    logger.info("Final signature (%d genes): %s", len(signature), ", ".join(signature))

    # Plot
    plot_selection_results(ranked, output_dir)

    # Summary of pathway overlap
    _pathway_summary(signature)

    return ranked


def _pathway_summary(signature: list[str]) -> None:
    """Log which candidate sets overlap with the final signature."""
    sig_set = set(signature)
    for set_name, genes in CANDIDATE_SETS.items():
        overlap = sig_set & set(genes)
        if overlap:
            logger.info(
                "  %s: %d/%d in signature (%s)",
                set_name, len(overlap), len(genes), ", ".join(sorted(overlap)),
            )


def main():
    parser = argparse.ArgumentParser(
        description="Feature selection for anti-TNF response prediction"
    )
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--de-dir", type=Path, default=DEFAULT_DE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-genes", type=int, default=30)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_feature_selection(
        batch_dir=args.batch_dir,
        de_dir=args.de_dir,
        output_dir=args.output_dir,
        max_genes=args.max_genes,
    )


if __name__ == "__main__":
    main()
