"""Publication figure generation for the transcriptomic biomarker panel.

Generates publication-ready figures:
  1. Volcano plots for DE results (combined + sex-stratified)
  2. ROC curves for all classifiers with AUC annotations
  3. Heatmaps of top classifier genes across sample groups
  4. Feature importance bar plots
  5. Pathway enrichment dotplots

Usage:
    uv run python -m bioagentics.models.figures --dest output/figures/
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = (
    REPO_ROOT / "output" / "pandas_pans" / "transcriptomic-biomarker-panel" / "figures"
)


def plot_volcano(
    de_df: pd.DataFrame,
    title: str = "Volcano Plot",
    alpha: float = 0.05,
    lfc_threshold: float = 1.0,
    n_label: int = 10,
    save_path: Path | None = None,
) -> None:
    """Publication-quality volcano plot with top gene labels.

    Parameters
    ----------
    de_df : pd.DataFrame
        DE results with columns: gene, log2FoldChange, padj.
    title : str
        Plot title.
    alpha : float
        Significance threshold for padj.
    lfc_threshold : float
        Log2 fold change threshold for coloring.
    n_label : int
        Number of top significant genes to label.
    save_path : Path, optional
        Save path for figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if de_df.empty:
        logger.warning("Empty DE results — skipping volcano for '%s'", title)
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    log2fc = de_df["log2FoldChange"].values
    padj = de_df["padj"].clip(lower=1e-300).values
    neg_log_p = -np.log10(padj)

    sig = de_df["padj"] < alpha
    up = sig & (de_df["log2FoldChange"] > lfc_threshold)
    down = sig & (de_df["log2FoldChange"] < -lfc_threshold)
    ns = ~(up | down)

    ax.scatter(log2fc[ns], neg_log_p[ns], c="#cccccc", alpha=0.4, s=6, label="NS", zorder=1)
    ax.scatter(log2fc[up], neg_log_p[up], c="#e74c3c", alpha=0.6, s=12,
               label=f"Up ({up.sum()})", zorder=2)
    ax.scatter(log2fc[down], neg_log_p[down], c="#3498db", alpha=0.6, s=12,
               label=f"Down ({down.sum()})", zorder=2)

    ax.axhline(-np.log10(alpha), color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(lfc_threshold, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
    ax.axvline(-lfc_threshold, color="black", linestyle="--", linewidth=0.5, alpha=0.5)

    # Label top genes
    if "gene" in de_df.columns and n_label > 0:
        sig_df = de_df[sig].nsmallest(n_label, "padj")
        for _, row in sig_df.iterrows():
            ax.annotate(
                row["gene"],
                (row["log2FoldChange"], -np.log10(max(row["padj"], 1e-300))),
                fontsize=6, alpha=0.8,
                arrowprops={"arrowstyle": "-", "alpha": 0.3, "lw": 0.5},
                textcoords="offset points", xytext=(5, 5),
            )

    ax.set_xlabel("log2 Fold Change", fontsize=10)
    ax.set_ylabel("-log10(adjusted p-value)", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved volcano: %s", save_path)
    plt.close(fig)


def plot_roc_multi(
    roc_data: list[dict],
    title: str = "Classifier ROC Curves",
    save_path: Path | None = None,
) -> None:
    """Publication-quality ROC curves for multiple classifiers.

    Parameters
    ----------
    roc_data : list[dict]
        Each dict has keys: name, y_true, y_prob, auc.
    title : str
        Plot title.
    save_path : Path, optional
        Save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    if not roc_data:
        return

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
    fig, ax = plt.subplots(figsize=(6, 6))

    for i, rd in enumerate(roc_data):
        y_true, y_prob = rd["y_true"], rd["y_prob"]
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        color = colors[i % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f"{rd['name']} (AUC = {rd['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved ROC curves: %s", save_path)
    plt.close(fig)


def plot_gene_heatmap(
    expr_df: pd.DataFrame,
    genes: list[str],
    group_labels: pd.Series,
    title: str = "Top Classifier Genes",
    n_genes: int = 30,
    save_path: Path | None = None,
) -> None:
    """Heatmap of top classifier genes across sample groups.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Expression matrix (samples x genes).
    genes : list[str]
        Genes to display (ordered by importance).
    group_labels : pd.Series
        Group labels per sample for column ordering.
    title : str
        Plot title.
    n_genes : int
        Max genes to show.
    save_path : Path, optional
        Save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    available = [g for g in genes[:n_genes] if g in expr_df.columns]
    if len(available) < 2:
        logger.warning("Fewer than 2 genes available for heatmap — skipping")
        return

    data = expr_df[available].copy()

    # Z-score normalize per gene
    data = (data - data.mean()) / data.std().replace(0, 1)
    data = data.clip(-3, 3)

    # Sort samples by group
    order = group_labels.sort_values().index
    order = [idx for idx in order if idx in data.index]
    data = data.loc[order]

    fig, ax = plt.subplots(figsize=(max(6, len(available) * 0.3), max(4, len(data) * 0.08)))

    norm = TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
    im = ax.imshow(data.T.values, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_yticks(range(len(available)))
    ax.set_yticklabels(available, fontsize=6)
    ax.set_xlabel("Samples (ordered by group)", fontsize=9)
    ax.set_title(title, fontsize=11)

    # Group separator
    sorted_labels = group_labels.loc[order]
    unique_groups = sorted_labels.unique()
    if len(unique_groups) == 2:
        boundary = (sorted_labels == unique_groups[0]).sum()
        ax.axvline(boundary - 0.5, color="black", linewidth=1.5)

    fig.colorbar(im, ax=ax, label="Z-score", shrink=0.6)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved heatmap: %s", save_path)
    plt.close(fig)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    title: str = "Feature Importance",
    n_top: int = 20,
    save_path: Path | None = None,
) -> None:
    """Bar plot of top feature importances.

    Parameters
    ----------
    feature_names : list[str]
        Gene names.
    importances : np.ndarray
        Importance scores (e.g., from Random Forest).
    title : str
        Plot title.
    n_top : int
        Number of top features to show.
    save_path : Path, optional
        Save path.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(feature_names) == 0:
        return

    df = pd.DataFrame({"gene": feature_names, "importance": importances})
    df = df.nlargest(n_top, "importance")

    fig, ax = plt.subplots(figsize=(6, max(3, len(df) * 0.3)))
    ax.barh(range(len(df)), df["importance"].values, color="#3498db", edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["gene"].values, fontsize=7)
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.invert_yaxis()

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved feature importance: %s", save_path)
    plt.close(fig)


def plot_enrichment_dotplot(
    enrichment_df: pd.DataFrame,
    title: str = "Pathway Enrichment",
    n_top: int = 15,
    save_path: Path | None = None,
) -> None:
    """Publication-quality enrichment dotplot.

    Parameters
    ----------
    enrichment_df : pd.DataFrame
        Must have columns: term, fdr. Optional: fold_enrichment or nes, overlap.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if enrichment_df.empty:
        return

    df = enrichment_df.nsmallest(n_top, "fdr").copy()
    neg_log_fdr = -np.log10(df["fdr"].clip(lower=1e-300))

    if "fold_enrichment" in df.columns:
        color_vals = df["fold_enrichment"].values
        color_label = "Fold Enrichment"
    elif "nes" in df.columns:
        color_vals = df["nes"].values
        color_label = "NES"
    else:
        color_vals = neg_log_fdr.values
        color_label = "-log10(FDR)"

    sizes = df["overlap"].values * 8 if "overlap" in df.columns else np.full(len(df), 60)
    sizes = np.clip(sizes, 20, 250)

    fig, ax = plt.subplots(figsize=(7, max(3, len(df) * 0.35)))
    scatter = ax.scatter(
        neg_log_fdr, range(len(df)),
        c=color_vals, s=sizes, cmap="YlOrRd",
        edgecolors="black", linewidth=0.4, alpha=0.85,
    )

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["term"].values, fontsize=7)
    ax.set_xlabel("-log10(FDR)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.axvline(-np.log10(0.05), color="grey", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.invert_yaxis()

    fig.colorbar(scatter, ax=ax, label=color_label, shrink=0.6)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info("Saved enrichment dotplot: %s", save_path)
    plt.close(fig)


def generate_all_figures(
    de_results: dict[str, pd.DataFrame] | None = None,
    classifier_roc_data: list[dict] | None = None,
    expr_df: pd.DataFrame | None = None,
    top_genes: list[str] | None = None,
    group_labels: pd.Series | None = None,
    feature_importances: dict[str, np.ndarray] | None = None,
    feature_names: list[str] | None = None,
    enrichment_df: pd.DataFrame | None = None,
    dest_dir: Path | None = None,
) -> list[Path]:
    """Generate all publication figures.

    Parameters
    ----------
    de_results : dict, optional
        Mode -> DE DataFrame (columns: gene, log2FoldChange, padj).
    classifier_roc_data : list[dict], optional
        Each with name, y_true, y_prob, auc.
    expr_df : pd.DataFrame, optional
        Expression matrix for heatmap.
    top_genes : list[str], optional
        Top genes for heatmap.
    group_labels : pd.Series, optional
        Sample group labels.
    feature_importances : dict, optional
        Classifier name -> importance array.
    feature_names : list[str], optional
        Feature names matching importance arrays.
    enrichment_df : pd.DataFrame, optional
        Enrichment results for dotplot.
    dest_dir : Path, optional
        Output directory.

    Returns
    -------
    List of saved file paths.
    """
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []

    # 1. Volcano plots
    if de_results:
        for mode, de_df in de_results.items():
            if de_df.empty:
                continue
            path = dest_dir / f"volcano_{mode}.png"
            plot_volcano(de_df, title=f"Differential Expression ({mode})", save_path=path)
            saved.append(path)

    # 2. ROC curves
    if classifier_roc_data:
        path = dest_dir / "roc_curves.png"
        plot_roc_multi(classifier_roc_data, save_path=path)
        saved.append(path)

    # 3. Gene heatmap
    if expr_df is not None and top_genes and group_labels is not None:
        path = dest_dir / "gene_heatmap.png"
        plot_gene_heatmap(expr_df, top_genes, group_labels, save_path=path)
        saved.append(path)

    # 4. Feature importance
    if feature_importances and feature_names:
        for clf_name, imp in feature_importances.items():
            path = dest_dir / f"importance_{clf_name}.png"
            plot_feature_importance(
                feature_names, imp, title=f"{clf_name} Feature Importance", save_path=path,
            )
            saved.append(path)

    # 5. Enrichment dotplot
    if enrichment_df is not None and not enrichment_df.empty:
        path = dest_dir / "enrichment_dotplot.png"
        plot_enrichment_dotplot(enrichment_df, save_path=path)
        saved.append(path)

    logger.info("Generated %d figures in %s", len(saved), dest_dir)
    return saved


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication figures for transcriptomic biomarker panel"
    )
    parser.add_argument("--de-results", type=Path, help="CSV with DE results")
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    de_results = None
    if args.de_results and args.de_results.exists():
        de_df = pd.read_csv(args.de_results)
        de_results = {"combined": de_df}

    saved = generate_all_figures(de_results=de_results, dest_dir=args.dest)
    print(f"Generated {len(saved)} figures")
    for p in saved:
        print(f"  {p}")


if __name__ == "__main__":
    main()
