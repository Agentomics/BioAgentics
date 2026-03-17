"""Label distribution profiling for CXR long-tail datasets.

Computes per-class sample counts, assigns head/body/tail bins,
generates frequency distribution plots and summary CSV.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from bioagentics.cxr_rare.config import (
    HEAD_THRESHOLD,
    LABEL_NAMES,
    OUTPUT_DIR,
    TAIL_THRESHOLD,
    classify_by_frequency,
    ensure_dirs,
)

logger = logging.getLogger(__name__)

# Colour mapping for frequency bins
_BIN_COLORS = {"head": "#2196F3", "body": "#FF9800", "tail": "#F44336"}


def compute_class_counts(
    labels_df: pd.DataFrame,
    label_names: list[str] | None = None,
) -> pd.Series:
    """Compute per-class positive sample counts from a labels DataFrame.

    Parameters
    ----------
    labels_df : pd.DataFrame
        DataFrame with binary columns for each label.
    label_names : list[str], optional
        Columns to count. Defaults to CXR-LT label set.

    Returns
    -------
    pd.Series
        Counts indexed by label name, sorted descending.
    """
    label_names = label_names or LABEL_NAMES
    cols = [c for c in label_names if c in labels_df.columns]
    counts = labels_df[cols].sum().astype(int).sort_values(ascending=False)
    return counts


def assign_bins(
    counts: pd.Series,
    head_thresh: int = HEAD_THRESHOLD,
    tail_thresh: int = TAIL_THRESHOLD,
) -> pd.DataFrame:
    """Assign each class to head/body/tail based on sample count.

    Returns a DataFrame with columns: label, count, bin.
    """
    bins = classify_by_frequency(counts.to_dict(), head_thresh, tail_thresh)
    df = pd.DataFrame({
        "label": counts.index,
        "count": counts.values,
        "bin": [bins.get(str(lbl), "unknown") for lbl in counts.index],
    })
    return df


def plot_class_distribution(
    summary_df: pd.DataFrame,
    output_path: Path,
    title: str = "CXR-LT Class Frequency Distribution",
    log_scale: bool = False,
) -> Path:
    """Generate a bar chart of per-class frequencies, colored by bin.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of assign_bins() with columns: label, count, bin.
    output_path : Path
        Path to save the figure.
    title : str
        Plot title.
    log_scale : bool
        If True, use log scale for y-axis.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = [_BIN_COLORS.get(b, "#999999") for b in summary_df["bin"]]
    ax.bar(range(len(summary_df)), summary_df["count"], color=colors)
    ax.set_xticks(range(len(summary_df)))
    ax.set_xticklabels(summary_df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Sample Count")
    ax.set_title(title)
    if log_scale:
        ax.set_yscale("log")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_BIN_COLORS["head"], label="Head"),
        Patch(facecolor=_BIN_COLORS["body"], label="Body"),
        Patch(facecolor=_BIN_COLORS["tail"], label="Tail"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved distribution plot: %s", output_path)
    return output_path


def profile_labels(
    labels_df: pd.DataFrame,
    label_names: list[str] | None = None,
    head_thresh: int = HEAD_THRESHOLD,
    tail_thresh: int = TAIL_THRESHOLD,
    output_dir: Path = OUTPUT_DIR,
    dataset_name: str = "combined",
) -> pd.DataFrame:
    """Run full label profiling: counts, bins, plots, CSV.

    Parameters
    ----------
    labels_df : pd.DataFrame
        DataFrame with binary label columns.
    label_names : list[str], optional
        Label columns.
    head_thresh, tail_thresh : int
        Thresholds for head/body/tail classification.
    output_dir : Path
        Directory for outputs.
    dataset_name : str
        Name for output files.

    Returns
    -------
    pd.DataFrame
        Summary with columns: label, count, bin.
    """
    ensure_dirs()

    counts = compute_class_counts(labels_df, label_names)
    summary_df = assign_bins(counts, head_thresh, tail_thresh)

    # Save CSV
    csv_path = output_dir / f"label_distribution_{dataset_name}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    logger.info("Saved label distribution CSV: %s", csv_path)

    # Generate plots
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_class_distribution(
        summary_df,
        fig_dir / f"class_distribution_{dataset_name}.png",
        title=f"Class Frequency — {dataset_name}",
    )
    plot_class_distribution(
        summary_df,
        fig_dir / f"class_distribution_{dataset_name}_log.png",
        title=f"Class Frequency (log) — {dataset_name}",
        log_scale=True,
    )

    # Summary stats
    n_head = (summary_df["bin"] == "head").sum()
    n_body = (summary_df["bin"] == "body").sum()
    n_tail = (summary_df["bin"] == "tail").sum()
    logger.info(
        "Label profiling [%s]: %d classes — %d head, %d body, %d tail",
        dataset_name, len(summary_df), n_head, n_body, n_tail,
    )

    return summary_df


def profile_multiple_datasets(
    dataset_dfs: dict[str, pd.DataFrame],
    label_names: list[str] | None = None,
    head_thresh: int = HEAD_THRESHOLD,
    tail_thresh: int = TAIL_THRESHOLD,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, pd.DataFrame]:
    """Profile label distributions across multiple datasets.

    Also computes a combined profile from all datasets concatenated.
    """
    results: dict[str, pd.DataFrame] = {}
    for name, df in dataset_dfs.items():
        results[name] = profile_labels(
            df, label_names, head_thresh, tail_thresh, output_dir, name
        )

    # Combined profile
    if len(dataset_dfs) > 1:
        combined = pd.concat(list(dataset_dfs.values()), ignore_index=True)
        results["combined"] = profile_labels(
            combined, label_names, head_thresh, tail_thresh, output_dir, "combined"
        )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile CXR label distributions")
    parser.add_argument("--labels-csv", required=True, help="Path to labels CSV")
    parser.add_argument("--dataset-name", default="dataset", help="Name for output files")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Output directory")
    parser.add_argument("--head-thresh", type=int, default=HEAD_THRESHOLD)
    parser.add_argument("--tail-thresh", type=int, default=TAIL_THRESHOLD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    df = pd.read_csv(args.labels_csv)
    profile_labels(
        df,
        head_thresh=args.head_thresh,
        tail_thresh=args.tail_thresh,
        output_dir=Path(args.output_dir),
        dataset_name=args.dataset_name,
    )
