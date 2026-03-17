"""VUS classification and visualization pipeline for NOD2 variants.

Applies the trained ensemble model to all NOD2 VUS and produces:
1. Ranked list of VUS with predicted functional impact
2. Feature importance bar chart
3. Domain distribution plot
4. Structure visualization data (colored by prediction)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioagentics.data.nod2.classifier import ALL_FEATURES, merge_features
from bioagentics.data.nod2.structure import NOD2_DOMAINS, get_domain
from bioagentics.data.nod2.varmeter2 import _parse_protein_change

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/crohns/nod2-variant-functional-impact")
DATA_DIR = Path("data/crohns/nod2-variant-functional-impact")


def classify_vus(
    data_dir: Path | None = None,
    model_dir: Path | None = None,
) -> pd.DataFrame:
    """Apply trained model to all NOD2 VUS.

    Returns DataFrame with columns: variant, predicted_class,
    prob_GOF, prob_neutral, prob_LOF, confidence_flag.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if model_dir is None:
        model_dir = OUTPUT_DIR

    # Load model
    model_data = joblib.load(model_dir / "model.pkl")
    model = model_data["model"]
    le = model_data["label_encoder"]
    features = model_data["features"]

    # Load all variants
    variants_df = pd.read_csv(data_dir / "nod2_variants.tsv", sep="\t")

    # Filter to VUS (uncertain significance)
    vus_mask = variants_df["clinvar_significance"].str.contains(
        r"[Uu]ncertain", na=False
    )
    vus_df = variants_df[vus_mask].copy()
    logger.info("Found %d VUS in dataset", len(vus_df))

    # Build feature matrix for VUS
    # Load all feature tables
    struct_df = pd.read_csv(data_dir / "nod2_structure_features.tsv", sep="\t")
    pred_df = pd.read_csv(data_dir / "nod2_predictor_scores.tsv", sep="\t")
    vm2_df = pd.read_csv(data_dir / "nod2_varmeter2_features.tsv", sep="\t")
    girdin_df = pd.read_csv(data_dir / "nod2_girdin_features.tsv", sep="\t")

    # Parse residue positions
    vus_df["residue_pos"] = vus_df["hgvs_p"].apply(
        lambda x: _parse_protein_change(str(x))[1]
        if _parse_protein_change(str(x)) is not None
        else _extract_fs_pos(str(x))
    )

    # Merge features
    merged = vus_df.merge(
        struct_df[["residue_pos", "plddt", "rsasa", "active_site_distance", "domain"]],
        on="residue_pos", how="left",
    )
    merged = merged.merge(pred_df, on=["chrom", "pos", "ref", "alt"],
                          how="left", suffixes=("", "_pred"))
    vm2_cols = ["residue_pos"] + [c for c in ["nsasa", "mutation_energy", "grantham_distance"] if c in vm2_df.columns]
    merged = merged.merge(vm2_df[vm2_cols].drop_duplicates(subset="residue_pos"),
                          on="residue_pos", how="left")
    girdin_cols = ["residue_pos"] + [c for c in ["girdin_interface_distance", "disrupts_girdin_domain", "ripk2_interface_distance"] if c in girdin_df.columns]
    merged = merged.merge(girdin_df[girdin_cols].drop_duplicates(subset="residue_pos"),
                          on="residue_pos", how="left")

    # One-hot domain
    if "domain" in merged.columns:
        domain_dummies = pd.get_dummies(merged["domain"], prefix="domain")
        for col in [f"domain_{d}" for d in ["CARD1", "CARD2", "NACHT", "WH", "LRR", "linker"]]:
            if col not in domain_dummies.columns:
                domain_dummies[col] = 0
        merged = pd.concat([merged, domain_dummies[[f"domain_{d}" for d in ["CARD1", "CARD2", "NACHT", "WH", "LRR", "linker"]]]], axis=1)

    if "disrupts_girdin_domain" in merged.columns:
        merged["disrupts_girdin_domain"] = merged["disrupts_girdin_domain"].astype(float)

    # Prepare feature matrix
    available = [f for f in features if f in merged.columns]
    X = merged[available].copy()
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if not np.isnan(median_val) else 0)

    # Add missing features as zeros
    for f in features:
        if f not in X.columns:
            X[f] = 0
    X = X[features]

    # Predict
    if len(X) > 0:
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        pred_classes = le.inverse_transform(y_pred)

        results = pd.DataFrame({
            "variant": merged["hgvs_p"].values,
            "chrom": merged["chrom"].values,
            "pos": merged["pos"].values,
            "ref": merged["ref"].values,
            "alt": merged["alt"].values,
            "residue_pos": merged["residue_pos"].values,
            "domain": merged.get("domain", pd.Series([""] * len(merged))).values,
            "predicted_class": pred_classes,
        })

        class_names = le.classes_.tolist()
        for i, cls in enumerate(class_names):
            results[f"prob_{cls}"] = y_proba[:, i]

        # Confidence flag: high if max probability > 0.8
        results["max_prob"] = y_proba.max(axis=1)
        results["confidence_flag"] = results["max_prob"].apply(
            lambda p: "high" if p > 0.8 else "moderate" if p > 0.6 else "low"
        )

        # Sort by predicted pathogenicity (GOF+LOF probability descending)
        if "prob_GOF" in results.columns and "prob_LOF" in results.columns:
            results["pathogenicity_score"] = results["prob_GOF"] + results["prob_LOF"]
            results = results.sort_values("pathogenicity_score", ascending=False)
    else:
        results = pd.DataFrame()

    logger.info("Classified %d VUS", len(results))
    high_conf = (results["confidence_flag"] == "high").sum() if not results.empty else 0
    logger.info("High confidence predictions: %d", high_conf)

    return results


def _extract_fs_pos(hgvs_p: str) -> int | None:
    """Extract position from frameshift notation."""
    import re
    m = re.search(r"(\d+)fs", hgvs_p)
    if m:
        return int(m.group(1))
    return None


def generate_feature_importance_plot(output_dir: Path | None = None) -> Path:
    """Generate feature importance bar chart."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    fi_path = output_dir / "feature_importance.tsv"
    fi_df = pd.read_csv(fi_path, sep="\t")

    fig, ax = plt.subplots(figsize=(10, 8))
    top_n = min(15, len(fi_df))
    top = fi_df.head(top_n)

    ax.barh(range(top_n), top["importance"].values, color="#2196F3")
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance (Mean Decrease Impurity)")
    ax.set_title("NOD2 Variant Classifier: Top Feature Importances")
    plt.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "feature_importance.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved feature importance plot to %s", fig_path)
    return fig_path


def generate_domain_distribution_plot(
    vus_results: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """Generate domain distribution plot for VUS predictions."""
    if output_dir is None:
        output_dir = OUTPUT_DIR

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: domain distribution of all VUS
    domain_counts = vus_results["domain"].value_counts()
    ax1 = axes[0]
    colors_map = {
        "CARD1": "#E57373", "CARD2": "#EF5350", "NACHT": "#FF9800",
        "WH": "#FFC107", "LRR": "#4CAF50", "linker": "#9E9E9E",
    }
    colors = [colors_map.get(d, "#757575") for d in domain_counts.index]
    ax1.bar(domain_counts.index, domain_counts.values, color=colors)
    ax1.set_title("VUS Distribution by NOD2 Domain")
    ax1.set_ylabel("Number of VUS")
    ax1.tick_params(axis="x", rotation=45)

    # Right: predicted class distribution by domain
    ax2 = axes[1]
    if "predicted_class" in vus_results.columns:
        class_domain = vus_results.groupby(["domain", "predicted_class"]).size().unstack(fill_value=0)
        class_colors = {"GOF": "#F44336", "neutral": "#9E9E9E", "LOF": "#2196F3"}
        class_domain.plot(kind="bar", ax=ax2, color=[class_colors.get(c, "#000") for c in class_domain.columns])
        ax2.set_title("Predicted Functional Class by Domain")
        ax2.set_ylabel("Count")
        ax2.tick_params(axis="x", rotation=45)
        ax2.legend(title="Predicted Class")

    plt.tight_layout()

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_path = fig_dir / "domain_distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved domain distribution plot to %s", fig_path)
    return fig_path


def generate_structure_coloring_data(
    vus_results: pd.DataFrame,
    output_dir: Path | None = None,
) -> Path:
    """Generate data for coloring NOD2 structure by predicted class.

    Outputs a CSV with residue_pos and color assignments
    (GOF=red, neutral=gray, LOF=blue) that can be used with PyMOL/py3Dmol.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    color_map = {
        "GOF": "red",
        "neutral": "gray",
        "LOF": "blue",
    }

    records = []
    for _, row in vus_results.iterrows():
        res_pos = row.get("residue_pos")
        pred_class = row.get("predicted_class", "neutral")
        if pd.notna(res_pos):
            records.append({
                "residue_pos": int(res_pos),
                "predicted_class": pred_class,
                "color": color_map.get(pred_class, "gray"),
                "max_prob": row.get("max_prob", 0),
                "confidence": row.get("confidence_flag", "low"),
            })

    color_df = pd.DataFrame(records)

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    color_path = fig_dir / "structure_coloring.csv"
    color_df.to_csv(color_path, index=False)
    logger.info("Saved structure coloring data to %s (%d residues)", color_path, len(color_df))

    return color_path


def run_vus_pipeline(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Full VUS classification and visualization pipeline.

    Returns dict with predictions DataFrame and figure paths.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Classify VUS
    vus_results = classify_vus(data_dir)

    # Save predictions
    pred_path = output_dir / "vus_predictions.tsv"
    vus_results.to_csv(pred_path, sep="\t", index=False)
    logger.info("Saved VUS predictions to %s", pred_path)

    # 2. Generate visualizations
    fi_path = generate_feature_importance_plot(output_dir)
    domain_path = generate_domain_distribution_plot(vus_results, output_dir)
    color_path = generate_structure_coloring_data(vus_results, output_dir)

    # 3. Summary stats
    high_conf = (vus_results["confidence_flag"] == "high").sum() if not vus_results.empty else 0
    class_dist = vus_results["predicted_class"].value_counts().to_dict() if not vus_results.empty else {}

    summary = {
        "total_vus": len(vus_results),
        "high_confidence": int(high_conf),
        "class_distribution": class_dist,
        "output_files": {
            "predictions": str(pred_path),
            "feature_importance": str(fi_path),
            "domain_distribution": str(domain_path),
            "structure_coloring": str(color_path),
        },
    }

    # Save summary
    summary_path = output_dir / "vus_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("VUS pipeline complete: %d VUS classified, %d high confidence",
                len(vus_results), high_conf)

    return {
        "predictions": vus_results,
        "summary": summary,
    }
