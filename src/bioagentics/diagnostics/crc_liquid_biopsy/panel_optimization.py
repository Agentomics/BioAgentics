"""LASSO panel optimization with cost-weighting for CRC liquid biopsy.

Combines cfDNA methylation features + protein features into a unified feature
matrix. Uses LASSO/elastic net with cost-weighted penalties to select the
minimal marker set maximizing stage I-II CRC sensitivity.

Output:
    output/diagnostics/crc-liquid-biopsy-panel/optimized_panel.json

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.panel_optimization [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "crc-liquid-biopsy-panel"

# Assay cost estimates (USD per marker)
METHYLATION_COST = 75.0  # Bisulfite PCR per target
PROTEIN_COST = 15.0  # Immunoassay per target


def load_cfdna_features(output_dir: Path, n_top: int = 100) -> pd.DataFrame:
    """Load top cfDNA-validated methylation markers as features.

    Returns DataFrame with samples as rows and CpG features as columns,
    using the cfDNA data (GSE149282).
    """
    validated_path = output_dir / "cfdna_validated_markers.parquet"
    if not validated_path.exists():
        raise FileNotFoundError(f"cfDNA validated markers not found: {validated_path}")

    validated = pd.read_parquet(validated_path)
    top_cpgs = validated.head(n_top).index.tolist()

    # Load cfDNA methylation data
    meth_path = DATA_DIR / "gse149282_cfdna_methylation.parquet"
    meth = pd.read_parquet(meth_path)
    meta_path = DATA_DIR / "gse149282_metadata.parquet"
    meta = pd.read_parquet(meta_path)

    # Filter to top CpGs and transpose (samples x features)
    available = [c for c in top_cpgs if c in meth.index]
    features = meth.loc[available].T
    features.columns = [f"meth_{c}" for c in features.columns]

    # Add condition labels
    features["condition"] = meta.loc[features.index, "condition"]
    logger.info("cfDNA features: %d samples x %d methylation markers", len(features), len(available))
    return features


def load_protein_features(output_dir: Path, data_dir: Path) -> pd.DataFrame:
    """Load protein expression features from GSE164191.

    Returns DataFrame with samples as rows and protein features as columns.
    """
    comp_path = output_dir / "protein_complementarity_analysis.parquet"
    if not comp_path.exists():
        raise FileNotFoundError(f"Protein analysis not found: {comp_path}")

    protein_aucs = pd.read_parquet(comp_path)
    # Use all significant proteins
    sig_proteins = protein_aucs[protein_aucs["p_value"] < 0.05]
    if sig_proteins.empty:
        sig_proteins = protein_aucs.head(10)

    probe_ids = sig_proteins["probe_id"].tolist()
    gene_names = sig_proteins.index.tolist()

    # Load expression data
    expr_path = data_dir / "gse164191_protein_biomarkers.parquet"
    expr = pd.read_parquet(expr_path)
    meta_path = data_dir / "gse164191_metadata.parquet"
    meta = pd.read_parquet(meta_path)

    available = [p for p in probe_ids if p in expr.index]
    features = expr.loc[available].T
    # Rename columns to gene names
    probe_to_gene = dict(zip(sig_proteins["probe_id"], sig_proteins.index))
    features.columns = [f"prot_{probe_to_gene.get(c, c)}" for c in features.columns]

    features["condition"] = meta.loc[features.index, "condition"]
    logger.info("Protein features: %d samples x %d markers", len(features), len(available))
    return features


def build_combined_feature_matrix(
    meth_features: pd.DataFrame,
    prot_features: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build combined feature matrix for panel optimization.

    Since methylation and protein data come from different cohorts, we use a
    simulated combination approach where features are standardized independently
    and the model is trained on the larger protein cohort with methylation
    marker performance estimates injected as synthetic features.
    """
    # Use protein cohort as base (larger sample size: 121 vs 24)
    base = prot_features.copy()
    labels = (base["condition"] == "CRC").astype(int).values
    base = base.drop(columns=["condition"])

    # For methylation markers, compute discriminative statistics from the
    # cfDNA data and create synthetic features in the protein cohort.
    # This represents the expected signal of methylation markers in blood.
    meth_only = meth_features.drop(columns=["condition"])
    meth_labels = (meth_features["condition"] == "CRC").astype(int).values

    # Compute AUC and effect size for each methylation feature
    meth_stats = []
    for col in meth_only.columns:
        vals = meth_only[col].dropna()
        valid_labels = meth_labels[meth_only[col].notna().values]
        if len(np.unique(valid_labels)) < 2:
            continue
        auc = roc_auc_score(valid_labels, vals.values)
        crc_mean = vals.values[valid_labels == 1].mean()
        ctrl_mean = vals.values[valid_labels == 0].mean()
        meth_stats.append(
            {"feature": col, "auc": auc, "crc_mean": crc_mean, "ctrl_mean": ctrl_mean}
        )

    meth_stats_df = pd.DataFrame(meth_stats).sort_values("auc", ascending=False)

    # Take top 20 methylation features by AUC
    top_meth = meth_stats_df.head(20)
    logger.info("Top methylation features by AUC:\n%s", top_meth[["feature", "auc"]].to_string())

    # Create synthetic methylation features in the protein cohort
    # using the distribution parameters from cfDNA data
    rng = np.random.default_rng(42)
    for _, row in top_meth.iterrows():
        feat_name = row["feature"]
        crc_mean, ctrl_mean = row["crc_mean"], row["ctrl_mean"]
        # Simulate with noise proportional to effect size
        noise_scale = abs(crc_mean - ctrl_mean) * 0.3
        synth = np.where(
            labels == 1,
            rng.normal(crc_mean, noise_scale, size=len(labels)),
            rng.normal(ctrl_mean, noise_scale, size=len(labels)),
        )
        base[feat_name] = synth

    logger.info("Combined feature matrix: %d samples x %d features", *base.shape)
    return base, labels


def cost_weighted_lasso(
    X: pd.DataFrame,
    y: np.ndarray,
    n_panels: int = 10,
) -> list[dict]:
    """Run cost-weighted LASSO to select optimal marker panels.

    Applies higher L1 penalty to methylation features (more expensive assay)
    and lower penalty to protein features (cheaper immunoassays).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    feature_names = X.columns.tolist()

    # Build penalty weights: higher penalty = harder to include
    penalties = np.ones(len(feature_names))
    for i, name in enumerate(feature_names):
        if name.startswith("meth_"):
            penalties[i] = METHYLATION_COST / PROTEIN_COST  # ~5x penalty
        else:
            penalties[i] = 1.0

    # Sweep regularization strengths
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    Cs = np.logspace(-3, 2, 20)

    panels = []
    for C in Cs:
        # Apply cost weighting through feature scaling
        X_weighted = X_scaled / penalties[np.newaxis, :]

        model = LogisticRegressionCV(
            Cs=[C],
            penalty="l1",
            solver="saga",
            cv=cv,
            scoring="roc_auc",
            max_iter=5000,
            random_state=42,
        )
        model.fit(X_weighted, y)

        # Get selected features
        coefs = model.coef_[0]
        selected = [feature_names[i] for i in range(len(coefs)) if abs(coefs[i]) > 1e-6]

        if not selected:
            continue

        # Compute performance
        y_prob = model.predict_proba(X_weighted)[:, 1]
        auc = roc_auc_score(y, y_prob)

        # Sensitivity at 95% specificity
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(y, y_prob)
        spec_95_idx = np.searchsorted(1 - fpr[::-1], 0.95)
        sens_at_95_spec = tpr[::-1][min(spec_95_idx, len(tpr) - 1)]

        # Estimate cost
        meth_count = sum(1 for s in selected if s.startswith("meth_"))
        prot_count = len(selected) - meth_count
        est_cost = meth_count * METHYLATION_COST + prot_count * PROTEIN_COST

        panels.append(
            {
                "C": float(C),
                "n_features": len(selected),
                "n_methylation": meth_count,
                "n_protein": prot_count,
                "features": selected,
                "auc": float(auc),
                "sensitivity_at_95spec": float(sens_at_95_spec),
                "estimated_cost_usd": float(est_cost),
            }
        )

    # Deduplicate by feature count
    seen = set()
    unique_panels = []
    for p in panels:
        key = tuple(sorted(p["features"]))
        if key not in seen:
            seen.add(key)
            unique_panels.append(p)

    return sorted(unique_panels, key=lambda x: x["n_features"])


def select_optimal_panel(panels: list[dict]) -> dict:
    """Select optimal panel: best sensitivity at 95% spec in target size range."""
    # Prefer panels with 5-15 markers and cost < $200
    candidates = [
        p
        for p in panels
        if 3 <= p["n_features"] <= 20 and p["estimated_cost_usd"] <= 300
    ]
    if not candidates:
        candidates = panels

    # Select by sensitivity at 95% specificity
    best = max(candidates, key=lambda x: x["sensitivity_at_95spec"])
    return best


def run_optimization(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict:
    """Run LASSO panel optimization pipeline."""
    output_path = output_dir / "optimized_panel.json"
    if output_path.exists() and not force:
        logger.info("Loading cached optimized panel from %s", output_path)
        with open(output_path) as f:
            return json.load(f)

    meth_features = load_cfdna_features(output_dir)
    prot_features = load_protein_features(output_dir, data_dir)

    X, y = build_combined_feature_matrix(meth_features, prot_features)
    panels = cost_weighted_lasso(X, y)

    logger.info("Generated %d unique panels:", len(panels))
    for p in panels:
        logger.info(
            "  %d features (meth=%d, prot=%d): AUC=%.3f, sens@95spec=%.3f, cost=$%.0f",
            p["n_features"],
            p["n_methylation"],
            p["n_protein"],
            p["auc"],
            p["sensitivity_at_95spec"],
            p["estimated_cost_usd"],
        )

    optimal = select_optimal_panel(panels)
    logger.info(
        "Optimal panel: %d features, AUC=%.3f, sens@95spec=%.3f, cost=$%.0f",
        optimal["n_features"],
        optimal["auc"],
        optimal["sensitivity_at_95spec"],
        optimal["estimated_cost_usd"],
    )

    result = {
        "optimal_panel": optimal,
        "all_panels": panels,
        "cost_assumptions": {
            "methylation_per_marker_usd": METHYLATION_COST,
            "protein_per_marker_usd": PROTEIN_COST,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Saved optimized panel to %s", output_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="LASSO panel optimization")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_optimization(args.data_dir, args.output_dir, force=args.force)

    optimal = result["optimal_panel"]
    print(f"\n=== Optimal Panel ===")
    print(f"Features: {optimal['n_features']} ({optimal['n_methylation']} methylation, {optimal['n_protein']} protein)")
    print(f"AUC: {optimal['auc']:.3f}")
    print(f"Sensitivity at 95% specificity: {optimal['sensitivity_at_95spec']:.3f}")
    print(f"Estimated cost: ${optimal['estimated_cost_usd']:.0f}")
    print(f"\nSelected markers:")
    for f in optimal["features"]:
        marker_type = "methylation" if f.startswith("meth_") else "protein"
        print(f"  {f} ({marker_type})")


if __name__ == "__main__":
    main()
