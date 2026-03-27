"""Two-tier diagnostic architecture: protein screen + methylation reflex.

Tier 1: Protein panel (7 markers, ~$105) as fast/cheap first-line screen.
Tier 2: Methylation reflex (~$60) for borderline cases only.

Decision logic:
  protein score > upper_threshold  -> CRC positive (high confidence)
  protein score < lower_threshold  -> CRC negative (high confidence)
  protein score in borderline zone -> methylation decides

Since protein (GSE164191, n=121) and methylation (GSE149282, n=24) are
separate cohorts, Tier 2 performance is simulated using the corrected
methylation AUC (0.882, from bootstrap validation with feature selection
inside LOO-CV).

Output:
    output/diagnostics/crc-liquid-biopsy-panel/two_tier_architecture_results.json

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.two_tier_architecture [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "crc-liquid-biopsy-panel"

PROTEIN_COST = 105.0  # 7 markers * $15/marker
METHYLATION_COST = 60.0  # 12 CpGs * $5/CpG
CORRECTED_METH_AUC = 0.882  # From bootstrap validation (feature selection inside LOO)
CORRECTED_METH_SENS95 = 0.750


PROTEIN_MARKERS = [
    "MMP9", "CXCL8", "S100A4", "TIMP1", "SEPT9", "ERBB2", "SPP1",
]


def load_protein_data(data_dir: Path, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load protein feature matrix and labels from GSE164191.

    Uses the 7 protein markers from complementarity analysis.
    """
    expr = pd.read_parquet(data_dir / "gse164191_protein_biomarkers.parquet")
    meta = pd.read_parquet(data_dir / "gse164191_metadata.parquet")

    prot_analysis = pd.read_parquet(output_dir / "protein_complementarity_analysis.parquet")
    available_genes = [g for g in PROTEIN_MARKERS if g in prot_analysis.index]
    probe_ids = prot_analysis.loc[available_genes, "probe_id"].tolist()

    available_probes = [p for p in probe_ids if p in expr.index]
    prot_features = expr.loc[available_probes].T
    common = prot_features.index.intersection(meta.index)
    prot_features = prot_features.loc[common]
    labels = (meta.loc[common, "condition"] == "CRC").astype(int).values

    logger.info("Protein data: %d samples, %d features (%s), %d CRC / %d control",
                prot_features.shape[0], prot_features.shape[1],
                available_genes, labels.sum(), len(labels) - labels.sum())
    return prot_features.values, labels


def get_protein_loo_scores(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Get LOO-CV probability scores from protein classifier."""
    loo = LeaveOneOut()
    probs = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        model = LogisticRegression(
            C=10.0, penalty="l2", solver="lbfgs", max_iter=5000, random_state=42,
        )
        model.fit(X_train, y[train_idx])
        probs[test_idx] = model.predict_proba(X_test)[:, 1]

    return probs


def simulate_methylation_decisions(
    y_true: np.ndarray, meth_auc: float = CORRECTED_METH_AUC, seed: int = 42
) -> np.ndarray:
    """Simulate methylation classifier predictions for a set of samples.

    Uses the corrected methylation AUC to generate realistic probability
    scores that match the observed classifier performance.

    Returns simulated methylation probabilities.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Derive score distributions from target AUC using binormal model
    # AUC = Phi(d'/sqrt(2)) where d' is the separation in standard normal space
    # For AUC=0.882, d' ≈ 1.56
    from scipy.stats import norm
    d_prime = norm.ppf(meth_auc) * np.sqrt(2)

    # Generate scores: controls ~ N(0, 1), CRC ~ N(d', 1)
    raw_scores = np.where(
        y_true == 1,
        rng.normal(d_prime, 1.0, n),
        rng.normal(0.0, 1.0, n),
    )
    # Convert to probabilities via sigmoid
    probs = 1.0 / (1.0 + np.exp(-raw_scores))
    return probs


def evaluate_two_tier(
    prot_scores: np.ndarray,
    y: np.ndarray,
    lower: float,
    upper: float,
    meth_scores: np.ndarray,
    meth_threshold: float = 0.5,
) -> dict:
    """Evaluate two-tier architecture at given borderline zone."""
    n = len(y)

    # Tier 1 decisions
    positive = prot_scores >= upper  # high confidence CRC
    negative = prot_scores <= lower  # high confidence normal
    borderline = ~positive & ~negative

    n_borderline = borderline.sum()
    pct_borderline = n_borderline / n

    # Final decisions
    final_pred = np.zeros(n)
    final_pred[positive] = 1  # protein says CRC
    final_pred[negative] = 0  # protein says normal
    # Borderline: methylation decides
    final_pred[borderline] = (meth_scores[borderline] >= meth_threshold).astype(float)

    # Metrics
    tp = ((final_pred == 1) & (y == 1)).sum()
    fp = ((final_pred == 1) & (y == 0)).sum()
    fn = ((final_pred == 0) & (y == 1)).sum()
    tn = ((final_pred == 0) & (y == 0)).sum()

    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    # Effective cost
    effective_cost = PROTEIN_COST + pct_borderline * METHYLATION_COST

    # Breakdown of where CRC cases ended up
    crc_mask = y == 1
    crc_in_positive = (positive & crc_mask).sum()
    crc_in_borderline = (borderline & crc_mask).sum()
    crc_in_negative = (negative & crc_mask).sum()
    crc_caught_by_meth = ((borderline & crc_mask) & (final_pred == 1)).sum()

    return {
        "lower_threshold": lower,
        "upper_threshold": upper,
        "n_positive": int(positive.sum()),
        "n_negative": int(negative.sum()),
        "n_borderline": int(n_borderline),
        "pct_borderline": float(pct_borderline),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "effective_cost_usd": float(effective_cost),
        "crc_caught_tier1": int(crc_in_positive),
        "crc_borderline": int(crc_in_borderline),
        "crc_missed_tier1": int(crc_in_negative),
        "crc_rescued_by_methylation": int(crc_caught_by_meth),
    }


def run_two_tier(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict:
    """Run two-tier architecture evaluation."""
    output_path = output_dir / "two_tier_architecture_results.json"
    if output_path.exists() and not force:
        logger.info("Loading cached results from %s", output_path)
        with open(output_path) as f:
            return json.load(f)

    # Load protein data and get LOO-CV scores
    X_prot, y = load_protein_data(data_dir, output_dir)
    prot_scores = get_protein_loo_scores(X_prot, y)

    prot_auc = roc_auc_score(y, prot_scores)
    fpr_p, tpr_p, _ = roc_curve(y, prot_scores)
    prot_sens95 = float(np.interp(0.05, fpr_p, tpr_p))
    logger.info("Protein LOO-CV: AUC=%.3f, sens@95spec=%.3f", prot_auc, prot_sens95)

    # Simulate methylation scores for protein cohort
    meth_scores = simulate_methylation_decisions(y)

    # Test multiple borderline zones
    # Use quantiles of protein score distribution for thresholds
    configs = [
        (0.3, 0.7),
        (0.25, 0.75),
        (0.2, 0.8),
        (0.35, 0.65),
        (0.4, 0.6),
    ]
    # Add percentile-based configs
    prot_p30 = float(np.percentile(prot_scores, 30))
    prot_p70 = float(np.percentile(prot_scores, 70))
    prot_p25 = float(np.percentile(prot_scores, 25))
    prot_p75 = float(np.percentile(prot_scores, 75))
    prot_p20 = float(np.percentile(prot_scores, 20))
    prot_p80 = float(np.percentile(prot_scores, 80))
    configs.extend([
        (prot_p30, prot_p70),
        (prot_p25, prot_p75),
        (prot_p20, prot_p80),
    ])

    tier_results = []
    for lower, upper in configs:
        result = evaluate_two_tier(prot_scores, y, lower, upper, meth_scores)
        result["config_label"] = f"[{lower:.3f}, {upper:.3f}]"
        tier_results.append(result)
        logger.info(
            "Zone %s: sens=%.3f, spec=%.3f, %d%% borderline, cost=$%.0f",
            result["config_label"], result["sensitivity"], result["specificity"],
            result["pct_borderline"] * 100, result["effective_cost_usd"],
        )

    # Find best config by sensitivity at >=90% specificity
    valid = [r for r in tier_results if r["specificity"] >= 0.90]
    if valid:
        best = max(valid, key=lambda r: r["sensitivity"])
    else:
        best = max(tier_results, key=lambda r: r["sensitivity"])

    # Compare against baselines
    # Protein-only at various thresholds
    prot_fpr, prot_tpr, _ = roc_curve(y, prot_scores)
    prot_only_sens90 = float(np.interp(0.10, prot_fpr, prot_tpr))
    prot_only_sens95 = float(np.interp(0.05, prot_fpr, prot_tpr))

    # Run repeated simulations for robustness
    sim_results = []
    for seed in range(50):
        ms = simulate_methylation_decisions(y, seed=seed + 100)
        r = evaluate_two_tier(
            prot_scores, y,
            best["lower_threshold"], best["upper_threshold"],
            ms,
        )
        sim_results.append(r)

    sim_sens = [r["sensitivity"] for r in sim_results]
    sim_spec = [r["specificity"] for r in sim_results]

    results = {
        "protein_baseline": {
            "auc": float(prot_auc),
            "sensitivity_at_90spec": float(prot_only_sens90),
            "sensitivity_at_95spec": float(prot_only_sens95),
            "cost_usd": PROTEIN_COST,
            "n_samples": len(y),
            "n_crc": int(y.sum()),
        },
        "methylation_baseline": {
            "corrected_auc": CORRECTED_METH_AUC,
            "corrected_sensitivity_at_95spec": CORRECTED_METH_SENS95,
            "cost_usd": METHYLATION_COST,
            "note": "From corrected LOO-CV (bootstrap validation, n=24)",
        },
        "two_tier_configs": tier_results,
        "best_config": {
            **best,
            "simulation_sensitivity_mean": float(np.mean(sim_sens)),
            "simulation_sensitivity_std": float(np.std(sim_sens)),
            "simulation_specificity_mean": float(np.mean(sim_spec)),
            "simulation_specificity_std": float(np.std(sim_spec)),
        },
        "comparison": {
            "protein_only_cost": PROTEIN_COST,
            "methylation_only_cost": METHYLATION_COST,
            "two_tier_effective_cost": best["effective_cost_usd"],
            "combined_cost": PROTEIN_COST + METHYLATION_COST,
            "protein_only_sens95": float(prot_only_sens95),
            "methylation_only_sens95": CORRECTED_METH_SENS95,
            "two_tier_sensitivity": best["sensitivity"],
            "two_tier_specificity": best["specificity"],
            "cost_savings_vs_combined": float(
                (PROTEIN_COST + METHYLATION_COST) - best["effective_cost_usd"]
            ),
            "pct_avoiding_tier2": float(1.0 - best["pct_borderline"]),
        },
        "methodology": {
            "tier1": "Protein panel LOO-CV (LogisticRegression, C=10, 7 markers, GSE164191 n=121)",
            "tier2": "Simulated methylation reflex using binormal model matching corrected AUC=0.882",
            "decision": "protein-high -> CRC, protein-low -> normal, borderline -> methylation decides",
            "simulation_runs": 50,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", output_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Two-tier CRC diagnostic architecture")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_two_tier(args.data_dir, args.output_dir, force=args.force)

    print("\n=== Two-Tier Architecture: Protein Screen + Methylation Reflex ===")
    prot = results["protein_baseline"]
    print(f"Protein-only: AUC={prot['auc']:.3f}, sens@95spec={prot['sensitivity_at_95spec']:.3f}, "
          f"cost=${prot['cost_usd']:.0f}")

    meth = results["methylation_baseline"]
    print(f"Methylation-only: AUC={meth['corrected_auc']:.3f}, "
          f"sens@95spec={meth['corrected_sensitivity_at_95spec']:.3f}, cost=${meth['cost_usd']:.0f}")

    best = results["best_config"]
    print(f"\nBest two-tier config: zone {best['config_label']}")
    print(f"  Sensitivity: {best['sensitivity']:.3f} (sim mean: {best['simulation_sensitivity_mean']:.3f} "
          f"+/- {best['simulation_sensitivity_std']:.3f})")
    print(f"  Specificity: {best['specificity']:.3f}")
    print(f"  Borderline: {best['pct_borderline']:.1%} of patients need Tier 2")
    print(f"  Effective cost: ${best['effective_cost_usd']:.0f}/test")

    cmp = results["comparison"]
    print(f"\n=== Comparison ===")
    print(f"  Protein-only: sens@95spec={cmp['protein_only_sens95']:.3f}, cost=${cmp['protein_only_cost']:.0f}")
    print(f"  Methylation-only: sens@95spec={cmp['methylation_only_sens95']:.3f}, cost=${cmp['methylation_only_cost']:.0f}")
    print(f"  Two-tier: sens={cmp['two_tier_sensitivity']:.3f}, spec={cmp['two_tier_specificity']:.3f}, "
          f"cost=${cmp['two_tier_effective_cost']:.0f}")
    print(f"  Combined (all patients): cost=${cmp['combined_cost']:.0f}")
    print(f"  Cost savings vs combined: ${cmp['cost_savings_vs_combined']:.0f}/test "
          f"({cmp['pct_avoiding_tier2']:.0%} avoid Tier 2)")


if __name__ == "__main__":
    main()
