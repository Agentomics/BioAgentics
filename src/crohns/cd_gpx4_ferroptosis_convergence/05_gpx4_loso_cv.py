"""Phase 2 Task 5: Test whether GPX4 improves anti-TNF response prediction via LOSO-CV.

Compares:
  (a) GPX4 alone
  (b) Top 30-gene signature + GPX4
  (c) Baseline top 30-gene signature (from anti-tnf-response-prediction project)
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path("output/crohns/anti-tnf-response-prediction/processed")
OUTPUT_DIR = Path("output/crohns/cd-gpx4-ferroptosis-convergence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TOP30_GENES = [
    "GSDMC", "ILK", "FAM136A", "NRM", "PLSCR3", "CHPF2", "NFS1", "BPIFB1",
    "PGC", "BCL2L2", "ZKSCAN5", "CASP6", "WDR70", "FAM127A", "PIGO", "TBK1",
    "RBPJ", "RHBDD2", "OSMR", "ACAD9", "TNIP2", "CDC40", "PSPC1", "DENR",
    "C6ORF58", "DESI2", "CSTF3", "SHC1", "CLDN18", "NOL10",
]

STUDIES = ["GSE16879", "GSE12251", "GSE73661"]


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load expression data and metadata for all 3 studies."""
    frames = []
    for study in STUDIES:
        expr = pd.read_csv(PROCESSED_DIR / f"{study}_expression.csv", index_col=0)
        meta = pd.read_csv(PROCESSED_DIR / f"{study}_metadata.csv")
        meta = meta.set_index("sample_id")
        expr_t = expr.T
        expr_t["study"] = study
        expr_t["response"] = meta.loc[expr_t.index, "response_status"].map(
            {"responder": 1, "non_responder": 0}
        )
        frames.append(expr_t)

    combined = pd.concat(frames)
    study_labels = combined["study"]
    response = combined["response"]
    combined = combined.drop(columns=["study", "response"])
    return combined, response, study_labels


def loso_cv(X: pd.DataFrame, y: pd.Series, studies: pd.Series,
            feature_cols: list[str], n_jobs: int = 1) -> dict:
    """Leave-One-Study-Out cross-validation with logistic regression."""
    available = [f for f in feature_cols if f in X.columns]
    if not available:
        return {"mean_auc": np.nan, "per_study": {}, "n_features": 0}

    results = {}
    for test_study in STUDIES:
        train_mask = studies != test_study
        test_mask = studies == test_study

        X_train = X.loc[train_mask, available].values
        y_train = y[train_mask].values
        X_test = X.loc[test_mask, available].values
        y_test = y[test_mask].values

        if len(np.unique(y_test)) < 2:
            continue

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(
            penalty="l2", C=1.0, max_iter=1000, solver="lbfgs",
            random_state=42, n_jobs=n_jobs,
        )
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = np.nan

        results[test_study] = {
            "auc": auc,
            "n_test": len(y_test),
            "n_resp": int(y_test.sum()),
            "n_nonresp": int((1 - y_test).sum()),
        }

    aucs = [v["auc"] for v in results.values() if not np.isnan(v["auc"])]
    mean_auc = np.mean(aucs) if aucs else np.nan

    return {"mean_auc": mean_auc, "per_study": results, "n_features": len(available)}


def bootstrap_auc_ci(X, y, studies, feature_cols, n_boot=200, seed=42):
    """Bootstrap 95% CI for LOSO-CV AUC."""
    rng = np.random.RandomState(seed)
    available = [f for f in feature_cols if f in X.columns]
    if not available:
        return np.nan, np.nan

    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(len(X), size=len(X), replace=True)
        X_b = X.iloc[idx]
        y_b = y.iloc[idx]
        s_b = studies.iloc[idx]
        result = loso_cv(X_b, y_b, s_b, available)
        if not np.isnan(result["mean_auc"]):
            aucs.append(result["mean_auc"])

    if len(aucs) < 10:
        return np.nan, np.nan
    return np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def main():
    print("Loading anti-TNF cohort data...")
    X, y, studies = load_data()
    print(f"  {len(X)} samples, {X.shape[1]} genes")
    print(f"  Responders: {y.sum():.0f}, Non-responders: {(1-y).sum():.0f}")

    # Check GPX4 availability
    gpx4_present = "GPX4" in X.columns
    print(f"\n  GPX4 in expression matrix: {gpx4_present}")
    if not gpx4_present:
        print("  ERROR: GPX4 not found. Cannot proceed.")
        return

    # Model configurations
    configs = {
        "GPX4_alone": ["GPX4"],
        "Top30_baseline": TOP30_GENES,
        "Top30_plus_GPX4": TOP30_GENES + ["GPX4"],
    }

    rows = []
    for name, features in configs.items():
        print(f"\n--- {name} ({len(features)} features) ---")
        result = loso_cv(X, y, studies, features)
        ci_lo, ci_hi = bootstrap_auc_ci(X, y, studies, features, n_boot=200)

        print(f"  Mean LOSO-CV AUC: {result['mean_auc']:.3f} (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")
        for study, perf in result["per_study"].items():
            print(f"    {study}: AUC={perf['auc']:.3f} (n={perf['n_test']}, {perf['n_resp']}R/{perf['n_nonresp']}NR)")

        rows.append({
            "model": name,
            "n_features": result["n_features"],
            "mean_auc": result["mean_auc"],
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            **{f"auc_{s}": result["per_study"].get(s, {}).get("auc", np.nan) for s in STUDIES},
        })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(OUTPUT_DIR / "gpx4_loso_cv_results.csv", index=False)

    # Compute delta-AUC
    baseline_auc = results_df.loc[results_df["model"] == "Top30_baseline", "mean_auc"].values[0]
    plus_gpx4_auc = results_df.loc[results_df["model"] == "Top30_plus_GPX4", "mean_auc"].values[0]
    gpx4_alone_auc = results_df.loc[results_df["model"] == "GPX4_alone", "mean_auc"].values[0]
    delta = plus_gpx4_auc - baseline_auc

    print(f"\n=== SUMMARY ===")
    print(f"GPX4 alone AUC: {gpx4_alone_auc:.3f}")
    print(f"Top30 baseline AUC: {baseline_auc:.3f}")
    print(f"Top30 + GPX4 AUC: {plus_gpx4_auc:.3f}")
    print(f"Delta-AUC (+ GPX4): {delta:+.3f}")
    print(f"Success criterion (AUC > 0.65): {plus_gpx4_auc > 0.65}")


if __name__ == "__main__":
    main()
