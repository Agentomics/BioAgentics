"""Phase 3 Tasks 7-9: Fibrosis association, ferroptosis-fibrosis index, iron metabolism.

Task 7: GPX4 association with fibrotic progression markers (scRNA-seq + bulk)
Task 8: Ferroptosis-fibrosis index as multivariate predictor
Task 9: Iron metabolism genes across CD subtypes
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("output/crohns/cd-gpx4-ferroptosis-convergence")
PROCESSED_DIR = Path("output/crohns/anti-tnf-response-prediction/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIBROSIS_MARKERS = ["COL1A1", "COL3A1", "ACTA2", "FAP", "ILK", "FN1", "TGFB1",
                    "MMP2", "MMP9", "TIMP1", "LOX", "CTGF", "PDGFRB"]
IRON_GENES = ["TFRC", "FTH1", "FTL", "HAMP", "SLC40A1"]
FERROPTOSIS_INDEX_GENES = ["GPX4", "NFS1", "GSDMC", "ACAD9", "ILK"]
STUDIES = ["GSE16879", "GSE12251", "GSE73661"]

TOP30_GENES = [
    "GSDMC", "ILK", "FAM136A", "NRM", "PLSCR3", "CHPF2", "NFS1", "BPIFB1",
    "PGC", "BCL2L2", "ZKSCAN5", "CASP6", "WDR70", "FAM127A", "PIGO", "TBK1",
    "RBPJ", "RHBDD2", "OSMR", "ACAD9", "TNIP2", "CDC40", "PSPC1", "DENR",
    "C6ORF58", "DESI2", "CSTF3", "SHC1", "CLDN18", "NOL10",
]


def load_bulk_data():
    """Load all anti-TNF cohort data."""
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


def task7_fibrosis_association():
    """Task 7: GPX4 association with fibrotic progression markers."""
    print("=" * 60)
    print("TASK 7: GPX4 vs Fibrotic Progression Markers (Bulk)")
    print("=" * 60)

    X, y, studies = load_bulk_data()

    # Check available fibrosis markers
    available_markers = [g for g in FIBROSIS_MARKERS if g in X.columns]
    print(f"Available fibrosis markers: {len(available_markers)}/{len(FIBROSIS_MARKERS)}")
    missing = [g for g in FIBROSIS_MARKERS if g not in X.columns]
    if missing:
        print(f"  Missing: {missing}")

    if "GPX4" not in X.columns:
        print("ERROR: GPX4 not in bulk data")
        return pd.DataFrame()

    rows = []
    for marker in available_markers:
        rho, pval = stats.spearmanr(X["GPX4"], X[marker])
        rows.append({
            "marker": marker,
            "spearman_rho": rho,
            "pvalue": pval,
            "n_samples": len(X),
        })
        marker_str = "*" if pval < 0.05 else ""
        print(f"  GPX4 vs {marker}: rho={rho:.3f}, p={pval:.4f} {marker_str}")

    result = pd.DataFrame(rows)
    # BH FDR
    if len(result) > 0:
        n = len(result)
        sorted_idx = result["pvalue"].sort_values().index
        fdr = result.loc[sorted_idx, "pvalue"].values * n / np.arange(1, n + 1)
        fdr = np.minimum.accumulate(fdr[::-1])[::-1]
        result.loc[sorted_idx, "FDR"] = np.clip(fdr, 0, 1)

    result.to_csv(OUTPUT_DIR / "gpx4_fibrosis_bulk_correlations.csv", index=False)

    # Also test per-study
    print("\n  Per-study GPX4 vs COL1A1:")
    for study in STUDIES:
        mask = studies == study
        if "COL1A1" in X.columns:
            rho, p = stats.spearmanr(X.loc[mask, "GPX4"], X.loc[mask, "COL1A1"])
            print(f"    {study}: rho={rho:.3f}, p={p:.4f}")

    n_sig = (result["pvalue"] < 0.05).sum() if len(result) > 0 else 0
    n_criterion = ((result["spearman_rho"].abs() > 0.3) & (result["pvalue"] < 0.05)).sum() if len(result) > 0 else 0
    print(f"\n  Significant (p<0.05): {n_sig}/{len(result)}")
    print(f"  Meeting criterion (|rho|>0.3, p<0.05): {n_criterion}/{len(result)}")
    return result


def task8_ferroptosis_fibrosis_index():
    """Task 8: Build ferroptosis-fibrosis index and test as predictor."""
    print("\n" + "=" * 60)
    print("TASK 8: Ferroptosis-Fibrosis Index as Predictor")
    print("=" * 60)

    X, y, studies = load_bulk_data()

    available = [g for g in FERROPTOSIS_INDEX_GENES if g in X.columns]
    print(f"Index genes available: {available}")
    missing = [g for g in FERROPTOSIS_INDEX_GENES if g not in X.columns]
    if missing:
        print(f"  Missing: {missing}")

    if len(available) < 3:
        print("ERROR: Too few index genes available")
        return {}

    # Build index: z-score and average
    X_idx = X[available].copy()
    scaler = StandardScaler()
    X_idx_z = pd.DataFrame(scaler.fit_transform(X_idx), index=X_idx.index, columns=X_idx.columns)
    ffi = X_idx_z.mean(axis=1)
    ffi.name = "ferroptosis_fibrosis_index"

    # Test index vs response
    resp_ffi = ffi[y == 1]
    nonresp_ffi = ffi[y == 0]
    stat, pval = stats.mannwhitneyu(resp_ffi, nonresp_ffi, alternative="two-sided")
    d = (resp_ffi.mean() - nonresp_ffi.mean()) / ffi.std()
    print(f"\n  FFI: responder mean={resp_ffi.mean():.3f}, non-responder mean={nonresp_ffi.mean():.3f}")
    print(f"  Cohen's d={d:.3f}, p={pval:.4f}")

    # LOSO-CV: FFI alone
    print("\n  LOSO-CV: FFI alone")
    aucs_ffi = {}
    for test_study in STUDIES:
        train_mask = studies != test_study
        test_mask = studies == test_study
        X_train = ffi[train_mask].values.reshape(-1, 1)
        y_train = y[train_mask].values
        X_test = ffi[test_mask].values.reshape(-1, 1)
        y_test = y[test_mask].values
        clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=1)
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, prob)
        except ValueError:
            auc = np.nan
        aucs_ffi[test_study] = auc
        print(f"    {test_study}: AUC={auc:.3f}")
    mean_ffi_auc = np.nanmean(list(aucs_ffi.values()))
    print(f"  Mean AUC (FFI alone): {mean_ffi_auc:.3f}")

    # LOSO-CV: Top30 + FFI
    print("\n  LOSO-CV: Top30 + FFI")
    top30_avail = [g for g in TOP30_GENES if g in X.columns]
    X_with_ffi = X[top30_avail].copy()
    X_with_ffi["FFI"] = ffi
    aucs_combined = {}
    for test_study in STUDIES:
        train_mask = studies != test_study
        test_mask = studies == test_study
        sc = StandardScaler()
        X_train = sc.fit_transform(X_with_ffi[train_mask])
        X_test = sc.transform(X_with_ffi[test_mask])
        y_train = y[train_mask].values
        y_test = y[test_mask].values
        clf = LogisticRegression(penalty="l2", C=1.0, random_state=42, max_iter=1000, n_jobs=1)
        clf.fit(X_train, y_train)
        prob = clf.predict_proba(X_test)[:, 1]
        try:
            auc = roc_auc_score(y_test, prob)
        except ValueError:
            auc = np.nan
        aucs_combined[test_study] = auc
        print(f"    {test_study}: AUC={auc:.3f}")
    mean_combined_auc = np.nanmean(list(aucs_combined.values()))
    print(f"  Mean AUC (Top30+FFI): {mean_combined_auc:.3f}")

    result = {
        "ffi_response_pvalue": pval,
        "ffi_cohens_d": d,
        "ffi_alone_auc": mean_ffi_auc,
        "top30_plus_ffi_auc": mean_combined_auc,
        "meets_criterion": mean_combined_auc > 0.65,
        "per_study_ffi": aucs_ffi,
        "per_study_combined": aucs_combined,
    }

    pd.DataFrame([result]).to_csv(OUTPUT_DIR / "ferroptosis_fibrosis_index_results.csv", index=False)

    print(f"\n  Success criterion (AUC > 0.65): {result['meets_criterion']}")
    return result


def task9_iron_metabolism():
    """Task 9: Iron metabolism genes across CD subtypes."""
    print("\n" + "=" * 60)
    print("TASK 9: Iron Metabolism Genes in Anti-TNF Cohorts")
    print("=" * 60)

    X, y, studies = load_bulk_data()

    available_iron = [g for g in IRON_GENES if g in X.columns]
    print(f"Available iron genes: {available_iron}")
    missing = [g for g in IRON_GENES if g not in X.columns]
    if missing:
        print(f"  Missing: {missing}")

    # Test responder vs non-responder for each iron gene
    rows = []
    for gene in available_iron:
        resp_vals = X.loc[y == 1, gene]
        nonresp_vals = X.loc[y == 0, gene]
        stat, pval = stats.mannwhitneyu(resp_vals, nonresp_vals, alternative="two-sided")
        pooled_std = np.sqrt(
            ((len(resp_vals) - 1) * resp_vals.std() ** 2 + (len(nonresp_vals) - 1) * nonresp_vals.std() ** 2)
            / (len(resp_vals) + len(nonresp_vals) - 2)
        )
        d = (resp_vals.mean() - nonresp_vals.mean()) / pooled_std if pooled_std > 0 else 0
        fc = np.log2((resp_vals.mean() + 1) / (nonresp_vals.mean() + 1))
        rows.append({
            "gene": gene,
            "responder_mean": resp_vals.mean(),
            "non_responder_mean": nonresp_vals.mean(),
            "log2FC_R_vs_NR": fc,
            "cohens_d": d,
            "pvalue": pval,
        })
        marker = "*" if pval < 0.05 else ""
        print(f"  {gene}: R={resp_vals.mean():.1f}, NR={nonresp_vals.mean():.1f}, d={d:.3f}, p={pval:.4f} {marker}")

    result = pd.DataFrame(rows)
    if len(result) > 0:
        n = len(result)
        sorted_idx = result["pvalue"].sort_values().index
        fdr = result.loc[sorted_idx, "pvalue"].values * n / np.arange(1, n + 1)
        fdr = np.minimum.accumulate(fdr[::-1])[::-1]
        result.loc[sorted_idx, "FDR"] = np.clip(fdr, 0, 1)

    result.to_csv(OUTPUT_DIR / "iron_metabolism_response.csv", index=False)

    # Per-study analysis
    print("\n  Per-study iron gene analysis:")
    per_study_rows = []
    for study in STUDIES:
        mask = studies == study
        X_s = X[mask]
        y_s = y[mask]
        for gene in available_iron:
            if gene not in X_s.columns:
                continue
            resp = X_s.loc[y_s == 1, gene]
            nonresp = X_s.loc[y_s == 0, gene]
            if len(resp) < 3 or len(nonresp) < 3:
                continue
            _, p = stats.mannwhitneyu(resp, nonresp, alternative="two-sided")
            per_study_rows.append({"study": study, "gene": gene, "pvalue": p})
    per_study_df = pd.DataFrame(per_study_rows)
    per_study_df.to_csv(OUTPUT_DIR / "iron_metabolism_per_study.csv", index=False)

    # Correlation: iron genes with GPX4
    print("\n  Iron gene correlations with GPX4:")
    if "GPX4" in X.columns:
        for gene in available_iron:
            rho, p = stats.spearmanr(X["GPX4"], X[gene])
            print(f"    GPX4 vs {gene}: rho={rho:.3f}, p={p:.4f}")

    n_sig = (result["pvalue"] < 0.05).sum() if len(result) > 0 else 0
    print(f"\n  Significant (p<0.05): {n_sig}/{len(result)}")
    return result


def main():
    t7 = task7_fibrosis_association()
    t8 = task8_ferroptosis_fibrosis_index()
    t9 = task9_iron_metabolism()

    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)
    t7_sig = (t7["pvalue"] < 0.05).sum() if len(t7) > 0 else 0
    t7_crit = ((t7["spearman_rho"].abs() > 0.3) & (t7["pvalue"] < 0.05)).sum() if len(t7) > 0 else 0
    print(f"Task 7: {t7_sig} fibrosis markers significantly correlated with GPX4, {t7_crit} meet criterion")
    print(f"Task 8: FFI alone AUC={t8.get('ffi_alone_auc', 'N/A'):.3f}, Top30+FFI AUC={t8.get('top30_plus_ffi_auc', 'N/A'):.3f}")
    print(f"Task 9: {(t9['pvalue'] < 0.05).sum() if len(t9) > 0 else 0} iron genes diff expressed by response")


if __name__ == "__main__":
    main()
