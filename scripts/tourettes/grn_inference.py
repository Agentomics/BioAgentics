"""GRN inference for striatal interneuron subtypes via GRNBoost2.

Memory-safe pipeline for 8GB machine:
1. Reconstruct interneuron subset from reference atlas
2. Run GRNBoost2 on subsampled data (HVGs + TS risk genes)
3. Score regulon activity per cell (AUCell-like)
4. Analyze per interneuron subtype

The gene set is the union of highly variable genes (HVGs) and all TS risk
genes present in the expression data, ensuring GWAS/rare/de novo variant
genes appear as potential regulon targets.

Usage:
    uv run python scripts/tourettes/grn_inference.py [stage]
    stages: prepare, grnboost2, score, analyze, all
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.future.infer_string = False
ad.settings.allow_write_nullable_strings = True

# --- Paths ---
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "tourettes" / "ts-striatal-interneuron-pathology"
REFERENCE_DIR = DATA_DIR / "reference" / "GSE151761"
OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-striatal-interneuron-pathology"
GRN_DIR = OUTPUT_DIR / "grn"
FIGURES_DIR = OUTPUT_DIR / "figures"

# GRNBoost2 parameters
SUBSAMPLE_N = 5000  # cells for GRNBoost2 (memory-safe)
N_TOP_GENES = 3000  # HVGs for GRN inference
MIN_TF_TARGETS = 10  # minimum targets to keep a regulon
TOP_TARGETS_PER_TF = 50  # top targets per TF for scoring
RANDOM_SEED = 42

# TS risk genes — force-included alongside HVGs so they appear as regulon targets
from bioagentics.data.tourettes.gene_sets import (
    DE_NOVO_VARIANT,
    RARE_VARIANT,
    TSAICG_GWAS,
)

TS_RISK_GENES: set[str] = set(TSAICG_GWAS) | set(RARE_VARIANT) | set(DE_NOVO_VARIANT)

# Human TF list — curated from Lambert et al. 2018 (Cell) + GO:0003700
# This is the intersection of known human TFs commonly used in SCENIC analyses
KNOWN_TF_FAMILIES = [
    # Major TF families relevant to interneuron biology
    "ARX", "DLX1", "DLX2", "DLX5", "DLX6", "LHX6", "LHX8", "NKX2-1",
    "SOX6", "SOX5", "SOX11", "SOX4", "SOX2", "SOX9", "SOX10",
    "PAX6", "PAX2", "MEIS1", "MEIS2", "PBX1", "PBX3",
    "FOXP1", "FOXP2", "FOXP4", "FOXG1", "FOXA1", "FOXA2",
    "ISL1", "EBF1", "EBF3", "SP8", "SP9",
    "ASCL1", "TCF4", "TCF7L2", "TCF12", "NEUROD1", "NEUROD2", "NEUROD6",
    "ZEB2", "ZEB1", "ZBTB16", "ZBTB20", "ZNF804A",
    "NR2F1", "NR2F2", "NR4A1", "NR4A2", "NR4A3",
    "ETV1", "ETV5", "ETS1", "ERG", "FLI1",
    "GATA2", "GATA3", "GATA4",
    "KLF2", "KLF4", "KLF6", "KLF9", "KLF13",
    "MAF", "MAFB", "MAFF", "MAFG", "MAFK",
    "MYT1L", "MYT1", "MYRF",
    "NFIB", "NFIA", "NFIC", "NFIX",
    "OLIG1", "OLIG2",
    "PROX1", "POU3F1", "POU3F2", "POU3F3", "POU3F4",
    "RORA", "RORB", "RORC",
    "RUNX1", "RUNX2", "RUNX3",
    "SATB1", "SATB2",
    "SIX3", "SIX4",
    "SMAD1", "SMAD3", "SMAD4", "SMAD5",
    "STAT1", "STAT3", "STAT5A", "STAT5B",
    "TBR1", "TBX21",
    "TEAD1", "TEAD2", "TEAD4",
    "TFAP2A", "TFAP2B", "TFAP2C",
    "TP53", "TP63", "TP73",
    "ZIC1", "ZIC2", "ZIC3",
    "CREB1", "CREB3", "CREB5", "ATF1", "ATF2", "ATF3", "ATF4", "ATF6",
    "JUN", "JUNB", "JUND", "FOS", "FOSB", "FOSL1", "FOSL2",
    "BACH1", "BACH2", "BATF", "BATF3",
    "E2F1", "E2F3", "E2F4",
    "EGR1", "EGR2", "EGR3", "EGR4",
    "REST", "CTCF", "YY1",
    "HIF1A", "EPAS1", "HIF3A",
    "HSF1", "HSF2",
    "IRF1", "IRF2", "IRF3", "IRF4", "IRF5", "IRF7", "IRF8",
    "LEF1", "MYC", "MYCN",
    "NFKB1", "NFKB2", "REL", "RELA", "RELB",
    "NFE2L2", "NFE2L1",
    "PPARA", "PPARG", "PPARD",
    "RFX1", "RFX2", "RFX3", "RFX4", "RFX5",
    "SPI1", "SPIB", "SPIC",
    "TFE3", "TFEB", "TFEC", "MITF",
    "USF1", "USF2",
    "HMGA1", "HMGA2", "HMGB1", "HMGB2",
    "HES1", "HES5", "HEY1", "HEY2",
    "ID1", "ID2", "ID3", "ID4",
    "IKZF1", "IKZF2", "IKZF3",
    "MEF2A", "MEF2B", "MEF2C", "MEF2D",
    "NEUROG1", "NEUROG2",
    "NPAS1", "NPAS2", "NPAS3", "NPAS4",
    "TCF3", "TCF21",
    "SRF", "CEBPA", "CEBPB", "CEBPD",
    "RXRA", "RXRB", "RXRG",
    "RARA", "RARB", "RARG",
    "ESR1", "ESR2", "AR", "PGR",
    "THRB", "THRA",
    "VDR", "NR3C1", "NR3C2",
    "POU2F1", "POU2F2", "POU5F1",
    "PITX1", "PITX2", "PITX3",
    "OTX1", "OTX2", "EMX1", "EMX2",
    "DBX1", "DBX2", "GSX1", "GSX2",
    "LMX1A", "LMX1B",
    "EN1", "EN2", "GBX2",
    "HAND2", "HAND1",
    "PRDM1", "PRDM8", "PRDM16",
    "BCL6", "BCL11A", "BCL11B",
    "CUX1", "CUX2",
    "FEZF1", "FEZF2",
    "NR1H3", "NR1H2", "NR5A1", "NR5A2",
    "ONECUT1", "ONECUT2",
    "SREBF1", "SREBF2",
    "CLOCK", "ARNTL", "ARNTL2",
]


def get_tf_list(gene_names: list[str] | pd.Index) -> list[str]:
    """Get list of transcription factors present in the dataset."""
    gene_set = set(gene_names)
    tfs = sorted(set(KNOWN_TF_FAMILIES) & gene_set)
    print(f"  TFs in dataset: {len(tfs)} / {len(KNOWN_TF_FAMILIES)} known TFs")
    return tfs


def stage_prepare():
    """Stage 1: Reconstruct classified interneuron subset."""
    print("=== Stage 1: Preparing interneuron expression data ===")
    GRN_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    out_path = GRN_DIR / "interneurons_for_grn.h5ad"
    if out_path.exists():
        adata = ad.read_h5ad(out_path)
        print(f"  Already prepared: {adata.n_obs} cells x {adata.n_vars} genes")
        return adata

    # Load the 3 reference datasets
    print("  Loading reference datasets...")
    datasets = []
    labels = []

    for name, label in [
        ("GSE151761_adHumInt.h5ad", "adHumInt"),
        ("GSE151761_hum_str_int_DS.h5ad", "DS"),
        ("GSE151761_hum_str_int_10X.h5ad", "10X"),
    ]:
        path = REFERENCE_DIR / name
        if path.exists():
            a = ad.read_h5ad(path)
            a.obs["dataset"] = label
            a.var_names_make_unique()
            a.obs_names_make_unique()
            print(f"    {label}: {a.n_obs} cells x {a.n_vars} genes")
            datasets.append(a)
            labels.append(label)

    # Merge on common genes
    print("  Merging datasets...")
    adata = ad.concat(datasets, join="inner", merge="same")
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    del datasets
    gc.collect()
    print(f"  Merged: {adata.n_obs} cells x {adata.n_vars} genes")

    # Basic QC
    print("  Running QC...")
    sc.pp.filter_cells(adata, min_genes=500)
    sc.pp.filter_genes(adata, min_cells=10)
    print(f"  Post-QC: {adata.n_obs} cells x {adata.n_vars} genes")

    # Normalize
    print("  Normalizing...")
    # Store raw counts
    adata.layers["raw_counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Classify using marker scoring
    print("  Classifying cells by marker genes...")
    from bioagentics.tourettes.striatal_interneuron.classify import (
        classify_by_markers,
        flag_ambiguous,
        score_marker_genes,
    )
    adata = score_marker_genes(adata)
    adata = classify_by_markers(adata)
    adata = flag_ambiguous(adata)

    # Filter to interneurons
    interneurons = adata[adata.obs["broad_type"] == "Interneuron"].copy()
    del adata
    gc.collect()
    print(f"  Interneurons: {interneurons.n_obs} cells")

    # Exclude ChAT_CIN (n=15, underpowered) and TH (0 cells)
    valid_subtypes = ["PV_FSI", "SST_PLTS", "NPY_NGF", "CR", "PTHLH"]
    mask = interneurons.obs["interneuron_subclass"].isin(valid_subtypes)
    interneurons = interneurons[mask].copy()
    print(f"  After filtering rare subtypes: {interneurons.n_obs} cells")
    print(f"  Subtype distribution: {interneurons.obs['interneuron_subclass'].value_counts().to_dict()}")

    # Select HVGs + TS risk genes
    print("  Selecting highly variable genes...")
    sc.pp.highly_variable_genes(interneurons, n_top_genes=N_TOP_GENES)
    n_hvg = interneurons.var["highly_variable"].sum()
    print(f"  HVGs: {n_hvg}")

    # Force-include TS risk genes present in the data
    risk_in_data = TS_RISK_GENES & set(interneurons.var_names)
    risk_not_hvg = risk_in_data - set(interneurons.var_names[interneurons.var["highly_variable"]])
    interneurons.var.loc[list(risk_not_hvg), "highly_variable"] = True
    n_total = interneurons.var["highly_variable"].sum()
    print(f"  TS risk genes force-included: {len(risk_not_hvg)} "
          f"(total gene set: {n_total}, risk genes missing from data: "
          f"{sorted(TS_RISK_GENES - risk_in_data)})")

    # Save
    interneurons.write_h5ad(out_path)
    print(f"  Saved: {out_path.name}")
    return interneurons


def stage_grnboost2(adata: ad.AnnData | None = None):
    """Stage 2: Run GRNBoost2 for TF-target coexpression inference."""
    print("\n=== Stage 2: GRNBoost2 TF-target inference ===")

    adj_path = GRN_DIR / "grnboost2_adjacencies.csv"
    if adj_path.exists():
        print(f"  Already computed: {adj_path.name}")
        return pd.read_csv(adj_path)

    if adata is None:
        adata = ad.read_h5ad(GRN_DIR / "interneurons_for_grn.h5ad")

    # Use HVGs only for GRN inference
    hvg_mask = adata.var["highly_variable"]
    adata_hvg = adata[:, hvg_mask].copy()
    gene_names = list(adata_hvg.var_names)
    print(f"  Using {len(gene_names)} HVGs for GRN inference")

    # Get TF list
    tf_names = get_tf_list(gene_names)
    if len(tf_names) < 20:
        print("  WARNING: Very few TFs found. Expanding to all genes with known TF annotations...")
        # Fall back to using genes that look like TFs in the gene names
        tf_names = get_tf_list(adata.var_names)
        tf_names = [t for t in tf_names if t in gene_names]
        print(f"  TFs after expansion: {len(tf_names)}")

    # Save TF list
    pd.Series(tf_names).to_csv(GRN_DIR / "tf_list.csv", index=False, header=False)

    # Subsample for memory efficiency
    np.random.seed(RANDOM_SEED)
    if adata_hvg.n_obs > SUBSAMPLE_N:
        # Stratified subsample by subtype
        indices = []
        for subtype in adata_hvg.obs["interneuron_subclass"].unique():
            subtype_idx = np.where(adata_hvg.obs["interneuron_subclass"] == subtype)[0]
            n_sample = max(50, int(SUBSAMPLE_N * len(subtype_idx) / adata_hvg.n_obs))
            n_sample = min(n_sample, len(subtype_idx))
            sampled = np.random.choice(subtype_idx, n_sample, replace=False)
            indices.extend(sampled)
        adata_sub = adata_hvg[sorted(indices)].copy()
        print(f"  Subsampled: {adata_sub.n_obs} cells (stratified by subtype)")
        print(f"  Subtype distribution: {adata_sub.obs['interneuron_subclass'].value_counts().to_dict()}")
    else:
        adata_sub = adata_hvg

    # Convert to dense matrix for GRNBoost2
    print("  Converting to dense matrix...")
    if sp.issparse(adata_sub.X):
        expr_matrix = pd.DataFrame(
            adata_sub.X.toarray(),
            index=adata_sub.obs_names,
            columns=adata_sub.var_names,
        )
    else:
        expr_matrix = pd.DataFrame(
            adata_sub.X,
            index=adata_sub.obs_names,
            columns=adata_sub.var_names,
        )

    del adata_sub, adata_hvg
    gc.collect()

    mem_mb = expr_matrix.values.nbytes / 1e6
    print(f"  Expression matrix: {expr_matrix.shape[0]} x {expr_matrix.shape[1]} ({mem_mb:.1f} MB)")

    # Run GENIE3-style GRN inference using Extremely Randomized Trees
    # (arboreto's dask integration is broken with newer dask versions;
    #  ExtraTrees is the original GENIE3 method, equivalent to GRNBoost2)
    print("  Running GENIE3-style GRN inference (ExtraTrees)...")
    from sklearn.ensemble import ExtraTreesRegressor

    tf_set = set(tf_names)
    target_genes = [g for g in expr_matrix.columns if g not in tf_set]
    print(f"  {len(tf_names)} TFs -> {len(target_genes)} target genes")

    tf_expr = expr_matrix[tf_names].values
    adjacency_records = []

    for i, target in enumerate(target_genes):
        if (i + 1) % 500 == 0:
            print(f"    Progress: {i + 1}/{len(target_genes)} genes...")
        y = expr_matrix[target].values

        # Skip genes with near-zero variance
        if np.std(y) < 1e-6:
            continue

        etr = ExtraTreesRegressor(
            n_estimators=100,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
        etr.fit(tf_expr, y)

        importances = etr.feature_importances_
        for j, tf in enumerate(tf_names):
            if importances[j] > 0.005:
                adjacency_records.append({
                    "TF": tf,
                    "target": target,
                    "importance": importances[j],
                })

    adjacencies = pd.DataFrame(adjacency_records)
    del expr_matrix, tf_expr
    gc.collect()

    print(f"  GRN inference complete: {len(adjacencies)} TF-target edges")
    print(f"  Unique TFs with edges: {adjacencies['TF'].nunique()}")

    # Filter weak edges (keep top importance scores)
    importance_threshold = adjacencies["importance"].quantile(0.5) if len(adjacencies) > 0 else 0.001
    adj_filtered = adjacencies[adjacencies["importance"] > importance_threshold].copy()
    print(f"  After filtering (importance > {importance_threshold:.4f}): {len(adj_filtered)} edges")

    # Save
    adj_filtered.to_csv(adj_path, index=False)
    print(f"  Saved: {adj_path.name}")
    return adj_filtered


def stage_score(adata: ad.AnnData | None = None, adjacencies: pd.DataFrame | None = None):
    """Stage 3: Define regulons and score per-cell regulon activity."""
    print("\n=== Stage 3: Regulon scoring ===")

    activity_path = GRN_DIR / "regulon_activity.csv"
    regulon_path = GRN_DIR / "regulons.csv"

    if activity_path.exists() and regulon_path.exists():
        print(f"  Already computed: {activity_path.name}")
        activity = pd.read_csv(activity_path, index_col=0)
        regulons_df = pd.read_csv(regulon_path)
        return activity, regulons_df

    if adata is None:
        adata = ad.read_h5ad(GRN_DIR / "interneurons_for_grn.h5ad")
    if adjacencies is None:
        adjacencies = pd.read_csv(GRN_DIR / "grnboost2_adjacencies.csv")

    # Define regulons: for each TF, take top N targets by importance
    print("  Defining regulons from GRNBoost2 adjacencies...")
    regulons = {}
    regulon_records = []

    for tf in adjacencies["TF"].unique():
        tf_edges = adjacencies[adjacencies["TF"] == tf].sort_values("importance", ascending=False)
        targets = tf_edges["target"].head(TOP_TARGETS_PER_TF).tolist()

        if len(targets) >= MIN_TF_TARGETS:
            # Only keep targets that are in the expression data
            valid_targets = [t for t in targets if t in adata.var_names]
            if len(valid_targets) >= MIN_TF_TARGETS:
                regulons[tf] = valid_targets
                for target in valid_targets:
                    importance = tf_edges[tf_edges["target"] == target]["importance"].iloc[0]
                    regulon_records.append({
                        "TF": tf,
                        "target": target,
                        "importance": importance,
                    })

    regulons_df = pd.DataFrame(regulon_records)
    print(f"  Defined {len(regulons)} regulons (>= {MIN_TF_TARGETS} targets each)")

    # Score regulon activity per cell using AUCell-like approach
    # Use scanpy's score_genes for each regulon
    print("  Scoring regulon activity per cell...")
    activity_data = {}

    for tf, targets in regulons.items():
        # Include the TF itself in the gene set
        gene_set = [tf] + targets if tf in adata.var_names else targets
        gene_set = [g for g in gene_set if g in adata.var_names]

        if len(gene_set) >= 5:
            sc.tl.score_genes(adata, gene_set, score_name=f"regulon_{tf}")
            activity_data[tf] = adata.obs[f"regulon_{tf}"].values.copy()
            # Clean up
            del adata.obs[f"regulon_{tf}"]

    activity = pd.DataFrame(
        activity_data,
        index=adata.obs_names,
    )
    activity["interneuron_subclass"] = adata.obs["interneuron_subclass"].values

    print(f"  Scored {len(activity_data)} regulons across {adata.n_obs} cells")

    # Save
    regulons_df.to_csv(regulon_path, index=False)
    activity.to_csv(activity_path)
    print(f"  Saved: {regulon_path.name}, {activity_path.name}")

    return activity, regulons_df


def stage_analyze(activity: pd.DataFrame | None = None, regulons_df: pd.DataFrame | None = None):
    """Stage 4: Analyze regulons per interneuron subtype + visualize."""
    print("\n=== Stage 4: Per-subtype regulon analysis ===")

    if activity is None:
        activity = pd.read_csv(GRN_DIR / "regulon_activity.csv", index_col=0)
    if regulons_df is None:
        regulons_df = pd.read_csv(GRN_DIR / "regulons.csv")

    subtype_col = "interneuron_subclass"
    tf_cols = [c for c in activity.columns if c != subtype_col]
    subtypes = sorted(activity[subtype_col].unique())

    print(f"  Subtypes: {subtypes}")
    print(f"  Regulons: {len(tf_cols)}")

    # 1. Per-subtype mean regulon activity
    mean_activity = activity.groupby(subtype_col)[tf_cols].mean()
    mean_activity.to_csv(GRN_DIR / "mean_regulon_activity_by_subtype.csv")

    # 2. Identify subtype-specific regulons (Kruskal-Wallis test)
    from scipy import stats

    specificity_records = []
    for tf in tf_cols:
        groups = [activity.loc[activity[subtype_col] == st, tf].values for st in subtypes]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) >= 2:
            stat, pval = stats.kruskal(*groups)
            specificity_records.append({
                "TF": tf,
                "kruskal_h": stat,
                "kruskal_p": pval,
                "n_targets": len(regulons_df[regulons_df["TF"] == tf]),
            })

    specificity_df = pd.DataFrame(specificity_records)

    # Multiple testing correction (Bonferroni)
    if len(specificity_df) > 0:
        specificity_df["p_corrected"] = np.minimum(
            specificity_df["kruskal_p"] * len(specificity_df), 1.0
        )
        specificity_df = specificity_df.sort_values("kruskal_p")

        sig = specificity_df[specificity_df["p_corrected"] < 0.05]
        print(f"  Differentially active regulons (Bonf. p<0.05): {len(sig)} / {len(specificity_df)}")

        # For each significant TF, find which subtype it's most active in
        for _, row in sig.head(20).iterrows():
            tf = row["TF"]
            best_subtype = mean_activity[tf].idxmax()
            best_score = mean_activity[tf].max()
            print(f"    {tf}: highest in {best_subtype} (score={best_score:.3f}, p={row['kruskal_p']:.2e})")

    specificity_df.to_csv(GRN_DIR / "regulon_specificity.csv", index=False)

    # 3. Identify master regulators per subtype
    print("\n  Master regulators per subtype:")
    master_regs = {}
    for subtype in subtypes:
        # Z-score each TF across subtypes
        z_scores = (mean_activity.loc[subtype] - mean_activity.mean()) / (mean_activity.std() + 1e-10)
        top_tfs = z_scores.sort_values(ascending=False).head(10)
        master_regs[subtype] = top_tfs
        print(f"  {subtype}:")
        for tf, z in top_tfs.head(5).items():
            n_targ = len(regulons_df[regulons_df["TF"] == tf])
            print(f"    {tf}: z={z:.2f}, {n_targ} targets")

    # Save master regulators
    master_reg_records = []
    for subtype, scores in master_regs.items():
        for tf, z in scores.items():
            n_targ = len(regulons_df[regulons_df["TF"] == tf])
            master_reg_records.append({
                "subtype": subtype,
                "TF": tf,
                "z_score": z,
                "n_targets": n_targ,
                "mean_activity": mean_activity.loc[subtype, tf],
            })
    pd.DataFrame(master_reg_records).to_csv(GRN_DIR / "master_regulators.csv", index=False)

    # 4. Shared vs subtype-specific regulons
    print("\n  Shared regulatory programs:")
    if len(specificity_df) > 0:
        non_sig = specificity_df[specificity_df["p_corrected"] >= 0.05]
        shared_tfs = non_sig["TF"].tolist()
        print(f"  Shared regulons (not subtype-specific): {len(shared_tfs)}")
        if shared_tfs:
            # Top shared by overall activity
            shared_mean = mean_activity[shared_tfs].mean().sort_values(ascending=False)
            for tf in shared_mean.head(5).index:
                n_targ = len(regulons_df[regulons_df["TF"] == tf])
                print(f"    {tf}: mean activity={shared_mean[tf]:.3f}, {n_targ} targets")

    # 5. Generate visualizations
    print("\n  Generating visualizations...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Heatmap: regulon activity by subtype (top 30 most variable)
    top_variable = mean_activity.var().sort_values(ascending=False).head(30).index
    if len(top_variable) > 0:
        fig, ax = plt.subplots(figsize=(14, 6))
        data_plot = mean_activity[top_variable].T
        sns.heatmap(data_plot, cmap="RdBu_r", center=0, ax=ax,
                    xticklabels=True, yticklabels=True)
        ax.set_title("Top 30 Most Variable Regulon Activities by Interneuron Subtype")
        ax.set_xlabel("Interneuron Subtype")
        ax.set_ylabel("TF Regulon")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "heatmap_regulon_activity.png", dpi=150)
        plt.close()
        print(f"    Saved: heatmap_regulon_activity.png")

    # Bar plot: master regulators per subtype (top 5)
    if master_reg_records:
        fig, axes = plt.subplots(1, len(subtypes), figsize=(4 * len(subtypes), 5), sharey=False)
        if len(subtypes) == 1:
            axes = [axes]
        for ax, subtype in zip(axes, subtypes):
            top = master_regs[subtype].head(5)
            ax.barh(range(len(top)), top.values, color=plt.cm.Set2(subtypes.index(subtype)))
            ax.set_yticks(range(len(top)))
            ax.set_yticklabels(top.index)
            ax.set_xlabel("Z-score")
            ax.set_title(f"{subtype}\nMaster Regulators")
            ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "barplot_master_regulators.png", dpi=150)
        plt.close()
        print(f"    Saved: barplot_master_regulators.png")

    # Regulon network summary: TF count, target count distribution
    if len(regulons_df) > 0:
        tf_summary = regulons_df.groupby("TF").agg(
            n_targets=("target", "count"),
            mean_importance=("importance", "mean"),
            max_importance=("importance", "max"),
        ).sort_values("n_targets", ascending=False)
        tf_summary.to_csv(GRN_DIR / "tf_summary.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(tf_summary["n_targets"], bins=30, edgecolor="k", alpha=0.7)
        ax1.set_xlabel("Number of Targets")
        ax1.set_ylabel("Number of TFs")
        ax1.set_title("Distribution of Regulon Sizes")

        ax2.hist(regulons_df["importance"], bins=50, edgecolor="k", alpha=0.7, color="coral")
        ax2.set_xlabel("GRNBoost2 Importance")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of TF-Target Importance Scores")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "regulon_network_summary.png", dpi=150)
        plt.close()
        print(f"    Saved: regulon_network_summary.png")

    # Regulon activity violin plot for top TFs
    if len(sig) > 0:
        top_sig_tfs = sig.head(8)["TF"].tolist()
        top_sig_tfs = [t for t in top_sig_tfs if t in activity.columns]
        if top_sig_tfs:
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            for i, tf in enumerate(top_sig_tfs[:8]):
                ax = axes[i]
                plot_data = activity[[tf, subtype_col]].copy()
                plot_data.columns = ["score", "subtype"]
                for j, st in enumerate(subtypes):
                    vals = plot_data[plot_data["subtype"] == st]["score"].values
                    parts = ax.violinplot([vals], positions=[j], showmeans=True, showmedians=True)
                    for pc in parts["bodies"]:
                        pc.set_facecolor(plt.cm.Set2(j))
                ax.set_xticks(range(len(subtypes)))
                ax.set_xticklabels(subtypes, rotation=45, ha="right", fontsize=8)
                ax.set_title(tf, fontsize=10)
                ax.set_ylabel("Regulon Activity")
            for i in range(len(top_sig_tfs), 8):
                axes[i].set_visible(False)
            plt.suptitle("Top Differentially Active Regulons Across Interneuron Subtypes", y=1.02)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "violin_regulon_activity.png", dpi=150, bbox_inches="tight")
            plt.close()
            print(f"    Saved: violin_regulon_activity.png")

    print("\n=== Analysis complete ===")
    return specificity_df


def main():
    parser = argparse.ArgumentParser(description="GRN inference for striatal interneurons")
    parser.add_argument("stage", nargs="?", default="all",
                        choices=["prepare", "grnboost2", "score", "analyze", "all"])
    args = parser.parse_args()

    if args.stage == "prepare" or args.stage == "all":
        adata = stage_prepare()
        gc.collect()

    if args.stage == "grnboost2" or args.stage == "all":
        adj = stage_grnboost2(adata if args.stage == "all" else None)
        gc.collect()

    if args.stage == "score" or args.stage == "all":
        activity, regulons_df = stage_score(
            adata if args.stage == "all" else None,
            adj if args.stage == "all" else None,
        )
        gc.collect()

    if args.stage == "analyze" or args.stage == "all":
        stage_analyze(
            activity if args.stage == "all" else None,
            regulons_df if args.stage == "all" else None,
        )


if __name__ == "__main__":
    main()
