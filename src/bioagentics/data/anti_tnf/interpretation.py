"""Gene signature biological interpretation and cell-type mapping.

Maps the final anti-TNF response gene signature to known biology, cell types,
published signatures, and generates interpretation visualizations.

Usage:
    uv run python -m bioagentics.data.anti_tnf.interpretation
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
import seaborn as sns

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DEFAULT_BATCH_DIR = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "batch_correction"
DEFAULT_FS_DIR = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "feature_selection"
DEFAULT_CLASSIFIER_DIR = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "classifier"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "interpretation"

# Published anti-TNF response signatures for overlap analysis
PUBLISHED_SIGNATURES = {
    "Porto_5gene": ["TREM1", "IL23R", "CCL7", "IL17F", "YES1"],
    "OSMR_oncostatin_M": ["OSMR", "OSM", "IL6ST", "IL13RA2", "LIFR"],
    "GIMATS_module": [
        "S100A8", "S100A9", "S100A12", "LCN2", "CXCL1", "CXCL2", "CXCL3",
        "CXCL5", "CXCL8", "IL1B", "IL6", "TREM1", "FCGR3A", "FCGR3B",
    ],
    "ECM_fibroblast": ["MMP2", "MMP9", "COL1A1", "COL3A1", "CD81", "FAP", "ACTA2"],
    "NAMPT_NAD": ["NAMPT", "NMNAT1", "NMNAT2", "NAPRT"],
}

# Cell-type marker gene associations (from Smillie 2019, Martin 2019, IBD atlases)
CELL_TYPE_MARKERS = {
    "Epithelial": [
        "CLDN18", "GSDMC", "BPIFB1", "PGC", "PLSCR3", "RHBDD2",
        "CDH1", "EPCAM", "KRT20",
    ],
    "Fibroblast/Stromal": [
        "ILK", "COL1A1", "MMP2", "CD81", "SHC1", "FAP", "ACTA2",
    ],
    "Myeloid/Macrophage": [
        "TREM1", "S100A8", "S100A9", "OSMR", "TNIP2", "TBK1",
        "FCGR3A", "CD14",
    ],
    "T cell": [
        "IL17F", "RORC", "IL23R", "RBPJ", "CD3D", "CD4", "CD8A",
    ],
    "B cell / Plasma": [
        "CD19", "MS4A1", "IGHG1",
    ],
    "General/Housekeeping": [
        "NRM", "FAM136A", "BCL2L2", "NFS1", "CHPF2", "ZKSCAN5",
        "CASP6", "WDR70", "ACAD9", "CDC40", "PSPC1", "DENR",
        "DESI2", "CSTF3", "NOL10", "NFYC", "PIGO", "FAM127A",
    ],
}

# MSigDB pathway annotations for ORA
PATHWAY_ANNOTATIONS = {
    "TNF_signaling": [
        "TNF", "TNFRSF1A", "TNFRSF1B", "TRADD", "TRAF2", "RIPK1",
        "NFKB1", "RELA", "TNIP2", "TBK1", "CASP6", "CASP8",
    ],
    "Th17_differentiation": [
        "RORC", "IL17F", "IL23R", "IL6ST", "STAT3", "CCR6",
    ],
    "Innate_immunity": [
        "TREM1", "TBK1", "TNIP2", "PLSCR3", "S100A8", "S100A9",
        "OSMR", "IL6ST",
    ],
    "Fibrosis_ECM": [
        "COL1A1", "MMP2", "CD81", "ILK", "SHC1", "FAP", "ACTA2",
    ],
    "Apoptosis": [
        "CASP6", "BCL2L2", "GSDMC", "PLSCR3",
    ],
    "Metabolic": [
        "NFS1", "ACAD9", "NAMPT", "PGC",
    ],
}


def load_signature(fs_dir: Path) -> list[str]:
    """Load the selected gene signature."""
    ranked = pd.read_csv(fs_dir / "ranked_genes.csv")
    return ranked[ranked["selected"]]["gene"].tolist()


def signature_overlap_analysis(signature: list[str]) -> pd.DataFrame:
    """Analyze overlap between our signature and published signatures."""
    sig_set = set(signature)
    records = []

    for pub_name, pub_genes in PUBLISHED_SIGNATURES.items():
        pub_set = set(pub_genes)
        overlap = sig_set & pub_set
        records.append({
            "published_signature": pub_name,
            "published_size": len(pub_set),
            "overlap_count": len(overlap),
            "overlap_genes": ", ".join(sorted(overlap)) if overlap else "none",
            "jaccard_index": len(overlap) / len(sig_set | pub_set) if (sig_set | pub_set) else 0,
        })

    return pd.DataFrame(records)


def cell_type_mapping(signature: list[str]) -> pd.DataFrame:
    """Map signature genes to putative cell types."""
    records = []
    sig_set = set(signature)

    for gene in signature:
        cell_types = []
        for ct, markers in CELL_TYPE_MARKERS.items():
            if gene in markers:
                cell_types.append(ct)
        records.append({
            "gene": gene,
            "cell_type": "; ".join(cell_types) if cell_types else "Unassigned",
        })

    return pd.DataFrame(records)


def pathway_enrichment_ora(signature: list[str]) -> pd.DataFrame:
    """Simple over-representation analysis against curated pathways."""
    sig_set = set(signature)
    records = []

    for pathway, genes in PATHWAY_ANNOTATIONS.items():
        pathway_set = set(genes)
        overlap = sig_set & pathway_set
        records.append({
            "pathway": pathway,
            "pathway_size": len(pathway_set),
            "overlap_count": len(overlap),
            "overlap_genes": ", ".join(sorted(overlap)) if overlap else "none",
            "fraction_in_signature": len(overlap) / len(sig_set) if sig_set else 0,
        })

    return pd.DataFrame(records).sort_values("overlap_count", ascending=False)


def plot_heatmap(
    signature: list[str],
    batch_dir: Path,
    output_dir: Path,
) -> None:
    """Generate heatmap of signature gene expression by study and response."""
    expr = pd.read_csv(batch_dir / "expression_combat.csv", index_col=0)
    metadata = pd.read_csv(batch_dir / "metadata.csv")

    available = [g for g in signature if g in expr.index]
    expr_sig = expr.loc[available]

    # Order samples by study then response
    meta = metadata.set_index("sample_id").loc[expr.columns]
    order = meta.sort_values(["study", "response_status"]).index
    expr_ordered = expr_sig[order]

    # Z-score per gene
    expr_z = expr_ordered.sub(expr_ordered.mean(axis=1), axis=0).div(expr_ordered.std(axis=1), axis=0)

    # Annotation bars
    col_colors_study = meta.loc[order, "study"].map({
        "GSE16879": "#1f77b4", "GSE12251": "#ff7f0e", "GSE73661": "#2ca02c",
    })
    col_colors_resp = meta.loc[order, "response_status"].map({
        "responder": "#2ca02c", "non_responder": "#d62728",
    })

    g = sns.clustermap(
        expr_z,
        col_cluster=False,
        row_cluster=True,
        col_colors=[col_colors_study, col_colors_resp],
        cmap="RdBu_r",
        vmin=-3, vmax=3,
        figsize=(16, 10),
        yticklabels=True,
        xticklabels=False,
    )
    g.fig.suptitle("Signature Gene Expression: Responders vs Non-Responders", y=1.02)
    g.savefig(output_dir / "signature_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close("all")
    logger.info("Saved signature heatmap")


def plot_risk_stratification(
    classifier_dir: Path,
    output_dir: Path,
) -> None:
    """Plot predicted probability distributions for R vs NR."""
    # Load elastic net predictions (best model)
    metrics = pd.read_csv(classifier_dir / "aggregate_metrics.csv")
    best = metrics.loc[metrics["auc"].idxmax(), "model"]

    # Load per-study results to reconstruct predictions
    per_study = pd.read_csv(classifier_dir / f"{best}_per_study.csv")

    # Since we don't store predictions per-sample, re-run briefly
    from bioagentics.data.anti_tnf.classifier import load_data, loso_cv

    batch_dir = Path(str(classifier_dir).replace("classifier", "batch_correction"))
    fs_dir = Path(str(classifier_dir).replace("classifier", "feature_selection"))

    X, y, study = load_data(batch_dir, fs_dir)
    res = loso_cv(X, y, study, best)

    fig, ax = plt.subplots(figsize=(8, 6))

    r_probs = res["y_prob"][res["y_true"] == 0]
    nr_probs = res["y_prob"][res["y_true"] == 1]

    ax.hist(r_probs, bins=15, alpha=0.6, color="#2ca02c", label=f"Responders (n={len(r_probs)})", density=True)
    ax.hist(nr_probs, bins=15, alpha=0.6, color="#d62728", label=f"Non-Responders (n={len(nr_probs)})", density=True)
    ax.axvline(0.5, ls="--", c="black", alpha=0.5, label="Decision threshold")
    ax.set_xlabel("Predicted Probability of Non-Response")
    ax.set_ylabel("Density")
    ax.set_title("Risk Stratification: Predicted Probability Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "risk_stratification.png", dpi=150)
    plt.close(fig)
    logger.info("Saved risk stratification plot")


def run_interpretation(
    batch_dir: Path = DEFAULT_BATCH_DIR,
    fs_dir: Path = DEFAULT_FS_DIR,
    classifier_dir: Path = DEFAULT_CLASSIFIER_DIR,
    output_dir: Path = DEFAULT_OUTPUT,
) -> None:
    """Run the full interpretation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load signature
    signature = load_signature(fs_dir)
    logger.info("Signature: %d genes", len(signature))

    # Overlap with published signatures
    overlap = signature_overlap_analysis(signature)
    overlap.to_csv(output_dir / "signature_overlap.csv", index=False)
    logger.info("Signature overlap analysis:")
    for _, row in overlap.iterrows():
        logger.info(
            "  %s: %d/%d overlap (%s)",
            row["published_signature"], row["overlap_count"],
            row["published_size"], row["overlap_genes"],
        )

    # Cell-type mapping
    ct_map = cell_type_mapping(signature)
    ct_map.to_csv(output_dir / "cell_type_mapping.csv", index=False)
    logger.info("Cell-type distribution:")
    for ct, count in ct_map["cell_type"].value_counts().items():
        logger.info("  %s: %d genes", ct, count)

    # Pathway ORA
    ora = pathway_enrichment_ora(signature)
    ora.to_csv(output_dir / "pathway_ora.csv", index=False)
    logger.info("Pathway enrichment (ORA):")
    for _, row in ora.iterrows():
        if row["overlap_count"] > 0:
            logger.info(
                "  %s: %d genes (%s)",
                row["pathway"], row["overlap_count"], row["overlap_genes"],
            )

    # Heatmap
    plot_heatmap(signature, batch_dir, output_dir)

    # Risk stratification
    plot_risk_stratification(classifier_dir, output_dir)

    logger.info("Interpretation pipeline complete")


def main():
    parser = argparse.ArgumentParser(
        description="Interpret anti-TNF response gene signature"
    )
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--fs-dir", type=Path, default=DEFAULT_FS_DIR)
    parser.add_argument("--classifier-dir", type=Path, default=DEFAULT_CLASSIFIER_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_interpretation(
        batch_dir=args.batch_dir,
        fs_dir=args.fs_dir,
        classifier_dir=args.classifier_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
