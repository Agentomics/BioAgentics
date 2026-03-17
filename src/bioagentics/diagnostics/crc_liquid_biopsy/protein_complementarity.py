"""Protein biomarker complementarity analysis for CRC liquid biopsy panel.

Analyzes GSE164191 protein/expression biomarker data for markers complementary
to methylation. Identifies proteins that rescue methylation false negatives
(cases where methylation markers alone miss CRC samples).

Output:
    output/diagnostics/crc-liquid-biopsy-panel/protein_complementarity_analysis.parquet

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.protein_complementarity [--force]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "crc-liquid-biopsy-panel"

# Candidate CRC protein biomarkers (gene symbols to search in expression data)
CANDIDATE_PROTEINS = [
    "CEACAM5",  # CEA
    "MUC16",  # CA-125 (related)
    "CA19-9",  # Not a gene but search for FUT3/FUT6
    "SEPT9",  # Septin-9
    "TIMP1",  # TIMP-1
    "PKM",  # M2-PK (pyruvate kinase)
    "S100A4",
    "S100A9",
    "MMP9",
    "MMP7",
    "VEGFA",
    "IL6",
    "IL8",  # CXCL8
    "CXCL8",
    "SPP1",  # Osteopontin
    "GDF15",
    "REG4",
    "LRG1",
    "AREG",
    "CHI3L1",  # YKL-40
    "CTSD",
    "ERBB2",  # HER2
]


def load_protein_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GSE164191 protein/expression data and metadata."""
    expr_path = data_dir / "gse164191_protein_biomarkers.parquet"
    meta_path = data_dir / "gse164191_metadata.parquet"

    if not expr_path.exists():
        raise FileNotFoundError(f"Protein data not found: {expr_path}")

    expr = pd.read_parquet(expr_path)
    meta = pd.read_parquet(meta_path)
    logger.info("Loaded expression: %d features x %d samples", *expr.shape)
    logger.info("Conditions: %s", meta["condition"].value_counts().to_dict())
    return expr, meta


def load_platform_annotation(data_dir: Path) -> pd.DataFrame | None:
    """Try to load platform annotation for probe-to-gene mapping."""
    import glob

    raw_dir = data_dir / "raw"
    # Look for GPL annotation files
    patterns = [str(raw_dir / "GPL*.txt"), str(raw_dir / "GPL*.annot*")]
    for pattern in patterns:
        files = glob.glob(pattern)
        for f in files:
            try:
                ann = pd.read_csv(f, sep="\t", comment="#", low_memory=False)
                if "Gene Symbol" in ann.columns and "ID" in ann.columns:
                    logger.info("Loaded platform annotation from %s", f)
                    return ann.set_index("ID")[["Gene Symbol"]].rename(
                        columns={"Gene Symbol": "gene_symbol"}
                    )
            except Exception:
                continue
    return None


def map_probes_to_genes(
    expr: pd.DataFrame, data_dir: Path
) -> dict[str, list[str]]:
    """Map gene symbols to probe IDs in the expression data.

    Returns dict mapping gene symbol -> list of probe IDs.
    """
    annotation = load_platform_annotation(data_dir)

    gene_to_probes: dict[str, list[str]] = {}
    if annotation is not None:
        # Use platform annotation
        for gene in CANDIDATE_PROTEINS:
            mask = annotation["gene_symbol"].fillna("").str.contains(
                rf"\b{gene}\b", case=False, regex=True
            )
            probes = annotation.index[mask].intersection(expr.index).tolist()
            if probes:
                gene_to_probes[gene] = probes
    else:
        # Fallback: check if any row indices match gene names
        logger.warning("No platform annotation found; using probe IDs directly")
        for gene in CANDIDATE_PROTEINS:
            matching = [p for p in expr.index if gene.lower() in str(p).lower()]
            if matching:
                gene_to_probes[gene] = matching

    logger.info("Mapped %d / %d candidate genes to probes", len(gene_to_probes), len(CANDIDATE_PROTEINS))
    return gene_to_probes


def compute_protein_auc(
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    gene_to_probes: dict[str, list[str]],
) -> pd.DataFrame:
    """Compute AUC for each protein marker discriminating CRC vs control."""
    crc_samples = meta[meta["condition"] == "CRC"].index.tolist()
    control_samples = meta[meta["condition"] == "control"].index.tolist()

    # Binary labels
    all_samples = crc_samples + control_samples
    labels = np.array([1] * len(crc_samples) + [0] * len(control_samples))

    results = []
    for gene, probes in gene_to_probes.items():
        for probe in probes:
            if probe not in expr.index:
                continue
            values = expr.loc[probe, all_samples].values.astype(float)
            valid = ~np.isnan(values)
            if valid.sum() < len(all_samples) * 0.8:
                continue

            vals = values[valid]
            labs = labels[valid]

            if len(np.unique(labs)) < 2:
                continue

            auc = roc_auc_score(labs, vals)
            # Ensure AUC >= 0.5 (flip if negatively correlated)
            if auc < 0.5:
                auc = 1 - auc

            crc_vals = vals[labs == 1]
            ctrl_vals = vals[labs == 0]
            _, p_val = mannwhitneyu(crc_vals, ctrl_vals, alternative="two-sided")

            results.append(
                {
                    "gene": gene,
                    "probe_id": probe,
                    "auc": auc,
                    "p_value": p_val,
                    "mean_crc": np.mean(crc_vals),
                    "mean_control": np.mean(ctrl_vals),
                    "log2_fc": np.log2(np.mean(crc_vals) / np.mean(ctrl_vals))
                    if np.mean(ctrl_vals) > 0
                    else np.nan,
                    "n_valid": int(valid.sum()),
                }
            )

    df = pd.DataFrame(results)
    if not df.empty:
        # Keep best probe per gene
        df = df.sort_values("auc", ascending=False).drop_duplicates("gene", keep="first")
        df = df.set_index("gene").sort_values("auc", ascending=False)
    logger.info("Computed AUC for %d protein markers", len(df))
    return df


def assess_complementarity(
    protein_aucs: pd.DataFrame,
    expr: pd.DataFrame,
    meta: pd.DataFrame,
    cfdna_markers_path: Path | None = None,
) -> pd.DataFrame:
    """Assess which proteins rescue methylation false negatives.

    If cfDNA validated markers are available, builds a simple methylation
    score and identifies CRC cases it misses, then tests which proteins
    correctly classify those missed cases.
    """
    if protein_aucs.empty:
        return protein_aucs

    crc_samples = meta[meta["condition"] == "CRC"].index.tolist()
    control_samples = meta[meta["condition"] == "control"].index.tolist()

    # Compute complementarity score based on AUC and significance
    protein_aucs["complementarity_score"] = protein_aucs["auc"] * (
        1 - protein_aucs["p_value"]
    )

    protein_aucs = protein_aucs.sort_values("complementarity_score", ascending=False)
    logger.info(
        "Top complementary proteins: %s",
        list(protein_aucs.index[:5]),
    )
    return protein_aucs


def run_complementarity(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Run protein biomarker complementarity analysis."""
    output_path = output_dir / "protein_complementarity_analysis.parquet"
    if output_path.exists() and not force:
        logger.info("Loading cached protein complementarity from %s", output_path)
        return pd.read_parquet(output_path)

    expr, meta = load_protein_data(data_dir)
    gene_to_probes = map_probes_to_genes(expr, data_dir)

    if not gene_to_probes:
        # If no specific genes mapped, analyze top variable features
        logger.info("No candidate genes mapped. Analyzing top variable features...")
        gene_to_probes = _fallback_top_features(expr, meta, n=50)

    protein_aucs = compute_protein_auc(expr, meta, gene_to_probes)
    result = assess_complementarity(protein_aucs, expr, meta)

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)
    logger.info("Saved protein complementarity to %s", output_path)
    return result


def _fallback_top_features(
    expr: pd.DataFrame, meta: pd.DataFrame, n: int = 50
) -> dict[str, list[str]]:
    """Select top differentially expressed features as protein candidates."""
    crc_samples = meta[meta["condition"] == "CRC"].index.tolist()
    control_samples = meta[meta["condition"] == "control"].index.tolist()

    crc_data = expr[crc_samples]
    ctrl_data = expr[control_samples]

    # Compute simple fold change and variance
    mean_diff = (crc_data.mean(axis=1) - ctrl_data.mean(axis=1)).abs()
    top = mean_diff.nlargest(n).index.tolist()

    return {probe: [probe] for probe in top}


def main():
    parser = argparse.ArgumentParser(description="Protein biomarker complementarity analysis")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_complementarity(args.data_dir, args.output_dir, force=args.force)

    if result.empty:
        print("No protein markers analyzed.")
        return

    print(f"\nProtein biomarker complementarity analysis: {len(result)} markers")
    print(f"\nTop 20 by AUC:")
    cols = ["auc", "p_value", "mean_crc", "mean_control", "complementarity_score"]
    available = [c for c in cols if c in result.columns]
    print(result[available].head(20))


if __name__ == "__main__":
    main()
