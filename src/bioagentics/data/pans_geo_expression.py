"""Download and preprocess GEO dataset GSE102482 (LPS-treated mouse microglia).

GSE102482 provides transcriptomic data from mouse microglia treated with LPS,
modeling neuroinflammatory conditions relevant to PANS. This module downloads
the dataset, normalizes expression, computes differential expression (LPS vs
control), and maps mouse genes to human orthologs for cross-referencing with
PANS variant genes.

Usage:
    uv run python -m bioagentics.data.pans_geo_expression [--dest DIR] [--force]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/pandas_pans/pans-genetic-variant-pathway-analysis")
PROCESSED_FILE = "gse102482_processed.csv"
DE_FILE = "gse102482_de_results.csv"

# Mouse-to-human ortholog mapping for PANS variant genes.
# Mouse gene symbols are typically the same but lowercase-initial (e.g., Trex1).
# This covers all 22 PANS variant genes plus common aliases.
MOUSE_TO_HUMAN: dict[str, str] = {
    "Cux1": "CUX1",
    "Usp45": "USP45",
    "Parp14": "PARP14",
    "Uvssa": "UVSSA",
    "Ep300": "EP300",
    "Trex1": "TREX1",
    "Samhd1": "SAMHD1",
    "Stk19": "STK19",
    "Pidd1": "PIDD1",
    "Fancd2": "FANCD2",
    "Rad54l": "RAD54L",
    "Prkn": "PRKN",
    "Park2": "PRKN",  # alias
    "Polg": "POLG",
    "Lgals4": "LGALS4",
    "Duox2": "DUOX2",
    "Ccr9": "CCR9",
    "Mbl2": "MBL2",
    "Mbl1": "MBL2",  # mouse Mbl1 maps to human MBL2
    "Masp1": "MASP1",
    "Masp2": "MASP2",
    "Myt1l": "MYT1L",
    "Tep1": "TEP1",
    "Adnp": "ADNP",
}

# Broader mouse-to-human mapping generated on first download by uppercasing
# mouse symbols (works for ~95% of orthologs with 1:1 mapping).
# The explicit dict above handles known exceptions.


def _uppercase_symbol_map(mouse_symbol: str) -> str:
    """Map mouse gene symbol to human by uppercasing (default ortholog heuristic)."""
    if mouse_symbol in MOUSE_TO_HUMAN:
        return MOUSE_TO_HUMAN[mouse_symbol]
    return mouse_symbol.upper()


def _build_probe_to_gene_map(gse: object, platform_name: str | None) -> dict[str, str]:
    """Build probe ID to gene symbol mapping from GPL platform annotation.

    Checks ``Gene Symbol``, ``gene_assignment``, and ``SPOT_ID.1`` columns
    in the GPL annotation table.  Returns ``{probe_id: gene_symbol}`` for
    every probe that can be resolved.
    """
    if platform_name is None or platform_name not in gse.gpls:
        return {}

    annot = gse.gpls[platform_name].table
    if annot is None or annot.empty:
        return {}

    probe_col = "ID" if "ID" in annot.columns else annot.columns[0]

    # --- Strategy 1: direct Gene Symbol column ---
    for candidate in ("Gene Symbol", "gene_symbol", "GENE_SYMBOL", "Symbol"):
        if candidate in annot.columns:
            mapping: dict[str, str] = {}
            for _, row in annot.iterrows():
                probe = str(row[probe_col]).strip()
                raw = str(row.get(candidate, "")).strip()
                if raw and raw not in ("---", "nan", ""):
                    symbol = raw.split("///")[0].strip()
                    if symbol:
                        mapping[probe] = symbol
            if mapping:
                return mapping

    # --- Strategy 2: gene_assignment (Affymetrix format) ---
    #  Typical format: "accession // Symbol // description // loc // geneID"
    if "gene_assignment" in annot.columns:
        mapping = {}
        for _, row in annot.iterrows():
            probe = str(row[probe_col]).strip()
            assignment = str(row.get("gene_assignment", "")).strip()
            if not assignment or assignment in ("---", "nan"):
                continue
            # Take the first assignment (before ///)
            first_assign = assignment.split("///")[0]
            parts = [p.strip() for p in first_assign.split("//")]
            if len(parts) >= 2 and parts[1] and parts[1] != "---":
                mapping[probe] = parts[1].strip()
        if mapping:
            return mapping

    # --- Strategy 3: SPOT_ID.1 ---
    if "SPOT_ID.1" in annot.columns:
        mapping = {}
        for _, row in annot.iterrows():
            probe = str(row[probe_col]).strip()
            raw = str(row.get("SPOT_ID.1", "")).strip()
            if raw and raw not in ("---", "nan", ""):
                mapping[probe] = raw.split("///")[0].strip()
        if mapping:
            return mapping

    return {}


def download_gse102482(dest_dir: Path,
                       force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download and parse GSE102482 using GEOparse.

    Returns (expression_df, metadata_df). Expression has gene symbols as index,
    sample IDs as columns.
    """
    import GEOparse

    processed_path = dest_dir / PROCESSED_FILE
    meta_path = dest_dir / "gse102482_sample_metadata.csv"
    if not force and processed_path.exists() and meta_path.exists():
        logger.info("Loading cached processed data from %s", processed_path)
        return pd.read_csv(processed_path, index_col=0), pd.read_csv(meta_path, index_col=0)

    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading GSE102482 from GEO...")
    gse = GEOparse.get_GEO(geo="GSE102482", destdir=str(dest_dir), silent=True)

    # Get the platform (typically GPL or similar)
    platform_name = list(gse.gpls.keys())[0] if gse.gpls else None
    logger.info("Platform: %s", platform_name)

    # Build probe-to-gene mapping from GPL annotation table
    probe_to_gene = _build_probe_to_gene_map(gse, platform_name)
    if probe_to_gene:
        logger.info("Built probe-to-gene mapping: %d probes mapped to gene symbols",
                     len(probe_to_gene))
    else:
        logger.warning("No probe-to-gene mapping available from GPL annotations")

    # Extract expression data from the first (usually only) GPL
    # GSMs contain the per-sample data
    samples: dict[str, pd.Series] = {}
    sample_metadata: dict[str, dict] = {}

    for gsm_name, gsm in gse.gsms.items():
        table = gsm.table
        if table.empty:
            logger.warning("Empty table for %s, skipping", gsm_name)
            continue

        # Get expression values keyed by probe/gene ID
        if "GENE_SYMBOL" in table.columns:
            id_col = "GENE_SYMBOL"
        elif "Gene Symbol" in table.columns:
            id_col = "Gene Symbol"
        elif "ID_REF" in table.columns:
            id_col = "ID_REF"
        else:
            id_col = table.columns[0]

        val_col = "VALUE" if "VALUE" in table.columns else table.columns[1]

        series = table.set_index(id_col)[val_col]
        series = pd.to_numeric(series, errors="coerce")
        samples[gsm_name] = series

        # Extract metadata
        meta = {
            "title": gsm.metadata.get("title", [""])[0],
            "source": gsm.metadata.get("source_name_ch1", [""])[0],
            "characteristics": gsm.metadata.get("characteristics_ch1", []),
        }
        sample_metadata[gsm_name] = meta

    if not samples:
        raise ValueError("No sample data extracted from GSE102482")

    expr_df = pd.DataFrame(samples)
    expr_df.index.name = "gene_symbol"

    # Map probe IDs to gene symbols using GPL annotation
    if probe_to_gene:
        mapped = [probe_to_gene.get(str(pid).strip()) for pid in expr_df.index]
        keep = [s is not None for s in mapped]
        n_mapped = sum(keep)
        n_total = len(expr_df)
        logger.info("Probe mapping: %d/%d probes resolved to gene symbols",
                     n_mapped, n_total)
        expr_df = expr_df[keep].copy()
        expr_df.index = pd.Index([m for m in mapped if m is not None],
                                 name="gene_symbol")

    # Drop rows with all NaN
    expr_df = expr_df.dropna(how="all")

    # For duplicate gene symbols, keep the one with highest mean expression
    expr_df["_mean"] = expr_df.mean(axis=1)
    expr_df = expr_df.sort_values("_mean", ascending=False)
    expr_df = expr_df[~expr_df.index.duplicated(keep="first")]
    expr_df = expr_df.drop(columns=["_mean"])

    logger.info("Expression matrix: %d genes x %d samples", *expr_df.shape)

    # Save metadata alongside
    meta_df = pd.DataFrame.from_dict(sample_metadata, orient="index")
    meta_df.to_csv(dest_dir / "gse102482_sample_metadata.csv")

    return expr_df, meta_df


def classify_samples(meta_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Classify samples into LPS-treated and control groups.

    Scans sample titles and characteristics for LPS/control indicators.
    Returns (lps_samples, control_samples).
    """
    lps_samples = []
    control_samples = []

    for sample_id, row in meta_df.iterrows():
        title = str(row.get("title", "")).lower()
        source = str(row.get("source", "")).lower()
        chars = row.get("characteristics", [])
        if isinstance(chars, str):
            chars = [chars]
        all_text = f"{title} {source} {' '.join(str(c) for c in chars)}".lower()

        if "lps" in all_text:
            lps_samples.append(sample_id)
        elif any(kw in all_text for kw in ["control", "untreated", "vehicle", "pbs", "baseline"]):
            control_samples.append(sample_id)

    logger.info("Sample classification: %d LPS, %d control",
                len(lps_samples), len(control_samples))
    return lps_samples, control_samples


def normalize_expression(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Log2 transform and quantile normalize expression data."""
    # Log2 transform (add 1 to avoid log(0))
    df = expr_df.copy()
    min_val = df[df > 0].min().min()
    offset = min_val / 2 if min_val > 0 else 1.0
    df = np.log2(df + offset)

    # Quantile normalization
    rank_mean = df.stack().groupby(df.rank(method="first").stack().astype(int)).mean()
    df_norm = df.rank(method="min").stack().astype(int).map(rank_mean).unstack()

    return df_norm


def compute_de(expr_df: pd.DataFrame, lps_samples: list[str],
               control_samples: list[str]) -> pd.DataFrame:
    """Compute differential expression between LPS and control samples.

    Uses Welch's t-test with BH FDR correction.
    Returns DataFrame with columns: gene_symbol, log2fc, pvalue, padj.
    """
    lps_data = expr_df[lps_samples]
    ctrl_data = expr_df[control_samples]

    results = []
    for gene in expr_df.index:
        lps_vals = lps_data.loc[gene].dropna().values
        ctrl_vals = ctrl_data.loc[gene].dropna().values

        if len(lps_vals) < 2 or len(ctrl_vals) < 2:
            continue

        log2fc = lps_vals.mean() - ctrl_vals.mean()
        t_stat, pval = stats.ttest_ind(lps_vals, ctrl_vals, equal_var=False)

        if np.isnan(pval):
            continue

        results.append({
            "gene_symbol": gene,
            "log2fc": log2fc,
            "pvalue": pval,
        })

    de_df = pd.DataFrame(results)

    if de_df.empty:
        de_df["padj"] = []
        return de_df

    # BH FDR correction
    de_df = de_df.sort_values("pvalue")
    n = len(de_df)
    de_df["padj"] = de_df["pvalue"] * n / (np.arange(1, n + 1))
    de_df["padj"] = de_df["padj"].clip(upper=1.0)
    # Ensure monotonicity
    de_df["padj"] = de_df["padj"].iloc[::-1].cummin().iloc[::-1]

    return de_df.sort_values("pvalue").reset_index(drop=True)


def get_neuroinflammation_de(dest_dir: Path | None = None,
                             force: bool = False) -> pd.DataFrame:
    """Return differential expression results with human gene symbol mapping.

    Columns: gene_symbol (human), mouse_symbol, log2fc, pvalue, padj.
    """
    if dest_dir is None:
        dest_dir = OUTPUT_DIR

    de_path = dest_dir / DE_FILE
    if not force and de_path.exists():
        return pd.read_csv(de_path)

    # Run full pipeline
    expr_df, meta_df = download_gse102482(dest_dir, force=force)
    lps_samples, control_samples = classify_samples(meta_df)

    if not lps_samples or not control_samples:
        raise ValueError(
            f"Could not classify samples: {len(lps_samples)} LPS, "
            f"{len(control_samples)} control. Check dataset metadata."
        )

    expr_norm = normalize_expression(expr_df)
    de_df = compute_de(expr_norm, lps_samples, control_samples)

    # Add human ortholog mapping
    de_df["mouse_symbol"] = de_df["gene_symbol"]
    de_df["gene_symbol"] = de_df["mouse_symbol"].apply(_uppercase_symbol_map)

    # Save
    dest_dir.mkdir(parents=True, exist_ok=True)
    expr_df.to_csv(dest_dir / PROCESSED_FILE)
    de_df.to_csv(de_path, index=False)
    logger.info("Saved DE results: %d genes, %d significant (FDR<0.05)",
                len(de_df), (de_df["padj"] < 0.05).sum())

    return de_df


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and preprocess GSE102482 (LPS-treated microglia)"
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download and reprocess even if files exist")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    de_df = get_neuroinflammation_de(dest_dir=args.dest, force=args.force)
    print(f"\nDE results: {len(de_df)} genes")
    print(f"Significant (FDR < 0.05): {(de_df['padj'] < 0.05).sum()}")
    print(f"Up in LPS: {((de_df['padj'] < 0.05) & (de_df['log2fc'] > 0)).sum()}")
    print(f"Down in LPS: {((de_df['padj'] < 0.05) & (de_df['log2fc'] < 0)).sum()}")


if __name__ == "__main__":
    main()
