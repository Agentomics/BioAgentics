"""Derive bulk tissue fibrosis gene signature from GEO microarray datasets.

Differential expression between stricturing (B2) and inflammatory (B1) Crohn's
disease using GSE16879 and GSE57945 series matrix data.

Pipeline:
1. Download GEO series matrix + platform annotation
2. Parse expression data and map probes to gene symbols
3. Classify samples by disease phenotype
4. Per-gene Mann-Whitney U test with BH-FDR correction
5. Filter for fibrosis-relevant pathways (MSigDB)
6. Output ranked signature as TSV

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.bulk_signature
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.cd_fibrosis.geo import (
    DEFAULT_DEST as GEO_DEST,
    download_series_matrix,
    download_soft,
    parse_series_matrix_samples,
)
from bioagentics.data.cd_fibrosis.msigdb import (
    DEFAULT_DEST as MSIGDB_DEST,
    filter_fibrosis_sets,
    get_combined_fibrosis_genes,
    parse_gmt,
)

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing" / "signatures"
GPL_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/platforms"

# Known fibrosis genes from literature — always included in signature
KNOWN_FIBROSIS_GENES = {
    # IBD Journal 2025 (doi:10.1093/ibd/izae295) fibrostenotic transcriptomic
    "HDAC1", "GREM1", "SERPINE1", "LY96", "AKAP11", "SRM", "EHD2", "FGF2",
    # JCI 2024 (doi:10.1172/JCI179472) TWIST1+FAP+ fibroblasts
    "TWIST1", "FAP",
    # JCI Insight 2026 protein-validated markers
    "CTHRC1", "POSTN", "TNC", "CPA3",
    # YAP/TAZ mechanotransduction (Cell 2025)
    "YAP1", "WWTR1", "CTGF", "CYR61",
    # CD38/PECAM1 axis (J Crohn's Colitis 2025)
    "CD38", "PECAM1",
    # Core fibrosis markers
    "COL1A1", "COL1A2", "COL3A1", "ACTA2", "TGFB1", "TGFBR1", "TGFBR2",
    "SMAD2", "SMAD3", "VIM", "FN1",
}


# ── Expression matrix parsing ──


def parse_expression_matrix(matrix_path: Path) -> pd.DataFrame:
    """Parse expression data from a GEO series matrix file.

    Returns DataFrame: probe IDs as index, sample IDs as columns.
    """
    rows: list[tuple[str, list[float]]] = []
    header: list[str] | None = None
    in_table = False

    opener = gzip.open if str(matrix_path).endswith(".gz") else open
    with opener(matrix_path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!series_matrix_table_begin"):
                in_table = True
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if not in_table:
                continue

            parts = line.split("\t")
            if header is None:
                header = [s.strip('"') for s in parts]
                continue

            probe_id = parts[0].strip('"')
            values: list[float] = []
            for v in parts[1:]:
                v = v.strip('" ')
                if v in ("", "null", "NA"):
                    values.append(np.nan)
                else:
                    values.append(float(v))
            rows.append((probe_id, values))

    if not rows or header is None:
        raise ValueError(f"No expression data in {matrix_path}")

    sample_ids = header[1:]
    data = np.array([r[1] for r in rows], dtype=np.float64)
    df = pd.DataFrame(data, index=[r[0] for r in rows], columns=sample_ids)
    df.index.name = "probe_id"
    return df


def auto_log2(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Log2-transform expression data if in linear scale.

    Affymetrix MAS5 data has values ~100-10000 (linear).
    RMA-normalized data has values ~2-15 (already log2).
    RNA-seq RPKM/FPKM can be 0-10000+ (linear, but median near 0 due to zeros).

    Uses both median and 99th percentile to detect linear scale — RPKM data
    often has median ~0 (many zero-expression genes) but max >> 20.
    """
    vals = expr_df.values[~np.isnan(expr_df.values)]
    median_val = float(np.median(vals)) if len(vals) > 0 else 0
    pct99 = float(np.percentile(vals, 99)) if len(vals) > 0 else 0
    if median_val > 25 or pct99 > 100:
        # Linear scale — log2(x + 1) transform
        return np.log2(expr_df + 1)
    return expr_df


def parse_rpkm_matrix(rpkm_path: Path) -> pd.DataFrame:
    """Parse a GEO supplementary RPKM/FPKM expression file.

    Handles GSE57945 format: Gene ID | Gene Symbol | CCFA_Risk_001 | ...
    Returns DataFrame with gene symbols as index, sample IDs as columns.
    """
    df = pd.read_csv(rpkm_path, sep="\t", compression="infer", low_memory=False)
    # Detect gene symbol column and use it as index
    cols = list(df.columns)
    gene_col = None
    for c in cols:
        if c.lower() in ("gene symbol", "gene_symbol", "symbol"):
            gene_col = c
            break

    if gene_col:
        # Drop rows with missing gene symbols, set as index
        df = df.dropna(subset=[gene_col])
        df = df[df[gene_col].astype(str).str.strip() != ""]
        df = df.set_index(gene_col)
        df.index = df.index.astype(str).str.upper()
    else:
        df = df.set_index(df.columns[0])
        df.index = df.index.astype(str)

    df.index.name = "gene"
    # Keep only numeric columns (expression values)
    df = df.select_dtypes(include=[np.number])
    # For duplicate gene symbols, keep the one with highest mean
    if df.index.duplicated().any():
        df["_mean"] = df.mean(axis=1)
        df = df.loc[df.groupby(level=0)["_mean"].idxmax()]
        df = df.drop(columns=["_mean"])
    return df


def parse_soft_sample_titles(soft_path: Path) -> dict[str, str]:
    """Parse SOFT file to get GSM ID -> sample title mapping.

    Also extracts embedded CCFA_Risk IDs from titles like
    "CD Female ... (CCFA_Risk_001)" -> gsm_to_risk_id mapping.
    """
    gsm_to_title: dict[str, str] = {}
    current_gsm = ""

    opener = gzip.open if str(soft_path).endswith(".gz") else open
    with opener(soft_path, "rt", errors="replace") as f:
        for line in f:
            if line.startswith("^SAMPLE = "):
                current_gsm = line.strip().split("= ")[1]
            elif line.startswith("!Sample_title = ") and current_gsm:
                gsm_to_title[current_gsm] = line.strip().split("= ", 1)[1]

    return gsm_to_title


def build_gsm_to_rpkm_map(
    soft_path: Path, rpkm_columns: list[str]
) -> dict[str, str]:
    """Build mapping from GSM IDs to RPKM column names.

    Parses SOFT titles for embedded IDs like "(CCFA_Risk_001)".
    Returns: gsm_id -> rpkm_column_name.
    """
    import re

    titles = parse_soft_sample_titles(soft_path)
    rpkm_set = set(rpkm_columns)
    mapping: dict[str, str] = {}

    for gsm, title in titles.items():
        # Look for embedded IDs in parentheses
        match = re.search(r"\(([^)]+)\)", title)
        if match:
            embedded_id = match.group(1)
            if embedded_id in rpkm_set:
                mapping[gsm] = embedded_id
                continue
        # Try matching the title directly to columns
        if title in rpkm_set:
            mapping[gsm] = title

    return mapping


def parse_platform_id(matrix_path: Path) -> str:
    """Extract platform ID (e.g. GPL570) from series matrix metadata."""
    opener = gzip.open if str(matrix_path).endswith(".gz") else open
    with opener(matrix_path, "rt") as f:
        for line in f:
            if line.startswith("!Series_platform_id") or line.startswith("!Sample_platform_id"):
                return line.split("\t")[1].strip().strip('"')
            if line.startswith("!series_matrix_table_begin"):
                break
    return "GPL570"  # default for CD microarray datasets


# ── Platform annotation ──


def download_gpl_annotation(gpl: str, dest_dir: Path) -> Path:
    """Download GPL platform annotation from GEO FTP."""
    numeric = gpl.replace("GPL", "")
    dir_prefix = numeric[:-3] if len(numeric) > 3 else ""
    url = f"{GPL_BASE}/GPL{dir_prefix}nnn/{gpl}/annot/{gpl}.annot.gz"

    dest_file = dest_dir / f"{gpl}.annot.gz"
    if dest_file.exists():
        return dest_file

    print(f"  Downloading {gpl} annotation...")
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest_file, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
    return dest_file


def parse_gpl_annotation(annot_path: Path) -> dict[str, str]:
    """Parse GPL annotation: probe_id -> gene_symbol.

    For probes with multiple gene symbols ("TP53 /// MDM2"), takes the first.
    """
    probe_to_gene: dict[str, str] = {}
    opener = gzip.open if str(annot_path).endswith(".gz") else open

    with opener(annot_path, "rt", errors="replace") as f:
        header_cols: list[str] | None = None
        gene_col_idx: int | None = None

        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")

            if header_cols is None:
                header_cols = [c.strip().lower() for c in parts]
                for i, col in enumerate(header_cols):
                    if col in ("gene symbol", "gene_symbol", "symbol"):
                        gene_col_idx = i
                        break
                if gene_col_idx is None:
                    raise ValueError(f"No Gene Symbol column in {annot_path}")
                continue

            if len(parts) <= gene_col_idx:
                continue

            probe_id = parts[0].strip()
            gene = parts[gene_col_idx].strip()
            if not gene or gene == "---":
                continue
            if " /// " in gene:
                gene = gene.split(" /// ")[0]
            probe_to_gene[probe_id] = gene.upper()

    return probe_to_gene


def parse_soft_platform_table(soft_path: Path) -> dict[str, str]:
    """Fallback: extract probe-to-gene mapping from SOFT file platform table."""
    probe_to_gene: dict[str, str] = {}
    in_platform = False
    header: list[str] | None = None
    gene_col_idx: int | None = None

    opener = gzip.open if str(soft_path).endswith(".gz") else open
    with opener(soft_path, "rt", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("!platform_table_begin"):
                in_platform = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if not in_platform:
                continue

            parts = line.split("\t")
            if header is None:
                header = [h.strip().lower() for h in parts]
                for i, col in enumerate(header):
                    if col in ("gene symbol", "gene_symbol", "symbol", "gene_assignment"):
                        gene_col_idx = i
                        break
                continue

            if gene_col_idx is None or len(parts) <= gene_col_idx:
                continue
            probe_id = parts[0].strip()
            gene_str = parts[gene_col_idx].strip()
            if not gene_str or gene_str == "---":
                continue
            gene = gene_str.split(" /// ")[0].split(" // ")[0].strip()
            if gene:
                probe_to_gene[probe_id] = gene.upper()

    return probe_to_gene


def get_probe_to_gene(gpl: str, geo_dir: Path, soft_path: Path) -> dict[str, str]:
    """Get probe-to-gene mapping, trying annotation file then SOFT fallback."""
    # Try standalone annotation
    try:
        annot_path = download_gpl_annotation(gpl, geo_dir)
        mapping = parse_gpl_annotation(annot_path)
        if mapping:
            print(f"  {gpl} annotation: {len(mapping)} probes mapped")
            return mapping
    except Exception as e:
        print(f"  {gpl} annotation unavailable: {e}")

    # Fallback to SOFT file
    try:
        mapping = parse_soft_platform_table(soft_path)
        print(f"  SOFT fallback: {len(mapping)} probes mapped")
        return mapping
    except Exception as e:
        print(f"  SOFT parsing failed: {e}", file=sys.stderr)
        return {}


def probes_to_genes(
    expr_df: pd.DataFrame, probe_to_gene: dict[str, str]
) -> pd.DataFrame:
    """Collapse probe-level expression to gene symbols.

    For multiple probes per gene, keeps the probe with highest mean expression.
    """
    genes = expr_df.index.map(lambda p: probe_to_gene.get(p, ""))
    expr = expr_df.copy()
    expr["_gene"] = genes
    expr = expr[expr["_gene"] != ""]

    expr["_mean"] = expr.drop(columns=["_gene"]).mean(axis=1)
    idx = expr.groupby("_gene")["_mean"].idxmax()
    result = expr.loc[idx].drop(columns=["_gene", "_mean"])
    result.index = expr.loc[idx, "_gene"].values
    result.index.name = "gene"
    return result


# ── Sample classification ──


def classify_samples_gse16879(
    metadata: dict[str, dict],
) -> tuple[list[str], list[str], str]:
    """Classify GSE16879 samples: stricturing (B2) vs inflammatory (B1).

    Falls back to CD-before-treatment vs normal control if B1/B2 unavailable.
    Returns: (fibrosis_group, control_group, comparison_label)
    """
    b2, b1 = [], []
    cd_before, controls = [], []

    for sid, meta in metadata.items():
        chars = " ".join(meta.get("characteristics", [])).lower()
        title = meta.get("title", "").lower()
        source = meta.get("source", "").lower()
        text = f"{chars} {title} {source}"

        if any(k in text for k in ("stricturing", "b2", "stenosing")):
            b2.append(sid)
        elif any(k in text for k in (" b1", "non-stricturing", "inflammatory")):
            b1.append(sid)

        if any(k in text for k in ("normal", "control", "healthy")):
            controls.append(sid)
        elif any(k in text for k in ("crohn", "cd ")) and "before" in text:
            cd_before.append(sid)

    if len(b2) >= 3 and len(b1) >= 3:
        print(f"  GSE16879: {len(b2)} stricturing (B2) vs {len(b1)} inflammatory (B1)")
        return b2, b1, "B2_vs_B1"

    if len(cd_before) >= 3 and len(controls) >= 3:
        print(f"  GSE16879 fallback: {len(cd_before)} CD pre-treatment vs {len(controls)} normal")
        return cd_before, controls, "CD_preTx_vs_Normal"

    # Broader CD vs control
    all_cd = [sid for sid, m in metadata.items()
              if any(k in " ".join(m.get("characteristics", [])).lower() + " "
                     + m.get("title", "").lower()
                     for k in ("crohn", "cd "))]
    if len(all_cd) >= 3 and len(controls) >= 3:
        print(f"  GSE16879 fallback: {len(all_cd)} CD vs {len(controls)} normal")
        return all_cd, controls, "CD_vs_Normal"

    raise ValueError(
        f"GSE16879: insufficient samples — B2={len(b2)}, B1={len(b1)}, "
        f"CD_before={len(cd_before)}, Control={len(controls)}"
    )


def classify_samples_gse57945(
    metadata: dict[str, dict],
) -> tuple[list[str], list[str], str]:
    """Classify GSE57945 (RISK cohort): progressors vs non-progressors.

    Falls back to CD vs control.
    Returns: (fibrosis_group, control_group, comparison_label)
    """
    progressors, non_progressors = [], []
    cd, controls = [], []

    for sid, meta in metadata.items():
        chars = " ".join(meta.get("characteristics", [])).lower()
        title = meta.get("title", "").lower()
        source = meta.get("source", "").lower()
        text = f"{chars} {title} {source}"

        if any(k in text for k in ("strictur", "b2", "stenosing")):
            progressors.append(sid)
        elif "non-progressor" in text or "non_progressor" in text:
            non_progressors.append(sid)
        elif any(k in text for k in ("b1", "inflammatory", "non-complicat")):
            non_progressors.append(sid)

        if any(k in text for k in ("crohn", " cd ", "ileal cd")):
            cd.append(sid)
        elif any(k in text for k in ("normal", "control", "healthy", "not ibd")):
            controls.append(sid)

    if len(progressors) >= 3 and len(non_progressors) >= 3:
        print(f"  GSE57945: {len(progressors)} progressors vs {len(non_progressors)} non-progressors")
        return progressors, non_progressors, "Progressor_vs_NonProgressor"

    if len(cd) >= 3 and len(controls) >= 3:
        print(f"  GSE57945 fallback: {len(cd)} CD vs {len(controls)} normal")
        return cd, controls, "CD_vs_Normal"

    raise ValueError(
        f"GSE57945: insufficient samples — Progressors={len(progressors)}, "
        f"NonProgressors={len(non_progressors)}, CD={len(cd)}, Control={len(controls)}"
    )


# ── Differential expression ──


def fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])

    valid = ~np.isnan(pvalues)
    fdr = np.full(n, np.nan)
    if valid.sum() == 0:
        return fdr

    valid_p = pvalues[valid]
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    m = len(ranked_p)

    adjusted = ranked_p / (np.arange(1, m + 1) / m)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    result = np.empty(m)
    result[order] = adjusted
    fdr[valid] = result
    return fdr


def differential_expression(
    expr_df: pd.DataFrame,
    group_a_ids: list[str],
    group_b_ids: list[str],
) -> pd.DataFrame:
    """Per-gene Mann-Whitney U with FDR correction.

    group_a: fibrosis/disease group, group_b: control/reference group.
    log2FC = mean_a - mean_b (data already log2-scale from microarray normalization).
    """
    avail_a = [s for s in group_a_ids if s in expr_df.columns]
    avail_b = [s for s in group_b_ids if s in expr_df.columns]
    if len(avail_a) < 3 or len(avail_b) < 3:
        raise ValueError(f"Too few samples: group_a={len(avail_a)}, group_b={len(avail_b)}")

    a_vals = expr_df[avail_a].values
    b_vals = expr_df[avail_b].values
    genes = expr_df.index.tolist()

    records: list[dict] = []
    for i in range(len(genes)):
        va = a_vals[i][~np.isnan(a_vals[i])]
        vb = b_vals[i][~np.isnan(b_vals[i])]

        if len(va) < 2 or len(vb) < 2:
            records.append({"gene": genes[i], "log2fc": np.nan, "pvalue": np.nan,
                            "mean_a": np.nan, "mean_b": np.nan,
                            "n_a": len(va), "n_b": len(vb)})
            continue

        mean_a, mean_b = float(np.mean(va)), float(np.mean(vb))
        try:
            _, pval = stats.mannwhitneyu(va, vb, alternative="two-sided")
        except ValueError:
            pval = np.nan

        records.append({"gene": genes[i], "log2fc": mean_a - mean_b,
                        "pvalue": pval, "mean_a": mean_a, "mean_b": mean_b,
                        "n_a": len(va), "n_b": len(vb)})

    de = pd.DataFrame(records)
    de["fdr"] = fdr_correction(de["pvalue"].values)
    return de.sort_values("pvalue")


def meta_combine_de(
    results: list[tuple[pd.DataFrame, str, str]],
) -> pd.DataFrame:
    """Combine DE results from multiple datasets using Fisher's method.

    Input: list of (de_df, dataset_name, comparison_label).
    """
    if len(results) == 1:
        df, name, comp = results[0]
        out = df.rename(columns={"log2fc": f"log2fc_{name}", "pvalue": f"pvalue_{name}",
                                 "fdr": f"fdr_{name}"})
        out["meta_pvalue"] = out[f"pvalue_{name}"]
        out["mean_log2fc"] = out[f"log2fc_{name}"]
        out["meta_fdr"] = out[f"fdr_{name}"]
        out["datasets"] = name
        out["comparison"] = comp
        return out

    merged: pd.DataFrame = pd.DataFrame()
    for df, name, _ in results:
        sub = df[["gene", "log2fc", "pvalue"]].rename(
            columns={"log2fc": f"log2fc_{name}", "pvalue": f"pvalue_{name}"}
        )
        if merged.empty:
            merged = sub
        else:
            merged = merged.merge(sub, on="gene", how="outer")

    pval_cols = [f"pvalue_{n}" for _, n, _ in results]
    fc_cols = [f"log2fc_{n}" for _, n, _ in results]

    def _fisher(row: pd.Series) -> float:
        ps = [float(row[c]) for c in pval_cols if pd.notna(row[c]) and float(row[c]) > 0]
        if not ps:
            return float("nan")
        if len(ps) == 1:
            return ps[0]
        chi2_val = -2 * sum(np.log(p) for p in ps)
        return float(stats.chi2.sf(chi2_val, 2 * len(ps)))

    merged["meta_pvalue"] = merged.apply(_fisher, axis=1)
    merged["mean_log2fc"] = merged[fc_cols].mean(axis=1)
    merged["meta_fdr"] = fdr_correction(merged["meta_pvalue"].to_numpy())
    merged["datasets"] = ",".join(n for _, n, _ in results)
    merged["comparison"] = ",".join(c for _, _, c in results)
    return merged.sort_values("meta_pvalue")


# ── Pathway annotation ──


def annotate_pathways(de_df: pd.DataFrame, msigdb_dir: Path) -> pd.DataFrame:
    """Add fibrosis pathway annotations to each gene."""
    pathway_map: dict[str, list[str]] = {}
    for gmt_file in msigdb_dir.glob("*.gmt"):
        if gmt_file.name == "fibrosis_relevant_sets.gmt":
            continue
        for set_name, genes in filter_fibrosis_sets(parse_gmt(gmt_file)).items():
            for g in genes:
                pathway_map.setdefault(g.upper(), []).append(set_name)

    de_df["pathways"] = de_df["gene"].map(
        lambda g: "; ".join(sorted(set(pathway_map.get(g, []))))
    )
    de_df["n_pathways"] = de_df["gene"].map(lambda g: len(set(pathway_map.get(g, []))))
    return de_df


# ── Main pipeline ──


def derive_bulk_signature(
    geo_dir: Path | None = None,
    msigdb_dir: Path | None = None,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Derive bulk tissue fibrosis gene signature end-to-end."""
    geo_dir = geo_dir or GEO_DEST
    msigdb_dir = msigdb_dir or MSIGDB_DEST
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Bulk Tissue Fibrosis Signature Derivation")
    print("=" * 60)

    de_results: list[tuple[pd.DataFrame, str, str]] = []
    classifiers = {
        "GSE16879": classify_samples_gse16879,
        "GSE57945": classify_samples_gse57945,
    }

    for gse in ("GSE16879", "GSE57945"):
        print(f"\n--- {gse} ---")
        gse_dir = geo_dir / gse
        gse_dir.mkdir(parents=True, exist_ok=True)

        # Download
        try:
            matrix_path = download_series_matrix(gse, gse_dir)
            soft_path = download_soft(gse, gse_dir)
        except Exception as e:
            print(f"  Download failed: {e}", file=sys.stderr)
            continue

        # Parse expression — try series matrix first, then supplementary RPKM
        print(f"  Parsing expression...")
        gene_expr: pd.DataFrame | None = None

        try:
            expr_df = parse_expression_matrix(matrix_path)
            print(f"  Series matrix: {expr_df.shape[0]} probes x {expr_df.shape[1]} samples")

            # Platform annotation and probe-to-gene mapping
            gpl = parse_platform_id(matrix_path)
            print(f"  Platform: {gpl}")
            probe_to_gene = get_probe_to_gene(gpl, geo_dir, soft_path)
            if not probe_to_gene:
                print(f"  No probe mapping, skipping")
                continue
            gene_expr = probes_to_genes(expr_df, probe_to_gene)
        except ValueError:
            # No expression in series matrix — try supplementary RPKM file
            print(f"  No expression in series matrix, checking supplementary...")
            rpkm_candidates = list(gse_dir.glob("**/GSE*RPKM*")) + list(
                gse_dir.glob("supplementary/*RPKM*")
            )
            if not rpkm_candidates:
                # Download supplementary files
                from bioagentics.data.cd_fibrosis.geo import download_supplementary
                try:
                    suppl_files = download_supplementary(gse, gse_dir)
                    rpkm_candidates = [f for f in suppl_files if "RPKM" in f.name.upper()]
                except Exception as e:
                    print(f"  Supplementary download failed: {e}", file=sys.stderr)

            if rpkm_candidates:
                rpkm_path = rpkm_candidates[0]
                print(f"  Parsing RPKM from {rpkm_path.name}...")
                try:
                    gene_expr = parse_rpkm_matrix(rpkm_path)
                    print(f"  RPKM matrix: {gene_expr.shape[0]} genes x {gene_expr.shape[1]} samples")
                except Exception as e:
                    print(f"  RPKM parse failed: {e}", file=sys.stderr)

        if gene_expr is None or gene_expr.empty:
            print(f"  No expression data available, skipping")
            continue

        # Log2 transform if needed (MAS5 or RPKM data)
        gene_expr = auto_log2(gene_expr)
        print(f"  {gene_expr.shape[0]} genes x {gene_expr.shape[1]} samples (log2-scale)")

        # Classify samples — handle GSM vs RPKM column ID mismatch
        metadata = parse_series_matrix_samples(matrix_path)
        expr_columns = set(gene_expr.columns)
        gsm_ids = set(metadata.keys())

        # If expression columns don't match GSM IDs, build a mapping
        gsm_to_expr_col: dict[str, str] = {}
        if not gsm_ids & expr_columns:
            print(f"  Sample IDs differ from expression columns, building mapping...")
            gsm_to_expr_col = build_gsm_to_rpkm_map(soft_path, list(gene_expr.columns))
            print(f"  Mapped {len(gsm_to_expr_col)} of {len(gsm_ids)} samples")

        try:
            group_a_gsm, group_b_gsm, comparison = classifiers[gse](metadata)
        except ValueError as e:
            print(f"  Classification failed: {e}", file=sys.stderr)
            continue

        # Translate GSM IDs to expression column names if needed
        if gsm_to_expr_col:
            group_a = [gsm_to_expr_col[g] for g in group_a_gsm if g in gsm_to_expr_col]
            group_b = [gsm_to_expr_col[g] for g in group_b_gsm if g in gsm_to_expr_col]
            print(f"  Mapped groups: {len(group_a)} vs {len(group_b)}")
        else:
            group_a, group_b = group_a_gsm, group_b_gsm

        # DE analysis
        print(f"  Running DE ({comparison})...")
        try:
            de = differential_expression(gene_expr, group_a, group_b)
            sig = (de["fdr"] < 0.05).sum()
            print(f"  {len(de)} genes, {sig} significant (FDR < 0.05)")
            de_results.append((de, gse, comparison))
        except ValueError as e:
            print(f"  DE failed: {e}", file=sys.stderr)

    if not de_results:
        raise RuntimeError("No DE results from any dataset")

    # Combine datasets
    print(f"\n--- Combining {len(de_results)} dataset(s) ---")
    combined = meta_combine_de(de_results)
    print(f"  {len(combined)} genes combined")

    # Pathway annotation
    print(f"  Annotating pathways...")
    try:
        combined = annotate_pathways(combined, msigdb_dir)
    except Exception as e:
        print(f"  Pathway annotation failed: {e}")
        combined["pathways"] = ""
        combined["n_pathways"] = 0

    # Fibrosis gene filter
    try:
        fibrosis_genes = get_combined_fibrosis_genes(msigdb_dir)
        print(f"  MSigDB fibrosis genes: {len(fibrosis_genes)}")
    except Exception:
        fibrosis_genes = set()

    all_fibrosis = fibrosis_genes | KNOWN_FIBROSIS_GENES
    combined["in_fibrosis_pathway"] = combined["gene"].isin(all_fibrosis)
    combined["is_known_target"] = combined["gene"].isin(KNOWN_FIBROSIS_GENES)

    # Build signature: significant fibrosis genes + known targets
    combined["in_signature"] = (
        ((combined["meta_fdr"] < 0.05) & combined["in_fibrosis_pathway"])
        | combined["is_known_target"]
    )
    combined["direction"] = np.where(combined["mean_log2fc"] > 0, "up", "down")
    combined = combined.sort_values(["in_signature", "meta_pvalue"], ascending=[False, True])

    # Save full results
    out_path = output_dir / "bulk_tissue_signature.tsv"
    combined.to_csv(out_path, sep="\t", index=False)
    print(f"\n  Full results: {out_path}")

    # Save signature gene list (for CMAP querying)
    sig_df = combined[combined["in_signature"]].copy()
    up_genes = sig_df[sig_df["direction"] == "up"]["gene"].tolist()
    down_genes = sig_df[sig_df["direction"] == "down"]["gene"].tolist()

    sig_path = output_dir / "bulk_tissue_signature_genes.tsv"
    sig_cols = ["gene", "direction", "mean_log2fc", "meta_pvalue", "meta_fdr",
                "pathways", "is_known_target"]
    sig_df[[c for c in sig_cols if c in sig_df.columns]].to_csv(
        sig_path, sep="\t", index=False
    )
    print(f"  Signature genes: {sig_path}")
    print(f"  {len(up_genes)} UP + {len(down_genes)} DOWN = {len(sig_df)} total")

    # Summary
    print(f"\n--- Summary ---")
    print(f"  Genes tested: {len(combined)}")
    print(f"  In fibrosis pathway: {combined['in_fibrosis_pathway'].sum()}")
    print(f"  Known targets found: {combined['is_known_target'].sum()}")

    print(f"\n  Top 15 upregulated fibrosis genes:")
    for _, r in sig_df[sig_df["direction"] == "up"].head(15).iterrows():
        m = "*" if r.get("is_known_target") else " "
        print(f"    {m} {r['gene']:12s} log2FC={r['mean_log2fc']:+.3f}  p={r['meta_pvalue']:.2e}")

    print(f"\n  Top 15 downregulated fibrosis genes:")
    for _, r in sig_df[sig_df["direction"] == "down"].head(15).iterrows():
        m = "*" if r.get("is_known_target") else " "
        print(f"    {m} {r['gene']:12s} log2FC={r['mean_log2fc']:+.3f}  p={r['meta_pvalue']:.2e}")

    return combined


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Derive bulk tissue fibrosis gene signature from GEO data"
    )
    parser.add_argument("--geo-dir", type=Path, default=GEO_DEST)
    parser.add_argument("--msigdb-dir", type=Path, default=MSIGDB_DEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    derive_bulk_signature(args.geo_dir, args.msigdb_dir, args.output_dir)


if __name__ == "__main__":
    main()
