"""Integrate Grotzinger et al. (Nature 2026) 5-factor model into stratified PRS.

Provides data ingestion helpers for the standardized factor GWAS, factor
loadings, and pleiotropic loci outputs, config templates for all five factors
plus the TS residual, and a runner that generates reference PRS weight files
wired to the existing stratified_prs pipeline.

Expected input directory layout (from data_curator tasks #381/#752):
  grotzinger2026/processed/
    factor_loadings.tsv         — trait, factor, loading, se, p, note
    pleiotropic_loci.tsv        — locus_id, chr, bp_start, bp_end, factor, beta, se, p
    factor_gwas/
      sb.tsv                    — Schizophrenia-Bipolar factor GWAS
      internalizing.tsv
      neurodevelopmental.tsv
      compulsive.tsv
      substance_use.tsv
      ts_residual.tsv           — TS-specific residual GWAS
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from bioagentics.config import REPO_ROOT
from bioagentics.pipelines.stratified_prs.pipeline import (
    DEFAULT_CLUMP_KB,
    DEFAULT_CLUMP_R2,
    DEFAULT_P_THRESHOLDS,
    compute_prs_weights,
    load_factor_gwas,
    run_stratified_prs,
)

logger = logging.getLogger(__name__)

# --- Published factor model constants ---

GROTZINGER_FACTORS = [
    "sb",
    "internalizing",
    "neurodevelopmental",
    "compulsive",
    "substance_use",
]

GROTZINGER_FACTOR_LABELS = {
    "sb": "Schizophrenia-Bipolar",
    "internalizing": "Internalizing",
    "neurodevelopmental": "Neurodevelopmental",
    "compulsive": "Compulsive",
    "substance_use": "Substance Use",
}

# TS dual-loads on Neurodevelopmental and Compulsive factors.
# ~87% of TS genetic variance is trait-specific residual.
TS_PRIMARY_FACTORS = ["neurodevelopmental", "compulsive"]
TS_RESIDUAL_VARIANCE_FRACTION = 0.87

# The three PRS strata the project builds
PRS_STRATA = ["compulsive", "neurodevelopmental", "ts_residual"]

# Default output sub-path
DEFAULT_OUTPUT_SUBDIR = (
    "output/tourettes/ts-comorbidity-genetic-architecture/phase3/grotzinger_factors"
)

# Required columns for each input file type
LOADINGS_REQUIRED_COLS = {"trait", "factor", "loading"}
LOCI_REQUIRED_COLS = {"locus_id", "chr", "factor"}


# --- Data classes ---


@dataclass
class FactorLoadingRecord:
    """Single row from the factor loadings table."""

    trait: str
    factor: str
    loading: float
    se: float = 0.0
    p: float = 1.0
    note: str = ""


@dataclass
class GrotzingerDataPackage:
    """Parsed contents of the Grotzinger 2026 standardized data."""

    loadings: pd.DataFrame  # trait, factor, loading, se, p, note
    pleiotropic_loci: pd.DataFrame  # locus_id, chr, bp_start, bp_end, factor, ...
    factor_gwas: dict[str, pd.DataFrame]  # factor_name -> GWAS DataFrame
    ts_residual_gwas: pd.DataFrame | None = None


@dataclass
class FactorPRSConfig:
    """Configuration for building a single factor-stratified PRS."""

    factor_name: str
    factor_label: str
    gwas_filename: str
    p_thresholds: list[float] = field(default_factory=lambda: list(DEFAULT_P_THRESHOLDS))
    clump_r2: float = DEFAULT_CLUMP_R2
    clump_kb: int = DEFAULT_CLUMP_KB
    include_pleiotropic_snps: bool = True
    min_factor_loading: float = 0.0
    metadata: dict[str, str] = field(default_factory=dict)


# --- Data ingestion helpers ---


def load_factor_loadings(path: Path) -> pd.DataFrame:
    """Load the factor loadings table.

    Expected columns: trait, factor, loading[, se, p, note]
    """
    sep = "\t" if path.suffix in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip().str.lower()

    missing = LOADINGS_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Factor loadings file missing columns: {missing}")

    df["loading"] = pd.to_numeric(df["loading"], errors="coerce")
    df = df.dropna(subset=["trait", "factor", "loading"])
    logger.info("Loaded %d factor loading records from %s", len(df), path)
    return df


def load_pleiotropic_loci(path: Path) -> pd.DataFrame:
    """Load the 238 pleiotropic loci manifest with per-factor annotations.

    Expected columns: locus_id, chr, bp_start, bp_end, factor[, beta, se, p]
    """
    sep = "\t" if path.suffix in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = df.columns.str.strip().str.lower()

    missing = LOCI_REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Pleiotropic loci file missing columns: {missing}")

    logger.info(
        "Loaded %d pleiotropic locus-factor records (%d unique loci) from %s",
        len(df),
        df["locus_id"].nunique(),
        path,
    )
    return df


def load_grotzinger_package(processed_dir: Path) -> GrotzingerDataPackage:
    """Load the full Grotzinger 2026 standardized data package.

    Parameters
    ----------
    processed_dir : Path
        Path to grotzinger2026/processed/ directory.

    Returns
    -------
    GrotzingerDataPackage with loadings, loci, and per-factor GWAS DataFrames.
    """
    loadings_path = processed_dir / "factor_loadings.tsv"
    loci_path = processed_dir / "pleiotropic_loci.tsv"
    gwas_dir = processed_dir / "factor_gwas"

    if not loadings_path.exists():
        raise FileNotFoundError(f"Factor loadings not found: {loadings_path}")
    if not loci_path.exists():
        raise FileNotFoundError(f"Pleiotropic loci not found: {loci_path}")
    if not gwas_dir.exists():
        raise FileNotFoundError(f"Factor GWAS directory not found: {gwas_dir}")

    loadings = load_factor_loadings(loadings_path)
    loci = load_pleiotropic_loci(loci_path)

    factor_gwas: dict[str, pd.DataFrame] = {}
    for factor in GROTZINGER_FACTORS:
        gwas_path = gwas_dir / f"{factor}.tsv"
        if gwas_path.exists():
            factor_gwas[factor] = load_factor_gwas(gwas_path)
            logger.info("Loaded %s factor GWAS: %d SNPs", factor, len(factor_gwas[factor]))
        else:
            logger.warning("No GWAS file for factor %s at %s", factor, gwas_path)

    ts_residual = None
    ts_res_path = gwas_dir / "ts_residual.tsv"
    if ts_res_path.exists():
        ts_residual = load_factor_gwas(ts_res_path)
        logger.info("Loaded TS residual GWAS: %d SNPs", len(ts_residual))

    return GrotzingerDataPackage(
        loadings=loadings,
        pleiotropic_loci=loci,
        factor_gwas=factor_gwas,
        ts_residual_gwas=ts_residual,
    )


# --- SNP filtering by pleiotropic loci and factor loading ---


def filter_snps_by_pleiotropic_loci(
    gwas_df: pd.DataFrame,
    loci_df: pd.DataFrame,
    factor: str,
) -> pd.DataFrame:
    """Flag or filter GWAS SNPs that fall within pleiotropic loci for a factor.

    Adds a 'pleiotropic' boolean column. Requires CHR and BP in gwas_df.
    """
    if "CHR" not in gwas_df.columns or "BP" not in gwas_df.columns:
        logger.warning("GWAS lacks CHR/BP — cannot filter by pleiotropic loci")
        gwas_df = gwas_df.copy()
        gwas_df["pleiotropic"] = False
        return gwas_df

    factor_loci = loci_df[loci_df["factor"].str.lower() == factor.lower()]
    if len(factor_loci) == 0:
        gwas_df = gwas_df.copy()
        gwas_df["pleiotropic"] = False
        return gwas_df

    gwas_df = gwas_df.copy()
    gwas_df["pleiotropic"] = False

    for _, locus in factor_loci.iterrows():
        chr_match = gwas_df["CHR"] == locus["chr"]
        bp_start = locus.get("bp_start", 0)
        bp_end = locus.get("bp_end", float("inf"))
        in_locus = chr_match & (gwas_df["BP"] >= bp_start) & (gwas_df["BP"] <= bp_end)
        gwas_df.loc[in_locus, "pleiotropic"] = True

    n_pleio = gwas_df["pleiotropic"].sum()
    logger.info(
        "Factor %s: %d/%d SNPs in pleiotropic loci",
        factor,
        n_pleio,
        len(gwas_df),
    )
    return gwas_df


def get_ts_factor_loading(loadings_df: pd.DataFrame, factor: str) -> float:
    """Get TS loading on a specific factor from the loadings table."""
    mask = (loadings_df["trait"].str.upper() == "TS") & (
        loadings_df["factor"].str.lower() == factor.lower()
    )
    rows = loadings_df[mask]
    if len(rows) == 0:
        return 0.0
    return float(rows.iloc[0]["loading"])


# --- Config template generation ---


def build_factor_prs_configs(
    loadings_df: pd.DataFrame | None = None,
    p_thresholds: list[float] | None = None,
    clump_r2: float = DEFAULT_CLUMP_R2,
    clump_kb: int = DEFAULT_CLUMP_KB,
) -> list[FactorPRSConfig]:
    """Build PRS config templates for the 5 Grotzinger factors + TS residual.

    If loadings_df is provided, the TS loading on each factor is recorded in
    the metadata so downstream code can weight or filter by it.
    """
    configs = []

    for factor in GROTZINGER_FACTORS:
        meta = {
            "source": "Grotzinger et al. Nature 649:406-415, 2026",
            "doi": "10.1038/s41586-025-09820-3",
            "factor_label": GROTZINGER_FACTOR_LABELS[factor],
            "is_ts_primary": str(factor in TS_PRIMARY_FACTORS),
        }

        if loadings_df is not None:
            ts_loading = get_ts_factor_loading(loadings_df, factor)
            meta["ts_loading"] = f"{ts_loading:.4f}"

        configs.append(
            FactorPRSConfig(
                factor_name=factor,
                factor_label=GROTZINGER_FACTOR_LABELS[factor],
                gwas_filename=f"{factor}.tsv",
                p_thresholds=p_thresholds or list(DEFAULT_P_THRESHOLDS),
                clump_r2=clump_r2,
                clump_kb=clump_kb,
                include_pleiotropic_snps=True,
                metadata=meta,
            )
        )

    # TS residual component
    configs.append(
        FactorPRSConfig(
            factor_name="ts_residual",
            factor_label="TS-specific residual",
            gwas_filename="ts_residual.tsv",
            p_thresholds=p_thresholds or list(DEFAULT_P_THRESHOLDS),
            clump_r2=clump_r2,
            clump_kb=clump_kb,
            include_pleiotropic_snps=False,
            metadata={
                "source": "Grotzinger et al. Nature 649:406-415, 2026",
                "doi": "10.1038/s41586-025-09820-3",
                "factor_label": "TS-specific residual (~87% unique variance)",
                "residual_variance_fraction": str(TS_RESIDUAL_VARIANCE_FRACTION),
            },
        )
    )

    return configs


def write_config_templates(
    configs: list[FactorPRSConfig],
    output_dir: Path,
) -> Path:
    """Serialize config templates to JSON for reproducibility.

    Returns the path to the written config file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "prs_factor_configs.json"

    serializable = []
    for cfg in configs:
        d = asdict(cfg)
        serializable.append(d)

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Wrote %d factor PRS configs to %s", len(configs), out_path)
    return out_path


# --- Reference PRS weight generation ---


def generate_reference_weights(
    gwas_dir: Path,
    configs: list[FactorPRSConfig],
    loci_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    ld_matrix: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Generate reference PRS weight files for each factor config.

    For each config, loads the factor GWAS, optionally annotates pleiotropic
    status, runs C+T, and writes weight TSV files with metadata headers.

    Returns dict mapping factor_name -> directory of weight files.
    """
    if output_dir is None:
        output_dir = REPO_ROOT / DEFAULT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    result_dirs: dict[str, Path] = {}

    for cfg in configs:
        gwas_path = gwas_dir / cfg.gwas_filename
        if not gwas_path.exists():
            logger.warning("Skipping %s — GWAS file not found: %s", cfg.factor_name, gwas_path)
            continue

        gwas_df = load_factor_gwas(gwas_path)

        # Annotate pleiotropic status if loci data available
        if loci_df is not None and cfg.include_pleiotropic_snps:
            gwas_df = filter_snps_by_pleiotropic_loci(gwas_df, loci_df, cfg.factor_name)

        # Compute C+T weights
        weights_list = compute_prs_weights(
            gwas_df,
            stratum=cfg.factor_name,
            p_thresholds=cfg.p_thresholds,
            ld_matrix=ld_matrix,
            clump_r2=cfg.clump_r2,
            clump_kb=cfg.clump_kb,
        )

        factor_dir = output_dir / cfg.factor_name
        factor_dir.mkdir(parents=True, exist_ok=True)

        for w in weights_list:
            wt_df = pd.DataFrame(
                {
                    "SNP": w.snp_ids,
                    "A1": w.effect_alleles,
                    "WEIGHT": w.weights,
                }
            )
            # Add pleiotropic column if available
            if "pleiotropic" in gwas_df.columns:
                pleio_map = dict(zip(gwas_df["SNP"], gwas_df["pleiotropic"]))
                wt_df["PLEIOTROPIC"] = wt_df["SNP"].map(pleio_map).fillna(False)

            safe_pt = f"{w.p_threshold:.0e}".replace("+", "")
            fname = f"weights_p{safe_pt}.tsv"
            wt_df.to_csv(factor_dir / fname, sep="\t", index=False, float_format="%.6f")

        # Write factor metadata
        meta = {**cfg.metadata, "n_thresholds": len(weights_list)}
        if weights_list:
            meta["max_snps"] = max(w.n_snps for w in weights_list)
        with open(factor_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        result_dirs[cfg.factor_name] = factor_dir
        logger.info(
            "Generated %d weight files for %s (%s)",
            len(weights_list),
            cfg.factor_name,
            cfg.factor_label,
        )

    return result_dirs


# --- End-to-end runner ---


def run_grotzinger_prs(
    processed_dir: Path | None = None,
    gwas_dir: Path | None = None,
    target_genotypes: Path | None = None,
    output_dir: Path | None = None,
    factors: list[str] | None = None,
    p_thresholds: list[float] | None = None,
    clump_r2: float = DEFAULT_CLUMP_R2,
    clump_kb: int = DEFAULT_CLUMP_KB,
) -> dict[str, Path]:
    """Run the full Grotzinger factor PRS integration.

    This is the main entry point. It can work in two modes:

    1. Full mode (processed_dir provided): loads factor loadings, pleiotropic
       loci, and factor GWAS from the standardized data package.
    2. GWAS-only mode (gwas_dir provided): loads factor GWAS files directly,
       without loadings or loci annotation.

    Parameters
    ----------
    processed_dir : Path | None
        Path to grotzinger2026/processed/ from data_curator.
    gwas_dir : Path | None
        Direct path to directory of factor GWAS files.
    target_genotypes : Path | None
        Target genotype file for scoring (optional).
    output_dir : Path | None
        Output directory. Defaults to output/tourettes/.../grotzinger_factors/
    factors : list[str] | None
        Subset of factors to process. Default: compulsive, neurodevelopmental,
        ts_residual (the three primary TS strata).
    p_thresholds : list[float] | None
        P-value thresholds for C+T.
    clump_r2, clump_kb
        LD clumping parameters.

    Returns
    -------
    Dict mapping factor_name -> output directory with weight files.
    """
    if output_dir is None:
        output_dir = REPO_ROOT / DEFAULT_OUTPUT_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if factors is None:
        factors = list(PRS_STRATA)

    loadings_df = None
    loci_df = None

    if processed_dir is not None and processed_dir.exists():
        pkg = load_grotzinger_package(processed_dir)
        loadings_df = pkg.loadings
        loci_df = pkg.pleiotropic_loci
        effective_gwas_dir = processed_dir / "factor_gwas"
    elif gwas_dir is not None and gwas_dir.exists():
        effective_gwas_dir = gwas_dir
    else:
        raise FileNotFoundError(
            "Must provide either processed_dir or gwas_dir with factor GWAS files"
        )

    # Build configs (filtered to requested factors)
    all_configs = build_factor_prs_configs(
        loadings_df=loadings_df,
        p_thresholds=p_thresholds,
        clump_r2=clump_r2,
        clump_kb=clump_kb,
    )
    configs = [c for c in all_configs if c.factor_name in factors]

    # Write config templates
    write_config_templates(all_configs, output_dir)

    # Generate reference weight files
    result_dirs = generate_reference_weights(
        gwas_dir=effective_gwas_dir,
        configs=configs,
        loci_df=loci_df,
        output_dir=output_dir,
    )

    # If target genotypes provided, also run the standard stratified PRS
    # comparison using the factor GWAS files remapped to the expected names
    if target_genotypes is not None and target_genotypes.exists():
        _run_scoring(effective_gwas_dir, target_genotypes, output_dir, factors, p_thresholds)

    # Write README
    _write_readme(output_dir, factors, processed_dir, gwas_dir)

    logger.info("Grotzinger factor PRS complete — output at %s", output_dir)
    return result_dirs


def _run_scoring(
    gwas_dir: Path,
    target_genotypes: Path,
    output_dir: Path,
    factors: list[str],
    p_thresholds: list[float] | None,
) -> None:
    """Run scoring via the standard stratified PRS pipeline.

    Creates a temporary GWAS directory with the expected filenames that
    run_stratified_prs expects (compulsive.tsv, neurodevelopmental.tsv, etc.).
    """
    import shutil
    import tempfile

    name_map = {
        "compulsive": "compulsive.tsv",
        "neurodevelopmental": "neurodevelopmental.tsv",
        "ts_residual": "ts_specific.tsv",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        for factor in factors:
            src_name = f"{factor}.tsv"
            dst_name = name_map.get(factor, src_name)
            src = gwas_dir / src_name
            if src.exists():
                shutil.copy2(src, tmp / dst_name)

        scoring_dir = output_dir / "scoring"
        run_stratified_prs(
            factor_gwas_dir=tmp,
            target_genotypes=target_genotypes,
            output_dir=scoring_dir,
            p_thresholds=p_thresholds,
        )


def _write_readme(
    output_dir: Path,
    factors: list[str],
    processed_dir: Path | None,
    gwas_dir: Path | None,
) -> None:
    """Write a README describing the output directory."""
    lines = [
        "# Grotzinger Factor PRS — Output",
        "",
        "Reference: Grotzinger et al. Nature 649:406-415, 2026",
        "DOI: 10.1038/s41586-025-09820-3",
        "",
        "## Factors processed",
        "",
    ]
    for f in factors:
        label = GROTZINGER_FACTOR_LABELS.get(f, f)
        lines.append(f"- **{f}**: {label}")
    lines.append("- **ts_residual**: TS-specific residual (~87% unique variance)")

    lines += [
        "",
        "## Directory layout",
        "",
        "```",
        "<factor_name>/",
        "  weights_p<threshold>.tsv   — SNP weights at each P-value threshold",
        "  metadata.json              — Factor metadata and provenance",
        "prs_factor_configs.json      — Full config templates for all factors",
        "scoring/                     — Stratified PRS scoring results (if genotypes provided)",
        "README.md                    — This file",
        "```",
        "",
        "## Input sources",
        "",
    ]
    if processed_dir:
        lines.append(f"- Standardized data: `{processed_dir}`")
    if gwas_dir:
        lines.append(f"- Factor GWAS directory: `{gwas_dir}`")

    lines += [
        "",
        "## Swapping in real genotype data",
        "",
        "To use UK Biobank target genotypes instead of synthetic data:",
        "1. Convert BGEN/PLINK to dosage TSV (rows=individuals, cols=SNP IDs)",
        "2. Place the file alongside a phenotype.tsv with IID and PHENO columns",
        "3. Re-run with --target-genotypes pointing to the dosage file",
        "",
    ]

    readme_path = output_dir / "README.md"
    readme_path.write_text("\n".join(lines))
    logger.info("Wrote README to %s", readme_path)


# --- Synthetic data generation for validation ---


def generate_synthetic_factor_gwas(
    n_snps: int = 500,
    n_sample: int = 50000,
    h2: float = 0.3,
    seed: int = 42,
    with_position: bool = True,
) -> pd.DataFrame:
    """Generate synthetic factor GWAS summary statistics for testing."""
    rng = np.random.default_rng(seed)

    snp_ids = [f"rs{i}" for i in range(1, n_snps + 1)]
    z = rng.normal(0, np.sqrt(1 + h2 * 2), size=n_snps)
    se = 1.0 / np.sqrt(n_sample)
    beta = z * se
    p = 2 * sp_stats.norm.sf(np.abs(z))

    df = pd.DataFrame(
        {
            "SNP": snp_ids,
            "A1": rng.choice(["A", "C", "G", "T"], size=n_snps),
            "A2": rng.choice(["A", "C", "G", "T"], size=n_snps),
            "BETA": beta,
            "SE": np.full(n_snps, se),
            "P": p,
            "N": np.full(n_snps, n_sample, dtype=int),
        }
    )

    if with_position:
        df["CHR"] = rng.integers(1, 23, size=n_snps)
        df["BP"] = rng.integers(1, 250_000_000, size=n_snps)

    return df


def generate_synthetic_loadings() -> pd.DataFrame:
    """Generate synthetic factor loadings table mimicking published values."""
    records = [
        ("TS", "compulsive", 0.35, 0.04),
        ("TS", "neurodevelopmental", 0.28, 0.05),
        ("TS", "sb", 0.08, 0.03),
        ("TS", "internalizing", 0.12, 0.04),
        ("TS", "substance_use", 0.05, 0.03),
        ("OCD", "compulsive", 0.72, 0.03),
        ("ADHD", "neurodevelopmental", 0.65, 0.03),
        ("ASD", "neurodevelopmental", 0.48, 0.04),
        ("SCZ", "sb", 0.85, 0.02),
        ("BIP", "sb", 0.68, 0.03),
        ("MDD", "internalizing", 0.78, 0.02),
        ("ANX", "internalizing", 0.62, 0.03),
        ("AUD", "substance_use", 0.71, 0.03),
    ]
    rows = []
    for trait, factor, loading, se in records:
        p = 2 * sp_stats.norm.sf(abs(loading / se)) if se > 0 else 1.0
        rows.append(
            {"trait": trait, "factor": factor, "loading": loading, "se": se, "p": p, "note": ""}
        )
    return pd.DataFrame(rows)


def generate_synthetic_pleiotropic_loci(n_loci: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic pleiotropic loci manifest."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(1, n_loci + 1):
        chrom = rng.integers(1, 23)
        bp_start = rng.integers(1_000_000, 240_000_000)
        bp_end = bp_start + rng.integers(50_000, 500_000)
        factor = rng.choice(GROTZINGER_FACTORS)
        beta = rng.normal(0, 0.05)
        se = abs(rng.normal(0.01, 0.003))
        p_val = 2 * sp_stats.norm.sf(abs(beta / se)) if se > 0 else 1.0
        rows.append(
            {
                "locus_id": f"LOC{i:04d}",
                "chr": int(chrom),
                "bp_start": int(bp_start),
                "bp_end": int(bp_end),
                "factor": factor,
                "beta": beta,
                "se": se,
                "p": p_val,
            }
        )
    return pd.DataFrame(rows)


def create_synthetic_data_package(output_dir: Path, seed: int = 42) -> Path:
    """Create a complete synthetic Grotzinger data package for testing.

    Returns the path to the processed/ directory.
    """
    processed = output_dir / "grotzinger2026" / "processed"
    gwas_dir = processed / "factor_gwas"
    gwas_dir.mkdir(parents=True, exist_ok=True)

    # Factor loadings
    loadings = generate_synthetic_loadings()
    loadings.to_csv(processed / "factor_loadings.tsv", sep="\t", index=False)

    # Pleiotropic loci
    loci = generate_synthetic_pleiotropic_loci(seed=seed)
    loci.to_csv(processed / "pleiotropic_loci.tsv", sep="\t", index=False)

    # Factor GWAS files
    for i, factor in enumerate(GROTZINGER_FACTORS):
        gwas = generate_synthetic_factor_gwas(seed=seed + i, with_position=True)
        gwas.to_csv(gwas_dir / f"{factor}.tsv", sep="\t", index=False)

    # TS residual GWAS
    ts_res = generate_synthetic_factor_gwas(seed=seed + 100, with_position=True)
    ts_res.to_csv(gwas_dir / "ts_residual.tsv", sep="\t", index=False)

    logger.info("Created synthetic Grotzinger data package at %s", processed)
    return processed
