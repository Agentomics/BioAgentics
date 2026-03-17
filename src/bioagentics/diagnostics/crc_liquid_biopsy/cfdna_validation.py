"""cfDNA signal validation for CRC liquid biopsy methylation markers.

Cross-references tissue-derived methylation signatures (from TCGA discovery)
against cfDNA methylation data (GSE149282) to identify markers detectable
in blood. Filters out markers significant in tissue but undetectable in cfDNA.

Output:
    output/diagnostics/crc-liquid-biopsy-panel/cfdna_validated_markers.parquet

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.cfdna_validation [--force]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel"
OUTPUT_DIR = REPO_ROOT / "output" / "diagnostics" / "crc-liquid-biopsy-panel"


def load_tissue_signatures(output_dir: Path) -> pd.DataFrame:
    """Load tissue-derived methylation signatures from discovery step."""
    path = output_dir / "methylation_signatures.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Tissue signatures not found: {path}. Run methylation_discovery first."
        )
    sigs = pd.read_parquet(path)
    logger.info("Loaded %d tissue methylation signatures", len(sigs))
    return sigs


def load_cfdna_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load GSE149282 cfDNA methylation data and metadata."""
    meth_path = data_dir / "gse149282_cfdna_methylation.parquet"
    meta_path = data_dir / "gse149282_metadata.parquet"

    if not meth_path.exists():
        raise FileNotFoundError(f"cfDNA methylation not found: {meth_path}")

    meth = pd.read_parquet(meth_path)
    meta = pd.read_parquet(meta_path)
    logger.info("Loaded cfDNA methylation: %d probes x %d samples", *meth.shape)
    logger.info("cfDNA conditions: %s", meta["condition"].value_counts().to_dict())
    return meth, meta


def validate_markers_in_cfdna(
    tissue_sigs: pd.DataFrame,
    cfdna_meth: pd.DataFrame,
    cfdna_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Validate tissue signatures against cfDNA data.

    For each tissue-derived CpG marker, compute discriminative power in cfDNA
    (CRC vs control). Returns markers with both tissue and blood effect sizes.
    """
    # Find overlapping CpG probes
    common_cpgs = tissue_sigs.index.intersection(cfdna_meth.index)
    logger.info(
        "Overlapping CpGs between tissue signatures and cfDNA: %d / %d",
        len(common_cpgs),
        len(tissue_sigs),
    )

    if len(common_cpgs) == 0:
        logger.warning("No overlapping CpGs found. Check probe ID formats.")
        return pd.DataFrame()

    # Split cfDNA samples by condition
    crc_samples = cfdna_meta[cfdna_meta["condition"] == "CRC"].index.tolist()
    control_samples = cfdna_meta[cfdna_meta["condition"] == "control"].index.tolist()
    logger.info("cfDNA: %d CRC samples, %d controls", len(crc_samples), len(control_samples))

    if len(crc_samples) < 2 or len(control_samples) < 2:
        raise ValueError("Insufficient cfDNA samples for validation")

    # Compute cfDNA effect sizes and p-values for overlapping markers
    results = []
    for cpg in common_cpgs:
        crc_vals = cfdna_meth.loc[cpg, crc_samples].dropna().values.astype(float)
        ctrl_vals = cfdna_meth.loc[cpg, control_samples].dropna().values.astype(float)

        if len(crc_vals) < 2 or len(ctrl_vals) < 2:
            continue

        cfdna_mean_crc = np.mean(crc_vals)
        cfdna_mean_ctrl = np.mean(ctrl_vals)
        cfdna_delta = cfdna_mean_crc - cfdna_mean_ctrl

        _, p_val = mannwhitneyu(crc_vals, ctrl_vals, alternative="two-sided")

        results.append(
            {
                "cpg_id": cpg,
                "tissue_delta_beta": tissue_sigs.loc[cpg, "delta_beta"],
                "tissue_mean_tumor": tissue_sigs.loc[cpg, "mean_tumor"],
                "tissue_mean_normal": tissue_sigs.loc[cpg, "mean_normal"],
                "tissue_q_value": tissue_sigs.loc[cpg, "q_value"],
                "cfdna_delta": cfdna_delta,
                "cfdna_mean_crc": cfdna_mean_crc,
                "cfdna_mean_control": cfdna_mean_ctrl,
                "cfdna_p_value": p_val,
                "direction_concordant": (
                    (tissue_sigs.loc[cpg, "delta_beta"] > 0 and cfdna_delta > 0)
                    or (tissue_sigs.loc[cpg, "delta_beta"] < 0 and cfdna_delta < 0)
                ),
                "gene_name": tissue_sigs.loc[cpg].get("gene_name"),
                "is_crc_gene": tissue_sigs.loc[cpg].get("is_crc_gene", False),
            }
        )

    validated = pd.DataFrame(results).set_index("cpg_id")

    # FDR correction for cfDNA p-values
    valid_p = validated["cfdna_p_value"].dropna()
    if len(valid_p) > 0:
        ranked = valid_p.rank()
        n_tests = len(valid_p)
        validated.loc[valid_p.index, "cfdna_q_value"] = valid_p * n_tests / ranked
        validated["cfdna_q_value"] = validated["cfdna_q_value"].clip(upper=1.0)

    # Compute combined score: tissue effect * cfDNA effect (concordant direction)
    validated["combined_score"] = (
        validated["tissue_delta_beta"].abs() * validated["cfdna_delta"].abs()
    )
    validated.loc[~validated["direction_concordant"], "combined_score"] = 0

    validated = validated.sort_values("combined_score", ascending=False)
    logger.info("Validated %d markers in cfDNA", len(validated))

    return validated


def filter_cfdna_validated(
    validated: pd.DataFrame,
    cfdna_p_threshold: float = 0.1,
    require_concordant: bool = True,
) -> pd.DataFrame:
    """Filter for markers with significant cfDNA signal."""
    mask = validated["cfdna_p_value"] <= cfdna_p_threshold
    if require_concordant:
        mask = mask & validated["direction_concordant"]

    filtered = validated[mask].copy()
    logger.info(
        "cfDNA validated markers (p < %.2f, concordant=%s): %d / %d",
        cfdna_p_threshold,
        require_concordant,
        len(filtered),
        len(validated),
    )
    return filtered


def run_validation(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Run full cfDNA validation pipeline."""
    output_path = output_dir / "cfdna_validated_markers.parquet"
    if output_path.exists() and not force:
        logger.info("Loading cached cfDNA validated markers from %s", output_path)
        return pd.read_parquet(output_path)

    tissue_sigs = load_tissue_signatures(output_dir)
    cfdna_meth, cfdna_meta = load_cfdna_data(data_dir)

    validated = validate_markers_in_cfdna(tissue_sigs, cfdna_meth, cfdna_meta)

    if validated.empty:
        logger.warning("No markers could be validated")
        output_dir.mkdir(parents=True, exist_ok=True)
        validated.to_parquet(output_path)
        return validated

    # Filter for reliable cfDNA markers (use relaxed threshold given small n)
    filtered = filter_cfdna_validated(validated, cfdna_p_threshold=0.1)

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered.to_parquet(output_path)

    # Also save full validation results
    validated.to_parquet(output_dir / "cfdna_validation_full.parquet")

    logger.info("Saved %d validated markers to %s", len(filtered), output_path)
    return filtered


def main():
    parser = argparse.ArgumentParser(description="cfDNA validation of methylation markers")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    validated = run_validation(args.data_dir, args.output_dir, force=args.force)

    if validated.empty:
        print("No validated markers found.")
        return

    print(f"\ncfDNA validated markers: {len(validated)}")
    print(f"  Direction concordant: {validated['direction_concordant'].sum()}")
    crc = validated[validated["is_crc_gene"]]
    if len(crc) > 0:
        print(f"  In known CRC genes: {len(crc)}")
        print("\nTop CRC gene markers:")
        print(crc[["tissue_delta_beta", "cfdna_delta", "cfdna_p_value", "gene_name"]].head(20))

    print(f"\nTop 20 markers by combined score:")
    print(
        validated[["tissue_delta_beta", "cfdna_delta", "cfdna_p_value", "gene_name", "combined_score"]].head(20)
    )


if __name__ == "__main__":
    main()
