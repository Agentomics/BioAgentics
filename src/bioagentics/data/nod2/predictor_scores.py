"""Fetch dbNSFP pre-computed functional prediction scores for NOD2 variants.

Uses the MyVariant.info API to retrieve dbNSFP annotations including
CADD, REVEL, PolyPhen-2, SIFT, PhyloP, GERP++, phastCons, and
AlphaMissense scores.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/crohns/nod2-variant-functional-impact")

MYVARIANT_URL = "https://myvariant.info/v1"

# dbNSFP fields to retrieve
DBNSFP_FIELDS = [
    "dbnsfp.cadd.phred",
    "dbnsfp.revel.score",
    "dbnsfp.polyphen2.hdiv.score",
    "dbnsfp.polyphen2.hvar.score",
    "dbnsfp.sift.score",
    "dbnsfp.phylop.100way_vertebrate.score",
    "dbnsfp.gerp",
    "dbnsfp.phastcons.100way_vertebrate.score",
    "dbnsfp.alphamissense.score",
]


def _build_variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Build MyVariant.info-compatible variant ID (HGVS-like).

    Format: chr{chrom}:g.{pos}{ref}>{alt} for SNVs.
    """
    return f"chr{chrom}:g.{pos}{ref}>{alt}"


def _extract_score(data: dict, field_path: str) -> float | None:
    """Extract a nested score from MyVariant.info response.

    Handles cases where the value is a list (takes max) or nested dict.
    """
    parts = field_path.split(".")
    current: Any = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            # Try to get the field from the first element
            values = []
            for item in current:
                if isinstance(item, dict) and part in item:
                    values.append(item[part])
            current = values if values else None
        else:
            return None
        if current is None:
            return None

    # Handle list values (take max for prediction scores)
    if isinstance(current, list):
        numeric = [float(x) for x in current if x is not None]
        return max(numeric) if numeric else None
    if isinstance(current, (int, float)):
        return float(current)
    try:
        return float(current)
    except (TypeError, ValueError):
        return None


def _extract_gerp_score(data: dict) -> float | None:
    """Extract GERP conservation score.

    Falls back to GERP 91-mammals score since GERP++ field name
    has special characters that the MyVariant.info batch API doesn't handle.
    """
    dbnsfp = data.get("dbnsfp")
    if not isinstance(dbnsfp, dict):
        return None
    # Try gerp++ first (available in full responses)
    gerp_pp = dbnsfp.get("gerp++")
    if isinstance(gerp_pp, dict):
        rs = gerp_pp.get("rs")
        if rs is not None:
            try:
                return float(rs)
            except (TypeError, ValueError):
                pass
    # Fallback to gerp 91-mammals
    gerp = dbnsfp.get("gerp")
    if isinstance(gerp, dict):
        mammals = gerp.get("91_mammals")
        if isinstance(mammals, dict):
            score = mammals.get("score")
            if score is not None:
                try:
                    return float(score)
                except (TypeError, ValueError):
                    pass
    return None


def fetch_scores_batch(variant_ids: list[str]) -> list[dict]:
    """Fetch dbNSFP scores for a batch of variant IDs via MyVariant.info POST API.

    Args:
        variant_ids: List of HGVS-formatted variant IDs.

    Returns:
        List of dicts with raw MyVariant.info responses.
    """
    fields = ",".join(DBNSFP_FIELDS)

    resp = requests.post(
        f"{MYVARIANT_URL}/variant",
        data={
            "ids": ",".join(variant_ids),
            "fields": fields,
            "assembly": "hg38",
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_predictor_scores(variants_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch dbNSFP predictor scores for all NOD2 variants.

    Args:
        variants_df: DataFrame with columns chrom, pos, ref, alt
            (from variant collection pipeline).

    Returns:
        DataFrame with columns: chrom, pos, ref, alt, cadd_phred,
        revel, polyphen2_hdiv, polyphen2_hvar, sift, phylop_100way,
        gerp_rs, phastcons_100way, alphamissense.
    """
    # Filter to SNVs (single nucleotide variants) with ref/alt
    snv_mask = (
        variants_df["ref"].str.len() == 1
    ) & (
        variants_df["alt"].str.len() == 1
    ) & (
        variants_df["ref"] != ""
    ) & (
        variants_df["alt"] != ""
    )
    snvs = variants_df[snv_mask].copy()
    logger.info("Querying dbNSFP for %d SNVs (of %d total variants)", len(snvs), len(variants_df))

    # Build variant IDs
    snvs["query_id"] = snvs.apply(
        lambda row: _build_variant_id(str(row["chrom"]), int(row["pos"]), row["ref"], row["alt"]),
        axis=1,
    )

    # Fetch in batches
    batch_size = 200
    all_results: list[dict] = []
    query_ids = snvs["query_id"].tolist()

    for i in range(0, len(query_ids), batch_size):
        batch = query_ids[i:i + batch_size]
        logger.info("Fetching batch %d/%d (%d variants)...", i // batch_size + 1,
                     (len(query_ids) + batch_size - 1) // batch_size, len(batch))
        try:
            results = fetch_scores_batch(batch)
            all_results.extend(results if isinstance(results, list) else [results])
        except requests.RequestException as e:
            logger.warning("Batch request failed: %s", e)
            all_results.extend([{"notfound": True}] * len(batch))
        time.sleep(0.5)  # Rate limit

    # Parse results
    score_records: list[dict] = []
    for idx, (_, row) in enumerate(snvs.iterrows()):
        result = all_results[idx] if idx < len(all_results) else {}
        dbnsfp = result.get("dbnsfp", {}) if isinstance(result, dict) else {}

        record = {
            "chrom": row["chrom"],
            "pos": row["pos"],
            "ref": row["ref"],
            "alt": row["alt"],
            "cadd_phred": _extract_score(result, "dbnsfp.cadd.phred"),
            "revel": _extract_score(result, "dbnsfp.revel.score"),
            "polyphen2_hdiv": _extract_score(result, "dbnsfp.polyphen2.hdiv.score"),
            "polyphen2_hvar": _extract_score(result, "dbnsfp.polyphen2.hvar.score"),
            "sift": _extract_score(result, "dbnsfp.sift.score"),
            "phylop_100way": _extract_score(result, "dbnsfp.phylop.100way_vertebrate.score"),
            "gerp_rs": _extract_gerp_score(result),
            "phastcons_100way": _extract_score(result, "dbnsfp.phastcons.100way_vertebrate.score"),
            "alphamissense": _extract_score(result, "dbnsfp.alphamissense.score"),
        }
        score_records.append(record)

    scores_df = pd.DataFrame(score_records)

    # Report coverage
    score_cols = ["cadd_phred", "revel", "polyphen2_hdiv", "sift", "phylop_100way", "alphamissense"]
    for col in score_cols:
        if col in scores_df.columns:
            n_valid = scores_df[col].notna().sum()
            logger.info("Score coverage - %s: %d/%d (%.1f%%)", col, n_valid, len(scores_df),
                        100 * n_valid / len(scores_df) if len(scores_df) > 0 else 0)

    return scores_df


def collect_predictor_scores(
    variants_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Main pipeline: load variants, fetch scores, save TSV.

    Args:
        variants_path: Path to variants TSV. Defaults to nod2_variants.tsv.
        output_path: Where to save. Defaults to nod2_predictor_scores.tsv.

    Returns:
        DataFrame of predictor scores.
    """
    if variants_path is None:
        variants_path = OUTPUT_DIR / "nod2_variants.tsv"
    if output_path is None:
        output_path = OUTPUT_DIR / "nod2_predictor_scores.tsv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load variants
    variants_df = pd.read_csv(variants_path, sep="\t")
    logger.info("Loaded %d variants from %s", len(variants_df), variants_path)

    # Fetch scores
    scores_df = fetch_predictor_scores(variants_df)

    # Save
    scores_df.to_csv(output_path, sep="\t", index=False)
    logger.info("Saved predictor scores to %s (%d variants)", output_path, len(scores_df))

    return scores_df
