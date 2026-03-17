"""Construct GOF/neutral/LOF functional spectrum training set for NOD2 variants.

Combines curated functional labels from multiple sources:
- LOF: ClinVar pathogenic CD-associated variants, SURF LOF variants
- GOF: Blau syndrome NACHT domain variants, SURF GOF variants
- Neutral: ClinVar benign variants, common gnomAD variants

Critical insight: R702W and G908R show GOF in autoinflammation but LOF
in microbial sensing — context-dependent effects.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/crohns/nod2-variant-functional-impact")

# Curated LOF variants (ClinVar pathogenic CD-associated)
LOF_VARIANTS = [
    {"hgvs_p": "R702W", "source": "ClinVar", "disease_context": "Crohn's disease",
     "nfkb_activity": "reduced_MDP_sensing", "notes": "Also GOF in autoinflammation context"},
    {"hgvs_p": "G908R", "source": "ClinVar", "disease_context": "Crohn's disease",
     "nfkb_activity": "reduced_MDP_sensing", "notes": "Also GOF in autoinflammation context"},
    {"hgvs_p": "L1007fs", "source": "ClinVar", "disease_context": "Crohn's disease",
     "nfkb_activity": "abolished", "notes": "Frameshift deletes girdin binding domain"},
    # SURF LOF variants (Front Immunol 2025, PMID 41357180)
    {"hgvs_p": "L682F", "source": "SURF", "disease_context": "Pediatric SURF",
     "nfkb_activity": "reduced", "notes": "NACHT domain LOF — discriminating example"},
    {"hgvs_p": "R587C", "source": "SURF", "disease_context": "Pediatric SURF",
     "nfkb_activity": "reduced", "notes": ""},
]

# Curated GOF variants (Blau syndrome + SURF enhanced)
GOF_VARIANTS = [
    # Blau syndrome variants (Current Rheumatology Reports 2026)
    {"hgvs_p": "R334Q", "source": "Blau_syndrome", "disease_context": "Blau syndrome",
     "nfkb_activity": "constitutive_activation", "notes": "NACHT domain GOF"},
    {"hgvs_p": "R334W", "source": "Blau_syndrome", "disease_context": "Blau syndrome",
     "nfkb_activity": "constitutive_activation", "notes": "NACHT domain GOF"},
    {"hgvs_p": "L469F", "source": "Blau_syndrome", "disease_context": "Blau syndrome",
     "nfkb_activity": "constitutive_activation", "notes": "NACHT domain GOF"},
    # Early-onset sarcoidosis GOF (overlaps with Blau)
    {"hgvs_p": "R334Q", "source": "EOS", "disease_context": "Early-onset sarcoidosis",
     "nfkb_activity": "constitutive_activation", "notes": "Same variant as Blau"},
    # Additional NACHT GOF from literature
    {"hgvs_p": "E383K", "source": "Literature", "disease_context": "Blau syndrome",
     "nfkb_activity": "enhanced", "notes": "Walker B motif adjacent"},
    {"hgvs_p": "R587C", "source": "Literature", "disease_context": "Blau syndrome",
     "nfkb_activity": "enhanced", "notes": "Reported in some Blau families"},
    # SURF GOF — context-dependent
    {"hgvs_p": "P268S", "source": "SURF", "disease_context": "Pediatric SURF",
     "nfkb_activity": "moderate_enhanced", "notes": "Compound het with V955I showed GOF"},
]

# GNOMAD_AF threshold for "common benign" variants
COMMON_AF_THRESHOLD = 0.01


def build_training_set(
    variants_path: Path | None = None,
) -> pd.DataFrame:
    """Construct the labeled training dataset.

    Returns DataFrame with columns: variant_id, hgvs_p, functional_class,
    source, disease_context, nfkb_activity.
    """
    if variants_path is None:
        variants_path = OUTPUT_DIR / "nod2_variants.tsv"

    variants_df = pd.read_csv(variants_path, sep="\t")
    logger.info("Loaded %d total variants", len(variants_df))

    records: list[dict] = []

    # Add LOF variants
    for lof in LOF_VARIANTS:
        matched = _match_variant(variants_df, lof["hgvs_p"])
        if matched is not None:
            records.append({
                **matched,
                "functional_class": "LOF",
                "source": lof["source"],
                "disease_context": lof["disease_context"],
                "nfkb_activity": lof["nfkb_activity"],
            })
        else:
            logger.warning("LOF variant %s not found in dataset", lof["hgvs_p"])

    # Add GOF variants (deduplicate by hgvs_p)
    seen_gof = set()
    for gof in GOF_VARIANTS:
        if gof["hgvs_p"] in seen_gof:
            continue
        matched = _match_variant(variants_df, gof["hgvs_p"])
        if matched is not None:
            records.append({
                **matched,
                "functional_class": "GOF",
                "source": gof["source"],
                "disease_context": gof["disease_context"],
                "nfkb_activity": gof["nfkb_activity"],
            })
            seen_gof.add(gof["hgvs_p"])
        else:
            # Create record without genomic position (literature-only variants)
            records.append({
                "variant_id": f"Literature:{gof['hgvs_p']}",
                "hgvs_p": gof["hgvs_p"],
                "chrom": "16",
                "pos": None,
                "ref": "",
                "alt": "",
                "functional_class": "GOF",
                "source": gof["source"],
                "disease_context": gof["disease_context"],
                "nfkb_activity": gof["nfkb_activity"],
            })
            seen_gof.add(gof["hgvs_p"])
            logger.info("GOF variant %s added from literature (no genomic match)", gof["hgvs_p"])

    # Add ClinVar benign variants
    benign_mask = variants_df["clinvar_significance"].str.contains(
        r"[Bb]enign", na=False
    ) & ~variants_df["clinvar_significance"].str.contains(
        r"[Pp]athogenic|[Cc]onflicting", na=False
    )
    benign_variants = variants_df[benign_mask]
    logger.info("Found %d ClinVar benign variants", len(benign_variants))

    for _, row in benign_variants.iterrows():
        hgvs_p = str(row.get("hgvs_p", ""))
        if not hgvs_p or hgvs_p == "nan":
            continue
        records.append({
            "variant_id": row.get("variant_id", ""),
            "hgvs_p": hgvs_p,
            "chrom": str(row["chrom"]),
            "pos": row["pos"],
            "ref": row.get("ref", ""),
            "alt": row.get("alt", ""),
            "functional_class": "neutral",
            "source": "ClinVar_benign",
            "disease_context": "",
            "nfkb_activity": "",
        })

    # Add common gnomAD variants (AF > threshold, no disease association)
    if "gnomad_af" in variants_df.columns:
        common_mask = (
            (variants_df["gnomad_af"] > COMMON_AF_THRESHOLD)
            & ~variants_df["clinvar_significance"].str.contains(
                r"[Pp]athogenic", na=False
            )
        )
        common_variants = variants_df[common_mask]
        # Exclude variants already in training set
        existing_positions = {r.get("pos") for r in records if r.get("pos") is not None}
        for _, row in common_variants.iterrows():
            if row["pos"] in existing_positions:
                continue
            hgvs_p = str(row.get("hgvs_p", ""))
            if not hgvs_p or hgvs_p == "nan":
                continue
            records.append({
                "variant_id": row.get("variant_id", ""),
                "hgvs_p": hgvs_p,
                "chrom": str(row["chrom"]),
                "pos": row["pos"],
                "ref": row.get("ref", ""),
                "alt": row.get("alt", ""),
                "functional_class": "neutral",
                "source": "gnomAD_common",
                "disease_context": "",
                "nfkb_activity": "",
            })
        logger.info("Added %d common gnomAD variants as neutral", common_mask.sum())

    df = pd.DataFrame(records)

    # Deduplicate by position (keep first occurrence)
    if not df.empty and "pos" in df.columns:
        df = df.drop_duplicates(subset=["pos", "ref", "alt"], keep="first")

    logger.info(
        "Training set: %d variants (GOF=%d, neutral=%d, LOF=%d)",
        len(df),
        (df["functional_class"] == "GOF").sum(),
        (df["functional_class"] == "neutral").sum(),
        (df["functional_class"] == "LOF").sum(),
    )

    return df


def _match_variant(variants_df: pd.DataFrame, hgvs_short: str) -> dict | None:
    """Match a short protein change notation to the variant dataset.

    Handles both single-letter (R702W) and 3-letter notation matching.
    """
    from bioagentics.data.nod2.varmeter2 import _AA3TO1

    # Build search patterns
    patterns = [hgvs_short]

    # If single-letter, also try 3-letter conversion
    import re
    m = re.match(r"([A-Z])(\d+)([A-Z])", hgvs_short)
    if m:
        ref1, pos, alt1 = m.group(1), m.group(2), m.group(3)
        # Reverse lookup: 1-letter to 3-letter
        aa1to3 = {v: k for k, v in _AA3TO1.items()}
        ref3 = aa1to3.get(ref1, "")
        alt3 = aa1to3.get(alt1, "")
        if ref3 and alt3:
            patterns.append(f"{ref3}{pos}{alt3}")
            patterns.append(f"p.{ref3}{pos}{alt3}")

    # Also handle frameshift
    m = re.match(r"([A-Z])(\d+)fs", hgvs_short)
    if m:
        ref1, pos = m.group(1), m.group(2)
        aa1to3 = {v: k for k, v in _AA3TO1.items()}
        ref3 = aa1to3.get(ref1, "")
        if ref3:
            patterns.append(f"{ref3}{pos}fs")
            patterns.append(f"p.{ref3}{pos}fs")

    for pattern in patterns:
        mask = variants_df["hgvs_p"].str.contains(pattern, case=False, na=False)
        if mask.any():
            row = variants_df[mask].iloc[0]
            return {
                "variant_id": row.get("variant_id", ""),
                "hgvs_p": str(row["hgvs_p"]),
                "chrom": str(row["chrom"]),
                "pos": row["pos"],
                "ref": row.get("ref", ""),
                "alt": row.get("alt", ""),
            }

    return None


def collect_training_set(
    variants_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Main pipeline: build and save training set."""
    if output_path is None:
        output_path = OUTPUT_DIR / "nod2_training_set.tsv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = build_training_set(variants_path)
    df.to_csv(output_path, sep="\t", index=False)
    logger.info("Saved training set to %s", output_path)

    return df
