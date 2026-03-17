"""ClinVar + gnomAD NOD2 variant data collection pipeline.

Fetches all known NOD2 variants from ClinVar and gnomAD v4,
merges annotations, and outputs a unified TSV.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# NOD2 gene identifiers
NOD2_GENE_ID = "64127"  # NCBI Gene ID for NOD2
NOD2_GENE_SYMBOL = "NOD2"
NOD2_CHROM = "16"
# GRCh38 coordinates for NOD2 (chr16:50,693,587-50,733,076)
NOD2_START = 50693587
NOD2_END = 50733076

OUTPUT_DIR = Path("data/crohns/nod2-variant-functional-impact")

# NCBI E-utilities base URLs
ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
ELINK_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi"

# gnomAD v4 API
GNOMAD_API_URL = "https://gnomad.broadinstitute.org/api"


def _ncbi_request(url: str, params: dict[str, Any], max_retries: int = 3) -> requests.Response:
    """Make a request to NCBI with rate limiting and retries."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                logger.warning("NCBI request failed (attempt %d), retrying in %ds: %s", attempt + 1, wait, e)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Unreachable")


def fetch_clinvar_variants() -> pd.DataFrame:
    """Fetch all NOD2 variants from ClinVar using E-utilities.

    Returns DataFrame with columns: variant_id, chrom, pos, ref, alt,
    hgvs_p, clinvar_significance, review_stars.
    """
    logger.info("Fetching NOD2 variants from ClinVar...")

    # Search for all NOD2 variants in ClinVar
    search_params = {
        "db": "clinvar",
        "term": f"NOD2[gene] AND \"single nucleotide variant\"[Type]",
        "retmax": 0,
        "retmode": "json",
    }
    resp = _ncbi_request(ESEARCH_URL, search_params)
    total = int(resp.json()["esearchresult"]["count"])
    logger.info("Found %d NOD2 SNVs in ClinVar", total)

    # Also search for other variant types (indels, etc.)
    search_params_other = {
        "db": "clinvar",
        "term": "NOD2[gene]",
        "retmax": 0,
        "retmode": "json",
    }
    resp_other = _ncbi_request(ESEARCH_URL, search_params_other)
    total_all = int(resp_other.json()["esearchresult"]["count"])
    logger.info("Found %d total NOD2 variants in ClinVar (all types)", total_all)

    # Fetch all variant IDs with pagination
    variant_ids: list[str] = []
    batch_size = 500
    for start in range(0, total_all, batch_size):
        search_params_batch = {
            "db": "clinvar",
            "term": "NOD2[gene]",
            "retmax": batch_size,
            "retstart": start,
            "retmode": "json",
        }
        resp = _ncbi_request(ESEARCH_URL, search_params_batch)
        ids = resp.json()["esearchresult"]["idlist"]
        variant_ids.extend(ids)
        time.sleep(0.4)  # NCBI rate limit: 3 requests/sec without API key

    logger.info("Retrieved %d variant IDs from ClinVar", len(variant_ids))

    # Fetch variant details in batches
    records: list[dict[str, Any]] = []
    fetch_batch_size = 50
    for i in range(0, len(variant_ids), fetch_batch_size):
        batch_ids = variant_ids[i:i + fetch_batch_size]
        fetch_params = {
            "db": "clinvar",
            "id": ",".join(batch_ids),
            "rettype": "vcv",
            "retmode": "xml",
            "is_variationid": "true",
        }
        resp = _ncbi_request(EFETCH_URL, fetch_params)
        records.extend(_parse_clinvar_xml(resp.text))
        time.sleep(0.4)

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("No ClinVar variants parsed")
        return pd.DataFrame(columns=[
            "variant_id", "chrom", "pos", "ref", "alt",
            "hgvs_p", "clinvar_significance", "review_stars",
        ])

    logger.info("Parsed %d ClinVar variant records", len(df))
    return df


def _parse_clinvar_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse ClinVar VCV XML response into variant records."""
    records = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        logger.error("Failed to parse ClinVar XML response")
        return records

    for vcv in root.iter("VariationArchive"):
        variant_id = vcv.get("VariationID", "")
        accession = vcv.get("Accession", "")

        # Get clinical significance
        clin_sig = ""
        review_status = ""
        for interp in vcv.iter("ClassifiedRecord"):
            for simple in interp.iter("SimpleAllele"):
                pass  # handled below
            for classifications in interp.iter("Classifications"):
                for germ in classifications.iter("GermlineClassification"):
                    desc = germ.find("Description")
                    if desc is not None and desc.text:
                        clin_sig = desc.text
                    rev = germ.find("ReviewStatus")
                    if rev is not None and rev.text:
                        review_status = rev.text

        # Map review status to stars
        review_stars = _review_status_to_stars(review_status)

        # Get genomic location and alleles
        chrom, pos, ref, alt = "", "", "", ""
        hgvs_p = ""

        for simple_allele in vcv.iter("SimpleAllele"):
            # Get HGVS protein
            for hgvs_expr in simple_allele.iter("HGVSlist"):
                for hgvs in hgvs_expr.iter("HGVS"):
                    expr_type = hgvs.get("Type", "")
                    if "protein" in expr_type.lower() or expr_type == "coding":
                        ne = hgvs.find("NucleotideExpression")
                        pe = hgvs.find("ProteinExpression")
                        if pe is not None:
                            change = pe.get("change", "")
                            if change:
                                hgvs_p = change

            # Get genomic location from SequenceLocation
            for loc_list in simple_allele.iter("Location"):
                for seq_loc in loc_list.iter("SequenceLocation"):
                    assembly = seq_loc.get("Assembly", "")
                    if assembly == "GRCh38":
                        chrom = seq_loc.get("Chr", "")
                        pos = seq_loc.get("positionVCF", seq_loc.get("start", ""))
                        ref = seq_loc.get("referenceAlleleVCF", "")
                        alt = seq_loc.get("alternateAlleleVCF", "")
                        break

        if chrom and pos:
            records.append({
                "variant_id": f"ClinVar:{accession}",
                "chrom": chrom,
                "pos": int(pos) if pos else None,
                "ref": ref,
                "alt": alt,
                "hgvs_p": hgvs_p,
                "clinvar_significance": clin_sig,
                "review_stars": review_stars,
            })

    return records


def _review_status_to_stars(status: str) -> int:
    """Convert ClinVar review status string to star rating."""
    status_lower = status.lower()
    if "practice guideline" in status_lower:
        return 4
    elif "reviewed by expert panel" in status_lower:
        return 3
    elif "criteria provided, multiple submitters" in status_lower:
        return 2
    elif "criteria provided, single submitter" in status_lower:
        return 1
    elif "criteria provided, conflicting" in status_lower:
        return 1
    else:
        return 0


def fetch_gnomad_variants() -> pd.DataFrame:
    """Fetch NOD2 variants from gnomAD v4 GraphQL API.

    Returns DataFrame with columns: chrom, pos, ref, alt,
    gnomad_af, gnomad_af_popmax, gnomad_hom_count.
    """
    logger.info("Fetching NOD2 variants from gnomAD v4...")

    query = """
    query {
      gene(gene_symbol: "NOD2", reference_genome: GRCh38) {
        variants(dataset: gnomad_r4) {
          variant_id
          chrom
          pos
          ref
          alt
          exome {
            ac
            an
            af
            homozygote_count
          }
          genome {
            ac
            an
            af
            homozygote_count
          }
        }
      }
    }
    """

    try:
        resp = requests.post(
            GNOMAD_API_URL,
            json={"query": query},
            timeout=120,
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error("gnomAD API request failed: %s", e)
        return pd.DataFrame(columns=[
            "chrom", "pos", "ref", "alt",
            "gnomad_af", "gnomad_af_popmax", "gnomad_hom_count",
        ])

    if "errors" in data:
        logger.error("gnomAD API errors: %s", data["errors"])
        return pd.DataFrame(columns=[
            "chrom", "pos", "ref", "alt",
            "gnomad_af", "gnomad_af_popmax", "gnomad_hom_count",
        ])

    variants = data.get("data", {}).get("gene", {}).get("variants", [])
    logger.info("Retrieved %d variants from gnomAD", len(variants))

    records = []
    for v in variants:
        # Combine exome and genome data — use max AF across datasets
        afs: list[float] = []
        hom_total = 0

        for dataset in ["exome", "genome"]:
            d = v.get(dataset)
            if d and d.get("af") is not None:
                afs.append(d["af"])
                hom_total += d.get("homozygote_count", 0) or 0

        af_total = max(afs) if afs else 0.0

        records.append({
            "chrom": v["chrom"],
            "pos": int(v["pos"]),
            "ref": v["ref"],
            "alt": v["alt"],
            "gnomad_af": af_total,
            "gnomad_af_popmax": af_total,  # popmax approx: use max across exome/genome
            "gnomad_hom_count": hom_total,
        })

    return pd.DataFrame(records)


def merge_variants(
    clinvar_df: pd.DataFrame,
    gnomad_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge ClinVar and gnomAD variant data on genomic position.

    Returns unified DataFrame with all columns from both sources.
    """
    if clinvar_df.empty and gnomad_df.empty:
        return pd.DataFrame()

    merge_cols = ["chrom", "pos", "ref", "alt"]

    # Ensure consistent types
    for df in [clinvar_df, gnomad_df]:
        if not df.empty:
            df["chrom"] = df["chrom"].astype(str)
            df["pos"] = pd.to_numeric(df["pos"], errors="coerce").astype("Int64")

    if clinvar_df.empty:
        merged = gnomad_df.copy()
        merged["variant_id"] = ""
        merged["hgvs_p"] = ""
        merged["clinvar_significance"] = ""
        merged["review_stars"] = 0
    elif gnomad_df.empty:
        merged = clinvar_df.copy()
        merged["gnomad_af"] = float("nan")
        merged["gnomad_af_popmax"] = float("nan")
        merged["gnomad_hom_count"] = 0
    else:
        merged = pd.merge(
            clinvar_df,
            gnomad_df,
            on=merge_cols,
            how="outer",
            suffixes=("", "_gnomad"),
        )
        # Fill missing values
        merged["variant_id"] = merged["variant_id"].fillna("")
        merged["hgvs_p"] = merged["hgvs_p"].fillna("")
        merged["clinvar_significance"] = merged["clinvar_significance"].fillna("")
        merged["review_stars"] = merged["review_stars"].fillna(0).astype(int)
        merged["gnomad_af"] = merged["gnomad_af"].fillna(float("nan"))
        merged["gnomad_af_popmax"] = merged["gnomad_af_popmax"].fillna(float("nan"))
        merged["gnomad_hom_count"] = pd.to_numeric(merged["gnomad_hom_count"].fillna(0), downcast="integer")

    # Ensure required column order
    output_cols = [
        "chrom", "pos", "ref", "alt", "hgvs_p",
        "clinvar_significance", "review_stars",
        "gnomad_af", "gnomad_af_popmax",
    ]
    # Add any extra columns
    extra_cols = [c for c in merged.columns if c not in output_cols]
    final_cols = output_cols + extra_cols

    for col in output_cols:
        if col not in merged.columns:
            merged[col] = ""

    merged = merged[[c for c in final_cols if c in merged.columns]]
    merged = merged.sort_values(by=["chrom", "pos"]).reset_index(drop=True)

    return merged


def validate_known_variants(df: pd.DataFrame) -> dict[str, bool]:
    """Verify that known pathogenic NOD2 variants are present.

    Returns dict mapping variant name to whether it was found.
    """
    # Match both single-letter (R702W) and 3-letter (Arg702Trp) notation
    known = {
        "R702W": ["R702W", "Arg702Trp"],
        "G908R": ["G908R", "Gly908Arg"],
        "L1007fs": ["L1007", "Leu1007"],  # partial match for frameshift
    }

    results = {}
    for name, patterns in known.items():
        mask = pd.Series(False, index=df.index)
        for pattern in patterns:
            mask = mask | df["hgvs_p"].str.contains(pattern, case=False, na=False)
        results[name] = bool(mask.any())
        if results[name]:
            logger.info("Found known variant %s in dataset", name)
        else:
            logger.warning("Known variant %s NOT found in dataset", name)

    return results


def collect_nod2_variants(output_path: Path | None = None) -> pd.DataFrame:
    """Main pipeline: fetch, merge, validate, and save NOD2 variants.

    Args:
        output_path: Where to save the TSV. Defaults to
            data/crohns/nod2-variant-functional-impact/nod2_variants.tsv

    Returns:
        Merged DataFrame of all NOD2 variants.
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "nod2_variants.tsv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch from both sources
    clinvar_df = fetch_clinvar_variants()
    gnomad_df = fetch_gnomad_variants()

    logger.info(
        "ClinVar: %d variants, gnomAD: %d variants",
        len(clinvar_df), len(gnomad_df),
    )

    # Merge
    merged = merge_variants(clinvar_df, gnomad_df)
    logger.info("Merged dataset: %d variants", len(merged))

    # Validate known variants are present
    if not merged.empty:
        validation = validate_known_variants(merged)
        found = sum(validation.values())
        logger.info(
            "Known variant validation: %d/%d found (%s)",
            found, len(validation),
            ", ".join(f"{k}={'OK' if v else 'MISSING'}" for k, v in validation.items()),
        )

    # Save
    merged.to_csv(output_path, sep="\t", index=False)
    logger.info("Saved %d variants to %s", len(merged), output_path)

    return merged
