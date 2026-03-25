"""Metabolite annotation utility for HMP2 untargeted metabolomics.

Extracts RT, m/z, and chromatographic method metadata from the HMP2 biom file
and cross-references unannotated features against the HMDB database using
m/z-based neutral mass matching (±10 ppm tolerance).

Only 592 of 81,867 HMP2 metabolomic features have compound names. This utility
provides putative annotations for the remaining features by matching observed
m/z values to known HMDB monoisotopic masses, accounting for ionization mode.

Usage::

    from bioagentics.data.metabolite_annotation import annotate_hmp2_metabolites

    # Full pipeline: extract biom metadata + HMDB matching
    annotations = annotate_hmp2_metabolites()

    # Or step-by-step:
    from bioagentics.data.metabolite_annotation import (
        extract_biom_metadata,
        load_hmdb_masses,
        match_features_to_hmdb,
    )
    meta = extract_biom_metadata()
    hmdb = load_hmdb_masses()
    matched = match_features_to_hmdb(meta, hmdb)
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from xml.etree.ElementTree import iterparse

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "hmp2"
DEFAULT_BIOM_PATH = DATA_DIR / "HMP2_metabolomics_w_metadata.biom.gz"
HMDB_CACHE_PATH = DATA_DIR / "hmdb_mass_lookup.csv"
DEFAULT_OUTPUT_PATH = DATA_DIR / "metabolite_annotations_extended.csv"

# ── Mass Constants ──

PROTON_MASS = 1.00727646677  # Da

# LC-MS method → ionization mode
METHOD_TO_MODE: dict[str, str] = {
    "C18-neg": "negative",
    "HILIC-neg": "negative",
    "C8-pos": "positive",
    "HILIC-pos": "positive",
}

# Primary adduct per ionization mode (most common for small molecules)
_ADDUCT_SHIFT: dict[str, float] = {
    "negative": PROTON_MASS,   # [M-H]⁻: m/z = M - H  →  M = m/z + H
    "positive": -PROTON_MASS,  # [M+H]⁺: m/z = M + H  →  M = m/z - H
}

# Additional common adducts to consider
_EXTRA_ADDUCTS: dict[str, list[tuple[str, float]]] = {
    "negative": [
        ("[M+FA-H]-", PROTON_MASS - 46.00548),  # formate adduct
        ("[M+Cl]-", -34.96885),                   # chloride adduct
    ],
    "positive": [
        ("[M+Na]+", -22.98922),                   # sodium adduct
        ("[M+NH4]+", -18.03437),                  # ammonium adduct
    ],
}


# ── Biom Metadata Extraction ──


def extract_biom_metadata(
    biom_path: Path | str = DEFAULT_BIOM_PATH,
) -> pd.DataFrame:
    """Extract observation metadata from the HMP2 metabolomics biom file.

    Parameters
    ----------
    biom_path : Path
        Path to HMP2_metabolomics_w_metadata.biom.gz.

    Returns
    -------
    DataFrame with columns: feature_id, RT, mz, Method, Metabolite,
    HMDB_ID, QC_CV, is_annotated.
    """
    import biom

    biom_path = Path(biom_path)
    logger.info("Loading biom file: %s", biom_path)
    table = biom.load_table(str(biom_path))

    obs_ids = table.ids("observation")
    logger.info("Total features: %d", len(obs_ids))

    rows = []
    for oid in obs_ids:
        md = table.metadata(oid, axis="observation") or {}
        rows.append({
            "feature_id": oid,
            "RT": _safe_float(md.get("RT", "")),
            "mz": _safe_float(md.get("m/z", "")),
            "Method": str(md.get("Method", "")),
            "Metabolite": str(md.get("Metabolite", "")).strip(),
            "HMDB_ID": str(md.get("HMDB (*Representative ID)", "")).strip(),
            "QC_CV": _safe_float(md.get("Pooled QC sample CV", "")),
        })

    df = pd.DataFrame(rows)
    df["is_annotated"] = df["Metabolite"].astype(bool)

    n_ann = df["is_annotated"].sum()
    logger.info(
        "Annotated: %d / %d (%.1f%%)",
        n_ann, len(df), n_ann / len(df) * 100,
    )
    return df


def _safe_float(val: object) -> float:
    """Convert to float, returning NaN on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return float("nan")


# ── Neutral Mass Computation ──


def compute_neutral_mass(mz: float, method: str) -> float:
    """Compute neutral monoisotopic mass from observed m/z.

    Uses the primary adduct for the ionization mode:
    - Negative mode [M-H]⁻: M = m/z + proton_mass
    - Positive mode [M+H]⁺: M = m/z - proton_mass

    Parameters
    ----------
    mz : float
        Observed mass-to-charge ratio.
    method : str
        LC-MS method (e.g., "C18-neg", "HILIC-pos").

    Returns
    -------
    Neutral monoisotopic mass in Da. NaN if method is unrecognized.
    """
    mode = METHOD_TO_MODE.get(method)
    if mode is None:
        return float("nan")
    return mz + _ADDUCT_SHIFT[mode]


def compute_all_neutral_masses(
    mz: float, method: str,
) -> list[tuple[str, float]]:
    """Compute neutral masses for all plausible adducts.

    Returns list of (adduct_name, neutral_mass) tuples.
    """
    mode = METHOD_TO_MODE.get(method)
    if mode is None:
        return []

    primary_name = "[M-H]-" if mode == "negative" else "[M+H]+"
    results = [(primary_name, mz + _ADDUCT_SHIFT[mode])]

    for adduct_name, shift in _EXTRA_ADDUCTS.get(mode, []):
        results.append((adduct_name, mz + shift))

    return results


# ── HMDB Mass Database ──


def download_hmdb_masses(
    output_path: Path | str = HMDB_CACHE_PATH,
    url: str = "https://hmdb.ca/system/downloads/current/hmdb_metabolites.zip",
    timeout: int = 600,
) -> pd.DataFrame:
    """Download HMDB metabolites and extract mass lookup table.

    Downloads the HMDB metabolites XML archive and stream-parses it to
    extract {hmdb_id, name, monoisotopic_weight, chemical_formula, super_class}.
    Saves a compact CSV cache for fast reloading.

    Parameters
    ----------
    output_path : Path
        Where to save the cached CSV.
    url : str
        HMDB download URL.
    timeout : int
        Download timeout in seconds.

    Returns
    -------
    DataFrame with HMDB mass lookup data.
    """
    output_path = Path(output_path)
    logger.info("Downloading HMDB metabolites from %s", url)

    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    buf = io.BytesIO()
    with tqdm(total=total, unit="B", unit_scale=True, desc="HMDB download") as pbar:
        for chunk in resp.iter_content(chunk_size=64 * 1024):
            buf.write(chunk)
            pbar.update(len(chunk))

    logger.info("Download complete. Parsing XML...")
    buf.seek(0)

    records = _parse_hmdb_zip(buf)

    df = pd.DataFrame(records)
    if len(df) == 0:
        logger.warning("No HMDB records parsed!")
        return df

    # Clean up mass column
    df["monoisotopic_weight"] = pd.to_numeric(
        df["monoisotopic_weight"], errors="coerce"
    )
    df = df.dropna(subset=["monoisotopic_weight"])
    df = df[df["monoisotopic_weight"] > 0]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(
        "HMDB mass lookup saved: %s (%d metabolites)", output_path, len(df)
    )
    return df


def _parse_hmdb_zip(buf: io.BytesIO) -> list[dict]:
    """Stream-parse HMDB zip archive, extracting mass data.

    Memory-efficient: processes one <metabolite> element at a time.
    """
    records: list[dict] = []
    ns = "{http://www.hmdb.ca}"

    with zipfile.ZipFile(buf) as zf:
        xml_names = [n for n in zf.namelist() if n.endswith(".xml")]
        if not xml_names:
            logger.error("No XML file found in HMDB zip")
            return records

        xml_name = xml_names[0]
        logger.info("Parsing %s from archive...", xml_name)

        with zf.open(xml_name) as xml_file:
            context = iterparse(xml_file, events=("end",))
            for event, elem in context:
                if elem.tag == f"{ns}metabolite":
                    record = _extract_metabolite(elem, ns)
                    if record:
                        records.append(record)
                    elem.clear()
                # Also handle non-namespaced tags (some HMDB versions)
                elif elem.tag == "metabolite":
                    record = _extract_metabolite(elem, "")
                    if record:
                        records.append(record)
                    elem.clear()

    logger.info("Parsed %d metabolite records", len(records))
    return records


def _extract_metabolite(elem, ns: str) -> dict | None:
    """Extract key fields from a single <metabolite> XML element."""
    prefix = f"{ns}" if ns else ""

    hmdb_id = _get_text(elem, f"{prefix}accession")
    name = _get_text(elem, f"{prefix}name")
    mass = _get_text(elem, f"{prefix}monisotopic_molecular_weight")
    if not mass:
        mass = _get_text(elem, f"{prefix}monoisotopic_molecular_weight")
    formula = _get_text(elem, f"{prefix}chemical_formula")

    if not hmdb_id or not mass:
        return None

    return {
        "hmdb_id": hmdb_id,
        "name": name or "",
        "monoisotopic_weight": mass,
        "chemical_formula": formula or "",
        "super_class": _get_text(elem, f"{prefix}super_class") or "",
    }


def _get_text(elem, tag: str) -> str:
    """Get text content of a child element, or empty string."""
    child = elem.find(tag)
    return child.text.strip() if child is not None and child.text else ""


def load_hmdb_masses(
    cache_path: Path | str = HMDB_CACHE_PATH,
) -> pd.DataFrame:
    """Load HMDB mass lookup table from cached CSV.

    If cache doesn't exist, attempts to download it.

    Returns
    -------
    DataFrame with columns: hmdb_id, name, monoisotopic_weight,
    chemical_formula, super_class.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        logger.info("Loaded HMDB cache: %d entries from %s", len(df), cache_path)
        return df

    logger.info("HMDB cache not found at %s — downloading...", cache_path)
    return download_hmdb_masses(output_path=cache_path)


# ── Mass Matching ──


def match_by_mass(
    neutral_mass: float,
    hmdb_df: pd.DataFrame,
    ppm_tolerance: float = 10.0,
) -> pd.DataFrame:
    """Find HMDB entries matching a neutral mass within PPM tolerance.

    Parameters
    ----------
    neutral_mass : float
        Query neutral monoisotopic mass in Da.
    hmdb_df : DataFrame
        HMDB mass lookup with 'monoisotopic_weight' column.
    ppm_tolerance : float
        Maximum allowed mass error in parts-per-million (default: 10).

    Returns
    -------
    DataFrame of matching HMDB entries with added 'ppm_error' column,
    sorted by ascending ppm error.
    """
    if np.isnan(neutral_mass) or neutral_mass <= 0:
        return pd.DataFrame()

    masses = hmdb_df["monoisotopic_weight"].values
    ppm_errors = np.abs(masses - neutral_mass) / neutral_mass * 1e6
    mask = ppm_errors <= ppm_tolerance

    if not mask.any():
        return pd.DataFrame()

    matches = hmdb_df.loc[mask].copy()
    matches["ppm_error"] = ppm_errors[mask]
    return matches.sort_values("ppm_error")


def match_features_to_hmdb(
    biom_meta: pd.DataFrame,
    hmdb_df: pd.DataFrame,
    ppm_tolerance: float = 10.0,
    feature_subset: list[str] | None = None,
    max_matches_per_feature: int = 5,
    include_extra_adducts: bool = True,
) -> pd.DataFrame:
    """Match unannotated metabolomic features to HMDB by mass.

    Parameters
    ----------
    biom_meta : DataFrame
        Output of extract_biom_metadata().
    hmdb_df : DataFrame
        HMDB mass lookup table.
    ppm_tolerance : float
        Mass tolerance in ppm (default: 10).
    feature_subset : list[str], optional
        Only annotate these feature IDs (e.g., top 1000 variance features).
    max_matches_per_feature : int
        Keep at most N HMDB candidates per feature (default: 5).
    include_extra_adducts : bool
        Also try secondary adduct types (default: True).

    Returns
    -------
    DataFrame combining original biom metadata with putative HMDB matches.
    Columns: feature_id, RT, mz, Method, Metabolite, HMDB_ID, QC_CV,
    is_annotated, neutral_mass, putative_hmdb_id, putative_name,
    putative_formula, putative_super_class, adduct, ppm_error, match_rank.
    """
    if feature_subset is not None:
        work = biom_meta[biom_meta["feature_id"].isin(feature_subset)].copy()
    else:
        work = biom_meta.copy()

    unannotated = work[~work["is_annotated"]]
    logger.info(
        "Matching %d unannotated features (of %d total) against %d HMDB entries",
        len(unannotated), len(work), len(hmdb_df),
    )

    match_rows = []
    n_matched = 0

    for _, row in tqdm(
        unannotated.iterrows(),
        total=len(unannotated),
        desc="HMDB matching",
        disable=len(unannotated) < 100,
    ):
        mz = row["mz"]
        method = row["Method"]

        if np.isnan(mz) or not method:
            continue

        if include_extra_adducts:
            adduct_masses = compute_all_neutral_masses(mz, method)
        else:
            nm = compute_neutral_mass(mz, method)
            adduct_name = "[M-H]-" if METHOD_TO_MODE.get(method) == "negative" else "[M+H]+"
            adduct_masses = [(adduct_name, nm)]

        best_matches: list[tuple[float, str, pd.Series]] = []
        for adduct_name, nm in adduct_masses:
            if np.isnan(nm):
                continue
            matches = match_by_mass(nm, hmdb_df, ppm_tolerance)
            for _, m in matches.head(max_matches_per_feature).iterrows():
                best_matches.append((m["ppm_error"], adduct_name, m))

        # Sort by ppm error across all adducts, keep top N
        best_matches.sort(key=lambda x: x[0])
        if best_matches:
            n_matched += 1

        for rank, (ppm_err, adduct_name, m) in enumerate(
            best_matches[:max_matches_per_feature], 1
        ):
            match_rows.append({
                "feature_id": row["feature_id"],
                "putative_hmdb_id": m["hmdb_id"],
                "putative_name": m["name"],
                "putative_formula": m.get("chemical_formula", ""),
                "putative_super_class": m.get("super_class", ""),
                "adduct": adduct_name,
                "ppm_error": round(ppm_err, 2),
                "match_rank": rank,
            })

    logger.info(
        "Matched %d / %d unannotated features (%.1f%%)",
        n_matched, len(unannotated),
        n_matched / max(len(unannotated), 1) * 100,
    )

    # Compute neutral mass for all features
    work["neutral_mass"] = work.apply(
        lambda r: compute_neutral_mass(r["mz"], r["Method"]), axis=1
    )

    if match_rows:
        matches_df = pd.DataFrame(match_rows)
        result = work.merge(matches_df, on="feature_id", how="left")
    else:
        result = work.copy()
        for col in [
            "putative_hmdb_id", "putative_name", "putative_formula",
            "putative_super_class", "adduct", "ppm_error", "match_rank",
        ]:
            result[col] = np.nan

    return result


# ── Main Pipeline ──


def annotate_hmp2_metabolites(
    biom_path: Path | str = DEFAULT_BIOM_PATH,
    hmdb_cache_path: Path | str = HMDB_CACHE_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    feature_subset: list[str] | None = None,
    ppm_tolerance: float = 10.0,
    max_matches: int = 5,
) -> pd.DataFrame:
    """Run the full HMP2 metabolite annotation pipeline.

    Steps:
    1. Extract observation metadata from biom file
    2. Load (or download) HMDB mass database
    3. Match unannotated features by neutral mass within tolerance
    4. Save extended annotations CSV

    Parameters
    ----------
    biom_path : Path
        HMP2 metabolomics biom file.
    hmdb_cache_path : Path
        Cached HMDB mass lookup CSV.
    output_path : Path
        Where to write the extended annotations.
    feature_subset : list[str], optional
        Only annotate these feature IDs.
    ppm_tolerance : float
        Mass matching tolerance in ppm (default: 10).
    max_matches : int
        Max HMDB candidates per feature (default: 5).

    Returns
    -------
    DataFrame with extended annotations.
    """
    logger.info("=== HMP2 Metabolite Annotation Pipeline ===")

    # Step 1: Extract biom metadata
    biom_meta = extract_biom_metadata(biom_path)

    # Step 2: Load HMDB masses
    hmdb_df = load_hmdb_masses(hmdb_cache_path)

    # Step 3: Match features
    result = match_features_to_hmdb(
        biom_meta,
        hmdb_df,
        ppm_tolerance=ppm_tolerance,
        feature_subset=feature_subset,
        max_matches_per_feature=max_matches,
    )

    # Step 4: Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    logger.info("Saved annotations: %s (%d rows)", output_path, len(result))

    # Summary
    n_putative = result["putative_name"].notna().sum()
    n_unique_features = result.loc[
        result["putative_name"].notna(), "feature_id"
    ].nunique()
    logger.info(
        "Summary: %d putative annotations for %d unique features",
        n_putative, n_unique_features,
    )

    return result
