"""Dataset definitions for anti-TNF response prediction.

Each dataset entry specifies the GEO/ArrayExpress accession, platform,
tissue type, and expected response annotations.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetInfo:
    accession: str
    platform: str
    tissue: str
    drug: str
    description: str
    source: str  # "geo" or "arrayexpress"
    response_column: str | None = None
    timepoint_column: str | None = None


# Core datasets for the anti-TNF response prediction classifier
DATASETS: dict[str, DatasetInfo] = {
    "GSE16879": DatasetInfo(
        accession="GSE16879",
        platform="GPL570",  # Affymetrix HG-U133 Plus 2.0
        tissue="mucosal_biopsy",
        drug="infliximab",
        description="Mucosal biopsies from 24 CD patients before/after infliximab (Arijs 2009)",
        source="geo",
    ),
    "GSE12251": DatasetInfo(
        accession="GSE12251",
        platform="GPL570",
        tissue="colonic_biopsy",
        drug="infliximab",
        description="Colonic biopsies from 22 CD patients pre-infliximab (Arijs 2009)",
        source="geo",
    ),
    "GSE73661": DatasetInfo(
        accession="GSE73661",
        platform="GPL570",
        tissue="mucosal_biopsy",
        drug="anti-tnf",
        description="Mucosal biopsies from 73 IBD patients pre-anti-TNF (Haberman 2014)",
        source="geo",
    ),
    "GSE100833": DatasetInfo(
        accession="GSE100833",
        platform="GPL10558",  # Illumina HumanHT-12 V4.0
        tissue="blood",
        drug="anti-tnf",
        description="Blood transcriptomics from IBD patients on anti-TNF therapy",
        source="geo",
    ),
    "GSE57945": DatasetInfo(
        accession="GSE57945",
        platform="GPL6244",  # Affymetrix HuGene 1.0 ST
        tissue="ileal_biopsy",
        drug="mixed",
        description="RISK cohort: Pediatric CD treatment-naive ileal biopsies with follow-up",
        source="geo",
    ),
}

# Known anti-TNF response genes for sanity checking
EXPECTED_RESPONSE_GENES = [
    "TREM1", "OSMR", "IL13RA2", "IL23R", "CCL7", "IL17F", "YES1",
    "OSM", "IL6ST", "MMP2", "COL1A1", "CD81", "RORC", "NAMPT",
]

# TNF signaling pathway genes for enrichment checking
TNF_PATHWAY_GENES = [
    "TNF", "TNFRSF1A", "TNFRSF1B", "TRADD", "TRAF2", "TRAF5",
    "RIPK1", "NFKB1", "NFKB2", "RELA", "RELB", "BIRC2", "BIRC3",
    "MAP3K7", "MAPK8", "MAPK9", "MAPK14", "CASP3", "CASP8",
    "IKBKB", "IKBKG", "CHUK",
]
