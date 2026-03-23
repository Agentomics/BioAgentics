"""HMBA Basal Ganglia cell-type taxonomy reference.

Provides a reusable cell-type annotation layer based on the Allen Institute's
Human and Mouse Brain Atlas (HMBA) Basal Ganglia reference taxonomy.  When
abc_atlas_access export files are present under
``data/tourettes/cstc-circuit-expression-atlas/hmba_bg_reference/``, the module
loads taxonomy metadata, marker tables, and expression data from disk.
Otherwise it falls back to a curated built-in taxonomy covering the cell types
most relevant to Tourette syndrome CSTC circuit analysis:

  - D1 / D2 medium spiny neurons (striosome vs matrix subtypes)
  - Striatal interneurons (cholinergic, PV+, SST+)
  - GPe / GPi projection neurons
  - STN glutamatergic neurons
  - Thalamic relay / reticular neurons
  - Major glial classes (astrocytes, oligodendrocytes, OPCs, microglia)

Hierarchy follows the ABC Atlas convention:
  class → subclass → supertype → type

Usage::

    from bioagentics.data.tourettes.hmba_reference import (
        get_taxonomy,
        get_marker_panel,
        list_cell_types,
        map_to_hmba,
    )

    tax = get_taxonomy()
    markers = get_marker_panel("D1_MSN_matrix")
    hmba_ids = map_to_hmba(["D1 MSN", "cholinergic interneuron"])
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from bioagentics.config import DATA_DIR

logger = logging.getLogger(__name__)

HMBA_DATA_DIR = (
    DATA_DIR / "tourettes" / "cstc-circuit-expression-atlas" / "hmba_bg_reference"
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CellType:
    """Single node in the HMBA taxonomy."""

    hmba_id: str
    label: str
    cell_class: str
    subclass: str
    supertype: str
    cell_type: str
    region: str
    markers: tuple[str, ...] = field(default_factory=tuple)
    description: str = ""


# ---------------------------------------------------------------------------
# Built-in taxonomy — curated from HMBA Basal Ganglia reference
# Covers the cell types most relevant for TS CSTC circuit analysis.
# ---------------------------------------------------------------------------

_BUILTIN_TAXONOMY: list[CellType] = [
    # === Striatal D1 MSNs (direct pathway / striatonigral) ===
    CellType(
        hmba_id="CS20260101_D1_MSN_matrix",
        label="D1_MSN_matrix",
        cell_class="GABAergic",
        subclass="D1 MSN",
        supertype="D1 MSN matrix",
        cell_type="D1 MSN matrix",
        region="striatum",
        markers=("DRD1", "TAC1", "PDYN", "ISL1", "CHRM4", "EPHA4"),
        description="D1 medium spiny neuron, matrix compartment (direct pathway)",
    ),
    CellType(
        hmba_id="CS20260101_D1_MSN_striosome",
        label="D1_MSN_striosome",
        cell_class="GABAergic",
        subclass="D1 MSN",
        supertype="D1 MSN striosome",
        cell_type="D1 MSN striosome",
        region="striatum",
        markers=("DRD1", "TAC1", "PDYN", "ISL1", "OPRM1", "MU_OPIOID"),
        description="D1 medium spiny neuron, striosome/patch compartment",
    ),
    # === Striatal D2 MSNs (indirect pathway / striatopallidal) ===
    CellType(
        hmba_id="CS20260101_D2_MSN_matrix",
        label="D2_MSN_matrix",
        cell_class="GABAergic",
        subclass="D2 MSN",
        supertype="D2 MSN matrix",
        cell_type="D2 MSN matrix",
        region="striatum",
        markers=("DRD2", "PENK", "ADORA2A", "GPR6", "SP9"),
        description="D2 medium spiny neuron, matrix compartment (indirect pathway)",
    ),
    CellType(
        hmba_id="CS20260101_D2_MSN_striosome",
        label="D2_MSN_striosome",
        cell_class="GABAergic",
        subclass="D2 MSN",
        supertype="D2 MSN striosome",
        cell_type="D2 MSN striosome",
        region="striatum",
        markers=("DRD2", "PENK", "ADORA2A", "OPRM1"),
        description="D2 medium spiny neuron, striosome/patch compartment",
    ),
    # === Striatal interneurons ===
    CellType(
        hmba_id="CS20260101_CHAT_IN",
        label="cholinergic_interneuron",
        cell_class="Cholinergic",
        subclass="Cholinergic interneuron",
        supertype="CHAT IN",
        cell_type="CHAT IN",
        region="striatum",
        markers=("CHAT", "SLC5A7", "SLC18A3", "LHX8", "GBX2"),
        description="Striatal cholinergic interneuron (tonically active neuron)",
    ),
    CellType(
        hmba_id="CS20260101_PV_IN",
        label="pv_interneuron",
        cell_class="GABAergic",
        subclass="PV interneuron",
        supertype="PV IN",
        cell_type="PV IN",
        region="striatum",
        markers=("PVALB", "KCNC1", "KCNC2", "EYA1", "TAC3"),
        description="Parvalbumin+ fast-spiking GABAergic interneuron",
    ),
    CellType(
        hmba_id="CS20260101_SST_IN",
        label="sst_interneuron",
        cell_class="GABAergic",
        subclass="SST interneuron",
        supertype="SST IN",
        cell_type="SST IN",
        region="striatum",
        markers=("SST", "NPY", "NOS1", "CALB2"),
        description="Somatostatin+ GABAergic interneuron (PLTS type)",
    ),
    # === Globus pallidus neurons ===
    CellType(
        hmba_id="CS20260101_GPe_prototypic",
        label="GPe_prototypic",
        cell_class="GABAergic",
        subclass="GPe neuron",
        supertype="GPe prototypic",
        cell_type="GPe prototypic",
        region="GPe",
        markers=("PVALB", "LHX6", "NKX2-1", "SOX6"),
        description="GPe prototypic neuron — tonic firing, projects to STN",
    ),
    CellType(
        hmba_id="CS20260101_GPe_arkypallidal",
        label="GPe_arkypallidal",
        cell_class="GABAergic",
        subclass="GPe neuron",
        supertype="GPe arkypallidal",
        cell_type="GPe arkypallidal",
        region="GPe",
        markers=("FOXP2", "NPAS1", "MEIS2"),
        description="GPe arkypallidal neuron — projects back to striatum",
    ),
    CellType(
        hmba_id="CS20260101_GPi_projection",
        label="GPi_projection",
        cell_class="GABAergic",
        subclass="GPi neuron",
        supertype="GPi projection",
        cell_type="GPi projection",
        region="GPi",
        markers=("PVALB", "LHX6", "NKX2-1", "GAD1", "GAD2"),
        description="GPi projection neuron — basal ganglia output to thalamus",
    ),
    # === STN neurons ===
    CellType(
        hmba_id="CS20260101_STN_glutamatergic",
        label="STN_glutamatergic",
        cell_class="Glutamatergic",
        subclass="STN neuron",
        supertype="STN glutamatergic",
        cell_type="STN glutamatergic",
        region="STN",
        markers=("SLC17A6", "PITX2", "FOXP2", "PBX3"),
        description="STN glutamatergic projection neuron (hyperdirect pathway input)",
    ),
    # === Thalamic neurons ===
    CellType(
        hmba_id="CS20260101_thal_relay",
        label="thalamic_relay",
        cell_class="Glutamatergic",
        subclass="Thalamic neuron",
        supertype="Thalamic relay",
        cell_type="Thalamic relay",
        region="thalamus",
        markers=("SLC17A6", "TCF7L2", "RORA", "CALB1"),
        description="Thalamic relay neuron — receives BG output, projects to cortex",
    ),
    CellType(
        hmba_id="CS20260101_thal_reticular",
        label="thalamic_reticular",
        cell_class="GABAergic",
        subclass="Thalamic neuron",
        supertype="Thalamic reticular",
        cell_type="Thalamic reticular",
        region="thalamus",
        markers=("PVALB", "GAD1", "GAD2", "SST"),
        description="Thalamic reticular nucleus — GABAergic inhibitory gating",
    ),
    # === Glia ===
    CellType(
        hmba_id="CS20260101_astrocyte",
        label="astrocyte",
        cell_class="Non-neuronal",
        subclass="Astrocyte",
        supertype="Astrocyte",
        cell_type="Astrocyte",
        region="basal_ganglia_wide",
        markers=("GFAP", "AQP4", "ALDH1L1", "S100B", "SLC1A2"),
        description="Astrocyte — neuron-astrocyte spatial coordination in striatum",
    ),
    CellType(
        hmba_id="CS20260101_oligodendrocyte",
        label="oligodendrocyte",
        cell_class="Non-neuronal",
        subclass="Oligodendrocyte",
        supertype="Oligodendrocyte",
        cell_type="Oligodendrocyte",
        region="basal_ganglia_wide",
        markers=("MBP", "PLP1", "MOG", "OLIG2", "CNP"),
        description="Oligodendrocyte — myelination of BG fiber tracts",
    ),
    CellType(
        hmba_id="CS20260101_OPC",
        label="OPC",
        cell_class="Non-neuronal",
        subclass="OPC",
        supertype="OPC",
        cell_type="OPC",
        region="basal_ganglia_wide",
        markers=("PDGFRA", "CSPG4", "OLIG1", "OLIG2"),
        description="Oligodendrocyte precursor cell",
    ),
    CellType(
        hmba_id="CS20260101_microglia",
        label="microglia",
        cell_class="Non-neuronal",
        subclass="Microglia",
        supertype="Microglia",
        cell_type="Microglia",
        region="basal_ganglia_wide",
        markers=("CX3CR1", "P2RY12", "TMEM119", "AIF1", "CSF1R"),
        description="Microglia — neuroimmune surveillance in BG",
    ),
]

# ---------------------------------------------------------------------------
# Alias mappings for map_to_hmba()
# Maps common free-text labels to canonical HMBA taxonomy labels.
# ---------------------------------------------------------------------------
_LABEL_ALIASES: dict[str, str] = {
    # D1 MSNs
    "d1 msn": "D1_MSN_matrix",
    "d1_msn": "D1_MSN_matrix",
    "d1 medium spiny neuron": "D1_MSN_matrix",
    "d1-msn": "D1_MSN_matrix",
    "direct pathway msn": "D1_MSN_matrix",
    "d1 striosome": "D1_MSN_striosome",
    "d1_msn_striosome": "D1_MSN_striosome",
    "d1 patch": "D1_MSN_striosome",
    "d1 matrix": "D1_MSN_matrix",
    "d1_msn_matrix": "D1_MSN_matrix",
    # D2 MSNs
    "d2 msn": "D2_MSN_matrix",
    "d2_msn": "D2_MSN_matrix",
    "d2 medium spiny neuron": "D2_MSN_matrix",
    "d2-msn": "D2_MSN_matrix",
    "indirect pathway msn": "D2_MSN_matrix",
    "d2 striosome": "D2_MSN_striosome",
    "d2_msn_striosome": "D2_MSN_striosome",
    "d2 patch": "D2_MSN_striosome",
    "d2 matrix": "D2_MSN_matrix",
    "d2_msn_matrix": "D2_MSN_matrix",
    # Interneurons
    "cholinergic interneuron": "cholinergic_interneuron",
    "cholinergic_interneuron": "cholinergic_interneuron",
    "chat+": "cholinergic_interneuron",
    "chat interneuron": "cholinergic_interneuron",
    "tan": "cholinergic_interneuron",
    "pv interneuron": "pv_interneuron",
    "pv_interneuron": "pv_interneuron",
    "pv+": "pv_interneuron",
    "parvalbumin interneuron": "pv_interneuron",
    "fast-spiking interneuron": "pv_interneuron",
    "sst interneuron": "sst_interneuron",
    "sst_interneuron": "sst_interneuron",
    "sst+": "sst_interneuron",
    "somatostatin interneuron": "sst_interneuron",
    # GP neurons
    "gpe prototypic": "GPe_prototypic",
    "gpe_prototypic": "GPe_prototypic",
    "gpe arkypallidal": "GPe_arkypallidal",
    "gpe_arkypallidal": "GPe_arkypallidal",
    "gpi projection": "GPi_projection",
    "gpi_projection": "GPi_projection",
    "gpi": "GPi_projection",
    "gpe": "GPe_prototypic",
    # STN
    "stn glutamatergic": "STN_glutamatergic",
    "stn_glutamatergic": "STN_glutamatergic",
    "stn": "STN_glutamatergic",
    "subthalamic": "STN_glutamatergic",
    # Thalamic
    "thalamic relay": "thalamic_relay",
    "thalamic_relay": "thalamic_relay",
    "thalamic reticular": "thalamic_reticular",
    "thalamic_reticular": "thalamic_reticular",
    "trn": "thalamic_reticular",
    # Glia
    "astrocyte": "astrocyte",
    "astrocytes": "astrocyte",
    "oligodendrocyte": "oligodendrocyte",
    "oligodendrocytes": "oligodendrocyte",
    "opc": "OPC",
    "oligodendrocyte precursor": "OPC",
    "microglia": "microglia",
}

# ---------------------------------------------------------------------------
# Internal index (built lazily)
# ---------------------------------------------------------------------------
_taxonomy_index: dict[str, CellType] | None = None


def _build_index() -> dict[str, CellType]:
    """Build or load the taxonomy index (label → CellType)."""
    global _taxonomy_index
    if _taxonomy_index is not None:
        return _taxonomy_index

    # Try loading from disk first
    taxonomy_file = HMBA_DATA_DIR / "cell_type_taxonomy.csv"
    if taxonomy_file.exists():
        _taxonomy_index = _load_taxonomy_csv(taxonomy_file)
        logger.info(
            "Loaded HMBA taxonomy from disk: %d cell types", len(_taxonomy_index)
        )
    else:
        logger.info("HMBA data not found at %s; using built-in taxonomy", HMBA_DATA_DIR)
        _taxonomy_index = {ct.label: ct for ct in _BUILTIN_TAXONOMY}

    return _taxonomy_index


def _load_taxonomy_csv(path: Path) -> dict[str, CellType]:
    """Load taxonomy from an abc_atlas_access-exported CSV.

    Expected columns: hmba_id, label, cell_class, subclass, supertype,
    cell_type, region, markers (semicolon-separated), description.
    """
    index: dict[str, CellType] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            markers_raw = row.get("markers", "")
            markers = tuple(
                m.strip() for m in markers_raw.split(";") if m.strip()
            )
            ct = CellType(
                hmba_id=row.get("hmba_id", ""),
                label=row.get("label", ""),
                cell_class=row.get("cell_class", ""),
                subclass=row.get("subclass", ""),
                supertype=row.get("supertype", ""),
                cell_type=row.get("cell_type", ""),
                region=row.get("region", ""),
                markers=markers,
                description=row.get("description", ""),
            )
            index[ct.label] = ct
    return index


def _load_marker_overrides() -> dict[str, tuple[str, ...]]:
    """Load per-cell-type marker overrides from an abc_atlas_access marker table."""
    marker_file = HMBA_DATA_DIR / "marker_genes.json"
    if not marker_file.exists():
        return {}
    with open(marker_file) as f:
        data = json.load(f)
    return {
        k: tuple(v) if isinstance(v, list) else (v,)
        for k, v in data.items()
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_taxonomy() -> dict[str, CellType]:
    """Return the full HMBA BG taxonomy as ``{label: CellType}``."""
    return dict(_build_index())


def list_cell_types(
    *,
    cell_class: str | None = None,
    region: str | None = None,
    subclass: str | None = None,
) -> list[str]:
    """Return cell-type labels, optionally filtered by class/region/subclass."""
    index = _build_index()
    result: list[str] = []
    for label, ct in index.items():
        if cell_class and ct.cell_class != cell_class:
            continue
        if region and ct.region != region:
            continue
        if subclass and ct.subclass != subclass:
            continue
        result.append(label)
    return sorted(result)


def get_cell_type(label: str) -> CellType:
    """Return a single CellType by its canonical label.

    Raises
    ------
    KeyError
        If *label* is not found in the taxonomy.
    """
    index = _build_index()
    if label not in index:
        raise KeyError(
            f"Unknown HMBA cell type {label!r}. "
            f"Available: {', '.join(sorted(index.keys()))}"
        )
    return index[label]


def get_marker_panel(label: str) -> tuple[str, ...]:
    """Return the marker gene panel for a cell type.

    If a ``marker_genes.json`` override file exists in the HMBA data
    directory, its entries take precedence over built-in markers.
    """
    ct = get_cell_type(label)
    overrides = _load_marker_overrides()
    if label in overrides:
        return overrides[label]
    return ct.markers


def map_to_hmba(labels: list[str]) -> dict[str, str | None]:
    """Map free-text cell-type labels to canonical HMBA taxonomy labels.

    Returns a dict ``{input_label: hmba_label_or_None}``.  Unrecognised
    labels map to ``None``.

    Matching is case-insensitive and uses an alias table covering common
    alternative names (e.g. "D1 MSN" → "D1_MSN_matrix").
    """
    index = _build_index()
    result: dict[str, str | None] = {}
    for label in labels:
        normalised = label.strip().lower()
        # Direct match (case-insensitive against canonical labels)
        matched = None
        for canon in index:
            if canon.lower() == normalised:
                matched = canon
                break
        # Alias match
        if matched is None:
            matched = _LABEL_ALIASES.get(normalised)
        # Verify the matched label actually exists
        if matched is not None and matched not in index:
            matched = None
        result[label] = matched
    return result


def get_striosome_matrix_types() -> dict[str, list[CellType]]:
    """Return striosome vs matrix cell types for compartment analysis.

    Returns ``{"striosome": [...], "matrix": [...]}``.
    """
    index = _build_index()
    compartments: dict[str, list[CellType]] = {"striosome": [], "matrix": []}
    for ct in index.values():
        if "striosome" in ct.label.lower():
            compartments["striosome"].append(ct)
        elif "matrix" in ct.label.lower() and "MSN" in ct.label:
            compartments["matrix"].append(ct)
    return compartments


def get_cstc_region_cell_types(region: str) -> list[CellType]:
    """Return all cell types associated with a CSTC region.

    Parameters
    ----------
    region
        One of: striatum, GPe, GPi, STN, thalamus, basal_ganglia_wide.
    """
    index = _build_index()
    return sorted(
        [ct for ct in index.values() if ct.region == region],
        key=lambda ct: ct.label,
    )


def reset() -> None:
    """Reset the cached taxonomy index (for testing)."""
    global _taxonomy_index
    _taxonomy_index = None
