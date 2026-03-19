"""Mimicry-HLA cross-reference integration module.

Cross-references differentially presented GAS peptides with molecular mimicry
candidates from gas-molecular-mimicry-mapping project. Identifies peptides that
are BOTH mimicry hits AND differentially presented by susceptibility HLA alleles,
linking the genetic susceptibility → peptide presentation → molecular mimicry →
autoimmune target pathway.

Usage:
    uv run python -m bioagentics.pipelines.mimicry_hla_integration [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from scipy import stats as scipy_stats

from bioagentics.config import DATA_DIR

logger = logging.getLogger(__name__)

MIMICRY_DIR = DATA_DIR / "pandas_pans" / "gas-molecular-mimicry-mapping" / "mimicry_screen"
HLA_DIR = DATA_DIR / "pandas_pans" / "hla-peptide-presentation"
DIFF_DIR = HLA_DIR / "differential_analysis"
PREDICTIONS_DIR = HLA_DIR / "binding_predictions"
OUTPUT_DIR = HLA_DIR / "mimicry_integration"

# Known PANDAS autoantibody targets (gene symbols)
PANDAS_TARGETS = {
    "DRD1", "DRD2",          # Dopamine receptors
    "TUBA1A", "TUBA1B", "TUBB", "TUBB2A", "TUBB3",  # Tubulins
    "CAMK2A", "CAMK2B", "CAMK2D", "CAMK2G",          # CaMKII isoforms
    "GAPDH",                  # Glycolytic enzyme (glyceraldehyde-3-P dehydrogenase)
    "ENO1", "ENO2", "ENO3",   # Enolase (glycolytic enzyme)
    "PKM", "ALDOA", "ALDOB",  # Other glycolytic enzymes
    "B4GALNT1",               # GM1 ganglioside synthase
}


def _extract_accession(seqid: str) -> str:
    """Extract UniProt accession from FASTA-style sequence ID.

    Handles: 'sp|Q1JKD6|DNAK_STRPC', 'tr|A0A0H2UWC3|A0A0H2UWC3_STRP3',
    'UPI0000165A3D', plain accessions like 'Q1JK23'.
    """
    # sp|ACC|NAME or tr|ACC|NAME format
    m = re.match(r"(?:sp|tr)\|([^|]+)\|", seqid)
    if m:
        return m.group(1)
    # UPI (UniParc) identifiers — keep as-is
    if seqid.startswith("UPI"):
        return seqid
    return seqid


def load_mimicry_hits(hits_path: Path) -> list[dict]:
    """Load filtered mimicry hits TSV."""
    rows = []
    with open(hits_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["gas_accession"] = _extract_accession(row["qseqid"])
            rows.append(row)
    return rows


def load_differential_peptides(diff_path: Path) -> list[dict]:
    """Load differential peptides TSV."""
    rows = []
    with open(diff_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def load_binding_predictions(pred_path: Path) -> list[dict]:
    """Load binding predictions TSV (all peptides, not just differential)."""
    rows = []
    with open(pred_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["percentile_rank"] = float(row["percentile_rank"])
            rows.append(row)
    return rows


def build_mimicry_protein_map(hits: list[dict]) -> dict[str, list[dict]]:
    """Map GAS protein accessions to their human mimicry targets."""
    protein_map: dict[str, list[dict]] = defaultdict(list)
    for hit in hits:
        acc = hit["gas_accession"]
        protein_map[acc].append({
            "human_accession": hit.get("human_accession", ""),
            "human_gene": hit.get("human_gene", ""),
            "known_target": hit.get("known_target", "False").lower() == "true",
            "pident": float(hit.get("pident", 0)),
            "evalue": float(hit.get("evalue", 1)),
            "qstart": int(hit.get("qstart", 0)),
            "qend": int(hit.get("qend", 0)),
        })
    return dict(protein_map)


def cross_reference(
    diff_peptides: list[dict],
    binding_predictions: list[dict],
    mimicry_map: dict[str, list[dict]],
) -> dict:
    """Cross-reference differential peptides with mimicry hits.

    Returns structured results with overlap analysis.
    """
    # All GAS accessions with mimicry hits
    mimicry_accessions = set(mimicry_map.keys())

    # Build set of accessions from differential peptides
    diff_accessions = {row["source_accession"] for row in diff_peptides}

    # Build map of accession -> best binding peptides from all predictions
    acc_to_peptides: dict[str, list[dict]] = defaultdict(list)
    for pred in binding_predictions:
        acc = pred.get("source_accession", "")
        acc_to_peptides[acc].append(pred)

    # Find overlapping proteins (both mimicry + have binding predictions)
    overlap_accessions = mimicry_accessions & set(acc_to_peptides.keys())
    diff_overlap = diff_accessions & mimicry_accessions

    # For each overlapping protein, find its best-binding peptides to susceptibility alleles
    overlap_details = []
    susceptibility_alleles = {"DRB1*07:01", "DRB1*04:01", "DRB1*01:01"}

    for acc in sorted(overlap_accessions):
        mimicry_targets = mimicry_map[acc]
        peptides = acc_to_peptides.get(acc, [])

        # Find strong binders to susceptibility alleles
        strong_binders = []
        for p in peptides:
            allele = p.get("allele", "").replace("HLA-", "")
            if allele in susceptibility_alleles and p["percentile_rank"] < 2.0:
                strong_binders.append({
                    "peptide": p["peptide"],
                    "allele": allele,
                    "rank": p["percentile_rank"],
                    "source_protein": p.get("source_protein", ""),
                    "serotype": p.get("serotype", ""),
                })

        human_genes = list({t["human_gene"] for t in mimicry_targets if t["human_gene"]})
        human_accessions = list({t["human_accession"] for t in mimicry_targets})
        is_known_target = any(t["known_target"] for t in mimicry_targets)
        is_pandas_target = any(g in PANDAS_TARGETS for g in human_genes)

        overlap_details.append({
            "gas_accession": acc,
            "gas_protein": peptides[0].get("source_protein", "") if peptides else "",
            "human_genes": human_genes,
            "human_accessions": human_accessions,
            "is_known_autoantibody_target": is_known_target,
            "is_pandas_target": is_pandas_target,
            "n_binding_predictions": len(peptides),
            "n_strong_binders_susceptibility": len(strong_binders),
            "has_differential_peptides": acc in diff_accessions,
            "best_binders": sorted(strong_binders, key=lambda x: x["rank"])[:5],
            "best_mimicry_pident": max((t["pident"] for t in mimicry_targets), default=0),
            "best_mimicry_evalue": min((t["evalue"] for t in mimicry_targets), default=1),
        })

    # Sort: PANDAS targets first, then by number of strong binders
    overlap_details.sort(
        key=lambda x: (-x["is_pandas_target"], -x["n_strong_binders_susceptibility"]),
    )

    return {
        "total_mimicry_proteins": len(mimicry_accessions),
        "total_proteins_with_predictions": len(acc_to_peptides),
        "overlap_count": len(overlap_accessions),
        "diff_overlap_count": len(diff_overlap),
        "diff_overlap_accessions": sorted(diff_overlap),
        "overlap_details": overlap_details,
    }


def overlap_significance_test(
    diff_peptides: list[dict],
    binding_predictions: list[dict],
    mimicry_map: dict[str, list[dict]],
) -> dict:
    """Fisher's exact test: is overlap between differential presenters and mimicry
    candidates statistically significant?

    Tests at the protein level: among all GAS proteins with binding predictions,
    are proteins that are mimicry hits enriched among those producing differential peptides?

    Contingency table:
                        Differential    Not differential
    Mimicry hit         a               b
    Not mimicry hit     c               d
    """
    mimicry_accessions = set(mimicry_map.keys())
    diff_accessions = {row["source_accession"] for row in diff_peptides}
    all_predicted = {row.get("source_accession", "") for row in binding_predictions}

    a = len(diff_accessions & mimicry_accessions)
    b = len(mimicry_accessions & all_predicted - diff_accessions)
    c = len(diff_accessions - mimicry_accessions)
    d = len(all_predicted - diff_accessions - mimicry_accessions)

    table = [[a, b], [c, d]]

    if sum(sum(r) for r in table) == 0:
        return {"odds_ratio": None, "p_value": None, "table": table,
                "significant": False, "note": "No data"}

    import numpy as np
    fe_result = scipy_stats.fisher_exact(np.array(table), alternative="greater")
    odds_ratio = float(fe_result[0])  # type: ignore[arg-type]
    p_value = float(fe_result[1])  # type: ignore[arg-type]

    return {
        "odds_ratio": odds_ratio if not np.isinf(odds_ratio) else None,
        "p_value": p_value,
        "table": table,
        "table_labels": ["[mimicry+diff, mimicry+non-diff]",
                         "[non-mimicry+diff, non-mimicry+non-diff]"],
        "significant": p_value < 0.05,
        "mimicry_differential": a,
        "mimicry_non_differential": b,
        "non_mimicry_differential": c,
        "non_mimicry_non_differential": d,
    }


def build_network_edges(overlap_details: list[dict]) -> list[dict]:
    """Build network edges: susceptibility allele -> GAS peptide -> human mimic -> autoimmune target.

    Returns list of edge dicts for downstream visualization.
    """
    edges = []
    for detail in overlap_details:
        if not detail["best_binders"]:
            continue
        for binder in detail["best_binders"]:
            for gene in detail["human_genes"]:
                edges.append({
                    "hla_allele": binder["allele"],
                    "gas_peptide": binder["peptide"],
                    "gas_protein": detail["gas_protein"] or detail["gas_accession"],
                    "gas_accession": detail["gas_accession"],
                    "serotype": binder.get("serotype", ""),
                    "binding_rank": binder["rank"],
                    "human_gene": gene,
                    "human_accession": detail["human_accessions"][0] if detail["human_accessions"] else "",
                    "is_pandas_target": gene in PANDAS_TARGETS,
                    "mimicry_pident": detail["best_mimicry_pident"],
                    "is_differential": detail["has_differential_peptides"],
                })
    return edges


def write_network_tsv(edges: list[dict], output_path: Path) -> None:
    """Write network edges to TSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not edges:
        logger.warning("No network edges to write")
        return
    fieldnames = list(edges[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for edge in edges:
            writer.writerow(edge)


def run_analysis(
    mimicry_path: Path,
    diff_path: Path,
    predictions_path: Path,
    output_dir: Path,
) -> dict:
    """Run full mimicry-HLA integration analysis. Returns summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading mimicry hits from %s", mimicry_path)
    mimicry_hits = load_mimicry_hits(mimicry_path)
    logger.info("Loaded %d mimicry hits", len(mimicry_hits))

    logger.info("Loading differential peptides from %s", diff_path)
    diff_peptides = load_differential_peptides(diff_path)
    logger.info("Loaded %d differential peptides", len(diff_peptides))

    logger.info("Loading binding predictions from %s", predictions_path)
    predictions = load_binding_predictions(predictions_path)
    logger.info("Loaded %d binding predictions", len(predictions))

    # Build mimicry protein map
    mimicry_map = build_mimicry_protein_map(mimicry_hits)
    logger.info("Unique GAS proteins with mimicry hits: %d", len(mimicry_map))

    # Cross-reference
    xref = cross_reference(diff_peptides, predictions, mimicry_map)
    logger.info("Overlap: %d GAS proteins appear in both mimicry and binding datasets",
                xref["overlap_count"])

    # Significance test
    sig_test = overlap_significance_test(diff_peptides, predictions, mimicry_map)
    logger.info("Overlap significance: OR=%s, p=%s, significant=%s",
                sig_test.get("odds_ratio"), sig_test.get("p_value"),
                sig_test.get("significant"))

    # Build network
    edges = build_network_edges(xref["overlap_details"])
    network_path = output_dir / "hla_mimicry_network.tsv"
    write_network_tsv(edges, network_path)
    logger.info("Wrote %d network edges to %s", len(edges), network_path)

    # Identify PANDAS-relevant findings
    pandas_hits = [d for d in xref["overlap_details"] if d["is_pandas_target"]]
    pandas_with_binders = [d for d in pandas_hits if d["n_strong_binders_susceptibility"] > 0]

    # Summary
    summary = {
        "mimicry_hits_total": len(mimicry_hits),
        "unique_mimicry_proteins": len(mimicry_map),
        "differential_peptides_count": len(diff_peptides),
        "binding_predictions_count": len(predictions),
        "overlap": {
            "proteins_in_both": xref["overlap_count"],
            "differential_and_mimicry": xref["diff_overlap_count"],
            "differential_overlap_accessions": xref["diff_overlap_accessions"],
        },
        "significance_test": sig_test,
        "network_edges": len(edges),
        "pandas_target_analysis": {
            "pandas_targets_in_mimicry": len(pandas_hits),
            "pandas_targets_with_susceptibility_binders": len(pandas_with_binders),
            "details": [
                {
                    "gas_accession": d["gas_accession"],
                    "gas_protein": d["gas_protein"],
                    "human_genes": d["human_genes"],
                    "strong_binders": d["n_strong_binders_susceptibility"],
                    "best_pident": d["best_mimicry_pident"],
                }
                for d in pandas_hits
            ],
        },
        "all_overlap_proteins": [
            {
                "gas_accession": d["gas_accession"],
                "gas_protein": d["gas_protein"],
                "human_genes": d["human_genes"],
                "is_pandas_target": d["is_pandas_target"],
                "strong_binders": d["n_strong_binders_susceptibility"],
                "has_differential": d["has_differential_peptides"],
            }
            for d in xref["overlap_details"]
        ],
        "notes": [
            "Current analysis uses limited binding prediction data (IEDB API test batch).",
            "Full analysis requires ~500K peptides predicted against all alleles.",
            "Network edges link: HLA allele -> GAS peptide -> human mimic protein.",
            "PANDAS targets checked: DRD1/2, tubulin, CaMKII, GAPDH, enolase, GM1 synthase.",
        ],
    }

    summary_path = output_dir / "integration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Wrote integration summary to %s", summary_path)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Mimicry-HLA cross-reference integration for PANDAS research",
    )
    parser.add_argument("--mimicry-hits", type=Path,
                        default=MIMICRY_DIR / "hits_filtered.tsv",
                        help="Filtered mimicry hits TSV (default: %(default)s)")
    parser.add_argument("--differential", type=Path,
                        default=DIFF_DIR / "differential_peptides.tsv",
                        help="Differential peptides TSV (default: %(default)s)")
    parser.add_argument("--predictions", type=Path,
                        default=PREDICTIONS_DIR / "binding_predictions_mhc_ii.tsv",
                        help="Binding predictions TSV (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_analysis(args.mimicry_hits, args.differential, args.predictions,
                           args.output_dir)

    print(f"\nMimicry-HLA integration complete.")
    print(f"Overlap: {summary['overlap']['proteins_in_both']} proteins in both datasets")
    print(f"PANDAS targets with susceptibility binders: "
          f"{summary['pandas_target_analysis']['pandas_targets_with_susceptibility_binders']}")
    print(f"Network edges: {summary['network_edges']}")


if __name__ == "__main__":
    main()
