"""mTOR convergence analysis for PANS variant genes.

Evaluates the mTOR signaling pathway as a convergence node connecting
infection -> inflammation -> CNS dysfunction in PANS. Tests whether PANS
variant genes or their PPI network neighbors cluster on mTOR-related pathways.

Reference: Fronticelli Baldelli G et al. 2025, PMID 41462744.

Usage:
    uv run python -m bioagentics.models.pans_mtor_convergence [--dest DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/pandas_pans/pans-genetic-variant-pathway-analysis")

# Core mTOR pathway genes from KEGG hsa04150 and Reactome mTOR signaling.
# Curated from public pathway databases.
MTOR_PATHWAY_GENES = {
    # Core mTOR complex components
    "MTOR", "RPTOR", "RICTOR", "MLST8", "DEPTOR", "PRR5", "PRR5L",
    "MAPKAP1",
    # Upstream regulators
    "TSC1", "TSC2", "RHEB", "AKT1", "AKT2", "AKT3", "PIK3CA", "PIK3CB",
    "PIK3CD", "PIK3CG", "PIK3R1", "PIK3R2", "PIK3R3", "PTEN", "PDK1",
    "PDPK1", "DDIT4", "STK11", "PRKAA1", "PRKAA2",
    # Downstream effectors
    "RPS6KB1", "RPS6KB2", "EIF4EBP1", "EIF4E", "RPS6", "EIF4B",
    "EIF4G1", "ULK1", "ULK2", "ATG13", "RB1CC1",
    # Autophagy connection (links to IVIG initiative ATG7/UVRAG)
    "ATG7", "UVRAG", "ATG5", "ATG12", "BECN1", "PIK3C3", "ATG14",
    "AMBRA1", "SQSTM1",
    # Immune regulation via mTOR
    "HIF1A", "STAT3", "FOXP3", "NFKB1", "RELA",
}

# Autophagy genes from IVIG initiative for cross-reference
IVIG_AUTOPHAGY_GENES = {"ATG7", "UVRAG"}


def compute_mtor_overlap(pans_genes: list[str]) -> dict:
    """Compute direct overlap between PANS variant genes and mTOR pathway."""
    pans_set = set(pans_genes)
    direct_overlap = pans_set & MTOR_PATHWAY_GENES

    return {
        "direct_overlap_genes": sorted(direct_overlap),
        "direct_overlap_count": len(direct_overlap),
        "pans_gene_count": len(pans_set),
        "mtor_pathway_size": len(MTOR_PATHWAY_GENES),
    }


def compute_ppi_mtor_connections(pans_genes: list[str],
                                  ppi_network_path: Path | None = None) -> dict:
    """Check if PPI neighbors of PANS genes are in mTOR pathway.

    If PPI network JSON exists, loads it; otherwise returns empty results.
    """
    if ppi_network_path is None:
        ppi_network_path = OUTPUT_DIR / "ppi_network.json"

    if not ppi_network_path.exists():
        logger.warning("PPI network not found at %s, skipping neighbor analysis",
                       ppi_network_path)
        return {"ppi_mtor_connections": [], "connected_via": {}}

    with open(ppi_network_path) as f:
        network_data = json.load(f)

    # Build adjacency from edges
    adjacency: dict[str, set[str]] = {}
    for edge in network_data.get("edges", []):
        src, tgt = edge["source"], edge["target"]
        adjacency.setdefault(src, set()).add(tgt)
        adjacency.setdefault(tgt, set()).add(src)

    # Find PANS genes whose PPI neighbors are in mTOR pathway
    connections = []
    connected_via: dict[str, list[str]] = {}
    pans_set = set(pans_genes)

    for gene in pans_set:
        neighbors = adjacency.get(gene, set())
        mtor_neighbors = neighbors & MTOR_PATHWAY_GENES
        if mtor_neighbors:
            connections.append({
                "pans_gene": gene,
                "mtor_neighbors": sorted(mtor_neighbors),
                "neighbor_count": len(mtor_neighbors),
            })
            connected_via[gene] = sorted(mtor_neighbors)

    return {
        "ppi_mtor_connections": connections,
        "connected_via": connected_via,
    }


def compute_mtor_enrichment(pans_genes: list[str],
                             de_df: pd.DataFrame | None = None) -> dict:
    """Test enrichment of mTOR pathway genes in neuroinflammation DE results.

    If DE data is available, checks overlap between mTOR pathway and DE genes.
    """
    if de_df is None or de_df.empty:
        return {"mtor_de_overlap": [], "enrichment_tested": False}

    all_genes = set(de_df["gene_symbol"])
    de_genes = set(de_df[de_df["padj"] < 0.05]["gene_symbol"])
    mtor_detected = MTOR_PATHWAY_GENES & all_genes
    mtor_de = MTOR_PATHWAY_GENES & de_genes

    if not mtor_detected:
        return {"mtor_de_overlap": [], "enrichment_tested": False}

    # Fisher exact test: mTOR genes vs DE
    a = len(mtor_de)
    b = len(mtor_detected - de_genes)
    c = len(de_genes - mtor_detected)
    d = len(all_genes - mtor_detected - de_genes)

    odds_ratio, p_value = scipy_stats.fisher_exact([[a, b], [c, d]])

    return {
        "mtor_de_overlap": sorted(mtor_de),
        "mtor_de_count": a,
        "mtor_detected_count": len(mtor_detected),
        "total_de_count": len(de_genes),
        "odds_ratio": odds_ratio,
        "fisher_pvalue": p_value,
        "enrichment_tested": True,
    }


def generate_convergence_diagram(results: dict, dest: Path) -> None:
    """Generate a pathway convergence diagram showing PANS-mTOR connections."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw conceptual pathway boxes
    boxes = {
        "PANS Variant\nGenes": (0.1, 0.7),
        "DDR / cGAS-STING": (0.1, 0.4),
        "mTOR Signaling": (0.5, 0.5),
        "Autophagy\n(ATG7, UVRAG)": (0.9, 0.7),
        "Immune\nRegulation": (0.5, 0.2),
        "Neuroinflammation": (0.9, 0.3),
    }

    for label, (x, y) in boxes.items():
        color = "#E8E8E8"
        if "mTOR" in label:
            color = "#FFD700"
        elif "PANS" in label:
            color = "#E74C3C"
        elif "Autophagy" in label:
            color = "#3498DB"
        elif "Immune" in label:
            color = "#2ECC71"
        elif "Neuroinflammation" in label:
            color = "#9B59B6"

        ax.add_patch(plt.Rectangle((x - 0.08, y - 0.06), 0.16, 0.12,
                                    facecolor=color, edgecolor="black",
                                    linewidth=1.5, alpha=0.7,
                                    transform=ax.transAxes))
        ax.text(x, y, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, fontweight="bold")

    # Draw arrows
    arrows = [
        ((0.18, 0.7), (0.42, 0.55)),   # PANS -> mTOR
        ((0.18, 0.4), (0.42, 0.48)),   # DDR -> mTOR
        ((0.58, 0.55), (0.82, 0.7)),   # mTOR -> Autophagy
        ((0.5, 0.38), (0.5, 0.26)),    # mTOR -> Immune
        ((0.58, 0.22), (0.82, 0.3)),   # Immune -> Neuroinflammation
        ((0.1, 0.64), (0.1, 0.46)),    # PANS -> DDR
    ]

    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # Add statistics
    direct = results.get("direct_overlap", {})
    stats_text = (
        f"Direct PANS-mTOR overlap: {direct.get('direct_overlap_count', 0)} genes\n"
        f"PPI-mediated connections: {len(results.get('ppi_connections', {}).get('connected_via', {}))}\n"
        f"IVIG autophagy link: ATG7, UVRAG"
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=8, verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("mTOR as Convergence Node: Infection → Inflammation → CNS Dysfunction",
                 fontsize=12, fontweight="bold")

    plt.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved convergence diagram: %s", dest)


def run_mtor_convergence(dest_dir: Path | None = None) -> dict:
    """Run the full mTOR convergence analysis.

    Returns dict with all analysis results.
    """
    from bioagentics.data.pans_variants import get_pans_gene_symbols

    if dest_dir is None:
        dest_dir = OUTPUT_DIR

    dest_dir.mkdir(parents=True, exist_ok=True)

    pans_genes = get_pans_gene_symbols()

    # Direct overlap
    direct_overlap = compute_mtor_overlap(pans_genes)
    logger.info("Direct PANS-mTOR overlap: %d genes — %s",
                direct_overlap["direct_overlap_count"],
                direct_overlap["direct_overlap_genes"])

    # PPI-mediated connections
    ppi_connections = compute_ppi_mtor_connections(pans_genes,
                                                   dest_dir / "ppi_network.json")

    # DE enrichment (if available)
    de_df = None
    de_path = Path("data/pandas_pans/pans-genetic-variant-pathway-analysis") / "gse102482_de_results.csv"
    if de_path.exists():
        de_df = pd.read_csv(de_path)
    mtor_enrichment = compute_mtor_enrichment(pans_genes, de_df)

    results = {
        "direct_overlap": direct_overlap,
        "ppi_connections": ppi_connections,
        "mtor_de_enrichment": mtor_enrichment,
        "ivig_autophagy_link": {
            "genes": sorted(IVIG_AUTOPHAGY_GENES),
            "in_mtor_pathway": True,
            "note": "ATG7 and UVRAG from IVIG initiative connect via mTOR-autophagy axis",
        },
    }

    # Save results
    results_path = dest_dir / "mtor_convergence.csv"
    summary_rows = []

    for gene in direct_overlap["direct_overlap_genes"]:
        summary_rows.append({
            "gene_symbol": gene,
            "connection_type": "direct_overlap",
            "details": "PANS variant gene in mTOR pathway",
        })

    for conn in ppi_connections.get("ppi_mtor_connections", []):
        summary_rows.append({
            "gene_symbol": conn["pans_gene"],
            "connection_type": "ppi_neighbor",
            "details": f"PPI neighbors in mTOR: {', '.join(conn['mtor_neighbors'])}",
        })

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(results_path, index=False)
    else:
        pd.DataFrame(columns=["gene_symbol", "connection_type", "details"]).to_csv(
            results_path, index=False)

    logger.info("Saved mTOR convergence results: %s", results_path)

    # Generate diagram
    generate_convergence_diagram(results, dest_dir / "mtor_convergence_diagram.png")

    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="mTOR convergence analysis for PANS variant genes"
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = run_mtor_convergence(dest_dir=args.dest)

    direct = results["direct_overlap"]
    print(f"\nmTOR Convergence Analysis:")
    print(f"  Direct overlap: {direct['direct_overlap_count']} genes "
          f"({direct['direct_overlap_genes']})")

    ppi = results["ppi_connections"]
    print(f"  PPI-mediated connections: {len(ppi.get('ppi_mtor_connections', []))}")

    mtor_de = results["mtor_de_enrichment"]
    if mtor_de.get("enrichment_tested"):
        print(f"  mTOR genes in neuroinflammation DE: {mtor_de['mtor_de_count']}")
        print(f"  Enrichment p-value: {mtor_de['fisher_pvalue']:.4f}")


if __name__ == "__main__":
    main()
