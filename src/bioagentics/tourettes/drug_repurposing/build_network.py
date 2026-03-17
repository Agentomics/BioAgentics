"""Build TS disease PPI network with functional module detection.

Using TS seed genes and the human PPI network:
1. Seed with TS risk genes
2. Extend by first-degree interactors from STRING/BioGRID
3. Run Louvain clustering to identify densely connected functional modules
4. Annotate modules with pathway enrichment

Output:
- ts_disease_network.graphml (NetworkX graph)
- ts_network_modules.csv (gene -> module assignment)

Usage:
    uv run python -m bioagentics.tourettes.drug_repurposing.build_network
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import community as community_louvain
import networkx as nx
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.tourettes.drug_repurposing.seed_genes import compile_seed_genes

DATA_DIR = REPO_ROOT / "data" / "tourettes" / "ts-drug-repurposing-network"
OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-drug-repurposing-network"

PPI_PATH = DATA_DIR / "human_ppi_network.tsv"
NETWORK_OUTPUT = OUTPUT_DIR / "ts_disease_network.graphml"
MODULES_OUTPUT = OUTPUT_DIR / "ts_network_modules.csv"

# TS-relevant pathway annotations for module labeling
TS_PATHWAY_GENES: dict[str, set[str]] = {
    "dopamine_signaling": {
        "DRD1", "DRD2", "DRD3", "DRD4", "DRD5", "SLC6A3", "TH", "DDC",
        "COMT", "MAOA", "MAOB", "SLC18A2", "PPP1R1B",
    },
    "serotonin_signaling": {
        "HTR1A", "HTR1B", "HTR2A", "HTR2C", "HTR3A", "HTR4", "HTR7",
        "SLC6A4", "TPH1", "TPH2",
    },
    "GABAergic_transmission": {
        "GAD1", "GAD2", "GABRA1", "GABRA2", "GABRB1", "GABRB2", "GABRG2",
        "SLC6A1", "SLC32A1", "PVALB", "SST", "NPY",
    },
    "glutamatergic_transmission": {
        "GRIN1", "GRIN2A", "GRIN2B", "GRIA1", "GRIA2", "GRM1", "GRM5",
        "SLC1A2", "SLC1A3", "SLC17A7", "SLC17A6",
    },
    "cholinergic_signaling": {
        "CHRM1", "CHRM4", "CHAT", "SLC5A7", "ACHE", "CHRNA4", "CHRNB2",
    },
    "PDE_cAMP_cGMP": {
        "PDE10A", "PDE1B", "PDE4B", "PDE4D", "ADCY5", "PRKG1",
        "PRKACA", "CREB1",
    },
    "histamine_signaling": {
        "HRH3", "HDC", "HNMT",
    },
    "endocannabinoid_system": {
        "CNR1", "CNR2", "FAAH", "MGLL", "DAGLA", "NAPEPLD",
    },
    "synaptic_adhesion_axon_guidance": {
        "NRXN1", "NLGN1", "NLGN2", "SLITRK1", "CNTN6", "CNTNAP2",
        "SEMA6D", "NTN4", "DCC", "ROBO1",
    },
    "immune_neuroinflammation": {
        "TNF", "IL1B", "IL6", "IL12A", "IL12B", "IL23A", "TGFB1",
        "JAK1", "JAK2", "JAK3", "STAT3", "CCL2", "CCR2", "IL1RN",
        "NFKB1", "TLR4", "C4A", "C4B",
    },
    "striatal_interneuron_markers": {
        "PVALB", "SST", "NPY", "CHAT", "NOS1", "CALB1", "TAC1", "PENK",
    },
}


def load_ppi_network(ppi_path: Path) -> nx.Graph:
    """Load PPI edge list into NetworkX graph."""
    print(f"Loading PPI network from {ppi_path}...")
    df = pd.read_csv(ppi_path, sep="\t")
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(
            row["protein_a"],
            row["protein_b"],
            source=row["source"],
            weight=row["confidence_score"],
        )
    print(f"  Full PPI: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def build_ts_disease_network(
    ppi: nx.Graph,
    seed_genes: set[str],
    extend_first_degree: bool = True,
) -> nx.Graph:
    """Build TS disease subnetwork from seed genes.

    1. Keep seed genes present in PPI
    2. Optionally extend with first-degree interactors
    3. Extract the induced subgraph
    """
    # Find seed genes present in PPI
    seeds_in_ppi = seed_genes & set(ppi.nodes())
    missing = seed_genes - seeds_in_ppi
    print(f"  Seed genes in PPI: {len(seeds_in_ppi)}/{len(seed_genes)}")
    if missing:
        print(f"  Missing from PPI ({len(missing)}): {', '.join(sorted(missing)[:10])}...")

    disease_nodes = set(seeds_in_ppi)

    if extend_first_degree:
        # Add first-degree interactors of seed genes
        for gene in seeds_in_ppi:
            neighbors = set(ppi.neighbors(gene))
            disease_nodes.update(neighbors)
        print(f"  After first-degree extension: {len(disease_nodes)} nodes")

    # Extract induced subgraph
    G = ppi.subgraph(disease_nodes).copy()

    # Mark seed genes
    for node in G.nodes():
        G.nodes[node]["is_seed"] = node in seed_genes

    print(f"  TS disease network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def detect_modules(G: nx.Graph, resolution: float = 1.0) -> dict[str, int]:
    """Run Louvain community detection on the disease network.

    Returns dict: gene -> module_id
    """
    print(f"  Running Louvain clustering (resolution={resolution})...")
    partition = community_louvain.best_partition(G, weight="weight", resolution=resolution)
    n_modules = len(set(partition.values()))
    print(f"  Detected {n_modules} modules")

    # Module size distribution
    sizes = Counter(partition.values())
    for mod_id, size in sizes.most_common(10):
        print(f"    Module {mod_id}: {size} genes")
    if n_modules > 10:
        print(f"    ... and {n_modules - 10} smaller modules")

    return partition


def annotate_modules(
    partition: dict[str, int],
    seed_genes: set[str],
) -> pd.DataFrame:
    """Annotate each gene with module ID and pathway membership."""
    rows = []
    for gene, module_id in partition.items():
        # Find pathway annotations
        pathways = []
        for pathway_name, pathway_genes in TS_PATHWAY_GENES.items():
            if gene in pathway_genes:
                pathways.append(pathway_name)

        rows.append({
            "gene": gene,
            "module_id": module_id,
            "is_seed": gene in seed_genes,
            "pathway_annotations": ";".join(pathways) if pathways else "",
        })

    df = pd.DataFrame(rows).sort_values(["module_id", "gene"]).reset_index(drop=True)
    return df


def label_modules(modules_df: pd.DataFrame) -> dict[int, str]:
    """Generate descriptive labels for modules based on pathway enrichment."""
    labels: dict[int, str] = {}
    for mod_id in modules_df["module_id"].unique():
        mod_genes = modules_df[modules_df["module_id"] == mod_id]
        # Count pathway annotations
        pathway_counts: Counter[str] = Counter()
        for pathways_str in mod_genes["pathway_annotations"]:
            if pathways_str:
                for p in pathways_str.split(";"):
                    pathway_counts[p] += 1

        n_seeds = mod_genes["is_seed"].sum()
        n_total = len(mod_genes)

        if pathway_counts:
            top_pathway = pathway_counts.most_common(1)[0][0]
            labels[mod_id] = f"{top_pathway} ({n_seeds} seeds / {n_total} total)"
        else:
            labels[mod_id] = f"module_{mod_id} ({n_seeds} seeds / {n_total} total)"

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TS disease PPI network")
    parser.add_argument("--ppi", type=Path, default=PPI_PATH)
    parser.add_argument("--resolution", type=float, default=1.0,
                        help="Louvain resolution parameter")
    parser.add_argument("--no-extend", action="store_true",
                        help="Don't extend with first-degree interactors")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    ppi = load_ppi_network(args.ppi)
    seed_genes_list = compile_seed_genes()
    seed_set = {g["gene_symbol"] for g in seed_genes_list}

    # Build disease network
    G = build_ts_disease_network(ppi, seed_set, extend_first_degree=not args.no_extend)

    # Detect modules
    partition = detect_modules(G, resolution=args.resolution)

    # Store module IDs as node attributes
    for node, mod_id in partition.items():
        G.nodes[node]["module_id"] = mod_id

    # Save network
    nx.write_graphml(G, str(NETWORK_OUTPUT))
    print(f"  Saved network to {NETWORK_OUTPUT}")

    # Save module assignments
    modules_df = annotate_modules(partition, seed_set)
    modules_df.to_csv(MODULES_OUTPUT, index=False)
    print(f"  Saved module assignments to {MODULES_OUTPUT}")

    # Module labels
    labels = label_modules(modules_df)
    print("\nModule labels:")
    for mod_id, label in sorted(labels.items()):
        print(f"  {mod_id}: {label}")

    # Key stats
    seed_coverage = sum(1 for g in seed_set if g in partition) / len(seed_set)
    print(f"\nSeed gene coverage: {seed_coverage:.1%}")
    print(f"Network density: {nx.density(G):.6f}")

    # Connected components
    components = list(nx.connected_components(G))
    print(f"Connected components: {len(components)}")
    print(f"Largest component: {len(components[0])} nodes")


if __name__ == "__main__":
    main()
