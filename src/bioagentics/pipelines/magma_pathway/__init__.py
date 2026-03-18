"""MAGMA gene-set analysis pipeline for pathway decomposition."""

from bioagentics.pipelines.magma_pathway.pipeline import (
    ALL_BUILTIN_GENE_SETS,
    BRAIN_CELL_TYPE_MARKERS,
    CSTC_GENE_SETS,
    GeneResult,
    GeneSetResult,
    NEUROTRANSMITTER_GENE_SETS,
    PathwayAnalysisResult,
    gene_analysis_snp_wise,
    gene_set_analysis,
    load_gene_annotations,
    load_gene_sets,
    map_snps_to_genes,
    run_pathway_analysis,
)

__all__ = [
    "ALL_BUILTIN_GENE_SETS",
    "BRAIN_CELL_TYPE_MARKERS",
    "CSTC_GENE_SETS",
    "GeneResult",
    "GeneSetResult",
    "NEUROTRANSMITTER_GENE_SETS",
    "PathwayAnalysisResult",
    "gene_analysis_snp_wise",
    "gene_set_analysis",
    "load_gene_annotations",
    "load_gene_sets",
    "map_snps_to_genes",
    "run_pathway_analysis",
]
