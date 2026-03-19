"""TS Developmental Trajectory Modeling — project pipeline.

Models the neurodevelopmental trajectory of Tourette syndrome to identify
molecular and circuit-level factors underlying tic onset, peak severity,
and spontaneous remission.

Pipeline steps:
  01_download_brainspan      — Download and validate BrainSpan RNA-seq data
  02_curate_gene_lists       — Assemble and validate TS developmental gene lists
  03_expression_trajectories — Extract and visualize expression trajectories
  04_temporal_clustering     — Cluster genes by developmental expression pattern
  05_enrichment_testing      — Test TS gene enrichment in temporal clusters
  06_wgcna_brainspan         — WGCNA co-expression analysis on BrainSpan data
"""
