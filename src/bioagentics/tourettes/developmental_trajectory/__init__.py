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
  07_critical_period_modules — Phase 2: Critical period gene module analysis
  08_celltype_deconvolution  — Phase 3: Cell-type developmental dynamics
  09_persistence_remission_model — Phase 4: Persistence vs. remission model
  10_therapeutic_window_prediction — Phase 5: Therapeutic window prediction
"""
