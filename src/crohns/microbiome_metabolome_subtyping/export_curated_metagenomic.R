#!/usr/bin/env Rscript
# Export species abundance tables and metadata from curatedMetagenomicData.
#
# Usage: Rscript export_curated_metagenomic.R <study_name> <output_dir>
#
# Outputs:
#   <output_dir>/<study_name>_species.tsv   - species abundance matrix (samples x species)
#   <output_dir>/<study_name>_metadata.tsv  - sample metadata with diagnosis column

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
  cat("Usage: Rscript export_curated_metagenomic.R <study_name> <output_dir>\n",
      file = stderr())
  quit(status = 1)
}

study_name <- args[1]
output_dir <- args[2]

# Ensure output directory exists
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# Load library (suppress startup messages to keep output clean)
suppressPackageStartupMessages({
  library(curatedMetagenomicData)
  library(SummarizedExperiment)
  library(TreeSummarizedExperiment)
})

cat(sprintf("Fetching %s from curatedMetagenomicData...\n", study_name))

# Build the query pattern: "StudyName.relative_abundance"
query_pattern <- paste0(study_name, ".relative_abundance")

# Retrieve the dataset — returns a named list of TreeSummarizedExperiment objects
tryCatch({
  dataset_list <- curatedMetagenomicData(query_pattern, dryrun = FALSE)
}, error = function(e) {
  cat(sprintf("ERROR: Could not retrieve %s: %s\n", study_name, e$message),
      file = stderr())
  quit(status = 1)
})

if (length(dataset_list) == 0) {
  cat(sprintf("ERROR: No data found for pattern '%s'\n", query_pattern),
      file = stderr())
  quit(status = 1)
}

# Take the first (typically only) element
se <- dataset_list[[1]]
cat(sprintf("Retrieved: %d features x %d samples\n", nrow(se), ncol(se)))

# --- Extract species-level abundance ---
# curatedMetagenomicData provides taxonomic profiles at all levels.
# Filter to species level (rows containing "s__" but not "t__" strain level).
tax_names <- rownames(se)
species_mask <- grepl("s__", tax_names) & !grepl("t__", tax_names)
se_species <- se[species_mask, ]
cat(sprintf("Species-level features: %d\n", nrow(se_species)))

# Extract abundance matrix: samples as rows, species as columns
abundance <- t(assay(se_species))

# Clean species names: extract just the "s__Species_name" part
raw_names <- colnames(abundance)
clean_names <- sub(".*\\|s__", "s__", raw_names)
colnames(abundance) <- clean_names

# Convert to data.frame for writing
abundance_df <- as.data.frame(abundance)
abundance_df <- cbind(sample_id = rownames(abundance_df), abundance_df)

# --- Extract metadata ---
metadata <- as.data.frame(colData(se))
metadata <- cbind(sample_id = rownames(metadata), metadata)

# Ensure diagnosis column is present (may be called study_condition or disease)
if (!"study_condition" %in% colnames(metadata)) {
  cat("WARNING: 'study_condition' column not found in metadata\n",
      file = stderr())
}

# --- Write output ---
species_file <- file.path(output_dir, paste0(study_name, "_species.tsv"))
metadata_file <- file.path(output_dir, paste0(study_name, "_metadata.tsv"))

write.table(abundance_df, species_file, sep = "\t", row.names = FALSE,
            quote = FALSE)
write.table(metadata, metadata_file, sep = "\t", row.names = FALSE,
            quote = FALSE)

cat(sprintf("Wrote species table: %s (%d samples x %d species)\n",
            species_file, nrow(abundance_df), ncol(abundance_df) - 1))
cat(sprintf("Wrote metadata:      %s (%d samples, %d columns)\n",
            metadata_file, nrow(metadata), ncol(metadata)))

# Print diagnosis distribution as a quick sanity check
if ("study_condition" %in% colnames(metadata)) {
  cat("\nDiagnosis distribution:\n")
  print(table(metadata$study_condition))
}

# Clean up to free memory before next study
rm(dataset_list, se, se_species, abundance, abundance_df, metadata)
gc(verbose = FALSE)

cat("\nDone.\n")
