"""False Positive Biomarker Mining — Pre-Clinical Disease Discovery from AI Discordances.

Systematically investigates whether 'false positive' predictions from diagnostic AI models
contain discoverable pre-clinical disease states, turning model 'errors' into novel early
biomarker candidates.

Modules:
    extract: False positive extraction framework at configurable operating points
    profile: Feature distribution and demographic profiling of FP vs TN groups
    cluster: Unsupervised clustering to identify coherent FP subgroups
    longitudinal: Longitudinal outcome tracking for MIMIC-IV FP patients
    adapters: Data adapters for upstream project prediction outputs
"""
