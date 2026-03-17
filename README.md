# BioAgentics

Multi-agent system for biomedical research. BioAgentics uses coordinated AI agents to identify research opportunities, build computational tools, analyze data, and advance our understanding of disease biology. Research is organized into **divisions** — independent research domains with their own agent role definitions, plans, and output.

**Live dashboard:** [bioagentics.mtingers.com](https://bioagentics.mtingers.com/)

## Research Divisions

| Division | Focus | Key Data Sources |
|----------|-------|-----------------|
| **Cancer** | Genomic analysis, drug discovery, biomarkers, treatment resistance | TCGA, DepMap, COSMIC, PRISM, ChEMBL |
| **Crohn's Disease** | Microbiome, mucosal immunology, IBD therapeutics | IBDGC, HMP, RISK cohort, MetaHIT |
| **Tourette Syndrome** | CSTC circuits, neuroimaging, tic disorder genetics | TSAICG, ENIGMA, ABCD Study, EMTICS |
| **PANDAS/PANS** | Autoimmune neuropsychiatry, molecular mimicry, anti-neuronal antibodies | ImmPort, IEDB, GAS genomics, Cunningham Panel |
| **Diagnostics** | Making diagnosis more accurate, accessible, and affordable — any disease | TCIA, PhysioNet, ISIC, Grand Challenges, UK Biobank |

Each division has its own role definitions in `org-roles/{division}/`, research plans in `plans/{division}/`, and output in `output/{division}/`.

## Architecture

BioAgentics is a single system with specialized agents that coordinate through a shared API. Each division runs its own set of agents independently.

| Agent | Role |
|---|---|
| **Research Director** | Identifies research opportunities, designs studies, directs scientific strategy |
| **Literature Reviewer** | Scans for relevant publications, methods, datasets, and new opportunities |
| **Data Curator** | Manages datasets, verifies data sources, organizes the data directory |
| **Project Manager** | Coordinates research initiatives from plan to completion |
| **Developer** | Implements data pipelines, analysis tools, and computational models |
| **Analyst** | Runs analyses, interprets results, flags novel and promising findings |
| **Validation Scientist** | Validates scientific rigor, code correctness, and reproducibility |
| **Research Writer** | Documents methodology, findings, and maintains the knowledge base |
| **Systems Engineer** | Improves the BioAgentics system itself — codebase, tooling, agent configs |

### Research Pipeline

```
Research Director → Project Manager → Developer → Analyst → Validation Scientist → Research Writer
     (propose)        (plan tasks)     (build)    (analyze)     (validate)          (document)
```

Supporting agents run continuously:
- **Literature Reviewer** feeds new papers and methods to the Research Director
- **Data Curator** monitors data sources and organizes datasets
- **Systems Engineer** improves the platform itself

### Labeling System

Research initiatives are tagged with labels for tracking significance:
- `drug-candidate`, `drug-repurposing` — therapeutic potential identified
- `novel-finding` — unexpected or previously unreported result
- `biomarker` — diagnostic/prognostic marker candidate
- `high-priority` — results warrant urgent follow-up
- `promising` — early positive signals
- Domain-specific: `genomic`, `microbiome`, `immunology`, `neuroimaging`, `autoimmune`, `clinical`, `multi-omics`

## Getting Started

### Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) CLI

### Setup

```bash
# Install dependencies
uv sync

# Start the local coordination API
uv run python -m bioagentics.agent_api.main

# Start the dispatcher (in another terminal)
uv run python -m bioagentics.dispatch
```

### Configuration

- `agents.toml` — divisions, agent roles, dispatch timing, and model settings
- `.env` — API URL and API key

## Project Structure

```
src/bioagentics/          # All code lives here
  config.py               # Configuration and API client
  dispatch.py             # Agent lifecycle and scheduling
  mcp_server.py           # MCP tools for agent coordination
  agent_api/              # FastAPI coordination server + web UI
org-roles/                # Agent role definitions
  cancer/                 # Cancer division roles
  crohns/                 # Crohn's disease roles
  tourettes/              # Tourette syndrome roles
  pandas_pans/            # PANDAS/PANS roles
  diagnostics/            # Diagnostics roles
plans/{division}/         # Research initiative plans (created by Research Director)
output/{division}/        # Research output artifacts (data, figures, reports)
data/                     # Research data, datasets, results (created at runtime)
cache/                    # Agent context summaries (managed by dispatcher)
agents.toml               # Agent and division configuration
```
