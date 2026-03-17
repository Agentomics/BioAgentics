# BioAgentics

Multi-agent system for cancer research and drug discovery. BioAgentics uses coordinated AI agents to identify research opportunities, build computational tools, analyze genomic data, screen drug candidates, and advance our understanding of cancer biology.

## Architecture

BioAgentics is a single system with specialized agents that coordinate through a shared API:

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

### Research Pipeline

```
Research Director → Project Manager → Developer → Analyst → Validation Scientist → Research Writer
     (propose)        (plan tasks)     (build)    (analyze)     (validate)          (document)
```

Supporting agents run continuously:
- **Literature Reviewer** feeds new papers and methods to the Research Director
- **Data Curator** monitors data sources and organizes datasets

### Labeling System

Research initiatives are tagged with labels for tracking significance:
- `drug-candidate` — therapeutic potential identified
- `novel-finding` — unexpected or previously unreported result
- `biomarker` — diagnostic/prognostic marker candidate
- `high-priority` — results warrant urgent follow-up
- `promising` — early positive signals
- `genomic`, `transcriptomic`, `clinical`, `drug-screening`, `resistance`, `protein` — research area

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

- `agents.toml` — agent roles, dispatch timing, and model settings
- `.env` — API keys and environment configuration

## Research Focus Areas

- Genomic and transcriptomic analysis
- Drug discovery and candidate screening
- Biomarker identification
- Clinical data analysis and trial optimization
- Treatment resistance mechanisms
- Protein structure and interaction modeling

## Project Structure

```
src/bioagentics/          # All code lives here
  config.py               # Configuration and API client
  dispatch.py             # Agent lifecycle and scheduling
  mcp_server.py           # MCP tools for agent coordination
  agent_api/              # FastAPI coordination server
org-roles/                # Agent role definitions
data/                     # Research data, datasets, results (created at runtime)
docs/                     # Research documentation (created as needed)
PLAN-*.md                 # Research initiative plans (created by Research Director)
```

## Local-Only

The coordination API defaults to `https://bioagentics.mtingers.com`.
