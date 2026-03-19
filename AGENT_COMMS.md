# Agent Coordination API (AGENT_COMMS)

This API is the **only coordination channel between agents**. Use it to read tasks, record progress, and coordinate work. Agents must not assume shared memory outside this API.

## Use the MCP tools

**Always use the MCP tools for ALL API interactions.** The BioAgentics MCP server handles authentication and endpoint routing automatically.

Available tools:

| Tool | Purpose |
|---|---|
| `list_tasks` | List tasks (filter by username, status, project, priority, search text; sort asc/desc) |
| `get_task` | Get a specific task by ID |
| `create_task` | Create a task assigned to an agent |
| `update_task` | Update task status, title, description, priority, and/or blocked_reason |
| `list_journal` | List journal entries (filter by username, project, search text; sort asc/desc) |
| `create_journal` | Write a journal entry |
| `list_projects` | List research initiatives (filter by status, labels) |
| `get_project` | Get a specific research initiative |
| `create_project` | Register a new research initiative (with labels, plan_content, findings_content) |
| `update_project` | Update initiative status, description, labels, plan_content, and/or findings_content |
| `get_status` | Aggregate system status |
| `list_agents` | List agent presence |
| `list_issues` | List open GitHub issues |
| `list_prs` | List open GitHub PRs |
| `comment_issue` | Comment on a GitHub issue |
| `close_issue` | Close a GitHub issue (with optional comment) |
| `merge_pr` | Merge a GitHub PR (squash/merge/rebase) |
| `review_pr` | Review a GitHub PR (approve/request-changes/comment) |
| `close_pr` | Close a GitHub PR (with optional comment) |
| `get_pr_diff` | Get the diff of a GitHub PR |
| `get_pr_checks` | Get CI check status for a GitHub PR |
| `create_release` | Create a GitHub release |
| `list_ci_runs` | List recent CI workflow runs |
| `list_data` | List files in the data directory |
| `ensure_data_dir` | Create a subdirectory under data/ |

**NEVER use raw curl, NEVER hardcode a URL, NEVER use localhost.**

## Single Repository Model

BioAgentics is a single system. All code lives in `src/bioagentics/`. Research data goes in `data/`. Research plans are in `plans/{division}/{initiative}.md`. Output artifacts go in `output/{division}/{project}/`. Plan and findings text should also be stored in the project record via `plan_content` and `findings_content` fields so they appear in the web dashboard.

## Divisions

Research is organized into divisions — each division is an independent research domain with its own agent role definitions, plans, and output.

| Division | Description | Role definitions |
|----------|-------------|-----------------|
| `cancer` | Cancer research and drug discovery | `org-roles/cancer/` |
| `crohns` | Crohn's disease research | `org-roles/crohns/` |
| `tourettes` | Tourette syndrome research | `org-roles/tourettes/` |
| `pandas_pans` | PANDAS/PANS autoimmune neuropsychiatric research | `org-roles/pandas_pans/` |
| `diagnostics` | Making medical diagnosis more accurate, accessible, and affordable | `org-roles/diagnostics/` |

All tasks, journal entries, projects, and runs should include a `division` field. The dispatcher automatically sets division based on each agent's config in `agents.toml`.

**CRITICAL: Always pass your division parameter.** Every call to `create_task`, `create_journal`, `create_project`, `list_tasks`, `list_projects`, and `list_journal` must include your division. Omitting it causes cross-division contamination. Examples:
- `create_task(username="developer", title="...", project="...", division="cancer")`
- `list_tasks(username="developer", status="pending", division="crohns")`
- `create_journal(content="...", project="...", division="tourettes")`

"Projects" in this API represent logical research initiatives (e.g., "gene-expression-classifier", "drug-interaction-model"). They are tracked for coordination, not as separate codebases.

## Research Pipeline

```
proposed → planning → development → analysis → validation → documentation → published
```

| Status | What's happening |
|---|---|
| proposed | Research Director identified an opportunity |
| planning | Project Manager is creating tasks |
| development | Developer is building tools and pipelines |
| analysis | Analyst is running experiments and interpreting results |
| validation | Validation Scientist is checking rigor and reproducibility |
| documentation | Research Writer is documenting methodology and findings |
| published | Research initiative is complete and documented |
| cancelled | Intentionally abandoned |

## Labels

Research initiatives can be tagged with labels. Use labels appropriate to your division:

| Division | Labels |
|----------|--------|
| `cancer` | `drug-candidate`, `novel-finding`, `biomarker`, `genomic`, `transcriptomic`, `clinical`, `drug-screening`, `resistance`, `protein`, `catalyst`, `high-priority`, `promising` |
| `crohns` | `biomarker`, `novel-finding`, `microbiome`, `genomic`, `immunology`, `clinical`, `drug-repurposing`, `multi-omics`, `catalyst`, `high-priority`, `promising` |
| `tourettes` | `biomarker`, `novel-finding`, `genomic`, `neuroimaging`, `clinical`, `drug-repurposing`, `multi-omics`, `comorbidity`, `catalyst`, `high-priority`, `promising` |
| `pandas_pans` | `biomarker`, `novel-finding`, `autoimmune`, `genomic`, `immunology`, `clinical`, `drug-repurposing`, `multi-omics`, `microbiome`, `catalyst`, `high-priority`, `promising` |
| `diagnostics` | `biomarker`, `novel-finding`, `imaging`, `point-of-care`, `screening`, `cost-reduction`, `accessibility`, `ai-diagnostic`, `rare-disease`, `multi-omics`, `catalyst`, `high-priority`, `promising` |

## Task lifecycle

```
pending → in_progress → done
                      → blocked → in_progress → done
```

Priority: 1 (lowest) to 5 (highest). Only work on tasks assigned to you (unless you are `project_manager`).

## Journal (Lab Notebook)

The journal is **shared memory between agents**. Use it to record research findings, decisions, blockers, and results. Entries are returned newest first. This is the lab notebook — be thorough and specific.

## Agent presence

**Dispatcher-managed.** Do not register or deregister yourself. The dispatcher sets you to `running` before invocation and `idle` after.

## Agents

| Username | Role |
|---|---|
| `research_director` | Scientific strategy and research design |
| `research_catalyst` | Cross-project synthesis and unconventional hypothesis generation |
| `literature_reviewer` | Literature scanning and opportunity discovery |
| `data_curator` | Dataset management and data quality |
| `project_manager` | Task coordination and pipeline management |
| `developer` | Code implementation |
| `analyst` | Analysis execution and result interpretation |
| `validation_scientist` | Scientific and code validation |
| `research_writer` | Documentation and research summaries |
| `systems_engineer` | Codebase analysis and system improvements |
| `human` | Manual tasks requiring human intervention |

## Pagination

List tools support `limit` and `offset` parameters. Default limit is 50.
