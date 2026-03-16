# Agent Coordination API (AGENT_COMMS)

This API is the **only coordination channel between agents**. Use it to read tasks, record progress, and coordinate work. Agents must not assume shared memory outside this API.

## Use the MCP tools

**Always use the MCP tools for ALL API interactions.** The BioAgentics MCP server handles authentication and endpoint routing automatically.

Available tools:

| Tool | Purpose |
|---|---|
| `list_tasks` | List tasks (filter by username, status, project) |
| `get_task` | Get a specific task by ID |
| `create_task` | Create a task assigned to an agent |
| `update_task` | Update task status (and blocked_reason when blocking) |
| `list_journal` | List journal entries (filter by username, project) |
| `create_journal` | Write a journal entry |
| `list_projects` | List research initiatives (filter by status, labels) |
| `get_project` | Get a specific research initiative |
| `create_project` | Register a new research initiative (with labels) |
| `update_project` | Update initiative status and/or labels |
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

BioAgentics is a single system. All code lives in `src/bioagentics/`. Research data goes in `data/`. Research plans are `PLAN-{initiative}.md` files in the repo root.

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

Research initiatives can be tagged with labels to track significance:
- `drug-candidate`, `novel-finding`, `biomarker`, `high-priority`, `promising`
- `genomic`, `transcriptomic`, `clinical`, `drug-screening`, `resistance`, `protein`

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
| `literature_reviewer` | Literature scanning and opportunity discovery |
| `data_curator` | Dataset management and data quality |
| `project_manager` | Task coordination and pipeline management |
| `developer` | Code implementation |
| `analyst` | Analysis execution and result interpretation |
| `validation_scientist` | Scientific and code validation |
| `research_writer` | Documentation and research summaries |
| `human` | Manual tasks requiring human intervention |

## Pagination

List tools support `limit` and `offset` parameters. Default limit is 50.
