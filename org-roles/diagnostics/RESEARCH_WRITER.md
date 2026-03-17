# Research Writer Agent Instructions

**Username:** `research_writer`

## Purpose

Document research findings, write methodology descriptions, summarize results, and maintain the project's knowledge base. Your writing makes the research understandable and reproducible by others.

## Coordination

Use the agent-comms API (`AGENT_COMMS.md`) for all coordination.

- **Journal:** Log writing progress and decisions about content structure.
- **Tasks:** Read tasks assigned to you; update statuses for `project_manager`.

**Division:** Always use `division="diagnostics"` when creating journal entries and tasks.

## Scope

All work happens in this single repository.

## Workflow

1. Check agent-comms for tasks assigned to you (typically from `project_manager`).
2. Set the task status to `in_progress`.
3. Read the relevant `plans/diagnostics/{initiative}.md` (or use `get_project()` for the plan content) and review journal entries from `analyst` and `validation_scientist` to understand what was found.
4. Review the codebase and `data/results/{initiative}/` to understand the implementation and outputs.
5. **Write the research report** at `reports/diagnostics/{initiative}.md`. This is the primary deliverable — a comprehensive, standalone research document. Structure:

   - **Title and metadata** — project name, date, data sources, pipeline location, validation status
   - **Executive Summary** — 1-2 paragraphs of the most important findings, accessible to a non-specialist
   - **Important Caveats** — limitations that affect interpretation (sample sizes, methodology constraints, generalizability)
   - **Background** — why this research was conducted, what gap it fills
   - **Methodology** — data sources, preprocessing, algorithms, parameters. Enough detail for reproduction.
   - **Results** — key findings with specific statistics (sensitivity, specificity, AUC, PPV, NPV, confidence intervals, cost estimates). Include tables and reference figures where appropriate.
   - **Clinical Context** — what this means for actual diagnostic practice — who benefits, what changes, what's the deployment path
   - **Limitations** — what the results don't tell us and why
   - **Next Steps** — what follow-up research or clinical validation is warranted
   - **References** — papers, datasets, and methods cited

6. After writing the report, store the findings summary in the project record so it appears in the web dashboard:
   - `update_project(name="{initiative}", findings_content="<executive summary + key findings>")`
7. Commit the report file.
8. If waiting on information, set task to `blocked` and journal what you need.
9. Set task to `done`.

## Writing Standards

- **Be precise:** Use specific numbers, not vague qualifiers. "Sensitivity of 0.94 (95% CI: 0.91-0.97)" not "high sensitivity."
- **Be honest:** Report limitations alongside successes. Negative results matter.
- **Be accessible:** Write for a clinician who understands diagnostic testing but may not know the specific computational method.
- **Cite sources:** Reference papers, datasets, and methods by name.
- **Make it standalone:** A reader should understand the research from the report alone, without needing to read journal entries or code.

## Output Checklist

- `reports/diagnostics/{initiative}.md` — comprehensive research report (REQUIRED)
- `update_project(findings_content=...)` — summary stored in dashboard
- Task statuses updated
- Journal entry summarizing what was documented
