# Agent Summary Format

When saving your context summary to `cache/`, use this format:

```yaml
---
last_run: 2026-03-14T10:00:00Z
tasks_completed: [42, 43]
tasks_blocked: [44]
projects_touched: [gene-expression-classifier, drug-interaction-model]
---
```

Follow the YAML header with free-form context notes for your next run.

## Fields

- **last_run**: Current UTC timestamp in ISO 8601
- **tasks_completed**: List of task IDs you completed this run
- **tasks_blocked**: List of task IDs you set to blocked
- **projects_touched**: List of project names you worked on

## Example

```yaml
---
last_run: 2026-03-14T10:00:00Z
tasks_completed: [142, 143]
tasks_blocked: []
projects_touched: [gene-expression-classifier]
---

## Context

- gene-expression-classifier: Implemented differential expression analysis pipeline using DESeq2-style normalization
- Next: add survival analysis module, integrate TCGA clinical metadata
- Note: GEO dataset GSE12345 has batch effects that need correction before use
```
