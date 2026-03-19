# Role: Human

**Username:** `human`

## Purpose

Handle tasks that require human intervention — setting up external systems, configuring data access, managing credentials, approving access, and anything else that agents cannot do autonomously.

## Coordination

Use the agent-comms API (`AGENT_COMMS.md`) for all coordination.

**Division:** Always use `division="cancer"` when creating tasks and journal entries.

- **Tasks:** Check for tasks assigned to you by other agents. Update statuses when complete.

## Examples of tasks for `human`

- Requesting access to restricted research databases or datasets
- Providing API keys, tokens, or credentials for data sources
- Setting up compute resources (GPU instances, cluster access)
- Approving data usage agreements or IRB compliance
- Downloading large datasets that require manual authentication
- Configuring external services (cloud storage, CI/CD, hosting)
