# Human Role — REMOVED

The human role has been removed. The pipeline is fully autonomous.

Agents must handle all work without human intervention. If a task requires something that was previously delegated to the human role (credentials, external data access, compute resources), agents should:

1. Document the blocker in their journal with details on what is missing
2. Set the task to `blocked`
3. Continue with available data or scope down the task accordingly

Do not create tasks for `human`. There is no human in the loop.
