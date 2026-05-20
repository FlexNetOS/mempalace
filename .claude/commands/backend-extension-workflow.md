---
name: backend-extension-workflow
description: Workflow command scaffold for backend-extension-workflow in mempalace.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /backend-extension-workflow

Use this workflow when working on **backend-extension-workflow** in `mempalace`.

## Goal

Add a new backend or extend backend-neutral functionality, then update backend-specific shims and ensure integration with server routing and embedding logic.

## Common Files

- `mempalace/embedding.py`
- `mempalace/backends/chroma.py`
- `mempalace/backends/ruvector_postgres.py`
- `mempalace/mcp_server.py`
- `tests/test_backend_routing.py`
- `tests/test_ruvector_postgres_migrate.py`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Implement or refactor backend-neutral logic in mempalace/embedding.py or similar shared modules.
- Update or create backend-specific modules (e.g., mempalace/backends/ruvector_postgres.py, mempalace/backends/chroma.py) to use the new logic or provide shims.
- Update server routing or integration points (e.g., mempalace/mcp_server.py) to support the new backend or abstraction.
- Write or update tests to cover new backend behavior and routing (e.g., tests/test_backend_routing.py, tests/test_ruvector_postgres_migrate.py).

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.