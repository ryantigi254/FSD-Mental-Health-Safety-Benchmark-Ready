---
tracker:
  kind: linear
  project_slug: "nlp-ready-symphony-e79daf83a8b1"
  active_states:
    - Todo
    - In Progress
    - Rework
    - Human Review
    - Merging
  terminal_states:
    - Done
    - Closed
    - Cancelled
    - Canceled
    - Duplicate
polling:
  interval_ms: 5000
workspace:
  root: /Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/.codex/symphony/workspaces
hooks:
  after_create: |
    git clone --depth 1 "${SOURCE_REPO_URL:-https://github.com/ryantigi254/FSD-Mental-Health-Safety-Benchmark-Ready.git}" .
agent:
  max_concurrent_agents: 8
  max_turns: 20
codex:
  command: codex --config shell_environment_policy.inherit=all --config model_reasoning_effort=high --model gpt-5.3-codex app-server
  approval_policy: never
  thread_sandbox: workspace-write
  turn_sandbox_policy:
    type: workspaceWrite
---

You are operating inside a Symphony unattended workflow run.

Execution policy:

1. Treat `swarm-planner` as the primary planning/orchestration skill.
2. Do not use DAG orchestration by default.
3. Use dependency-first execution and run unblocked tasks in parallel where safe.
4. Use helper skills when appropriate: `linear`, `pull`, `commit`, `push`, `land`, `debug`.
5. Keep progress and validation evidence in the active tracker workpad comment.

Issue context:

- Identifier: {{ issue.identifier }}
- Title: {{ issue.title }}
- State: {{ issue.state }}
- URL: {{ issue.url }}
- Labels: {{ issue.labels }}

Description:
{% if issue.description %}
{{ issue.description }}
{% else %}
No description provided.
{% endif %}

Deliverables:

- Implementation that satisfies acceptance criteria
- Validation evidence
- Clean handoff notes with blockers (if any)
