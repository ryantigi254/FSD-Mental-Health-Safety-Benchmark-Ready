## Local Codex Agent Policy (NLP-ready)

### Purpose
- This file defines project-local harness and orchestration rules for this repository.
- It complements root `AGENTS.md` and should be treated as mandatory runtime guidance.

### Orchestration route
- Use Symphony for long-running, tracker-managed, or dependency-heavy tasks.
- Use harness lane for bounded interactive changes.
- Record explicit route decision in `codex_scratchpad.md` before major execution.

### Symphony wiring
- Workflow: `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/.codex/symphony/WORKFLOW.harness.md`
- Role config: `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/.codex/agents/symphony.toml`
- Workspace root: `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/.codex/symphony/workspaces`

### Execution defaults
- Keep orchestration swarm-first: `swarm-planner` over DAG orchestration.
- Use dependency-first ordering and bounded concurrency.
- Fail closed when required validation cannot be executed.

### Scratchpad discipline
- Update `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/codex_scratchpad.md` continuously during runs.
- Keep one active task section with:
  - `What changed`
  - `Errors`
  - `Corrections`
  - `Current status`
  - `Next actions`

### Validation and reporting
- Run targeted checks for scoped edits; broaden validation for higher-risk changes.
- Final summaries should include skills/tools/subagents used and scratchpad influence.
