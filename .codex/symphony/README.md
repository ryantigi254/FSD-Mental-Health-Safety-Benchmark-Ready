# Symphony Harness Integration (Project-local)

This folder bootstraps Symphony for this repository.

## Files

- Workflow contract: `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/.codex/symphony/WORKFLOW.harness.md`
- Symphony role config: `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/.codex/agents/symphony.toml`
- Project role wiring: `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/.codex/config.toml`

## Prerequisites

- `LINEAR_API_KEY` exported in your shell.
- Codex CLI available on `PATH`.
- Symphony source available at `/Users/ryangichuru/.codex/vendor_imports/symphony`.
- Helper skills available in Codex home: `linear`, `pull`, `commit`, `push`, `land`, `debug`.

## Run

```bash
cd /Users/ryangichuru/.codex/vendor_imports/symphony/elixir
mise trust
mise install
mise exec -- mix setup
SOURCE_REPO_URL=https://github.com/ryantigi254/FSD-Mental-Health-Safety-Benchmark-Ready.git \
  mise exec -- ./bin/symphony \
  /Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/.codex/symphony/WORKFLOW.harness.md
```

## Notes

- Replace `project_slug` in `WORKFLOW.harness.md` before starting the service.
- Workspace sandboxes are created under `.codex/symphony/workspaces/`.
