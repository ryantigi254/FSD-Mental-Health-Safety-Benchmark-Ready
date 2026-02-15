## Runtime Agent Rules (NLP-ready)

### Environment naming (mandatory)
- Always create uniquely named Python environments for this repo.
- Never use canonical/shared startup names (`base`, `default`, `test`, `env`, `.venv`).
- Required naming pattern:
  - Conda: `mh-llm-<purpose>-<user>-<yyyymmdd>`
  - venv: `.venv-<purpose>-<user>-<yyyymmdd>`

### Scratchpad discipline (mandatory)
- Update `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/codex_scratchpad.md` during active runs.
- Keep one active section per task and rewrite that section as work progresses.
- Section format:
  - `What changed`
  - `Errors`
  - `Corrections`
  - `Current status`
  - `Next actions`
- Log strategy switches explicitly (cause -> switch -> expected gain).

### Validation behaviour
- Fail closed when required validation cannot run.
- Record blockers in scratchpad before retrying with a new strategy.
