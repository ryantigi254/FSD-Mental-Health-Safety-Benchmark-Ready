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

## Cursor Cloud specific instructions

### Overview
This is a Python 3.10 ML/NLP benchmark codebase (**Reliable Clinical Reasoning in LLMs**) at `benchmark/runtime/`. No database, Docker, or external services are required for tests or metric calculation.

### Python version
The codebase pins `transformers==4.40.1` and `pydantic==1.10.26`, which require **Python 3.10**. The Cloud VM ships with 3.12 by default, so Python 3.10 must be installed via `deadsnakes` PPA. The update script handles this.

### Virtual environment
The venv lives at `benchmark/runtime/.venv-benchmark-cloud-20260307`. Activate with:
```bash
source /workspace/benchmark/runtime/.venv-benchmark-cloud-20260307/bin/activate
```
All commands below assume activation.

### Running tests
```bash
cd /workspace/benchmark/runtime
PYTHONPATH=src pytest tests/unit -v
PYTHONPATH=src pytest tests/integration -v
```
- `conftest.py` adds `src/` to `sys.path` automatically, but `PYTHONPATH=src` is still needed for some import paths.
- `tests/unit/review/test_v4_reference_review.py` requires `data/verification/v4/rubric_rules_v2.json` which is not tracked; it will fail at collection. Exclude it with `--ignore=tests/unit/review/test_v4_reference_review.py`.
- Several unit tests under `tests/unit/review/` and `tests/unit/study_c/` fail due to pre-existing missing data or code issues (not environment-related).
- Integration tests that need GPU/model weights or build scripts skip gracefully.

### Linting
```bash
cd /workspace/benchmark/runtime
black --check src/ tests/
flake8 src/ --max-line-length=120
mypy src/
```
Pre-existing formatting and lint warnings exist; these are not environment issues.

### Running metrics (hello-world demo)
The core "application" is metric calculation over cached JSONL generation outputs:
```bash
cd /workspace/benchmark/runtime
PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_metrics.py --model qwq
```
This loads 2000 vignettes, computes faithfulness gap / Step-F1 / bootstrap CIs, and writes results to `metric-results/study_a/`.

### Gotchas
- The scispaCy model `en_core_sci_sm` must be installed separately via its S3 URL (see `docs/environment/ENVIRONMENT.md`). The update script handles this.
- No GPU is available in Cloud VMs, so smoke tests (under `src/tests/`) and model inference scripts will not run. Unit and integration tests work fine without GPU.
- The `test_stale_paths_guard` test runs a CI script that checks for legacy path tokens in repo files; its failures are pre-existing content issues, not environment problems.
