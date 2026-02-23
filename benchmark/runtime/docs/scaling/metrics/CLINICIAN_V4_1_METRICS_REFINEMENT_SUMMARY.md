# Clinician v4.1 Metrics Refinement Summary

## Scope

This document summarises metrics-pipeline updates made to align evaluation with the clinician-ready v4.1 dataset transition (from pre-clinician `v0.2_preclinician` to `v4.1` release data).

No metric formulas were changed.  
All changes are source resolution, reproducibility, and operational reliability refinements.

## Why this was required

- Study A content changed materially in v4.1 (prompt/label/reasoning updates on a large subset of IDs).
- Legacy metric scripts defaulted to old working paths, which could score against stale Study A gold data.
- This created measurable score drift when comparing old-vs-v4.1 labels.

## Implemented updates

### 1. Shared release-aware data resolver

Added:
- `src/reliable_clinical_benchmark/data/release_data_resolver.py`

Behaviour:
- Supports `data_source` modes:
  - `latest_release` (default)
  - `working_data` (legacy)
- Supports explicit override:
  - `data_root` path
- Resolves latest release via:
  - `data/releases/LATEST.md`
- Fail-closed checks for required folders/files before metrics run.

### 2. Study A/B/C metrics CLI source controls

Updated scripts:
- `scripts/studies/study_a/metrics/calculate_metrics.py`
- `scripts/studies/study_b/metrics/calculate_metrics.py`
- `scripts/studies/study_c/metrics/calculate_metrics.py`

New CLI options:
- `--data-source {latest_release,working_data}` (default: `latest_release`)
- `--data-root <path>`

Outcome:
- Running with no data flags now evaluates against the canonical release pointed to by `LATEST.md`.
- Legacy behaviour remains available via `--data-source working_data`.

### 3. Study C fail-closed dependency behaviour

Study C metrics now exits cleanly with explicit error messaging if MedicalNER/scispaCy is unavailable, rather than failing with opaque runtime errors.

### 4. Study B deterministic smoke refinement

Added to Study B metrics script:
- `--no-nli`

Purpose:
- Skip heavy NLI model initialisation for deterministic/fast smoke runs in constrained environments.
- When set, `H_Ev` is forced to `0.0` by design.
- Default behaviour remains unchanged (NLI-enabled when available).

## Per-study metric calculation impact

### Study A (Faithfulness)

Changed:
- Input selection now defaults to release-backed:
  - `openr1_psy_splits/study_a_test.json`
  - `study_a_gold/gold_diagnosis_labels.json`
- Uses resolver-selected root (`latest_release` by default).

Unchanged:
- Faithfulness gap calculation logic (`acc_cot - acc_early`).
- Step-F1 extraction/scoring logic.
- Refusal handling, extraction handling, bootstrap CI behaviour.

### Study B (Sycophancy)

Changed:
- Input root now resolved via shared resolver (`latest_release` default).
- Added optional `--no-nli` execution path for deterministic smoke runs.

Unchanged (default run):
- Default still initialises NLI when available.
- `P_Syc`, agreement logic, ToF/ToF proxy flow, and output schema.
- `H_Ev` logic in normal runs remains NLI-backed as before.

Changed only when `--no-nli` is explicitly used:
- NLI initialisation is skipped.
- `H_Ev` is set to `0.0` by design for that run.

### Study C (Longitudinal Drift)

Changed:
- Input root now resolved via shared resolver (`latest_release` default).
- Startup now fails closed with explicit dependency errors when MedicalNER/scispaCy is unavailable.

Unchanged:
- Entity recall computation and recall-curve aggregation logic.
- Knowledge conflict calculation logic (when NLI is enabled).
- Continuity/alignment calculation flow and result schema.

## Documentation updates

Updated canonical runtime metrics docs:
- `docs/metrics/METRIC_CALCULATION_PIPELINE.md`
- `docs/metrics/METRICS_SUMMARY.md`

Updated environment setup doc:
- `docs/environment/ENVIRONMENT.md`

Environment policy now documented as:
- Install metrics dependencies in `mh-llm-benchmark-env` via `requirements.txt`.
- Install `en_core_sci_sm` separately in `mh-llm-benchmark-env` (required for Study C metrics).
- `mh-llm-local-env` does not require `en_core_sci_sm` unless Study C metrics are run there.

## Test coverage added

Added tests:
- `tests/unit/data/test_release_data_resolver.py`
- `tests/unit/metrics/test_study_a_metrics_data_source.py`
- `tests/unit/metrics/test_study_b_metrics_data_source.py`
- `tests/unit/metrics/test_study_c_metrics_data_source.py`

Coverage includes:
- latest release resolution
- explicit root override
- malformed/missing release pointer fail-closed behaviour
- per-study source selection (`latest_release` vs `working_data`)
- Study B `--no-nli` path

## Operational result

Metrics pipeline is now release-aware by default and reproducible against the active clinician release pointer, while retaining legacy path mode for back-compat and explicit overrides for controlled reruns.
