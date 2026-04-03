# Plan: Pairwise Secondary Evaluation

**Generated**: 2026-04-03

## Overview
Implement a notebook-driven pairwise subsystem in `benchmark/runtime` that reads cached candidate generations, evaluates them with the fixed four-judge local panel, writes canonical outputs under `metric-results/pairwise/`, and exposes dedicated notebooks plus runtime docs. The work stays isolated to the `pairwise-secondary-eval` branch/worktree and does not touch dissertation files.

## Prerequisites
- Existing worktree: `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready-pairwise`
- Branch: `pairwise-secondary-eval`
- PR base: `clinical-readiness`
- Local LM Studio endpoint for the fixed judge panel
- Cached study outputs already present under `benchmark/runtime/processed/`

## Dependency Graph

```text
T1 ──┬── T3 ──┬── T6 ── T8
     │        └── T7 ──┘
T2 ──┴── T4 ──┬── T6
              └── T7
T5 ───────────────┘
```

## Tasks

### T1: Finalise runtime contracts
- **depends_on**: []
- **location**: `benchmark/runtime/src/reliable_clinical_benchmark/pairwise/`
- **description**: Align config, rubric, parser, and package exports with the approved four-judge, manifest-driven contract.
- **validation**: Import the pairwise package and load a config fixture without schema errors.
- **status**: In Progress
- **log**:
- **files edited/created**:

### T2: Create frozen manifest and config builders
- **depends_on**: []
- **location**: `benchmark/runtime/scripts/pairwise/`, `benchmark/runtime/src/reliable_clinical_benchmark/pairwise/`
- **description**: Build case-manifest and judge-manifest tooling for the core, controllability, and invariance slices, with explicit missing-cache boundaries.
- **validation**: Builder script emits manifest JSON for an existing slice and fails closed for absent inputs.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T3: Replace the runner pipeline
- **depends_on**: [T1, T2]
- **location**: `benchmark/runtime/src/reliable_clinical_benchmark/pairwise/runner.py`, `benchmark/runtime/scripts/pairwise/run_pairwise.py`
- **description**: Run `AB` and `BA` comparisons against LM Studio-backed judges, persist raw and parsed outputs, and keep candidate responses separate from judges.
- **validation**: Dry-run and tiny smoke-run both execute and produce raw plus parsed artefacts.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T4: Implement aggregation and reporting
- **depends_on**: [T1]
- **location**: `benchmark/runtime/src/reliable_clinical_benchmark/pairwise/statistics.py`, `aggregator.py`, `report.py`
- **description**: Compute win rates, Bradley–Terry, swap consistency, verbosity diagnostics, agreement, and `95%` bootstrap intervals.
- **validation**: Synthetic fixtures produce stable aggregate JSON and report files.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T5: Write docs and fixed manifests
- **depends_on**: []
- **location**: `benchmark/runtime/docs/studies/pairwise/`, `benchmark/runtime/metric-results/pairwise/manifests/`
- **description**: Document permitted use, prohibited use, judge panel, commands, and canonical manifest layout.
- **validation**: Docs reference only real scripts and paths in this branch.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T6: Generate dedicated notebooks
- **depends_on**: [T2, T4, T5]
- **location**: `benchmark/runtime/notebooks/pairwise/`
- **description**: Generate the five core notebooks, five controllability notebooks, three invariance notebooks, plus cross-layer and judge-diagnostics notebooks against the canonical output schema.
- **validation**: Notebook JSON is valid and each notebook points at canonical pairwise result paths only.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T7: Add tests
- **depends_on**: [T1, T2, T3, T4]
- **location**: `benchmark/runtime/tests/unit/pairwise/`, `benchmark/runtime/tests/integration/`
- **description**: Add config, parser, statistics, aggregation, and smoke integration coverage.
- **validation**: Targeted `pytest` passes for the new pairwise test set.
- **status**: Not Completed
- **log**:
- **files edited/created**:

### T8: Validate branch state and PR readiness
- **depends_on**: [T3, T4, T6, T7]
- **location**: repo root / git state
- **description**: Run targeted validation, confirm no dissertation files changed, then prepare branch, push, and draft PR state.
- **validation**: Clean test output, expected file diff only, branch still `pairwise-secondary-eval`.
- **status**: Not Completed
- **log**:
- **files edited/created**:

## Parallel Execution Groups

| Wave | Tasks | Can Start When |
|------|-------|----------------|
| 1 | T1, T2, T5 | Immediately |
| 2 | T3, T4 | T1 complete; T3 also requires T2 |
| 3 | T6, T7 | T6 after T2/T4/T5; T7 after T1/T2/T3/T4 |
| 4 | T8 | T3, T4, T6, T7 complete |

## Testing Strategy
- Config validation for judge count, orders, and prohibited criteria.
- Parser validation for `A`, `B`, tie, invalid, and order normalisation.
- Statistics validation for Bradley–Terry, swap consistency, verbosity bias, and intervals.
- Integration smoke run using a tiny frozen manifest with all four judges and both `AB`/`BA`.

## Risks & Mitigations
- **Missing invariance/control caches**: fail closed in manifest builders and reports with explicit boundary notes.
- **LM Studio model-id drift**: keep a dedicated judge manifest mapping Hugging Face sources to local model ids.
- **Notebook drift**: generate notebooks from one template family tied to canonical output paths only.
