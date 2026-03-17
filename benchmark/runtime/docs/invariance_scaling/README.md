# Invariance Scaling Docs

This folder captures the split-building and sampling notes for the repo's
invariance layer.

It covers two source profiles:

- the canonical frozen `v5` benchmark split under `data/frozen_splits/v5`
- the controllability suite under `data/controllability_splits_large`

## Top-Level Overview

- `INVARIANCE_SCALING.md`

## Study Folders

- `study_a/`
- `study_b/`
- `study_c/`

## Scope

These docs cover:

- deterministic sampling manifests and materialised invariance roots
- the default `v5` and `controllability` sampling-budget profiles
- why the controllability-backed invariance sample is intentionally slightly
  smaller than the main `v5` invariance sample
- where the sampled roots land on disk for generation and comparison runs

They do not duplicate the runnable generation/evaluation commands. Those live
under:

- `docs/studies/invariance/README.md`
- `docs/studies/invariance/invariance_commands.md`

Use the scaling docs for sampling provenance and budget choices, and the study
docs for the canonical `study_*_invariance` generation workflow.
