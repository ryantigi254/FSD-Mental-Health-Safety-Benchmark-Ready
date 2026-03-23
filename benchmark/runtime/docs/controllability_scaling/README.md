# Controllability Scaling Docs

This folder documents how the controllability splits were built and what was actually frozen into the artefacts under `data/controllability/controllability_splits_v2_1/`.

## Top-Level Overview

- `CONTROLLABILITY_SCALING.md`

## Study Folders

- `study_a/`
- `study_b/`
- `study_c/`

## Scope

These docs cover:

- stronger condition-resolution based regeneration
- underrepresented-category priority sampling
- which prompt IDs were explicitly or effectively injected by study
- the realised Study B pressure matrix in the frozen split
- split-construction provenance for the current small and large resolved suites

They do not describe the runnable arm-aware generation/evaluation path
directly. That lives under:

- `docs/studies/controllability/README.md`
- `docs/studies/controllability/controllability_commands.md`
- `docs/studies/controllability/gold_generation.md`

Use the scaling docs for split construction provenance, and the study docs for
the canonical `ctrl_study_*` controllability workflow.
