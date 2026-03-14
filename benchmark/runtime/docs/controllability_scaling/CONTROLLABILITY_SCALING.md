# Controllability Scaling — Frozen Test Sets

> Purpose: document how the controllability splits were built, what changed in the stronger regeneration pass, and where the study-specific scaling details now live.

## Overview

Controllability is a compliance layer over the benchmark’s existing metrics. It measures whether a model follows an explicit reasoning constraint injected into the prompt, rather than replacing the underlying Study A, B, or C task.

Conceptually:

```text
controllability = compliant traces / total traces
```

The frozen controllability artefacts are:

| File | Study | Size | Notes |
|------|-------|------|-------|
| `study_a_controllability_test.json` | Study A | 300 samples | reasoning-adherence split |
| `study_a_bias_controllability_test.json` | Study A Bias | 300 cases | includes explicit condition-injection notes |
| `study_b_controllability_test.json` | Study B single-turn | 300 samples | condition-targeted incorrect opinions |
| `study_b_multi_turn_controllability_test.json` | Study B multi-turn | 30 cases × 20 turns | pressure-style and schedule metadata stored per case |
| `study_c_controllability_test.json` | Study C | 30 cases × 20 turns | stronger patient-summary and entity-anchor construction |

All split and gold artefacts live under `data/controllability_splits/`.

## Current Build State

The current controllability splits were regenerated with the stronger condition-resolution pipeline, not the earlier weak heuristic-only path.

Current manifest state:

- total OpenR1-Psy rows seen: `19309`
- previously used post IDs excluded: `3045`
- unused rows with patient text: `15854`
- resolved rows available for controllability sampling: `8193`
- unresolved rows excluded from resolved-only sampling: `7661`

The manifest also records resolution-source counts in `data/controllability_splits/build_manifest.json`.

## Stronger Condition Resolution

The current split build no longer relies on the earlier weak `infer_condition()` path described in older notes.

It now uses the shared resolver in `src/reliable_clinical_benchmark/utils/condition_resolution.py`, which combines:

- explicit condition extraction from reasoning/patient text
- symptom-pattern heuristics
- shared alias normalisation
- optional NLI-backed resolution for ambiguous cases

This is the same shift that removed the earlier `unspecified`/fallback-heavy behaviour that would have weakened the controllability benchmark.

## Underrepresented Category Priority Sampling

The build still prioritises the long-tail categories identified from the benchmark distribution analysis:

- `Sleep-Wake Disorders`
- `Substance Use Disorders`
- `Self-Harm & Suicidality`
- `Personality Disorders`
- `Psychotic Spectrum`
- `Neurodevelopmental`
- `Eating Disorders`
- `Somatic & Health-Related`

Sampling remains `40%` from the underrepresented pool and `60%` from the standard pool where availability allows.

## What “Injected” Means Here

The phrase “injected” is study-specific and should not be read as one uniform mechanism.

### Study A Bias

This is the only controllability split that stores an explicit `condition_injection_note` field in the JSON. In other words, these are the cases where the underrepresented condition was deliberately carried into the bias-oriented controllability setup and marked as such in metadata.

Detailed docs and the exact injected IDs:

- `docs/controllability_scaling/study_a/README.md`
- `docs/controllability_scaling/study_a/injected_prompt_ids.md`

### Study B Single-Turn

The current single-turn Study B controllability artefact does **not** store a dedicated pressure-style or pressure-context note per item. Instead:

- each sample stores the resolved `inferred_condition`
- each sample stores the sampled `incorrect_opinion`
- underrepresented-condition cases can be identified from `metadata.inferred_category`

So the “injected” cases for Study B single-turn are the underrepresented-category items documented from the built split itself, not a separate metadata flag.

Detailed docs and exact IDs:

- `docs/controllability_scaling/study_b/README.md`
- `docs/controllability_scaling/study_b/injected_prompt_ids.md`

### Study B Multi-Turn

This split does store explicit pressure metadata:

- `pressure_style`
- `pressure_schedule`
- `turns`

The build follows the existing Study B pressure-scaling machinery, but the realised frozen split is the concrete source of truth. In the current artefact, the 30 sampled cases realise 11 style/schedule combinations rather than a full even 5 × 3 grid.

Detailed docs:

- `docs/controllability_scaling/study_b/README.md`
- `docs/controllability_scaling/study_b/pressure_matrix.md`
- `docs/controllability_scaling/study_b/injected_prompt_ids.md`

### Study C

Study C does not have a separate “condition injected” prompt-ID track in the same sense as Study A Bias or Study B pressure. Its controllability strength comes from:

- stronger resolved-condition sampling
- richer `patient_summary` construction
- improved `critical_entities` extraction

Detailed docs:

- `docs/controllability_scaling/study_c/README.md`

## Folder Layout

This directory is now split by study:

- `study_a/`
- `study_b/`
- `study_c/`

Use this file as the high-level overview, then go into the study folders for the exact prompt IDs, realised pressure combinations, and study-specific scaling notes.
