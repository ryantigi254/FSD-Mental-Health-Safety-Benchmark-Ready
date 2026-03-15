# Invariance Study Guides

This folder documents how to build sampled invariance roots and run the
`study_*_invariance` generation targets.

Two source profiles are now supported:

- `v5`: sampled from `data/frozen_splits/v5`
- `controllability`: sampled from
  `data/controllability_splits_large_resolved`

The controllability-backed path exists so we can run the same invariance
comparison machinery over the control-conditioned suite as well as the main
frozen benchmark split.

## Study Folders

- `study_a/`
- `study_b/`
- `study_c/`

## Background references

- `docs/metrics/INVARIANCE_README.md`
- `docs/invariance_scaling/README.md`
- `docs/metrics/CONTROLLABILITY_META_METRIC.md`

## Shared prerequisites

1. Build the sampled invariance root you want to use.
2. Point the generation runner at that sampled root with `--data-dir`.
3. Keep outputs for each sampled root separate.

Recommended output directories:

- `results_invariance_v5/`
- `results_invariance_controllability/`

## Canonical command note

The runnable commands live in:

- `docs/studies/invariance/invariance_commands.md`

Per-study notes live in:

- `docs/studies/invariance/study_a/study_a_invariance.md`
- `docs/studies/invariance/study_b/study_b_invariance.md`
- `docs/studies/invariance/study_b/study_b_multi_turn_invariance.md`
- `docs/studies/invariance/study_c/study_c_invariance.md`

That note covers:

- building the sampled roots
- generation commands for all four invariance studies
- running the paired comparison scripts afterward

Canonical direct runner:

- `hf-local-scripts/run_invariance_generate_only.py`
