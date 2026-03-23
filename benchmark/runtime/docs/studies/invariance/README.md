# Invariance Study Guides

This folder documents how to build sampled invariance roots and run the
`study_*_invariance` generation targets.

Two source profiles are now supported:

- `v5`: sampled from `data/frozen_splits/v5`
- `controllability`: sampled from
  `data/invariance/variant_family`

The controllability-backed path exists so we can run the same invariance
comparison machinery over the control-conditioned suite as well as the main
frozen benchmark split.

The intended workflow is:

1. build one frozen sampled root
2. fan out multiple variant families from that same root
3. run base and variant generations against matched IDs

Do not resample separately for each family unless you are deliberately running
a second replication track.

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
2. Build one or more variant roots from that sampled root.
3. Point the generation runner at the chosen base or variant root with `--data-dir`.
4. Use `--variant-tag` or a separate output directory for each family run.

Recommended output directories:

- `results_invariance/`
- `results_invariance/base/` for `data/invariance/variant_family/base`
- `results_invariance/variant-family/` for `data/invariance/variant_family/study_*/*`

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
- fanning out the variant-family matrix from a fixed sample
- generation commands for all four invariance studies
- the automatic `base` and `variant-family` save roots under `results_invariance/`
- running the paired comparison scripts afterward

Canonical direct runner:

- `hf-local-scripts/run_invariance_generate_only.py`
