# Invariance Sampling Profiles

The repo now supports two deterministic invariance sampling profiles:

1. `v5`
2. `controllability`

Both profiles reuse the same manifest builder and the same canonical sampled
output layout. The difference is the source root and the default study budgets.

## Source roots

### `v5`

- Source root: `data/frozen_splits/v5`
- Materialised sampled root:
  `data/invariance/base`

### `controllability`

- Source root: `data/controllability_splits_large`
- Materialised sampled root:
  `data/controllability_splits_large/base`

## Default budgets

The main `v5` profile keeps the existing defaults:

- Study A: `150`
- Study B: `160`
- Study B multi-turn: `12`
- Study C: `15`

The controllability-backed profile uses a deliberately smaller diagnostic
budget:

- Study A: `140`
- Study B: `150`
- Study B multi-turn: `10`
- Study C: `12`

These are intentionally a little below the `v5` profile. Even where the raw
controllability split has enough rows, it is still a narrower control-conditioned
subset rather than the full frozen benchmark release, so we keep the invariance
slice slightly lighter.

## Canonical builder commands

Build the standard `v5` invariance sample:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/studies/v5_review/build_invariance_splits.py \
  --sample-profile v5
```

Build the controllability-backed invariance sample:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/studies/v5_review/build_invariance_splits.py \
  --sample-profile controllability \
  --data-root data/controllability_splits_large \
  --output-root data/controllability_splits_large/base
```

Generate a single manifest only:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/studies/v5_review/generate_invariance_manifest.py \
  --study study_b \
  --sample-profile controllability \
  --data-root data/controllability_splits_large
```

## Output layout

Both profiles materialise a frozen-layout-compatible root so the existing
invariance generation runners can be reused unchanged.

That means the sampled output root always contains canonical names such as:

- `study_a_test.json`
- `study_b_test.json`
- `study_b_multi_turn_test.json`
- `study_c_test.json`
- `study_a/gold_diagnosis_labels.json`
- `study_c/target_plans.json`

The source root may use controllability-specific names such as
`study_a_controllability_test.json` or `ctrl_target_plans.json`, but the
materialised sampled root normalises them back to the canonical invariance
layout.
