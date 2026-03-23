# Study B Invariance

## Overview

Study B single-turn invariance checks whether the model's behaviour stays stable
under harmless wording changes while keeping the same single-turn evaluation
unit.

The layer is diagnostic only. It does not replace the base Study B benchmark
run.

## Source roots

- `v5`: `data/frozen_splits/v5`
- `controllability`: `data/invariance/v5_variants_v2_1`

Both are materialised into the canonical sampled layout before generation, so
the runner still reads `study_b_test.json`.

## Sampled roots and default budgets

- `data/invariance/v5_base_v2_1` with default budget `160`
- `data/invariance/v5_variants_v2_1/base` with default
  budget `150`

## Pairing unit

Study B single-turn invariance pairs on sample `id`.

## Variant focus

Study B single-turn can now fan out these concrete variants from one fixed
sampled root:

- `paraphrase`
- `mild`
- `moderate`
- `strong`
- `question`
- `cultural`

## Related files

- Commands: `docs/studies/invariance/study_b/study_b_invariance_commands.md`
- Multi-turn note: `docs/studies/invariance/study_b/study_b_multi_turn_invariance.md`
- Base Study B guide: `docs/studies/study_b/study_b_sycophancy.md`
