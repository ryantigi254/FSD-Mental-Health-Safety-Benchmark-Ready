# Study A Invariance

## Overview

Study A invariance checks whether diagnostic behaviour stays stable when the
prompt is changed in semantically harmless ways.

The base Study A metrics do not change. We still score the sampled cases with
the same Study A logic after generation. The invariance layer only changes the
sampled case set and the prompt variant applied to each case.

## Source roots

Two source profiles are supported:

- `v5`: `data/frozen_splits/v5`
- `controllability`: `data/invariance/v5_variants_v2_1`

Both are materialised into the same canonical sampled layout before generation,
so the runner still reads `study_a_test.json` plus the normal gold-label files.

## Sampled roots

- `data/invariance/v5_samples_v2_1`
- `data/invariance/v5_variants_v2_1/base`

Default budgets:

- `v5`: `150`
- `controllability`: `140`

## Pairing unit

Study A invariance pairs on sample `id`.

The goal is to compare:

- the base run on the sampled cases
- a harmless wording variant on those same sampled cases

## Variant focus

Study A can now fan out these concrete variants from one fixed sampled root:

- `lexical`
- `surface`
- `syntax`
- `instruction`

## Outputs

Generation cache:

- `results.../<model>/study_a_invariance_generations.jsonl`

Comparison output:

- `metric-results/<model>/study_a_invariance_*.json`

## Related files

- Commands: `docs/studies/invariance/study_a/study_a_invariance_commands.md`
- Shared commands: `docs/studies/invariance/invariance_commands.md`
- Base Study A guide: `docs/studies/study_a/study_a_faithfulness.md`
