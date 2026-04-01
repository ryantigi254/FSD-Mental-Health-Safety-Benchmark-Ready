# Study B Multi-Turn Invariance

## Overview

Study B multi-turn invariance keeps the same conversation-level evaluation unit
and checks whether harmless changes to schedule/tone phrasing alter the
conversation outcome more than they should.

## Source roots

- `v5`: `data/frozen_splits/v5`
- `controllability`: `data/invariance_variants/variant_family`

The sampled output root is always normalised back to the canonical layout, so
the runner still reads `study_b_multi_turn_test.json`.

## Sampled roots and default budgets

- `v5`: budget `12`
- `controllability`: budget `10`

## Pairing unit

Study B multi-turn invariance pairs on conversation `case_id`.

The comparison surface keeps `turn_of_flip` as the governing outcome metric and
adds scalar diagnostic companions:

- `stance_shift_slope`
- `sycophancy_auc`
- `soften_before_flip`

These metrics refine how pressure-response stability is described across
variants, but they do not replace the primary `ToF` interpretation. The full
per-turn `turn_stance_mean` vector remains local to the main Study B post-hoc
analysis rather than being exported through invariance payloads.

## Variant focus

The concrete multi-turn variants are:

- `schedule_earlier`
- `schedule_later`
- `tone_gentle`
- `tone_direct`
- `tone_confrontational`
- `pressure_milder`
- `pressure_stronger`

## Related files

- Commands: `docs/studies/invariance/study_b/study_b_multi_turn_invariance_commands.md`
- Base Study B multi-turn guide: `docs/studies/study_b/study_b_multi_turn.md`
