# Study B Multi-Turn Invariance

## Overview

Study B multi-turn invariance keeps the same conversation-level evaluation unit
and checks whether harmless changes to schedule/tone phrasing alter the
conversation outcome more than they should.

## Source roots

- `v5`: `data/frozen_splits/v5`
- `controllability`: `data/controllability_splits_large_resolved`

The sampled output root is always normalised back to the canonical layout, so
the runner still reads `study_b_multi_turn_test.json`.

## Sampled roots and default budgets

- `v5`: budget `12`
- `controllability`: budget `10`

## Pairing unit

Study B multi-turn invariance pairs on conversation `case_id`.

## Variant focus

The default variants are:

- `pressure_schedule_shift`
- `pressure_tone`
- `pressure_intensity`

## Related files

- Commands: `docs/studies/invariance/study_b/study_b_multi_turn_invariance_commands.md`
- Base Study B multi-turn guide: `docs/studies/study_b/study_b_multi_turn.md`
