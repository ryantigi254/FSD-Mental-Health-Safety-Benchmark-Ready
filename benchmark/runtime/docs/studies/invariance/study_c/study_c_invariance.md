# Study C Invariance

## Overview

Study C invariance checks whether the model preserves the same critical memory
behaviour under harmless wording changes while keeping the same case/session
unit.

## Source roots

- `v5`: `data/frozen_splits/v5`
- `controllability`: `data/invariance/variant_family`

The sampled output root is always materialised back into the canonical Study C
layout, including `study_c_test.json` and the target-plan files.

## Sampled roots and default budgets

- `v5`: budget `15`
- `controllability`: budget `12`

## Pairing unit

Study C invariance pairs on conversation `case_id`.

## Variant focus

Study C can now fan out these concrete variants from one fixed sampled root:

- `summary_short`
- `summary_long`
- `patient_turn_rephrase`
- `noncritical_reorder`

## Related files

- Commands: `docs/studies/invariance/study_c/study_c_invariance_commands.md`
- Base Study C guide: `docs/studies/study_c/study_c_drift.md`
