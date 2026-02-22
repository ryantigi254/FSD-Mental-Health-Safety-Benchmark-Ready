# Runtime Data Reproducibility Guide

## Canonical reproducibility path (`v0.2_preclinician` -> `v4_1_resampled`)
This is the canonical rebuild path from the pre-clinician freeze to the full v4.1 resampled snapshot.

## Script map (canonical labels)
These scripts are already correctly scoped. Keep filenames as-is and use the labels below in docs/runbooks.

| Label | Script path | Purpose |
|---|---|---|
| `clinician_sendoff:stage2_gates` | `benchmark/runtime/scripts/studies/clinician_sendoff/run_stage2_gates.py` | Runs clinician send-off Stage 2 integrity gates before packaging/review. |
| `clinician_sendoff:preflight` | `benchmark/runtime/scripts/studies/clinician_sendoff/run_sendoff_preflight.py` | Runs end-to-end clinician send-off preflight checks and emits preflight report artefacts. |
| `v4_review:cross_study` | `benchmark/runtime/scripts/studies/v4_review/run_v4_cross_study_review.py` | Deterministic rubric-based cross-study scoring (Study A/B/C) and SSV output generation. |
| `v4_review:resample_study_a_v4_1` | `benchmark/runtime/scripts/studies/v4_review/run_v4_1_resample_study_a.py` | Deterministic replacement of non-acceptable Study A rows to produce `v4_1_resampled`. |

### Naming note
- `v4_review` and `clinician_sendoff` are sibling workstreams under `benchmark/runtime/scripts/studies/`.
- No file rename is required for reproducibility; use the labels above for clarity in reports and command logs.

### Preconditions
- Run commands from `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready`.
- Ensure `PYTHONPATH=src`.
- Baseline snapshots exist:
  - `benchmark/runtime/data/frozen_splits/v0.2_preclinician`
  - `benchmark/runtime/data/frozen_splits/v0.3_postclinician_audit`

### Steps
1. Validate baseline snapshot/manifests
```bash
PYTHONPATH=src python -m pytest \
  tests/unit/data/test_frozen_snapshot_v03_manifest.py \
  tests/unit/data/test_clinician_package_v03_manifest.py -q
```

2. Run clinician sendoff gates
```bash
PYTHONPATH=src python scripts/studies/clinician_sendoff/run_stage2_gates.py
PYTHONPATH=src python scripts/studies/clinician_sendoff/run_sendoff_preflight.py
```

3. Build v4 review outputs on audited v0.3 input
```bash
PYTHONPATH=src python scripts/studies/v4_review/run_v4_cross_study_review.py \
  --clean \
  --input-root benchmark/runtime/data/frozen_splits/v0.3_postclinician_audit \
  --out-dir benchmark/runtime/data/verification/v4 \
  --rules benchmark/runtime/data/verification/v4/rubric_rules_v2.json
```

4. Build v4.1 deterministic Study A resample
```bash
PYTHONPATH=src python scripts/studies/v4_review/run_v4_1_resample_study_a.py --clean
```

5. Re-run v4 review on v4.1 snapshot
```bash
PYTHONPATH=src python scripts/studies/v4_review/run_v4_cross_study_review.py \
  --clean \
  --input-root benchmark/runtime/data/frozen_splits/v4_1_resampled \
  --out-dir benchmark/runtime/data/verification/v4_1 \
  --rules benchmark/runtime/data/verification/v4_1/rubric_rules_v2.json
```

6. Run v4/v4.1 review contracts
```bash
PYTHONPATH=src python -m pytest \
  tests/unit/review/test_v4_reference_review.py \
  tests/unit/review/test_v4_ssv_contract.py \
  tests/unit/review/test_v4_1_resample.py \
  tests/unit/review/test_v4_1_contract.py -q
```

### Expected final state
- `benchmark/runtime/data/verification/v4_1/study_a_reference_verdicts.ssv`: 2000 rows, all `ACCEPTABLE`.
- `benchmark/runtime/data/verification/v4_1/study_b_single_verdicts.ssv`: 2000 rows, all `ACCEPTABLE`.
- `benchmark/runtime/data/verification/v4_1/study_b_multi_verdicts.ssv`: 120 rows, all `ACCEPTABLE`.
- `benchmark/runtime/data/verification/v4_1/study_c_verdicts.ssv`: 100 rows, all `ACCEPTABLE`.

### Related traceability artefacts
- `benchmark/runtime/data/verification/v4_1/V4_1_HISTORY_AUDIT.md`
- `benchmark/runtime/data/verification/v4_1/V4_1_UPDATE_NOTES.md`
- `benchmark/runtime/data/frozen_splits/v4/manifest.json`
- `benchmark/runtime/data/frozen_splits/v4_1_resampled/manifest.json`
