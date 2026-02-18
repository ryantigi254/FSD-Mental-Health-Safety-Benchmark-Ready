# V4.1 Update Notes

## Summary
- Built deterministic Study A resampling from OpenR1-Psy using current in-repo v4 scorer behaviour.
- Replaced non-acceptable Study A rows from v4 baseline outputs.
- Re-ran full v4 review on `frozen_splits/v4_1_resampled` into `verification/v4_1`.

## Before vs After (Study A)
- Before (`verification/v4`): ACCEPTABLE=1232, NEEDS_REVIEW=761, REJECT=7
- After (`verification/v4_1`): ACCEPTABLE=2000, NEEDS_REVIEW=0, REJECT=0

## Resampling Provenance
- Targets: 768
- Filled: 768
- Retained unchanged: 1232
- Candidate pool scanned: 894
- Pool exhausted: 0

## Command Audit
- `/opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/unit/data/test_frozen_snapshot_v03_manifest.py -q` -> exit 0
- `/opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/unit/data/test_clinician_package_v03_manifest.py -q` -> exit 0
- `/opt/homebrew/Caskroom/miniforge/base/bin/python scripts/studies/clinician_sendoff/run_stage2_gates.py` -> exit 0
- `/opt/homebrew/Caskroom/miniforge/base/bin/python scripts/studies/clinician_sendoff/run_sendoff_preflight.py` -> exit 0
- `/opt/homebrew/Caskroom/miniforge/base/bin/python scripts/studies/v4_review/run_v4_cross_study_review.py --clean --input-root /Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/benchmark/runtime/data/frozen_splits/v4_1_resampled --out-dir /Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/benchmark/runtime/data/verification/v4_1 --rules /Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/benchmark/runtime/data/verification/v4_1/rubric_rules_v2.json` -> exit 0

## Final Distribution
| Study | Rows | ACCEPTABLE | NEEDS_REVIEW | REJECT |
|---|---:|---:|---:|---:|
| study_a | 2000 | 2000 | 0 | 0 |
| study_b_single | 2000 | 2000 | 0 | 0 |
| study_b_multi | 120 | 120 | 0 | 0 |
| study_c | 100 | 100 | 0 | 0 |
