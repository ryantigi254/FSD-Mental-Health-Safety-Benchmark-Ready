# Frozen Snapshot v4.1 (Study A Resampled)

## Scope
- Baseline snapshot: `v0.3_postclinician_audit`.
- Study A rows replaced: non-acceptable (`NEEDS_REVIEW` + `REJECT`) from `verification/v4/study_a_reference_verdicts.ssv`.
- Study B/Study C retained unchanged from baseline.

## Deterministic policy
- Preserve Study A IDs and row order.
- Retain all baseline `ACCEPTABLE` rows unchanged.
- Replace only targeted rows using OpenR1-Psy candidate stream in split/index order (`train` then `test`).
- Candidate acceptance authority: current in-repo `score_study_a` rubric-v2 scorer.
- Exclude source IDs already used anywhere in baseline Study A and by earlier accepted replacements.

## Result
- Targets: 768
- Retained Study A rows: 1232
- Filled replacements: 768
- Candidate pool scanned: 894

See `manifest.json` for checksums and row counts.
