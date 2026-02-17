# Frozen Snapshot: v0.2_preclinician

## What this folder is
This is the pre-clinician frozen data snapshot used as the baseline before the v0.3 clinician-audit fixes.

## Version metadata
- Snapshot ID: `v0.2_preclinician`
- Created at (UTC): `2026-02-15T18:31:21.431068+00:00`
- Source manifest: `manifest.json`
- Superseded by: `../v0.3_postclinician_audit/`
- Superseded note: `SUPERSEDED.md`

## Files in this snapshot
- `study_a_test.json` (row_count: 2000)
- `study_b_test.json` (row_count: 2000)
- `study_b_multi_turn_test.json` (row_count: 120)
- `study_c_test.json` (row_count: 100)
- `gold_diagnosis_labels.json` (row_count: 2000)
- `gold_diagnosis_metadata.json` (row_count: null in this manifest)
- `target_plans.json` (row_count: 100)
- `manifest.json`
- `SUPERSEDED.md`

## Study folders (navigation copies)
To make study ownership obvious, this snapshot also includes:
- `study_a/`
  - `study_a_test.json`
  - `gold_diagnosis_labels.json`
  - `gold_diagnosis_metadata.json`
- `study_c/`
  - `study_c_test.json`
  - `target_plans.json`

These are convenience copies only. The canonical files used by the frozen manifest stay at the snapshot root.

## Provenance (git)
Primary commits touching this folder:
- `3cbc5ed` `feat(clinician-sendoff): apply readiness updates and build release dataset`
- `bec71fc` `chore(runtime): finalise clinician v0.3 snapshots, release package, and stage2 gates`

## What changed after this version
`v0.3_postclinician_audit` introduced clinician-audit fixes including:
- Tier-2 Study A relabels and metadata enrichment
- Study A mapping sync
- Study C entity evidence map and stricter validation support
- Updated frozen manifest and diff summary

## Verification
To verify integrity, use this folder's `manifest.json` hashes against local file bytes.
Example from repo root:

```bash
python -m pytest benchmark/runtime/tests/unit/data/test_frozen_snapshot_v03_manifest.py -q
```

Note: the above unit test validates v0.3. For v0.2 historical checks, compare hashes directly from this folder's `manifest.json`.
