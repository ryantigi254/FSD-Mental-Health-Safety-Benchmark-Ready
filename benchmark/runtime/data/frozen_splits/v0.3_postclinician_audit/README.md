# Frozen Snapshot: v0.3_postclinician_audit

## What this folder is
This is the post-audit frozen data snapshot for clinician send-off readiness in the v0.3 line.

## Version metadata
- Snapshot ID: `v0.3_postclinician_audit`
- Created at (UTC): `2026-02-16T00:08:24.414245+00:00`
- Description: `Post-audit frozen snapshot. Includes tier-2 label corrections, mapping sync, entity evidence map.`
- Supersedes: `v0.2_preclinician`
- Source manifest: `manifest.json`
- Audit summary: `audit_fix_diff_summary.json`
- Boundary clarification note: `README_NOTE.txt`

## Files in this snapshot
- `gold_diagnosis_labels.json` (row_count: 2000)
- `gold_diagnosis_metadata.json` (row_count: 27)
- `gold_labels_mapping.json` (row_count: 300)
- `study_a_test.json` (row_count: 2000)
- `study_b_test.json` (row_count: 2000)
- `study_b_multi_turn_test.json` (row_count: 120)
- `study_c_test.json` (row_count: 100)
- `study_c_target_plans.json` (row_count: 100)
- `entity_evidence_map.json` (row_count: 100)
- `audit_fix_diff_summary.json`
- `manifest.json`
- `README_NOTE.txt`

## Study folders (navigation copies)
To make study ownership obvious, this snapshot also includes:
- `study_a/`
  - `study_a_test.json`
  - `gold_diagnosis_labels.json`
  - `gold_diagnosis_metadata.json`
  - `gold_labels_mapping.json`
- `study_c/`
  - `study_c_test.json`
  - `study_c_target_plans.json`
  - `entity_evidence_map.json`

These are convenience copies only. The canonical files used by the frozen manifest stay at the snapshot root.

## Provenance (git)
Primary commits touching this folder:
- `bec71fc` `chore(runtime): finalise clinician v0.3 snapshots, release package, and stage2 gates`
- `fd3a955` `fix(sendoff): tighten v0.3 package gates and preflight`

## What was done in this version
As recorded in `audit_fix_diff_summary.json`:
- Applied 8 Tier-2 Study A label corrections
- Synced 176 Study A mapping labels (300 total mapped entries)
- Enriched 10 credibility-pass metadata entries with provenance fields
- Added Study C entity evidence artefact for anchoring
- Captured frozen snapshot source-of-truth hashes

## Verification
Use package and snapshot tests from repo root:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src conda run -n mh-llm-benchmark-env python -m pytest benchmark/runtime/tests/unit/data/test_frozen_snapshot_v03_manifest.py -q
PYTHONNOUSERSITE=1 conda run -n mh-llm-benchmark-env python benchmark/runtime/scripts/studies/clinician_sendoff/rebuild_release_manifest.py --verify
```
