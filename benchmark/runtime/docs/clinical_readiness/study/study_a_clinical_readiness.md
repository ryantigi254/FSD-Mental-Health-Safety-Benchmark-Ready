# Study A Clinician-Readiness Updates

## Scope

Study A changes in v0.3 were focused on reference-label integrity and review-ready metadata, not prompt rewriting.

## What Changed

- Canonical Study A split remains 2000 rows.
- Gold labels were regenerated/reconciled in the clinician line.
- v0.2 -> v0.3 frozen comparison shows 560/2000 label value differences.
- Tier-2 audit relabel set applied for 8 named IDs.
- Mapping consistency was restored for the legacy 300-row mapping surface (`entries_synced=176`).
- Label canonical map introduced for export normalisation (`17` canonical labels, `6` aliases).

## Metadata Sidecar

- `gold_diagnosis_metadata.json` remains sparse by design (`27` explicit IDs).
- v0.2 -> v0.3 metadata object count is stable (`27 -> 27`), with `10` enriched entries.
- Current explicit metadata profile:
  - `review_status`: `requires_clinician=19`, `internally_confirmed=8`
  - `certainty`: `low=17`, `medium=10`
- v0.3 packaging resolves defaults for missing IDs at load/export boundaries.

## Why These Changes Were Made

- Eliminate label-provenance incoherence that blocks external audit.
- Make safety/triage uncertainty explicit instead of implicit.
- Remove string-format drift in clinician-facing exports (for example ADHD alias normalisation).
- Keep raw payload stable while improving defensibility of reported metrics.

## Output-Level Consequences

- `study_a_review.csv` is deterministic and join-free (2000 rows).
- `safety_priority_review.csv` includes metadata IDs plus policy-derived high-priority IDs under fixed rules.
- Release/frozen manifests carry local hash-verifiable artefacts.
