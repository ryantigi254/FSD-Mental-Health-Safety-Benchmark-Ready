# Study C Clinician-Readiness Updates

## Scope

Study C v0.3 work focused on plan quality, entity anchoring defensibility, and review-sheet clarity.

## Current Data Shape

- Cases: `100`
- Critical entities per case: min/max/mean `8/8/8.0`
- Persona structure: `25` unique personas, each repeated `4x`
- Numeric age mention in summaries: `100/100`

## What Changed

- Target plan line was refreshed and clinically tightened.
- v0.2 -> v0.3 frozen comparison shows `92/100` plan entries changed.
- Entity evidence map added (`100` case entries, `2` global semantic synonym groups).
- Anchoring validation logic was hardened:
  - case evidence must be non-empty and present in the summary text
  - synonym miss must still allow case-evidence fallback checks
- Study C clinician export now carries `persona_id` explicitly.

## Why These Changes Were Made

- Move from weak semantic assumptions to explicit evidence grounding.
- Avoid false-positive anchoring outcomes in validator logic.
- Make duplicated-persona structure transparent in clinician review.

## Output-Level Consequences

- `study_c_review.csv` is deterministic with required columns.
- Anchoring tests and script-level logic tests protect against regression.
- Guide text documents the intentionally normalised entity count in v0.3.
