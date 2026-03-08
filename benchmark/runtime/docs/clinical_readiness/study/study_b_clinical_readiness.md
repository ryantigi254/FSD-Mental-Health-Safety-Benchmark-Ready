# Study B Clinician-Readiness Updates

## Scope

Study B v0.3 work targeted schema safety, pressure-construct reliability, and clinician export usability.

## Current Data Shape

- Single-turn set: `2000` rows.
- Multi-turn set: `120` cases.
- Multi-turn turn depth: fixed `20` turns per case.
- Persona coverage:
  - single-turn: no missing `persona_id`
  - multi-turn: no missing `persona_id`

## What Changed

- Runtime schema checks were hardened to fail closed on malformed payloads.
- Study B validation was made explicit and reportable:
  - uniqueness validator (duplicate full-turn signatures)
  - construct validator (including incorrect-opinion quality checks)
- Multi-turn exports now include deterministic `turns_text` to remove JSON-cell ambiguity in spreadsheets.
- Pressure controls remain balanced in v0.3 multi-turn:
  - `pressure_style`: 40/40/40
  - `pressure_schedule`: 40/40/40

## Why These Changes Were Made

- Prevent quiet runtime acceptance followed by downstream metric failure.
- Make pressure-test artefacts reviewable without additional joins or custom parsers.
- Preserve reproducible behavioural checks for every package build.

## Output-Level Consequences

- `study_b_single_turn_review.csv` has stable required columns and row count.
- `study_b_multi_turn_review.csv` exposes pressure fields and deterministic turn rendering.
- Stage-2 gate report documents uniqueness/construct pass outcomes.
