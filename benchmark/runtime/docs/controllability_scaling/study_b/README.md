# Study B Scaling

This folder covers the controllability scaling details for:

- Study B single-turn controllability
- Study B multi-turn controllability

## Single-Turn Study B

- Split: `data/controllability_splits/study_b_controllability_test.json`
- Size: `300` samples
- Constraint source: `STUDY_B_CONSTRAINT`

Important implementation detail:

- the frozen single-turn artefact stores the same pressure-dimension metadata family used by the controllability multi-turn variant
- realised single-turn fields include:
  - `pressure_style`
  - `pressure_schedule`
  - `pressure_level`
- because this is still a single-turn artefact, `pressure_level` is a flat metadata marker rather than a turn trajectory
- underrepresented-condition “injection” for this split is still documented from the realised category membership of the frozen IDs

Current underrepresented-category single-turn cases:

- `120 / 300`

Exact IDs:

- `injected_prompt_ids.md`

## Multi-Turn Study B

- Split: `data/controllability_splits/study_b_multi_turn_controllability_test.json`
- Size: `30` cases × `20` turns
- Constraint source: `STUDY_B_MULTI_CONSTRAINT`

This artefact does store explicit pressure metadata:

- `pressure_style`
- `pressure_schedule`
- per-turn conversation payloads

It uses the existing pressure-scaling machinery with:

- styles: `self_doubt`, `social_proof`, `authority_pressure`, `evidence_citation`, `disbelief`
- schedules: `early_spike`, `gradual`, `late_spike`
- turn-level pressure progression through the 20-turn dialogue builder

## Protocol vs Realised Frozen Split

The protocol family is still “5 styles × 3 schedules”, but the **realised frozen split** is what matters for documentation.

In the current 30-case multi-turn artefact:

- 11 style/schedule combinations are represented
- 4 combinations are absent from this specific frozen sample

See:

- `pressure_matrix.md`

## Underrepresented-Condition Pressure Cases

The multi-turn split contains `17 / 30` cases from the prioritised underrepresented categories.

Those exact IDs, with style/schedule annotations, are documented in:

- `injected_prompt_ids.md`

## Notes

- If you need the exact prompt IDs that carried underrepresented conditions into Study B, use the appendix file rather than inferring from summary prose.
- For single-turn Study B, “injected” refers to the underrepresented-condition cases documented from the frozen artefact, not a dedicated note field.
- For multi-turn Study B, the pressure fields in the JSON are the authoritative record.
