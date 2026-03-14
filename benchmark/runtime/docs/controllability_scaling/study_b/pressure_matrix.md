# Study B Pressure Matrix

This file documents the **realised** pressure-style / pressure-schedule matrix in the frozen multi-turn controllability split.

## Protocol Family

The builder uses the existing Study B pressure-scaling framework:

- styles: `self_doubt`, `social_proof`, `authority_pressure`, `evidence_citation`, `disbelief`
- schedules: `early_spike`, `gradual`, `late_spike`

This gives a protocol family of `5 × 3 = 15` possible style/schedule combinations.

## Realised Frozen Split

The current `study_b_multi_turn_controllability_test.json` contains `30` cases and realises the following combinations:

| Pressure style | Pressure schedule | Case count |
|---|---|---:|
| `authority_pressure` | `early_spike` | 5 |
| `authority_pressure` | `gradual` | 2 |
| `authority_pressure` | `late_spike` | 2 |
| `disbelief` | `early_spike` | 2 |
| `disbelief` | `late_spike` | 1 |
| `evidence_citation` | `early_spike` | 2 |
| `evidence_citation` | `gradual` | 5 |
| `self_doubt` | `early_spike` | 4 |
| `self_doubt` | `gradual` | 3 |
| `social_proof` | `early_spike` | 3 |
| `social_proof` | `gradual` | 1 |

## Important Caveat

The protocol family supports 15 combinations, but the frozen sample does **not** realise all 15 in this specific build.

Absent combinations in the current frozen artefact:

- `disbelief / gradual`
- `evidence_citation / late_spike`
- `self_doubt / late_spike`
- `social_proof / late_spike`

That means the correct documentation claim is:

- the split follows the existing pressure-scaling framework
- the frozen artefact realises a subset of the full 5 × 3 matrix

It would be inaccurate to document the current frozen file as evenly covering all 15 combinations.
