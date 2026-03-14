# Study C Scaling

This folder covers the controllability scaling details for Study C.

## Split

- Split: `data/controllability_splits/study_c_controllability_test.json`
- Size: `30` cases × `20` turns
- Gold plans: `data/controllability_splits/ctrl_target_plans.json`

## What Matters for Study C

Study C does not use “condition injection” in the same sense as Study A Bias or Study B pressure.

Its controllability scaling depends on three upstream improvements:

1. stronger condition resolution before sampling
2. richer `patient_summary` construction
3. stronger `critical_entities` extraction from both the summary and source patient text

That is the key reason the regeneration pass mattered for Study C. Without those changes, Controlled Entity Recall can become artificially easy because the required entities are too thin or too generic.

## Stronger Patient Summary Construction

The current build path:

- normalises the resolved condition
- builds a structured summary with demographic framing
- adds anchor entities extracted from the patient text
- adds medication mentions when present

This is implemented in `build_patient_summary()` inside `scripts/preprocessing/build_controllability_splits.py`.

## Stronger Critical Entity Construction

The current split build no longer depends on a thin summary-only extraction path.

`extract_critical_entities()` now uses:

- the generated patient summary
- the raw patient text
- anchor-pattern extraction
- medication extraction

This gives Study C a more meaningful retained-entity target for controllability scoring.

## Notes

- There is no separate prompt-ID appendix here because Study C’s scaling change is structural rather than a bias-style injected subset.
- The relevant study-level questions for Study C are therefore about entity quality and summary quality, not “which IDs were injected”.
