# Study A Scaling

This folder covers the controllability scaling details for:

- Study A controllability
- Study A Bias controllability

## What Changed

The current controllability artefacts for Study A were built after the stronger condition-resolution pass. That matters most for Study A Bias, because condition-aware injections are only meaningful if the underlying condition is resolved reliably enough to trust the metadata.

## Study A

- Split: `data/controllability_splits/study_a_controllability_test.json`
- Size: `300` samples
- Constraint source: `STUDY_A_CONSTRAINT`
- Sampling path: balanced selection from resolved unused OpenR1-Psy rows with a 40% underrepresented-category target

This split does not store a separate “condition injection note” field. Its scaling comes from the sampling mix and the stronger resolved-condition metadata.

## Study A Bias

- Split: `data/controllability_splits/study_a_bias_controllability_test.json`
- Size: `300` cases
- Constraint source: `STUDY_A_BIAS_CONSTRAINT`

This is the only controllability artefact that stores an explicit `condition_injection_note` in metadata. Those marked rows are the concrete cases where an underrepresented condition was deliberately carried into the bias-oriented controllability setup.

## Injected Prompt IDs

The exact injected Study A Bias IDs are documented in:

- `injected_prompt_ids.md`

Current realised injected-case count:

- `91 / 300` Study A Bias cases

These injected rows span:

- `Psychotic Spectrum`
- `Self-Harm & Suicidality`
- `Sleep-Wake Disorders`
- `Neurodevelopmental`
- `Somatic & Health-Related`
- `Substance Use Disorders`
- `Personality Disorders`
- `Eating Disorders`

## Notes

- “Injected” here means explicitly marked in the JSON metadata, not merely sampled from an underrepresented category.
- The base Study A controllability split remains a clean reasoning-adherence set; the explicit condition injection path is specific to Study A Bias.
