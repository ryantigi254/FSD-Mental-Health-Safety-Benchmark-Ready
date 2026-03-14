# Study A Bias Controllability

## Overview

This document describes the canonical controllability path for Study A Bias.
The study now runs the canonical adversarial bias cases under matched arms:

- `spontaneous`
- `generic_control`
- `explicit_control`

Unlike the main Study A controllability split, this study focuses on **bias-aware reasoning under control**, not direct-vs-CoT pairing.

This benchmark adaptation follows the CoT controllability framing introduced in Chen et al., *Reasoning Models Struggle to Control their Chains of Thought*:
[https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf](https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf)

## Data

- Legacy split file: `data/controllability_splits/study_a_bias_controllability_test.json`
- Source build script: `scripts/preprocessing/build_controllability_splits.py`

Each case stores:

- `prompt`
- `bias_feature`
- `bias_label`
- `cot_controlled_constraint`
- source provenance and inferred condition metadata

For underrepresented conditions, the split builder also records a `condition_injection_note` in metadata so the controllability set documents where condition-aware scaling was deliberately introduced.

The canonical bias cases for the arm-aware run come from:

- `data/frozen_splits/v4_1_resampled/adversarial_bias/biased_vignettes.json`

## Generation Path

- Unified runner: `hf-local-scripts/run_ctrl_generate_only.py`
- Auto launcher: `scripts/dev/run_generation_auto.py`

The legacy `hf-local-scripts/run_ctrl_v2_generate_only.py` wrapper remains as a
compatibility shim only.

Study A Bias controllability generates one constrained reasoning output per item:

- `mode="cot_controlled"`

The runner appends a short diagnosis instruction to the stored prompt, injects the case-specific reasoning constraint, and writes the result to the standard cache layout.

The three-arm path uses:

- `spontaneous`: plain `cot`
- `generic_control`: neutral evidence-only control prompt
- `explicit_control`: explicit feature-acknowledgement prompt

## Output

Per-model cache output:

- `results/<model>/ctrl_study_a_bias_generations.jsonl`

Compatibility aliases may still write
`results/<model>/ctrl_v2_study_a_bias_generations.jsonl` for older tooling.

Resume semantics use `id + arm + mode`, so successful controllability rows are
not re-run on restart.

## Metric Status

The canonical pipeline writes structured outputs:

- `results/<model>/ctrl_study_a_bias_results.json`

Reported metrics:

- `silent_bias_rate`
- `biased_outcome_rate`
- `feature_mention_rate`

Compatibility aliases may still write
`results/<model>/ctrl_v2_study_a_bias_results.json`.

Interpretation rule:

- compare `spontaneous` vs `generic_control` for the cleanest silent-bias claim
- treat `explicit_control` as a transparency-focused arm because it explicitly asks the model to acknowledge the personal feature

## Related Files

- Commands: `docs/studies/controllability/study_a/study_a_bias_controllability_commands.md`
- Base Study A Bias guide: `docs/studies/study_a/study_a_bias.md`
