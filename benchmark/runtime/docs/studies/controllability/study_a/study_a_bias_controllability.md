# Study A Bias Controllability

## Overview

This document describes the controllability variant of Study A Bias. It reuses the controlled-CoT infrastructure to test whether a model can follow an explicit anti-bias reasoning constraint while handling clinically realistic bias probes.

Unlike the main Study A controllability split, this variant focuses on **bias-aware reasoning under control**, not direct-vs-CoT pairing.

This benchmark adaptation follows the CoT controllability framing introduced in Chen et al., *Reasoning Models Struggle to Control their Chains of Thought*:
[https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf](https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf)

## Data

- Split file: `data/controllability_splits/study_a_bias_controllability_test.json`
- Source build script: `scripts/preprocessing/build_controllability_splits.py`

Each case stores:

- `prompt`
- `bias_feature`
- `bias_label`
- `cot_controlled_constraint`
- source provenance and inferred condition metadata

For underrepresented conditions, the split builder also records a `condition_injection_note` in metadata so the controllability set documents where condition-aware scaling was deliberately introduced.

## Generation Path

- Unified runner: `hf-local-scripts/run_ctrl_generate_only.py`
- Auto launcher: `scripts/dev/run_generation_auto.py`

Study A Bias controllability generates one constrained reasoning output per item:

- `mode="cot_controlled"`

The runner appends a short diagnosis instruction to the stored prompt, injects the case-specific reasoning constraint, and writes the result to the standard cache layout.

## Output

Per-model cache output:

- `results/<model>/ctrl_study_a_bias_generations.jsonl`

Resume semantics use `id + mode`, so successful controllability rows are not re-run on restart.

## Metric Status

This variant currently has a dedicated split and runner path, but does not yet have a standalone exported metric helper equivalent to the Study A, Study B single-turn, or Study C controllability helpers.

## Related Files

- Commands: `docs/studies/controllability/study_a/study_a_bias_controllability_commands.md`
- Base Study A Bias guide: `docs/studies/study_a/study_a_bias.md`
