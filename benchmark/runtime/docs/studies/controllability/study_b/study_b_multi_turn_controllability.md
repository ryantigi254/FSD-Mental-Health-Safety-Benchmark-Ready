# Study B Multi-Turn Controllability

## Overview

This document describes the multi-turn controllability variant of Study B. It extends the single-turn controllability idea into rolling-context conversations, where the user applies sustained diagnostic pressure across many turns.

The goal is to test whether a model can keep following the anti-sycophancy reasoning constraint throughout a pressured conversation rather than only in a one-shot prompt.

The repo now has:

- a legacy `v1` path that runs the original controlled multi-turn cache
- a `v2` path that reruns the same cases under `spontaneous`, `generic_control`, and `explicit_control`

This benchmark adaptation follows the CoT controllability framing introduced in Chen et al., *Reasoning Models Struggle to Control their Chains of Thought*:
[https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf](https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf)

## Data

- Split file: `data/controllability_splits/study_b_multi_turn_controllability_test.json`
- Source build script: `scripts/preprocessing/build_controllability_splits.py`

Each case stores:

- `gold_answer`
- `incorrect_opinion`
- `pressure_style`
- `pressure_schedule`
- `turns`
- `cot_controlled_constraint`
- source provenance and inferred condition metadata

## Generation Path

- Unified runner: `hf-local-scripts/run_ctrl_generate_only.py`
- Auto launcher: `scripts/dev/run_generation_auto.py`
- Arm-aware runner: `hf-local-scripts/run_ctrl_v2_generate_only.py`

This variant uses:

- rolling `conversation_history`
- `runner.chat(...)`
- `mode="cot_controlled"`

For each turn, the runner appends the next user message, applies the case-specific constraint, generates the assistant response from the full dialogue history, and stores a per-turn cache entry.

In `v2`, the control text is injected once in the initial system/setup message rather than being repeated every turn.

## Output

Per-model cache output:

- `results/<model>/ctrl_study_b_multi_turn_generations.jsonl`
- `results/<model>/ctrl_v2_study_b_multi_turn_generations.jsonl`

Resume semantics use `case_id + variant + turn_num`, so interrupted conversations can continue from the first missing successful turn.

## Metric Status

The arm-aware `v2` path now exports cached-response metrics:

- `no_flip_rate`
- `turn_of_flip_censored`
- `per_turn_agreement_rate`

and writes:

- `results/<model>/ctrl_v2_study_b_multi_turn_results.json`

`turn_of_flip_censored` uses `T + 1` when the model never flips inside the conversation window.

## Related Files

- Commands: `docs/studies/controllability/study_b/study_b_multi_turn_controllability_commands.md`
- Single-turn controllability guide: `docs/studies/controllability/study_b/study_b_controllability.md`
- Base Study B multi-turn guide: `docs/studies/study_b/study_b_multi_turn.md`
