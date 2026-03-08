# Study B Multi-Turn Controllability

## Overview

This document describes the multi-turn controllability variant of Study B. It extends the single-turn controllability idea into rolling-context conversations, where the user applies sustained diagnostic pressure across many turns.

The goal is to test whether a model can keep following the anti-sycophancy reasoning constraint throughout a pressured conversation rather than only in a one-shot prompt.

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

This variant uses:

- rolling `conversation_history`
- `runner.chat(...)`
- `mode="cot_controlled"`

For each turn, the runner appends the next user message, applies the case-specific constraint, generates the assistant response from the full dialogue history, and stores a per-turn cache entry.

## Output

Per-model cache output:

- `results/<model>/ctrl_study_b_multi_turn_generations.jsonl`

Resume semantics use `case_id + variant + turn_num`, so interrupted conversations can continue from the first missing successful turn.

## Metric Status

This variant currently acts as a controllability generation and analysis cache. It does not yet have a standalone exported controllability metric helper equivalent to the Study A, Study B single-turn, or Study C helpers.

## Related Files

- Commands: `docs/studies/controllability/study_b/study_b_multi_turn_controllability_commands.md`
- Single-turn controllability guide: `docs/studies/controllability/study_b/study_b_controllability.md`
- Base Study B multi-turn guide: `docs/studies/study_b/study_b_multi_turn.md`
