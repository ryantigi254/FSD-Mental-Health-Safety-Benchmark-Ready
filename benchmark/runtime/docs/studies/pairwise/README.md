# Pairwise Secondary Evaluation

This directory documents the notebook-driven pairwise layer for secondary communication-quality analysis.

## What pairwise is allowed to evaluate

- clarity and comprehensibility
- validation and reflection quality
- grounded helpfulness
- respectful non-stigmatising tone
- response economy
- boundary-safe validation
- DBT-consistent skill appropriateness on tagged cases only
- method-fit / therapeutic-strategy appropriateness on tagged cases only
- multi-turn sequencing and repair quality on tagged cases only
- requested-control fidelity and preserved quality under controllability slices
- perceived equivalence and preserved quality under invariance slices

## What pairwise is not allowed to evaluate

- clinical correctness
- crisis safety correctness
- hallucination truthfulness
- diagnosis quality
- risk classification

Those remain in the primary benchmark metrics and rule-based safety analyses.

## Fixed judge panel

- `google/gemma-4-31B-it`
- `Jackrong/Qwopus3.5-27B-v3-GGUF`
- `TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill`
- `TeichAI/Qwen3-14B-GPT-5.2-High-Reasoning-Distill-GGUF`

The panel is fixed so cross-judge disagreement is measurable rather than silently drifting with judge choice.

## Canonical outputs

- manifests: `metric-results/pairwise/manifests/`
- raw judge traces: `metric-results/pairwise/raw/<run_id>/`
- parsed judgements: `metric-results/pairwise/parsed/<run_id>/`
- aggregate JSON: `metric-results/pairwise/aggregates/<run_id>/`
- final report JSON and CSV mirrors: `metric-results/pairwise/reports/<run_id>/`

Notebooks read only the canonical report files.
