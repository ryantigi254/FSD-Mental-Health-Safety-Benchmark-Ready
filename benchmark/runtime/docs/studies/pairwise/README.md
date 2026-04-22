# Pairwise Secondary Evaluation

This directory documents the notebook-driven pairwise layer for secondary communication-quality analysis.

## Folder layout

- `core/`: one note per core study part
- `controllability/`: one note per controllability study part
- `invariance/`: one note for each invariance family

## Canonical execution mode

The default run mode is `stacked`, not “all judges all the time”.

Stacked execution uses the fixed panel as an uncertainty-and-audit stack:

- `primary`: `Jackrong/Qwopus3.5-27B-v3.5-GGUF`
- `audit`: `TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill`
- `escalation_1`: `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF`
- `escalation_2`: `TeichAI/gemma-4-31B-it-Claude-Opus-Distill-GGUF`

Routine flow:

1. Run the primary and audit judges on every pair in both `AB` and `BA`.
2. Accept routine secondary evidence only when they agree after order normalisation.
3. Escalate to the two escalation judges when there is disagreement, tie, invalid output, swap failure, or a configured high-risk tag.
4. Mark the case `uncertain` if escalation does not produce a clean unanimous decisive outcome.

`all_judges` remains available as an explicit audit/debug override only.

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

- `Jackrong/Qwopus3.5-27B-v3.5-GGUF`
- `TeichAI/GLM-4.7-Flash-Claude-Opus-4.5-High-Reasoning-Distill`
- `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF`
- `TeichAI/gemma-4-31B-it-Claude-Opus-Distill-GGUF`

The panel is fixed so cross-judge disagreement is measurable rather than silently drifting with judge choice.

Pooled summaries are secondary views. Per-judge outputs, escalation rates, and persistent disagreement cases are the primary reporting surfaces.

## Canonical outputs

- manifests: `metric-results/pairwise/manifests/`
- raw judge traces: `metric-results/pairwise/raw/<run_id>/`
- parsed judgements: `metric-results/pairwise/parsed/<run_id>/`
- aggregate JSON: `metric-results/pairwise/aggregates/<run_id>/`
- final report JSON and CSV mirrors: `metric-results/pairwise/reports/<run_id>/`
- judge-audit manifests: `metric-results/pairwise/judge_audit/manifests/`
- judge-audit reports: `metric-results/pairwise/judge_audit/reports/`

Notebooks read only the canonical report files.

## Judge-audit layer

The pairwise subsystem also carries a separate judge meta-evaluation layer. It does not treat the judges as ground truth. Instead it scores each judge against a small labelled audit slice on:

- gold agreement
- swap consistency
- prompt invariance
- sensitivity on clearly separated cases
- repeat-run stability
- calibration proxy on ambiguous versus easy cases

Judge-audit outputs justify the current primary/audit/escalation role assignment and provide the evidence needed to revisit it later.
