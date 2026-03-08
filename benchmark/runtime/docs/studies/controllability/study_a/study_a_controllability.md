# Study A Controllability: Reasoning Adherence

## Overview

This document describes the controllability variant of Study A. The goal is not to re-measure the base faithfulness gap, but to test whether a model can follow an explicit reasoning constraint under `cot_controlled` mode and still produce useful clinical reasoning.

The controllability target for this study is **Reasoning Adherence (RA)**: the fraction of controlled-CoT traces that cover enough of the expected gold reasoning steps.

This benchmark adaptation follows the CoT controllability framing introduced in Chen et al., *Reasoning Models Struggle to Control their Chains of Thought*:
[https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf](https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf)

## Data and Gold Artefacts

- Split file: `data/controllability_splits/study_a_controllability_test.json`
- Gold labels file: `data/controllability_splits/ctrl_gold_diagnosis_labels.json`
- Source build script: `scripts/preprocessing/build_controllability_splits.py`
- Gold-label script: `scripts/studies/controllability/generate_gold_labels.py`

The split builder samples unused OpenR1-Psy rows, resolves the underlying condition with the shared condition-resolution pipeline, and stores:

- `prompt`
- `gold_reasoning`
- `cot_controlled_constraint`
- source metadata, including inferred condition/category

The gold-label script mirrors the main Study A gold-label workflow:

- loads `GMLHUHE/OpenR1-Psy`
- pulls `counselor_think` for the source rows
- ranks diagnosis candidates with the DeBERTa NLI model
- falls back to heuristics and the shared case resolver when needed
- fails closed if any controllability labels remain unresolved

## Metric

- Compliance check: `check_reasoning_adherence()` in `src/reliable_clinical_benchmark/metrics/faithfulness.py`
- Aggregate metric: `calculate_reasoning_adherence()` in `src/reliable_clinical_benchmark/metrics/faithfulness.py`

`check_reasoning_adherence()` treats a trace as compliant when it recalls enough of the gold reasoning steps. It reuses the existing step-extraction and token-overlap matching logic, then enforces a minimum matched fraction.

Conceptually:

```text
RA = compliant controlled-CoT traces / total traces
```

## Generation Path

- Unified runner: `hf-local-scripts/run_ctrl_generate_only.py`
- Auto launcher: `scripts/dev/run_generation_auto.py`

Study A controllability generates paired outputs per item:

- `mode="cot_controlled"` for the constrained reasoning run
- `mode="direct"` for the direct comparison run

The runner injects the per-sample constraint via `runner.cot_controlled_constraint`, and the base model prompt formatter applies it through `ModelRunner._format_prompt()`.

## Output

Per-model cache output:

- `results/<model>/ctrl_study_a_generations.jsonl`

Per-model evaluation output:

- `results/<model>/ctrl_study_a_results.json`

Resume semantics use composite keys of `id + mode`, so interrupted runs can continue cleanly without duplicating successful entries.

## Threshold and Reporting Notes

- `RA` remains the primary controllability metric for this study.
- The controllability evaluation pipeline also writes a controlled performance profile for Study A metrics such as the controlled faithfulness gap and Step-F1.
- Formal benchmark gates still come from `docs/spec/Metrics and Evaluation.tex`.
- Any study-level controllability roll-up is marked experimental and does not replace the base Study A faithfulness verdict.

## Related Files

- Commands: `docs/studies/controllability/study_a/study_a_controllability_commands.md`
- Base Study A guide: `docs/studies/study_a/study_a_faithfulness.md`
- Shared prompt logic: `src/reliable_clinical_benchmark/models/base.py`
