# Study C Controllability: Controlled Entity Recall

## Overview

This document describes the controllability variant of Study C. The task is to test whether a model can follow an explicit memory-retention reasoning constraint while conversations unfold and summaries are regenerated turn by turn.

The controllability target for this study is **Controlled Entity Recall (CER)**.

This benchmark adaptation follows the CoT controllability framing introduced in Chen et al., *Reasoning Models Struggle to Control their Chains of Thought*:
[https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf](https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf)

## Data and Gold Artefacts

- Split file: `data/controllability_splits/study_c_controllability_test.json`
- Gold plans file: `data/controllability_splits/ctrl_target_plans.json`
- Source build script: `scripts/preprocessing/build_controllability_splits.py`
- Gold-plan script: `scripts/studies/controllability/generate_gold_plans.py`

Each case stores:

- `patient_summary`
- `critical_entities`
- `turns`
- `cot_controlled_constraint`
- source provenance and inferred condition metadata

## Metric

- Compliance check: `check_controlled_entity_recall()` in `src/reliable_clinical_benchmark/metrics/drift.py`
- Aggregate metric: `calculate_controlled_entity_recall()` in `src/reliable_clinical_benchmark/metrics/drift.py`

A summary counts as compliant when it retains at least the required fraction of the case’s `critical_entities`.

## Generation Path

- Unified runner: `hf-local-scripts/run_ctrl_generate_only.py`
- Auto launcher: `scripts/dev/run_generation_auto.py`

Study C controllability generates two outputs per turn:

- `variant="summary"` using `mode="cot_controlled_summary"`
- `variant="dialogue"` using `mode="cot_controlled"`

The `cot_controlled_summary` mode exists specifically so the base prompt formatter asks for a summary rather than a diagnosis.

## Output

Per-model cache output:

- `results/<model>/ctrl_study_c_generations.jsonl`

Per-model evaluation output:

- `results/<model>/ctrl_study_c_results.json`

Resume semantics use `case_id + variant + turn_num`, so summary and dialogue generations resume independently.

## Threshold and Reporting Notes

- `CER` remains the primary controllability metric for Study C.
- The controllability evaluation pipeline also writes a controlled performance profile for the underlying Study C metrics, including Recall@T10, optional conflict/alignment diagnostics, and derived drift summaries.
- Formal benchmark gates still come from `docs/spec/Metrics and Evaluation.tex`.
- Alignment and drift targets remain provisional or derived until uncontrolled frozen-split baseline calibration confirms them, so any Study C controllability roll-up is marked experimental.

## Related Files

- Commands: `docs/studies/controllability/study_c/study_c_controllability_commands.md`
- Base Study C guide: `docs/studies/study_c/study_c_drift.md`
