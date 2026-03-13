# Metrics Summary (v3.2)

**Last updated**: 2026-02-23
**Scope**: Current metric contracts, inputs/outputs, and interpretation boundaries for Studies A, B, and C.

## Canonical metrics docs

This folder is intentionally kept to two canonical files:

- `METRICS_SUMMARY.md` (this file): contract and interpretation scope
- `METRIC_CALCULATION_PIPELINE.md`: operational calculation pipeline and commands

Supporting v5 invariance workflow:

- `INVARIANCE_README.md`: deterministic sampling and paired-delta comparison on frozen v5

## Inputs and outputs

Primary inputs:

- default source for Study A/B/C metrics: latest release from `data/releases/LATEST.md`
- override options:
  - `--data-source working_data` to read legacy `data/` paths
  - `--data-root <root>` for explicit release/custom root
- required dataset layout at resolved root:
  - `openr1_psy_splits/study_a_test.json`
  - `openr1_psy_splits/study_b_test.json`
  - `openr1_psy_splits/study_b_multi_turn_test.json`
  - `openr1_psy_splits/study_c_test.json`
  - `study_a_gold/gold_diagnosis_labels.json`
  - `study_c_gold/`
- explicit `--data-root` may also point at the frozen snapshot layout used by `data/frozen_splits/v5`
- `data/adversarial_bias/biased_vignettes.json` (canonical v3.2 Study A bias set)

Primary outputs:

- `metric-results/study_a/all_models_metrics.json`
- `metric-results/study_a/study_a_bias_metrics.json`
- `metric-results/study_b/sycophancy_metrics.json`
- `metric-results/study_c/drift_metrics.json`

## Study A metric contract

Primary:

- `faithfulness_gap = acc_cot - acc_early`
- `acc_cot`
- `acc_early`

Diagnostic:

- `step_f1`

Supplementary:

- `silent_bias_rate` (`R_SB`)

### Study A bias scope note (current contract)

- `R_SB` is computed from `bias_label` and `bias_feature` behaviour in generated reasoning/output.
- Current contract is proxy-based and does **not** require `correct_diagnosis`.
- `correct_diagnosis` is reserved for a future, separate metric version (accuracy-aware/pair-aware bias scoring), not for current v3.x reporting.

## Study B metric contract

Primary:

- `sycophancy_probability` (`P_Syc`)

Diagnostic:

- `evidence_hallucination` (`H_Ev`) when NLI is enabled

Supplementary:

- `flip_rate`
- `turn_of_flip` (where applicable in analysis outputs)
- Study B CLI supports `--no-nli` for deterministic smoke runs; this forces `H_Ev` to `0.0`

## Study C metric contract

Primary:

- `entity_recall_t10`
- entity recall curve metrics

Diagnostic:

- `knowledge_conflict_rate`

Supplementary:

- `session_goal_alignment` family (only when required target-plan data is available)

## Reproducibility and omission behaviour

- Bootstrap CIs are computed for supported metrics via the shared stats utilities.
- Deterministic sampling/index order is used where applicable in offline scoring stages.
- When a supplementary metric cannot be computed from available data, it is treated as missing and omitted from result payloads instead of being forced to `0.0`.

## Quick reference

Core calculation scripts:

- `scripts/studies/study_a/metrics/calculate_metrics.py`
- `scripts/studies/study_a/metrics/calculate_bias.py`
- `scripts/studies/study_b/metrics/calculate_metrics.py`
- `scripts/studies/study_c/metrics/calculate_metrics.py`

Core metric modules:

- `src/reliable_clinical_benchmark/metrics/faithfulness.py`
- `src/reliable_clinical_benchmark/metrics/sycophancy.py`
- `src/reliable_clinical_benchmark/metrics/drift.py`
- `src/reliable_clinical_benchmark/utils/stats.py`

## Related docs

- `benchmark/runtime/docs/studies/study_a/study_a_bias.md`
- `benchmark/runtime/docs/scaling/study/study a bias/BIAS_SCALING_GUIDE.md`
- `benchmark/runtime/docs/spec/Metrics and Evaluation.tex`
