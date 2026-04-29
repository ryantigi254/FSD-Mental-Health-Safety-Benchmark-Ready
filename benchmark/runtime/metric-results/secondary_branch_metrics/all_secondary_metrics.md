# Secondary Branch Metrics Summary

## Status Counts

| Lane | ok | missing_cache | error | total |
|---|---:|---:|---:|---:|
| controllability | 24 | 6 | 0 | 30 |
| invariance | 23 | 7 | 0 | 30 |
| ctrl-invariance | 21 | 9 | 0 | 30 |

## Provenance

- `controllability`: `/tmp/nlp-ready-controllability/benchmark/runtime/data/controllability/misc/controllability_splits_large/base` (controllability-v2.1 branch worktree).
- `invariance`: `/tmp/nlp-ready-metric-invariance/benchmark/runtime/data/invariance/v5/base/v2_1` (metric-invariance branch worktree). Directory is labelled v5, but manifest source_root is frozen_splits/v6_1 and parent_version is v6.1_strict_no_generation.
- `ctrl-invariance`: `/tmp/nlp-ready-metric-invariance/benchmark/runtime/data/invariance/ctrl/base/v2_1` (metric-invariance branch worktree). Manifest source profile is controllability and parent_version is v6.1_strict_no_generation.
- NLI: enabled for all written jobs (`use_nli=True`, `nli_stride=1`); Study C NLI uses the project `NLIModel` path and MPS when available.

## Missing Coverage

| Lane | Model | Missing Studies |
|---|---|---|
| controllability | deepseek-r1-lmstudio | study_a, study_b, study_b_multi_turn, study_c |
| controllability | psyche-r1-local | study_b_multi_turn, study_c |
| invariance | deepseek-r1-lmstudio | study_a, study_a_bias, study_b, study_b_multi_turn, study_c |
| invariance | psyche-r1-local | study_b_multi_turn, study_c |
| ctrl-invariance | deepseek-r1-lmstudio | study_a, study_a_bias, study_b, study_b_multi_turn, study_c |
| ctrl-invariance | psyche-r1-local | study_a, study_b, study_b_multi_turn, study_c |

## Flat Metric Rows

- Rows: `268`
- Payloads: `68`
