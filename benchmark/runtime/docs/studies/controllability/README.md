# Controllability Study Guides

This folder is organised by study:

- `study_a/`
- `study_b/`
- `study_c/`

## Background Reference

The benchmark-specific controllability framing in this folder is adapted from:

- Chen et al., *Reasoning Models Struggle to Control their Chains of Thought*:
  [https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf](https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf)

## Shared Prerequisites

1. Build the controllability splits under `data/controllability_splits/`
1. Generate controllability gold labels:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/studies/controllability/generate_gold_labels.py
```

1. Generate controllability gold plans:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/studies/controllability/generate_gold_plans.py
```

## Unified Runner

Per-study command-only docs:

- `study_a/study_a_controllability_commands.md`
- `study_a/study_a_bias_controllability_commands.md`
- `study_b/study_b_controllability_commands.md`
- `study_b/study_b_multi_turn_controllability_commands.md`
- `study_c/study_c_controllability_commands.md`

All controllability generation uses:

```bash
cd benchmark/runtime
PYTHONPATH=src python hf-local-scripts/run_ctrl_generate_only.py \
    --study <ctrl_study_name> \
    --model-id <model_id>
```

Useful arguments:

- `--max-cases`
- `--max-tokens`
- `--output-dir`
- `--cache-out`

## Threshold and Reporting Layer

Controllability now has a separate evaluation/reporting pass after generation:

```bash
cd benchmark/runtime
PYTHONPATH=src python scripts/evaluation/run_controllability_pipeline.py \
    --model <results_model_dir>
```

The threshold contract is intentionally split:

- formal benchmark deployment gates still come from `docs/spec/Metrics and Evaluation.tex`
- stricter bands on the metric pages remain interpretation guidance
- missing controllability-only targets are stored as provisional or derived and stay reporting-only until an uncontrolled frozen-split baseline confirms them

Result files written by the controllability pipeline:

- `results/<model>/ctrl_study_a_results.json`
- `results/<model>/ctrl_study_b_results.json`
- `results/<model>/ctrl_study_c_results.json`
- `results/<model>/controllability_summary.json`

Primary controllability metrics stay unchanged:

- Study A: `RA`
- Study B single-turn: `CHR`
- Study C: `CER`

Study-level and benchmark-level controllability roll-ups are written as experimental summaries. They do not replace the base benchmark safety card.
