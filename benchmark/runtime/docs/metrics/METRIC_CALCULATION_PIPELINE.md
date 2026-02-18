# Metric Calculation Pipeline (v3.2)

This document defines the current operational pipeline from cached generations to metric artefacts.

## 1. Input/output contract

Inputs (read-only runtime data and generation caches):

- Study A main cache: `results/{model-id}/study_a_generations.jsonl`
- Study A bias cache: `results/{model-id}/study_a_bias_generations.jsonl`
- Study B cache: `results/{model-id}/study_b_generations.jsonl`
- Study C cache: `results/{model-id}/study_c_generations.jsonl`
- Gold/test data under `data/openr1_psy_splits/` and `data/study_a_gold/`

Outputs:

- `metric-results/study_a/all_models_metrics.json`
- `metric-results/study_a/study_a_bias_metrics.json`
- `metric-results/study_b/sycophancy_metrics.json`
- `metric-results/study_c/drift_metrics.json`

## 2. Calculation flow

1. Generate and cache model outputs in `results/{model-id}/...`.
2. Run metric scripts by study.
3. Persist metric JSON artefacts under `metric-results/`.
4. Use downstream analysis notebooks/reports from `metric-results/` outputs.

`results/` stores raw generations; metric scripts do not rewrite those cached generation files.

## 3. Study-level commands

From `benchmark/runtime`:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_bias.py --bias-dir results --output-dir metric-results/study_a
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_metrics.py
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/studies/study_b/metrics/calculate_metrics.py
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/studies/study_c/metrics/calculate_metrics.py
```

Optional cleaned-path mode is available where supported (`--use-cleaned`).

## 4. Study A bias metric contract

Metric script:

- `scripts/studies/study_a/metrics/calculate_bias.py`

Dataset contract for active runs:

- `data/adversarial_bias/biased_vignettes.json` (canonical v3.2 set)

Current `R_SB` semantics:

- biased outcome keyed from `bias_label`
- silent outcome keyed from non-mention of `bias_feature`

`correct_diagnosis` is not required for current `R_SB` computation.

## 5. Known caveats and resolved issues

The following operational points are already reflected in current metric code paths:

- Reasoning extraction supports `<think>...</think>` blocks and fallback reasoning markers for Study A.
- Study B agreement detection includes nuanced agreement phrasing and contradiction-aware handling.
- Study B evidence hallucination scoring exposes diagnostic counters (`..._n_attempted`, `..._n_scored`) in output.
- Study C session-goal alignment metrics are omitted when required target-plan inputs are unavailable, rather than emitted as misleading defaults.
- Study A Step-F1 enforces one-to-one matching behaviour to avoid repeated-step inflation.

## 6. Validation checkpoints

Recommended checks after metric runs:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python -m pytest tests/unit/metrics/test_faithfulness_metrics.py -q
PYTHONNOUSERSITE=1 PYTHONPATH=src python -m pytest tests/unit/metrics -q
```

Then verify expected artefacts exist under `metric-results/` for all studies.

## 7. Non-goals for this doc

- This pipeline doc does not redefine metric formulas.
- This pipeline doc does not introduce new bias metric scope (for example, accuracy-aware bias variants).
- Any future metric expansion should be versioned separately from the current v3.x contract.
