# Metric Calculation Pipeline (v3.2)

This document defines the current operational pipeline from cached generations to metric artefacts.

## 1. Input/output contract

Inputs (read-only runtime data and generation caches):

- Study A main cache: `results/{model-id}/study_a_generations.jsonl`
- Study A bias cache: `results/{model-id}/study_a_bias_generations.jsonl`
- Study B cache: `results/{model-id}/study_b_generations.jsonl`
- Study C cache: `results/{model-id}/study_c_generations.jsonl`
- Gold/test data resolved from:
  - default: latest release pointed to by `data/releases/LATEST.md`
  - fallback mode: legacy working data under `data/`

Outputs:

- `metric-results/study_a/all_models_metrics.json`
- `metric-results/study_a/study_a_bias_metrics.json`
- `metric-results/study_b/sycophancy_metrics.json`
- `metric-results/study_c/drift_metrics.json`

## 2. Calculation flow

1. Generate and cache model outputs in `results/{model-id}/...`.
2. Resolve dataset root:
   - default: `--data-source latest_release`
   - legacy mode: `--data-source working_data`
   - explicit override: `--data-root <release-root-or-frozen-v5-root>`
3. Run metric scripts by study.
4. Persist metric JSON artefacts under `metric-results/`.
5. Use downstream analysis notebooks/reports from `metric-results/` outputs.

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
For Study B deterministic smoke runs in constrained environments, use `--no-nli`.

Data-source options (Study A/B/C metric scripts):

```bash
# Use legacy in-tree working data
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/studies/study_a/metrics/calculate_metrics.py --data-source working_data

# Use explicit data root (release directory or custom root with required subfolders)
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/studies/study_b/metrics/calculate_metrics.py \
  --data-root data/releases/clinician_readiness_v4_2026-02-22

# Use the frozen v5 snapshot directly
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/studies/study_c/metrics/calculate_metrics.py \
  --data-root data/frozen_splits/v5
```

## 3.1 Invariance workflow on frozen v5

The clinician-ready invariance tooling sits alongside the canonical metric pipeline and keeps the metric definitions unchanged:

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/analysis/analyse_clinical_distribution.py --study study_a
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/studies/v5_review/generate_invariance_manifest.py --study study_a --sample-size 150
PYTHONNOUSERSITE=1 PYTHONPATH=src python scripts/evaluation/run_invariance_comparison.py \
  --study study_a \
  --base-cache results/<model>/study_a_generations.jsonl \
  --variant-cache results/<model>/study_a_invariance_lexical.jsonl \
  --out metric-results/<model>/study_a_invariance_lexical.json
```

Operational note:

- Treat invariance as a secondary diagnostic layer over the frozen split, not as a replacement for full benchmark runs.
- Suggested subset percentages are heuristic budgeting guidance only; the stronger constraint is metric-aligned sampling units plus coverage over the relevant persona / risk / age / condition axes.
- Base and variant caches must preserve one-to-one pairing by evaluation unit; the comparison wrapper fails closed on ID mismatches.

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
