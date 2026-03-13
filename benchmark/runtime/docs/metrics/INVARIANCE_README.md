# Invariance Evaluation (v5 frozen splits)

This note covers the clinician-ready invariance workflow built on the frozen `v5` snapshot at `data/frozen_splits/v5`.

It is a diagnostic stress-test layer. It does not replace the canonical full-run benchmark metrics.

## Entry points

- Distribution summary: `scripts/analysis/analyse_clinical_distribution.py`
- Deterministic sampling manifests: `scripts/studies/v5_review/generate_invariance_manifest.py`
- Paired cache comparison: `scripts/evaluation/run_invariance_comparison.py`

## Defaults

- Data source defaults to `data/frozen_splits/v5`.
- Sampling is deterministic (`--seed`, default `42`).
- Paired deltas use percentile bootstrap CIs with `>=1000` resamples by default.
- Gold labels and metric definitions stay unchanged; only the sampled prompt/case set varies.

## Budgeting note

- Suggested subset sizes such as `5-10%` for single-turn studies or `10-20%` for multi-turn studies are engineering heuristics for a first-pass diagnostic screen.
- Those percentages are not benchmark-mandated thresholds.
- The hard requirement is to sample from the frozen clinician-ready split using metric-aligned units:
  - Study A: vignette
  - Study B single-turn: control/injected pair
  - Study B multi-turn: full conversation
  - Study C: full case/session
- Coverage matters more than raw percentage. Preserve persona, risk, age-band, and condition coverage where the split exposes those axes.

## Output locations

- Sampling manifests: `data/frozen_splits/v5_invariance_samples/`
- Variant caches: `results/{model-id}/study_*_invariance_*.jsonl`
- Comparison outputs: `metric-results/{model-id}/...` or an explicit `--out` path
- Materialised sampled split root: `data/frozen_splits/v5_invariance_samples/`

Manifest metadata includes:

- `coverage_axes`: the primary balancing axes for that study
- `available_counts` / `selected_counts`: quick coverage summaries
- `sampling_role=diagnostic_subset`: explicit marker that this is an add-on diagnostic layer

## Pairing contract

- Base and variant runs must contain the same paired evaluation units for the metric being compared.
- The comparison wrapper now fails closed on mismatched IDs / case IDs instead of silently intersecting the two caches.

## Explicit data-root support

The metric scripts now accept either:

- the packaged release layout (`openr1_psy_splits/`, `study_a_gold/`, `study_c_gold/`), or
- the frozen snapshot layout used by `data/frozen_splits/v5` (top-level `study_*` files with `study_a/` and `study_c/` subdirectories).

That keeps the existing metric contract intact whilst allowing invariance runs directly against the frozen v5 snapshot.

## Building the sampled split root

Use the all-in-one builder to regenerate both manifests and sampled split files:

- `PYTHONPATH=src python scripts/studies/v5_review/build_invariance_splits.py`

The generated root is a valid frozen-snapshot-style `data-root`, containing:

- `study_a_test.json`
- `study_b_test.json`
- `study_b_multi_turn_test.json`
- `study_c_test.json`
- `study_a/gold_diagnosis_labels.json`
- `study_c/study_c_target_plans.json`

## Generation commands

Dedicated generation runners now mirror the normal study entrypoints:

- `hf-local-scripts/run_study_a_invariance_generate_only.py`
- `hf-local-scripts/run_study_b_invariance_generate_only.py`
- `hf-local-scripts/run_study_b_multi_turn_invariance_generate_only.py`
- `hf-local-scripts/run_study_c_invariance_generate_only.py`

And the auto launcher supports:

- `study_a_invariance`
- `study_b_invariance`
- `study_b_multi_turn_invariance`
- `study_c_invariance`
