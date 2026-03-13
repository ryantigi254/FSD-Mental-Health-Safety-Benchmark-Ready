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

## Explicit data-root support

The metric scripts now accept either:

- the packaged release layout (`openr1_psy_splits/`, `study_a_gold/`, `study_c_gold/`), or
- the frozen snapshot layout used by `data/frozen_splits/v5` (top-level `study_*` files with `study_a/` and `study_c/` subdirectories).

That keeps the existing metric contract intact whilst allowing invariance runs directly against the frozen v5 snapshot.
