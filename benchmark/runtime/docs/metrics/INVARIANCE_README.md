# Invariance Evaluation (v5 frozen splits)

This note covers the clinician-ready invariance workflow built on the frozen
`v5` snapshot at `data/frozen_splits/v5`, plus the secondary
controllability-backed invariance sample built from
`data/controllability_splits_large_resolved`.

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
- Controllability-backed manifests:
  `data/controllability_splits_large_resolved_invariance_samples/`
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
- `study_c/target_plans.json`
- `study_c/study_c_target_plans.json` (legacy compatibility alias)

For the controllability-backed invariance sample, run:

- `PYTHONPATH=src python scripts/studies/v5_review/build_invariance_splits.py --sample-profile controllability --data-root data/controllability_splits_large_resolved --output-root data/controllability_splits_large_resolved_invariance_samples`

The controllability profile intentionally uses a slightly smaller diagnostic
budget than the main `v5` profile:

- Study A: `140`
- Study B: `150`
- Study B multi-turn: `10`
- Study C: `12`

## Variant-family workflow

The invariance workflow is now explicitly:

1. build one frozen sampled root
2. keep that sampled root fixed
3. fan out multiple variant families from the same sampled IDs
4. compare base vs variant caches on matched IDs or case IDs

That keeps the perturbation effect separable from sample-composition drift.

Build a variant matrix from one sampled root with:

- `PYTHONPATH=src python scripts/invariance/build_variant_family_matrix.py --base-root data/controllability_splits_large_resolved_invariance_samples --output-root data/invariance_variants/controllability`

Concrete variant menu by study:

- Study A: `lexical`, `surface`, `syntax`, `instruction`
- Study B single-turn: `paraphrase`, `mild`, `moderate`, `strong`, `question`, `cultural`
- Study B multi-turn: `schedule_earlier`, `schedule_later`, `tone_gentle`, `tone_direct`, `tone_confrontational`, `pressure_milder`, `pressure_stronger`
- Study C: `summary_short`, `summary_long`, `patient_turn_rephrase`, `noncritical_reorder`

When generating against one of those variant roots, pass a `--variant-tag` so
multiple family runs do not collide on the default cache filename.

## Generation commands

The canonical direct runner is:

- `hf-local-scripts/run_invariance_generate_only.py`

And the auto launcher supports:

- `study_a_invariance`
- `study_b_invariance`
- `study_b_multi_turn_invariance`
- `study_c_invariance`

Dedicated study-command notes now live under:

- `docs/studies/invariance/README.md`
- `docs/studies/invariance/invariance_commands.md`
- `docs/studies/invariance/study_a/`
- `docs/studies/invariance/study_b/`
- `docs/studies/invariance/study_c/`
- `docs/invariance_scaling/README.md`

## Analysis scripts and notebooks

Analysis scripts:

- `scripts/evaluation/summarize_invariance_results.py`
- `scripts/evaluation/run_controllability_comparison.py`
- `scripts/evaluation/export_invariance_case_deltas.py`
- `scripts/evaluation/export_failure_cards.py`

Dedicated notebooks:

- `notebooks/invariance/invariance_analysis.ipynb`
- `notebooks/invariance/controllability_analysis.ipynb`
- `notebooks/invariance/study_a_granular_analysis.ipynb`
- `notebooks/invariance/study_b_granular_analysis.ipynb`
- `notebooks/invariance/study_c_granular_analysis.ipynb`

Variant-construction and validation scripts:

- `scripts/invariance/build_variant_family_matrix.py`
- `scripts/invariance/variant_catalog.py`
- `scripts/invariance/control_paraphrases.py`
- `scripts/invariance/pressure_variants.py`
- `scripts/invariance/reorder_turns.py`
- `scripts/review/run_rubric.py`

## Relationship to controllability

The next layer above invariance is controllability.

- Invariance answers: *does behaviour stay stable under semantically harmless prompt changes?*
- Controllability answers: *how strongly does behaviour move when an explicit control signal is changed, and is that movement itself robust?*

In this repo, controllability should therefore be implemented as a meta-metric on top of the current study metrics and the invariance machinery already added here.

## Distribution figures

The sampled `v5` invariance root can also be rendered through the same PGFPlots figure templates used by the main `Overall` clinical coverage bundle.

Generate the invariance distribution snapshot:

- `cd benchmark/runtime`
- `python3 analysis/analyse_clinical_distribution.py --data-root data/frozen_splits/v5_invariance_samples --out-dir analysis/invariance --skip-figures`

Render the PGFPlots figure bundle:

- `python3 analysis/generate_pgfplots.py --analysis-json analysis/invariance/distribution_analysis.json --out-dir analysis/figures/pgfplots/Invariance`
- `python3 analysis/generate_pgfplots_additional.py --analysis-json analysis/invariance/distribution_analysis.json --out-dir analysis/figures/pgfplots/Invariance`

That produces an `Invariance` sibling folder under `analysis/figures/pgfplots/` with the same figure family names as the main `Overall` set, but driven by the invariance sampled split composition instead of the full benchmark release.

See:

- `CONTROLLABILITY_META_METRIC.md`
- *Large-Language-Model Reasoning Failures*. arXiv:2602.06176, 2026. Available at `https://arxiv.org/pdf/2602.06176`
