# Clinical Readiness Next

This note marks the handoff point after controllability scaling and before the clinician-readiness phase.

## What Controllability Scaling Has Guaranteed

- The controllability splits were rebuilt from unused OpenR1-Psy rows with the stronger shared condition-resolution path, rather than the earlier weaker fallback-heavy path.
- The rebuilt controllability splits are study-specific and frozen under `data/controllability/controllability_splits_v2_1/`, with the clinician-readiness candidate snapshot now fixed to `data/controllability/controllability_splits_v2_1/`.
- Study A controllability gold diagnosis labels are now regenerated with the probe-backed classifier path in `scripts/studies/controllability/generate_gold_labels.py`, with `BiomedBERT` as the canonical primary model and `BiomedBERT + BioClinicalBERT` as the recorded robustness pair.
- Study C controllability target plans are now regenerated with the probe-backed condition-recovery path in `scripts/studies/controllability/generate_gold_plans.py`, with `BiomedBERT` as the canonical primary model and `BiomedBERT + BioLinkBERT` as the recorded robustness pair.
- The generation path for controllability variants is now explicit and separable from the base studies through `hf-local-scripts/run_ctrl_generate_only.py` and `scripts/dev/run_generation_auto.py`.
- The large resolved controllability suite now lives under `data/controllability/controllability_splits_v2_1/`, replacing the earlier `data/controllability/controllability_splits_v2_1/scaled/` artefacts.
- The docs now describe the controllability variants per study and cite the CoT controllability reference paper:
  [https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf](https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf)

## What It Does Not Guarantee

- It does not prove clinical correctness of every OpenR1-derived prompt, label, or plan.
- It does not auto-repair rejected controllability rows; the clinician-readiness line for controllability is explicitly review-first and blocking.
- It does not replace clinician review, release gating, or audit packaging.
- It does not guarantee that future reruns stay clean unless the artefact and smoke tests continue to pass.

## Controllability Clinician-Readiness Line

The controllability clinician-readiness path is now separate from the base-study `v0.3` sendoff stack.

- Candidate snapshot: `data/controllability/controllability_splits_v2_1/`
- Verification outputs: `data/verification/controllability_v0.1_large_resolved/`
- Clinician package: `docs/reports/clinician_package/controllability_v0.1_large_resolved/`
- Policy: deterministic review first, no auto-resample step, release blocked on any `NEEDS_REVIEW` or `REJECT`

Operational order:

1. `python scripts/studies/controllability_review/run_ctrl_cross_study_review.py --ctrl-dir data/controllability_splits_large_resolved --out-dir data/verification/controllability_v0.1_large_resolved --clean`
2. `python scripts/studies/controllability_review/run_ctrl_stage2_gates.py --ctrl-dir data/controllability_splits_large_resolved --verification-dir data/verification/controllability_v0.1_large_resolved`
3. `python scripts/studies/controllability_review/build_ctrl_clinician_package.py --ctrl-dir data/controllability_splits_large_resolved --verification-dir data/verification/controllability_v0.1_large_resolved --output-dir docs/reports/clinician_package/controllability_v0.1_large_resolved`
4. `python scripts/studies/controllability_review/run_ctrl_sendoff_preflight.py --ctrl-dir data/controllability_splits_large_resolved --verification-dir data/verification/controllability_v0.1_large_resolved --package-dir docs/reports/clinician_package/controllability_v0.1_large_resolved`

## Source-of-Truth Artefacts

- Split builder: `scripts/preprocessing/build_controllability_splits.py`
- Threshold register: `src/reliable_clinical_benchmark/metrics/thresholds.py`
- Controllability evaluation runner: `scripts/evaluation/run_controllability_pipeline.py`
- Controllability result writer: `src/reliable_clinical_benchmark/pipelines/controllability.py`
- Rebuilt controllability splits: `data/controllability/controllability_splits_v2_1/study_a_controllability_test.json`
- Rebuilt controllability splits: `data/controllability/controllability_splits_v2_1/study_a_bias_controllability_test.json`
- Rebuilt controllability splits: `data/controllability/controllability_splits_v2_1/study_b_controllability_test.json`
- Rebuilt controllability splits: `data/controllability/controllability_splits_v2_1/study_b_multi_turn_controllability_test.json`
- Rebuilt controllability splits: `data/controllability/controllability_splits_v2_1/study_c_controllability_test.json`
- Study A controllability gold labels: `data/controllability/controllability_splits_v2_1/ctrl_gold_diagnosis_labels.json`
- Study C controllability gold plans: `data/controllability/controllability_splits_v2_1/ctrl_target_plans.json`
- Large resolved Study A controllability gold labels: `data/controllability/controllability_splits_v2_1/ctrl_gold_diagnosis_labels.json`
- Large resolved Study C controllability gold plans: `data/controllability/controllability_splits_v2_1/ctrl_target_plans.json`
- Controllability cross-study review runner: `scripts/studies/controllability_review/run_ctrl_cross_study_review.py`
- Controllability stage-2 gates: `scripts/studies/controllability_review/run_ctrl_stage2_gates.py`
- Controllability clinician package builder: `scripts/studies/controllability_review/build_ctrl_clinician_package.py`
- Controllability preflight runner: `scripts/studies/controllability_review/run_ctrl_sendoff_preflight.py`
- Study docs index: `docs/studies/controllability/README.md`
- Scaling docs index: `docs/controllability_scaling/README.md`
- Gold-generation rationale: `docs/studies/controllability/gold_generation.md`
- Structured controllability outputs: `results/<model>/ctrl_study_a_results.json`, `results/<model>/ctrl_study_b_results.json`, `results/<model>/ctrl_study_c_results.json`, and `results/<model>/controllability_summary.json`

## Entry Criterion For Clinical Readiness Work

Treat controllability scaling as complete only when the controllability artefact regression tests and gold-script smoke tests pass against the checked-in files in this branch. After that point, clinician-readiness work should focus on auditability, release discipline, and clinician-facing review quality rather than split-regeneration mechanics.
