# Clinical Readiness Next

This note marks the handoff point after controllability scaling and before the clinician-readiness phase.

## What Controllability Scaling Has Guaranteed

- The controllability splits were rebuilt from unused OpenR1-Psy rows with the stronger shared condition-resolution path, rather than the earlier weaker fallback-heavy path.
- The rebuilt controllability splits are study-specific and frozen under `data/controllability_splits/`.
- Study A controllability gold diagnosis labels are now regenerated with the probe-backed classifier path in `scripts/studies/controllability/generate_gold_labels.py`, with `BiomedBERT` as the canonical primary model and `BiomedBERT + BioClinicalBERT` as the recorded robustness pair.
- Study C controllability target plans are now regenerated with the probe-backed condition-recovery path in `scripts/studies/controllability/generate_gold_plans.py`, with `BiomedBERT` as the canonical primary model and `BiomedBERT + BioLinkBERT` as the recorded robustness pair.
- The generation path for controllability variants is now explicit and separable from the base studies through `hf-local-scripts/run_ctrl_generate_only.py` and `scripts/dev/run_generation_auto.py`.
- The large resolved controllability suite now lives under `data/controllability_splits_large_resolved/`, replacing the earlier `data/controllability_splits/scaled/` artefacts.
- The docs now describe the controllability variants per study and cite the CoT controllability reference paper:
  [https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf](https://cdn.openai.com/pdf/a21c39c1-fa07-41db-9078-973a12620117/cot_controllability.pdf)

## What It Does Not Guarantee

- It does not prove clinical correctness of every OpenR1-derived prompt, label, or plan.
- It does not turn Study A Bias or Study B multi-turn into fully formalised controllability metric tracks yet; those remain generation-first tracks unless dedicated helpers are added later.
- It does not replace clinician review, release gating, or audit packaging.
- It does not guarantee that future reruns stay clean unless the artefact and smoke tests continue to pass.

## Source-of-Truth Artefacts

- Split builder: `scripts/preprocessing/build_controllability_splits.py`
- Threshold register: `src/reliable_clinical_benchmark/metrics/thresholds.py`
- Controllability evaluation runner: `scripts/evaluation/run_controllability_pipeline.py`
- Controllability result writer: `src/reliable_clinical_benchmark/pipelines/controllability.py`
- Rebuilt controllability splits: `data/controllability_splits/study_a_controllability_test.json`
- Rebuilt controllability splits: `data/controllability_splits/study_a_bias_controllability_test.json`
- Rebuilt controllability splits: `data/controllability_splits/study_b_controllability_test.json`
- Rebuilt controllability splits: `data/controllability_splits/study_b_multi_turn_controllability_test.json`
- Rebuilt controllability splits: `data/controllability_splits/study_c_controllability_test.json`
- Study A controllability gold labels: `data/controllability_splits/ctrl_gold_diagnosis_labels.json`
- Study C controllability gold plans: `data/controllability_splits/ctrl_target_plans.json`
- Large resolved Study A controllability gold labels: `data/controllability_splits_large_resolved/ctrl_gold_diagnosis_labels.json`
- Large resolved Study C controllability gold plans: `data/controllability_splits_large_resolved/ctrl_target_plans.json`
- Study docs index: `docs/studies/controllability/README.md`
- Scaling docs index: `docs/controllability_scaling/README.md`
- Gold-generation rationale: `docs/studies/controllability/gold_generation.md`
- Structured controllability outputs: `results/<model>/ctrl_study_a_results.json`, `results/<model>/ctrl_study_b_results.json`, `results/<model>/ctrl_study_c_results.json`, and `results/<model>/controllability_summary.json`

## Entry Criterion For Clinical Readiness Work

Treat controllability scaling as complete only when the controllability artefact regression tests and gold-script smoke tests pass against the checked-in files in this branch. After that point, clinician-readiness work should focus on auditability, release discipline, and clinician-facing review quality rather than split-regeneration mechanics.
