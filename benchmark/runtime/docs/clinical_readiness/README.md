# Clinician-Readiness Documentation (v0.3)

This folder records the final process that moved the benchmark from the pre-scaling small-scope run, through scaling, into the clinician-readiness v0.3 line.

The aim is not to claim full clinician adjudication of OpenR1-Psy. The aim is to make this benchmark line reproducible, auditable, and operationally clean for clinician-facing review.

## Why This Work Exists

The clinician-readiness line exists because upstream OpenR1-Psy validation and downstream benchmark fitness are not the same thing.

- OpenR1-Psy was validated for counselling-quality training use, not for direct reuse as a reference-standard layer in a safety benchmark.
- This benchmark uses the derived artefacts more aggressively: labels, target plans, pressure probes, and reference reasoning traces can all directly change metric outcomes.
- Some of those artefacts are LLM-generated or probe-backed rather than fully clinician-authored, so they need an explicit downstream defensibility layer before release.
- The audit found concrete benchmark-facing failures that upstream validation would not be expected to catch: safety-critical label and plan corrections, mapping incoherence, schema gaps, and rubric false positives.

So the clinical-readiness claim here is deliberately narrow: preserve comparability with OpenR1-Psy, add an auditable downstream verification layer, and avoid overclaiming full clinician adjudication.

## Canonical Documents

- Detailed process report (LaTeX): [`tex/CLINICIAN_READINESS_PROCESS_REPORT.tex`](tex/CLINICIAN_READINESS_PROCESS_REPORT.tex)
- Short index pointer: [`clinical_readiness_summary.md`](clinical_readiness_summary.md)

## Study-Wise Write-ups

- Study A updates: [`study/study_a_clinical_readiness.md`](study/study_a_clinical_readiness.md)
- Study B updates: [`study/study_b_clinical_readiness.md`](study/study_b_clinical_readiness.md)
- Study C updates: [`study/study_c_clinical_readiness.md`](study/study_c_clinical_readiness.md)

## Data and Release Anchors

- Frozen snapshot (audit source-of-truth): `benchmark/runtime/data/frozen_splits/v0.3_postclinician_audit/`
- Current canonical release: `benchmark/runtime/data/releases/clinician_readiness_v0.3_2026-02-16/`
- Release pointer: `benchmark/runtime/data/releases/LATEST.md`
- Clinician package outputs: `benchmark/runtime/docs/reports/clinician_package/v0.3/`
- Controllability candidate snapshot: `benchmark/runtime/data/controllability_splits_large/`
- Controllability verification outputs: `benchmark/runtime/data/verification/controllability_v0.1_large_resolved/`
- Controllability clinician package outputs: `benchmark/runtime/docs/reports/clinician_package/controllability_v0.1_large_resolved/`

## Process Line Covered

1. Small-scope baseline (documented 300-ish run)
2. Scaling to benchmark-ready sample sizes (A/B/C expansion)
3. Clinician-readiness hardening:
   - label/mapping/metadata integrity
   - Study B and Study C validator hardening
   - frozen snapshot and release manifest discipline
   - deterministic clinician package with preflight gates
4. Controllability clinician-readiness hardening:
   - large resolved controllability suite only
   - deterministic five-study review outputs
   - controllability-specific stage-2 gates
   - review-first blocked release policy with no auto-repair

## Controllability Release Line

The controllability line is not a copy of the old base-study `v0.3` packaging path. It has its own review, gates, package, and preflight entrypoints:

- `scripts/studies/controllability_review/run_ctrl_cross_study_review.py`
- `scripts/studies/controllability_review/run_ctrl_stage2_gates.py`
- `scripts/studies/controllability_review/build_ctrl_clinician_package.py`
- `scripts/studies/controllability_review/run_ctrl_sendoff_preflight.py`

This line only certifies `benchmark/runtime/data/controllability_splits_large/`.
It treats the probe-backed gold diagnosis labels and target plans as weakly supervised artefacts and blocks release on any `NEEDS_REVIEW` or `REJECT`.

## Branch Evidence Basis

The report is grounded in the current branch commit path for runtime data/docs/scripts, especially:

- `3cbc5ed` -> initial clinician-sendoff packaging line
- `88c4310`, `11f4028`, `6a660fe` -> pre/post freeze and rerun evidence
- `bec71fc`, `3f99a38`, `fd3a955` -> v0.3 finalisation, stage gates, and delta tightening

The write-up uses process-level interpretation (why changes were made), not commit-by-commit narration.
