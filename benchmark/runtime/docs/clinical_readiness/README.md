# Clinician-Readiness Documentation (v0.3)

This folder records the final process that moved the benchmark from the pre-scaling small-scope run, through scaling, into the clinician-readiness v0.3 line.

The aim is not to claim full clinician adjudication of OpenR1-Psy. The aim is to make this benchmark line reproducible, auditable, and operationally clean for clinician-facing review.

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

## Process Line Covered

1. Small-scope baseline (documented 300-ish run)
2. Scaling to benchmark-ready sample sizes (A/B/C expansion)
3. Clinician-readiness hardening:
   - label/mapping/metadata integrity
   - Study B and Study C validator hardening
   - frozen snapshot and release manifest discipline
   - deterministic clinician package with preflight gates

## Branch Evidence Basis

The report is grounded in the current branch commit path for runtime data/docs/scripts, especially:

- `3cbc5ed` -> initial clinician-sendoff packaging line
- `88c4310`, `11f4028`, `6a660fe` -> pre/post freeze and rerun evidence
- `bec71fc`, `3f99a38`, `fd3a955` -> v0.3 finalisation, stage gates, and delta tightening

The write-up uses process-level interpretation (why changes were made), not commit-by-commit narration.
