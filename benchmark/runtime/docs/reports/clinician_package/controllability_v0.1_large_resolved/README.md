# Controllability Clinician-Readiness Package

- Release line: `controllability_v0.1_large_resolved`
- Scope: large resolved controllability suite only at `data/controllability_splits_large_resolved`
- Verification root: `data/verification/controllability_v0.1_large_resolved`
- Review mode: deterministic review-first, no auto-repair or resampling
- This package is separate from the base-study clinician-ready v0.3 line.
- Gold diagnosis labels and target plans are probe-backed, weakly supervised artefacts.
- Robust `.robust.json` files are diagnostic appendices and are not release-gating inputs.
- Gate status: `ready`
- Review status: `all studies acceptable`
- Package release status: `ready`

The package remains blocked unless every study is fully `ACCEPTABLE` and every stage-2 gate passes.
