# Latest Release

Current canonical release: `clinician_readiness_v0.3_2026-02-16`

Primary path:
- `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/benchmark/runtime/data/releases/clinician_readiness_v0.3_2026-02-16`

Use `manifest.json` in that directory as the source-of-truth inventory.

Notes:
- `clinician_readiness_v0.2_2026-02-15` is retained for historical comparison.
- Study splits live in `data/openr1_psy_splits/`. Adversarial bias, Study A gold, and Study C gold
  artefacts ship under `data/releases/<release_id>/` (see manifest); optional working copies under
  `data/adversarial_bias/`, `data/study_a_gold/`, and `data/study_c_gold/` are not required when the
  release bundle is present.
- Frozen snapshot for audit reproducibility is `data/frozen_splits/v0.3_postclinician_audit/`.
