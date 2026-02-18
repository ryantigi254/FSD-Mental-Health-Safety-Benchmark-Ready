# Frozen Snapshot v4 (Rubric v2 ACCEPTABLE-Only)

This snapshot freezes all data rows that passed the v4 cross-study deterministic review (rubric v2).
It is a historical acceptable-only freeze and remains immutable.

## Source
- Input snapshot: `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/benchmark/runtime/data/frozen_splits/v0.3_postclinician_audit`
- Verification outputs: `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/benchmark/runtime/data/verification/v4`
- Rule file: `rubric_rules_v2.json` (`rule_version=v2`, `matching_mode=word_boundary`)

## Selection Policy
- Include only rows with `verdict=ACCEPTABLE` from the v4 SSV outputs.
- No source rows were edited; this is a filtered freeze.

## Adversarial Bias Traceability
- `v4` does not include an `adversarial_bias` subtree.
- Canonical adversarial bias source remains:
  - `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/benchmark/runtime/data/frozen_splits/v0.3_postclinician_audit/adversarial_bias`
- Promoted full resampled traceable successor snapshot is:
  - `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/benchmark/runtime/data/frozen_splits/v4_1_resampled`
  - includes: `/Users/ryangichuru/Documents/SSD-K/Uni/3rd year/NLP-ready/benchmark/runtime/data/frozen_splits/v4_1_resampled/adversarial_bias`

## Counts
- Study A: 1232 / 2000 kept
- Study B single-turn: 2000 / 2000 kept
- Study B multi-turn: 120 / 120 kept
- Study C: 100 / 100 kept

See `manifest.json` for checksums and provenance.
