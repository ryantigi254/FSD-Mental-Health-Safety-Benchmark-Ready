# Study A Bias Scaling Guide (v3.2 Canonical)

This guide is the source-of-truth for Study A bias scaling status and usage.

## Current status

The scaling work is complete for the current line. Runtime canonical data is:

- `benchmark/runtime/data/adversarial_bias/biased_vignettes.json`
- row count: `2000`
- duplicate `(prompt,bias_feature,bias_label)`: `0`
- duplicate `(prompt,bias_feature)`: `0`
- dimensions present: `44`
- explicit pairing: `pair_group_id` (1000 groups, 2 rows per group)
- persona coverage: `40` personas, `50` rows each
- structure version: `v3.2`

Legacy archive retained for provenance:

- `benchmark/runtime/data/adversarial_bias/biased_vignettes_legacy_2016.json` (`2016` rows)

## What this means for runs

- You can run Study A bias generations on the current canonical dataset.
- Do not re-run the old “58/138/synthetic_fill” restructuring plans; they are stale.
- Treat this as a synthetic stress-test dataset, not clinician-adjudicated diagnostic ground truth.

## Metric contract (current)

Current Study A bias metric is `R_SB` in:

- `benchmark/runtime/scripts/studies/study_a/metrics/calculate_bias.py`

Current behaviour is proxy-based:

- biased outcome keyed from `bias_label`
- “silent” keyed from non-mention of `bias_feature`

This contract does not require `correct_diagnosis`.

## What is optional (future expansion)

Not required for current v3.x or current metric:

- `correct_diagnosis` field
- `pair_id` field name (already represented by `pair_group_id`)

Only add `correct_diagnosis` if introducing a new accuracy-aware/pair-aware bias metric version.

## Decision summary

- Keep current v3.2 canonical dataset as active source.
- Keep current `R_SB` contract unchanged for this line.
- If evolving beyond proxy fairness, do it as a separate metric version (v5+), not by retrofitting v3.x outputs.

## Companion docs

- `IMPLEMENTATION_DETAILS.md` in this same folder
- `benchmark/runtime/data/adversarial_bias/README.md`
- `benchmark/runtime/data/adversarial_bias/dimension_catalog_v3_2.json`

*Last Updated: 2026-02-17*
