# Adversarial Bias Dataset (Study A Bias)

## Purpose
Synthetic adversarial bias probes used for Study A bias evaluation.

## Scope and caveat
- Synthetic stress-test data only.
- Not clinician-validated gold data.
- Not treatment-plan ground truth.

## Active vs legacy files
- Canonical active file: `biased_vignettes.json` (v3.2 structure-quality set; 2000 non-duplicate rows).
- Legacy archive: `biased_vignettes_legacy_2016.json` (historical 2016-row source retained for reproducibility and audit).

## Pass/fail interpretation
This set is used by the Study A bias metric to detect biased agreement/refusal behaviour under controlled prompts. It is not a clinical diagnosis benchmark.

## File inventory
- `biased_vignettes.json`: canonical active dataset for current runs.
- `biased_vignettes_legacy_2016.json`: legacy source archive.
- `dimension_catalog_v3_2.json`: explicit v3.2 dimension contract and floor/weight policy.
- `BIAS_DIMENSIONS.md`: dimension taxonomy and notes.
- `README.md`: this contract note.

## Expected schema (per case)
Required top-level keys:
- `id`
- `prompt`
- `bias_feature`
- `bias_label`
- `metadata`

Required metadata key:
- `dimension`

v3.2 canonical rows include:
- `pair_group_id`
- `template_signature`
- `structure_version`
- `source_variant_count`
