# Study A Bias Implementation Details (v3.2)

This file captures the implementation shape behind the canonical Study A bias dataset.

## Dataset and provenance

Canonical dataset:

- `benchmark/runtime/data/adversarial_bias/biased_vignettes.json`

Legacy archive:

- `benchmark/runtime/data/adversarial_bias/biased_vignettes_legacy_2016.json`

Pinned upstream source used in rebuild metadata:

- OpenR1 revision: `56fc0ef2fa5926df86713ed9b35f8689a6f85425`

## Structural fields in canonical rows

Required top-level fields:

- `id`
- `prompt`
- `bias_feature`
- `bias_label`
- `pair_group_id`
- `template_signature`
- `structure_version`
- `source_variant_count`
- `metadata`

Required metadata fields:

- `dimension`
- `persona_id`
- `source_openr1_split`
- `source_openr1_id`
- `openr1_revision`
- `dimension_family`

## Coverage contract

- `2000` total rows
- `2000` unique `(prompt,bias_feature,bias_label)` triples
- `40` personas, exactly `50` rows each
- `44` dimensions represented with configured floors
- `1000` `pair_group_id` values, exactly 2 rows per group

## Validation and checks

Primary checks are covered in:

- `benchmark/runtime/tests/unit/data/test_data_splits_invariants.py`
- `benchmark/runtime/tests/unit/data/test_adversarial_bias_structure_quality.py`
- `benchmark/runtime/tests/unit/data/test_adversarial_bias_v03_contract.py`

## Future expansion boundary

For current v3.x reporting, no schema expansion is required for `R_SB`.

Future metric expansion candidates (separate version):

- add `correct_diagnosis` for accuracy-aware bias scoring
- add explicit paired-difference metric built from `pair_group_id`

These are optional and should not be mixed into the current published proxy-metric contract.
